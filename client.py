"""Typed decision adapters and bounded, non-retrying key rotation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import aiohttp

from .policy import fit_state

ENDPOINTS = {
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "jev-latest"),
    "nanbei": ("https://hajimi.165201.xyz/v1/systemone", "jev-latest"),
}
# The only judgment services: official, Nanbei, or a custom Jev-format relay.
CHANNELS = ("typesafe", "nanbei", "custom")


def channel_configured(config, kind: str) -> bool:
    if kind == "custom":
        relay = config.get("relay", {}) or {}
        return bool(
            str(relay.get("base_url") or "").strip()
            and str(relay.get("api_key") or "").strip()
        )
    keys = (config.get(kind, {}) or {}).get("keys", [])
    return isinstance(keys, list) and any(str(k).strip() for k in keys)


def normalize_relay_url(base_url: str) -> str:
    """Normalize a custom Jev-format relay base URL to its /v1/systemone endpoint."""
    value = str(base_url or "").strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError("invalid relay base URL") from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or any(c.isspace() or ord(c) < 32 for c in value)
        or (port is not None and port == 0)
    ):
        raise ValueError("relay base URL must be HTTPS without credentials/query")
    path = parsed.path.rstrip("/")
    if path.endswith("/systemone"):
        path = path[: -len("/systemone")]
    if path.endswith("/v1"):
        path = path[:-3]
    path = path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}/v1/systemone"


class DecisionError(Exception):
    """Safe public error: never contains headers, state, key, or response body."""

    def __init__(self, code: str, status: int = 0):
        self.code, self.status = code, status
        super().__init__(f"{code} (HTTP {status})" if status else code)


def retry_delay(value: str | None, wall_time: float | None = None) -> float:
    try:
        result = float(value or "60")
    except ValueError:
        try:
            stamp = parsedate_to_datetime(value or "")
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            result = stamp.timestamp() - (wall_time or time.time())
        except (ValueError, TypeError, OverflowError):
            result = 60
    return min(86400, max(1, result if math.isfinite(result) else 60))


@dataclass(frozen=True)
class Adapter:
    kind: str
    endpoint: str
    model: str

    @classmethod
    def create(cls, kind: str, config: dict) -> Adapter:
        if kind == "custom":
            endpoint = normalize_relay_url(config.get("base_url", ""))
            model = str(config.get("model") or "").strip() or "jev-latest"
            return cls(kind, endpoint, model)
        if kind not in ENDPOINTS:
            raise ValueError("unknown channel type")
        endpoint, model = ENDPOINTS[kind]
        endpoint = str(config.get("endpoint") or endpoint).strip()
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("endpoint must be an HTTPS URL without credentials/query")
        if parsed.path.rstrip("/") != urlsplit(ENDPOINTS[kind][0]).path:
            raise ValueError("endpoint path does not match the selected adapter")
        return cls(kind, endpoint, str(config.get("model") or model).strip())

    def encode(self, state: dict, questions: dict) -> dict:
        # Official, Nanbei and custom relays all speak the same systemone contract.
        return {"model": self.model, "state": state, "questions": questions}

    def decode(self, body: object, questions: dict) -> dict:
        if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
            raise DecisionError("invalid_response")
        values = {}
        for name, question in questions.items():
            answer = body["answers"].get(name)
            if not isinstance(answer, dict) or answer.get("type") != question["type"]:
                raise DecisionError("missing_answer")
            value = answer.get("noul")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise DecisionError("invalid_probability")
            values[name] = float(value)
        return {
            "values": values,
            "model": str(body.get("model", ""))[:100],
            "usage": body.get("usage", {}),
        }

    def classify(
        self, status: int, body: object, headers: dict
    ) -> tuple[str, float, bool]:
        error = body.get("error", {}) if isinstance(body, dict) else {}
        if not isinstance(error, dict):
            error = {}
        code = error.get("code") or error.get("type")
        if status == 403 and (not isinstance(body, dict) or code == "edge_blocked"):
            return "edge_rejected", 60, True
        if self.kind == "nanbei" and status == 403:
            return "site_access_or_balance", 60, True
        if status in (401, 403):
            return "auth_disabled", math.inf, False
        if (
            status == 402
            or headers.get("x-should-retry", "").lower() == "false"
            or code
            in (
                "wallet_empty",
                "included_allowance_exhausted",
                "free_air_daily_spend_fuse_exceeded",
            )
        ):
            return "funding_cooldown", 3600, True
        if status in (429, 529):
            return (
                "rate_limited" if status == 429 else "overloaded",
                retry_delay(headers.get("retry-after")),
                True,
            )
        if status in (400, 404, 422):
            return "configuration_error", 300, True
        if status >= 500:
            return "upstream_error", 30, True
        return "http_error", 30, False


@dataclass
class KeySlot:
    secret: str = field(repr=False)
    ident: str = ""
    requests: deque = field(default_factory=lambda: deque(maxlen=10000))
    active: int = 0
    until: float = 0


class KeyPool:
    """One quota group; all keys share its organization/account budget."""

    def __init__(self, kind: str, config: dict, clock=time.monotonic):
        self.adapter = Adapter.create(kind, config)
        raw = config.get("keys", [])
        if not isinstance(raw, list) or any(not isinstance(k, str) for k in raw):
            raise ValueError("keys must be a list of strings")
        keys = list(dict.fromkeys(k.strip() for k in raw if k.strip()))
        if any("\n" in k or "\r" in k for k in keys):
            raise ValueError("key contains newline")
        if kind == "nanbei" and any(k.startswith(("apikey_", "mdb_")) for k in keys):
            raise ValueError(
                "Nanbei requires a site API key, not an upstream credential"
            )
        if kind == "typesafe" and any(k.startswith(("sk-", "mdb_")) for k in keys):
            raise ValueError(
                "TypeSafe requires an official API key, not a relay credential"
            )
        self.keys = [
            KeySlot(k, hashlib.sha256(k.encode()).hexdigest()[:8]) for k in keys
        ]
        self.rpm = max(0, min(10000, int(config.get("quota_rpm", 0))))
        self.key_rpm = max(0, min(10000, int(config.get("key_rpm", 0))))
        self.concurrency = max(1, min(20, int(config.get("concurrency", 8))))
        self.requests = deque(maxlen=10000)
        self.active = 0
        self.until = 0.0
        self.cursor = 0
        self.clock = clock
        self.lock = asyncio.Lock()
        self.ledger = None
        self.daily_limit = max(0, int(config.get("daily_request_limit", 0)))
        self.scope = "channel:" + kind + str(config.get("_quota_scope", ""))
        self.changed = None
        self.health_loaded = False

    async def restore_health(self):
        if not self.health_loaded:
            if self.ledger and hasattr(self.ledger, "acooldown"):
                wall_until = await self.ledger.acooldown(self.scope)
                self.until = max(
                    self.until, self.clock() + max(0, wall_until - time.time())
                )
            self.health_loaded = True

    async def acquire(self) -> KeySlot | None:
        async with self.lock:
            await self.restore_health()
            now = self.clock()
            while self.requests and self.requests[0] <= now - 60:
                self.requests.popleft()
            if (
                now < self.until
                or self.active >= self.concurrency
                or (self.rpm and len(self.requests) >= self.rpm)
            ):
                return None
            if (
                self.ledger
                and self.daily_limit
                and await self._count() >= self.daily_limit
            ):
                return None
            for offset in range(len(self.keys)):
                index = (self.cursor + offset) % len(self.keys)
                key = self.keys[index]
                while key.requests and key.requests[0] <= now - 60:
                    key.requests.popleft()
                if now < key.until or (
                    self.key_rpm and len(key.requests) >= self.key_rpm
                ):
                    continue
                # Commit quota before taking an in-memory lease. A failed or
                # cancelled write can overcount an attempt, never leak a slot.
                if self.ledger:
                    await self._bump()
                key.active += 1
                self.active += 1
                key.requests.append(now)
                self.requests.append(now)
                self.cursor = (index + 1) % len(self.keys)
                return key
        return None

    async def _count(self):
        if hasattr(self.ledger, "acount"):
            return await self.ledger.acount(self.scope, "requests")
        return await asyncio.to_thread(self.ledger.count, self.scope, "requests")

    async def _bump(self):
        if hasattr(self.ledger, "abump"):
            await self.ledger.abump(self.scope, "requests")
        else:
            await asyncio.to_thread(self.ledger.bump, self.scope, "requests")

    async def release(self, key: KeySlot, cooldown=0.0, shared=False):
        async with self.lock:
            key.active = max(0, key.active - 1)
            self.active = max(0, self.active - 1)
            key.until = max(key.until, self.clock() + cooldown)
            if shared:
                self.until = max(self.until, self.clock() + cooldown)
        if self.changed:
            self.changed()
        if shared and cooldown and self.ledger and hasattr(self.ledger, "acooldown"):
            try:
                await self.ledger.acooldown(self.scope, time.time() + cooldown)
            except Exception:
                pass  # The in-memory cooldown and released lease remain authoritative.

    def status(self):
        return {
            "channel": self.adapter.kind,
            "keys": len(self.keys),
            "disabled_keys": sum(math.isinf(k.until) for k in self.keys),
            "in_flight": self.active,
            "window_used": sum(t > self.clock() - 60 for t in self.requests),
            "quota_rpm": self.rpm,
            "cooldown_seconds": max(0, round(self.until - self.clock())),
        }


class JevClient:
    def __init__(self, config: dict, transport=None, clock=time.monotonic):
        self.config = config
        primary = config.get("relay_mode")
        if primary in (None, "", "jev"):
            # Unmigrated dicts: the old Jev mode used primary_channel.
            primary = config.get("primary_channel") or "nanbei"
        if primary not in CHANNELS:
            raise ValueError("invalid relay_mode")
        self.order = [primary]
        if config.get("allow_channel_fallback", False):
            # Other Jev-format channels are optional backups, only when configured.
            self.order += [
                k for k in CHANNELS if k != primary and channel_configured(config, k)
            ]
        self.pools = {}
        self.config_errors = []
        for kind in self.order:
            if kind == "custom":
                relay = config.get("relay", {}) or {}
                key = str(relay.get("api_key") or "").strip()
                if channel_configured(config, kind):
                    try:
                        self.pools[kind] = KeyPool(
                            kind,
                            {
                                **relay,
                                "keys": [key],
                                "concurrency": relay.get("concurrency", 4),
                            },
                            clock,
                        )
                    except ValueError:
                        # An invalid custom URL must not stop the plugin from loading;
                        # the channel just stays unavailable and status says so.
                        self.config_errors.append(kind)
                continue
            cfg = config.get(kind, {})
            groups = cfg.get("quota_group_ids", [])
            keys = cfg.get("keys", [])
            if not groups:
                self.pools[kind] = KeyPool(kind, cfg, clock)
                continue
            if len(groups) != len(keys) or any(
                not isinstance(g, str) or not g.strip() for g in groups
            ):
                raise ValueError("quota_group_ids must match keys one-for-one")
            grouped, assignments = {}, {}
            for key, group in zip(keys, groups):
                if not isinstance(key, str):
                    raise ValueError("keys must be strings")
                key, group = key.strip(), group.strip()
                if key in assignments and assignments[key] != group:
                    raise ValueError("one key cannot belong to multiple quota groups")
                assignments[key] = group
                grouped.setdefault(group, []).append(key)
            for group, group_keys in grouped.items():
                ident = hashlib.sha256(group.encode()).hexdigest()[:12]
                self.pools[kind + ":" + ident] = KeyPool(
                    kind,
                    {**cfg, "keys": group_keys, "_quota_scope": ":" + ident},
                    clock,
                )
        self.timeout = min(30, max(1, float(config.get("timeout_seconds", 8))))
        self.max_body = min(
            524288, max(8000, int(config.get("request_byte_budget", 192000)))
        )
        self.transport = transport or self._post
        self.session = None
        self.closed = False
        self.waiters = []
        self.sequence = 0
        self.revision = 0
        self.capacity_changed = asyncio.Event()
        self.group_cursor = 0
        for pool in self.pools.values():
            pool.changed = self._notify

    def _notify(self):
        self.revision += 1
        self.capacity_changed.set()

    async def review_ready(self):
        for pool in self.pools.values():
            await pool.restore_health()
            if pool.adapter.kind not in self.order or pool.until > pool.clock():
                continue
            if not any(not math.isinf(key.until) for key in pool.keys):
                continue
            if (
                pool.ledger
                and pool.daily_limit
                and await pool._count() >= pool.daily_limit
            ):
                continue
            return True
        return False

    async def reset_cooldowns(self):
        for pool in self.pools.values():
            async with pool.lock:
                if pool.ledger and hasattr(pool.ledger, "acooldown"):
                    await pool.ledger.acooldown(pool.scope, 0)
                pool.until = 0
                pool.health_loaded = True
                for key in pool.keys:
                    if not math.isinf(key.until):
                        key.until = 0
        self._notify()

    async def _reserve(self, review=False):
        if self.closed:
            raise DecisionError("client_closed")
        if len(self.waiters) >= 64:
            raise DecisionError("queue_full")
        self.sequence += 1
        ticket = (0 if review else 1, self.sequence)
        self.waiters.append(ticket)
        deadline = time.monotonic() + min(
            10,
            max(
                0,
                float(
                    self.config.get(
                        "review_queue_seconds" if review else "decision_queue_seconds",
                        4 if review else 1,
                    )
                ),
            ),
        )
        try:
            while not self.closed:
                version = self.revision
                if ticket == min(self.waiters):
                    for kind in self.order:
                        candidates = [
                            p for p in self.pools.values() if p.adapter.kind == kind
                        ]
                        if candidates:
                            offset = self.group_cursor % len(candidates)
                            candidates = candidates[offset:] + candidates[:offset]
                        for pool in candidates:
                            key = await pool.acquire()
                            if key is not None:
                                self.group_cursor += 1
                                return pool, key
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if version == self.revision:
                    self.capacity_changed.clear()
                # Expiry can release a minute quota without an in-flight completion.
                delays = [remaining]
                for pool in self.pools.values():
                    now = pool.clock()
                    if pool.until > now:
                        delays.append(pool.until - now)
                    if pool.rpm and pool.requests and len(pool.requests) >= pool.rpm:
                        delays.append(pool.requests[0] + 60 - now)
                    for key in pool.keys:
                        if math.isfinite(key.until) and key.until > now:
                            delays.append(key.until - now)
                        if (
                            pool.key_rpm
                            and key.requests
                            and len(key.requests) >= pool.key_rpm
                        ):
                            delays.append(key.requests[0] + 60 - now)
                try:
                    await asyncio.wait_for(
                        self.capacity_changed.wait(), max(0.01, min(delays))
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            self.waiters.remove(ticket)
            self._notify()
        raise DecisionError("client_closed" if self.closed else "no_capacity")

    async def _post(self, endpoint, key, payload):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                trust_env=False,
                json_serialize=lambda value: json.dumps(value, ensure_ascii=False),
            )
        async with self.session.post(
            endpoint,
            json=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "User-Agent": "JevActiveReply/0.10.4",
            },
            allow_redirects=False,
        ) as response:
            raw = bytearray()
            async for chunk in response.content.iter_chunked(16384):
                raw.extend(chunk)
                if len(raw) > 262144:
                    raise DecisionError("response_too_large")
            try:
                body = json.loads(raw)
            except (ValueError, UnicodeError):
                body = None
            return (
                response.status,
                {k.lower(): v for k, v in response.headers.items()},
                body,
            )

    async def evaluate(self, state: dict, questions: dict):
        try:
            state = fit_state(state, questions, self.config)
        except ValueError:
            raise DecisionError("protected_context_exceeds_safety_budget") from None
        review = "candidate_reply" in state
        attempts = 2 if self.config.get("retry_explicit_rejection", False) else 1
        for attempt in range(attempts):
            # Never retry timeouts, disconnects, malformed successes or ambiguous 5xx.
            wire_size = len(
                json.dumps(
                    {"model": "jev-latest", "state": state, "questions": questions},
                    ensure_ascii=False,
                ).encode()
            )
            if wire_size > self.max_body:
                raise DecisionError("request_budget_exceeded")
            try:
                pool, key = await self._reserve(review=review)
            except DecisionError:
                raise
            except Exception:
                raise DecisionError("quota_storage_unavailable") from None
            payload = pool.adapter.encode(state, questions)
            cooldown, shared = 0.0, False
            retry = False
            try:
                if (
                    len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                    > self.max_body
                ):
                    raise DecisionError("request_budget_exceeded")
                status, headers, body = await asyncio.wait_for(
                    self.transport(pool.adapter.endpoint, key.secret, payload),
                    self.timeout,
                )
                if not 200 <= status < 300:
                    code, cooldown, shared = pool.adapter.classify(
                        status, body, headers
                    )
                    if status in (401, 403, 429, 529) and attempt + 1 < attempts:
                        retry = True
                    else:
                        raise DecisionError(code, status)
                if retry:
                    continue
                result = pool.adapter.decode(body, questions)
                return {**result, "channel": pool.adapter.kind, "key_id": key.ident}
            except (asyncio.TimeoutError, aiohttp.ClientError):
                cooldown = 15
                raise DecisionError("transport_uncertain") from None
            except asyncio.CancelledError:
                cooldown = 0
                raise
            except DecisionError:
                raise
            except Exception:
                if pool.adapter.kind != "custom":
                    raise
                # A third-party relay must never leak its exception text.
                cooldown = 15
                raise DecisionError("transport_uncertain") from None
            finally:
                await pool.release(key, cooldown, shared)
        raise DecisionError("no_capacity")

    async def close(self):
        self.closed = True
        self._notify()
        if self.session:
            await self.session.close()
