"""Runtime state, daily aggregates and bounded audit details."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from .burst import BurstBook


def audit_preview(text):
    text = str(text or "")
    text = re.sub(r"(?i)(?:sk-|apikey_)[A-Za-z0-9_-]+", "[redacted]", text)
    # Discard the tail at a credential field: quoted/multiline/structured values
    # must not leave fragments in the preview, even when the input is not JSON.
    text = re.sub(
        r"""(?is)["']?(?:authorization|api[_-]?key|\bkey|token|password|cookie)["']?\s*[:= ]+.*""",
        "[redacted]",
        text,
    )
    text = re.sub(r"(?i)bearer\s+[^\s,，;；]+", "[redacted]", text)
    return text[:50]


def scope_id(umo: str) -> str:
    return hashlib.sha256(umo.encode()).hexdigest()[:24]


@dataclass
class Room:
    rows: deque = field(default_factory=lambda: deque(maxlen=64))
    seen: deque = field(default_factory=lambda: deque(maxlen=100))
    generation: int = 0
    cid: str = ""
    pending: str = ""
    pending_since: float = 0
    evaluated_at: float = -1e12
    last_sent: float = 0
    last_sender: str = ""
    last_text: str = ""
    delivery_backoff_until: float = 0
    sent_times: deque = field(default_factory=lambda: deque(maxlen=30))
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    phase: str = "idle"
    bursts: BurstBook = field(default_factory=BurstBook)
    ingress_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    quiet_until: float = 0
    quiet_loaded: bool = False
    quiet_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    quiet_senders: dict = field(default_factory=dict)
    direct_wakes: dict = field(default_factory=dict)
    last_judged_text: str = ""
    last_judged_at: float = 0

    def notify(self):
        signal, self.changed = self.changed, asyncio.Event()
        signal.set()

    def claim(self, ticket: str, now: float):
        self.pending, self.pending_since, self.phase = ticket, now, "judging"
        self.notify()

    def observe(self, row: dict, message_id: str) -> bool:
        if message_id in self.seen:
            return False
        self.seen.append(message_id)
        row = dict(row)
        row["generation"] = self.generation + 1
        self.rows.append(row)
        # Bound stored text by bytes, not destructive per-message previews.
        while (
            len(self.rows) > 1
            and sum(len(json.dumps(r, ensure_ascii=False).encode()) for r in self.rows)
            > 262144
        ):
            self.rows.popleft()
        self.generation += 1
        self.notify()
        return True

    def busy(self, now: float, timeout=120) -> bool:
        # Only the owner/lease can release a ticket; never expire a 180s request
        # through this old 120s housekeeping default.
        return bool(self.pending)

    def release(self, ticket: str):
        if self.pending == ticket:
            self.pending = ""
            self.phase = "idle"
            self.notify()


class RoomBook:
    def __init__(self, capacity=256):
        self.capacity = capacity
        self.rooms = OrderedDict()

    def get(self, umo: str) -> Room | None:
        if umo in self.rooms:
            self.rooms.move_to_end(umo)
            return self.rooms[umo]
        if len(self.rooms) >= self.capacity:
            free = next(
                (
                    key
                    for key, room in self.rooms.items()
                    if not room.busy(time.monotonic())
                ),
                None,
            )
            if free is None:
                return None
            del self.rooms[free]
        room = Room()
        self.rooms[umo] = room
        return room


class Ledger:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.closed = False
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA busy_timeout=500")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS counts(day TEXT, scope TEXT, kind TEXT, n INTEGER, PRIMARY KEY(day,scope,kind));
        CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, ts REAL, scope TEXT, reason TEXT, details TEXT);
        CREATE TABLE IF NOT EXISTS cooldowns(scope TEXT PRIMARY KEY, until REAL);
        CREATE TABLE IF NOT EXISTS daily_summary(
            day TEXT NOT NULL, scope TEXT NOT NULL, reason TEXT NOT NULL,
            n INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(day,scope,reason)
        );
        CREATE TABLE IF NOT EXISTS jev_migrations(version INTEGER PRIMARY KEY);
        """)
        with self.db:
            if not self.db.execute(
                "SELECT 1 FROM jev_migrations WHERE version=1"
            ).fetchone():
                for ts, scope, reason in self.db.execute(
                    "SELECT ts,scope,reason FROM audit"
                ).fetchall():
                    self._summarize(ts, scope, reason)
                self.db.execute("INSERT INTO jev_migrations VALUES(1)")
        self.db.commit()

    @staticmethod
    def day():
        return datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()

    def count(self, umo, kind):
        with self.lock:
            row = self.db.execute(
                "SELECT n FROM counts WHERE day=? AND scope=? AND kind=?",
                (self.day(), scope_id(umo), kind),
            ).fetchone()
        return row[0] if row else 0

    def bump(self, umo, kind):
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO counts VALUES(?,?,?,1) ON CONFLICT(day,scope,kind) DO UPDATE SET n=n+1",
                (self.day(), scope_id(umo), kind),
            )

    def record(self, umo, reason, details=None):
        # Explicit allowlist; preview is the sole bounded message-content field.
        allowed = {
            k: v
            for k, v in (details or {}).items()
            if k
            in (
                "channel",
                "key_id",
                "model",
                "values",
                "mode",
                "status",
                "truncated",
                "turn_id",
                "elapsed_ms",
                "stage_ms",
                "trigger_preview",
                "ingress_id",
                "sender_hash",
                "send_sequence",
                "receipt_id",
                "wake_kind",
                "sender_is_admin",
                "delivery_state",
            )
        }
        if "trigger_preview" in allowed:
            allowed["trigger_preview"] = audit_preview(allowed["trigger_preview"])
        stamp = time.time()
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO audit(ts,scope,reason,details) VALUES(?,?,?,?)",
                (
                    stamp,
                    scope_id(umo),
                    reason,
                    json.dumps(allowed, ensure_ascii=False),
                ),
            )
            self.db.execute(
                "DELETE FROM audit WHERE id <= (SELECT COALESCE(MAX(id),0)-5000 FROM audit)"
            )
            self.db.execute(
                "DELETE FROM counts WHERE day < date(?, '-30 days')", (self.day(),)
            )
            self._summarize(stamp, scope_id(umo), reason)

    def _summarize(self, stamp, scope, reason):
        day = (
            datetime.fromtimestamp(stamp, ZoneInfo("Asia/Shanghai")).date().isoformat()
        )
        self.db.execute(
            "INSERT INTO daily_summary(day,scope,reason,n) VALUES(?,?,?,1) "
            "ON CONFLICT(day,scope,reason) DO UPDATE SET n=n+1",
            (day, scope, reason),
        )

    def recent(self, umo):
        with self.lock:
            return self.db.execute(
                "SELECT reason FROM audit WHERE scope=? ORDER BY id DESC LIMIT 5",
                (scope_id(umo),),
            ).fetchall()

    async def acount(self, umo, kind):
        return await asyncio.to_thread(self.count, umo, kind)

    async def abump(self, umo, kind):
        await asyncio.to_thread(self.bump, umo, kind)

    async def arecord(self, umo, reason, details=None):
        await asyncio.to_thread(self.record, umo, reason, details)

    async def arecent(self, umo):
        return await asyncio.to_thread(self.recent, umo)

    def cooldown(self, scope, until=None):
        with self.lock, self.db:
            if until is not None:
                self.db.execute(
                    "INSERT INTO cooldowns VALUES(?,?) ON CONFLICT(scope) DO UPDATE SET until=excluded.until",
                    (scope, until),
                )
            row = self.db.execute(
                "SELECT until FROM cooldowns WHERE scope=?", (scope,)
            ).fetchone()
            return row[0] if row else 0

    async def acooldown(self, scope, until=None):
        return await asyncio.to_thread(self.cooldown, scope, until)

    async def read_quiet(self, scope):
        """Retry policy reads independently of a failing accounting connection."""

        def read():
            from pathlib import Path

            uri = Path(self.path).resolve().as_uri() + "?mode=ro"
            db = sqlite3.connect(uri, uri=True, timeout=0.5)
            try:
                row = db.execute(
                    "SELECT until FROM cooldowns WHERE scope=?", (scope,)
                ).fetchone()
                return row[0] if row else 0
            finally:
                db.close()

        return await asyncio.to_thread(read)

    def close(self):
        with self.lock:
            if not self.closed:
                self.db.close()
                self.closed = True

    async def aclose(self):
        await asyncio.to_thread(self.close)
