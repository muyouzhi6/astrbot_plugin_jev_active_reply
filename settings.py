"""Grouped dashboard settings with lossless legacy-key compatibility."""

from collections.abc import MutableMapping
from copy import deepcopy
import json
import os
from pathlib import Path
from datetime import datetime
import tempfile

try:
    from astrbot.api import logger
except Exception:  # pragma: no cover - only outside AstrBot
    import logging

    logger = logging.getLogger("astrbot")


GROUPS = {
    "judgment": (
        "decision_mode",
        "second_review",
        "natural_reply_threshold",
        "question_overrides",
        "uncertain_adjudication",
        "address_threshold",
        "continuation_threshold",
        "response_value_threshold",
        "intrusion_threshold",
        "join_threshold_offset",
        "join_intrusion_ceiling",
        "review_appropriate_threshold",
        "review_redundancy_ceiling",
        "image_decision_mode",
        "max_autonomous_reply_chars",
    ),
    "timing": (
        "debounce_seconds",
        "burst_merge_enabled",
        "burst_max_messages",
        "candidate_wait_seconds",
        "continuation_min_interval",
        "request_lifetime_seconds",
        "evaluation_interval",
        "min_reply_interval",
        "continuation_seconds",
        "direct_wake_interval",
        "local_prefilter_enabled",
        "prefilter_cooldown_seconds",
    ),
    "context": (
        "persona_override",
        "read_life_schedule",
        "interest_topics",
        "context_mode",
        "context_message_limit",
        "overflow_evidence_policy",
        "overflow_summary_timeout",
        "protected_recent_messages",
        "context_safety_margin",
        "request_byte_budget",
    ),
    "guards": (
        "admin_wake_names",
        "reply_to_bot_wakes",
        "skip_reply_to_others",
        "blocked_targets",
        "guard_exempt_targets",
        "block_official_qq_bots",
        "block_repeated_input",
        "blocked_keywords",
        "block_builtin_commands",
        "blocked_commands",
        "block_prefixed_commands",
        "block_prefixed_llm",
        "skip_command_prefixes",
        "respect_quiet_requests",
        "quiet_request_seconds",
        "other_bot_ids",
        "strict_conflicts",
        "proactive_tool_allowlist",
    ),
    "network": (
        "allow_channel_fallback",
        "timeout_seconds",
        "decision_queue_seconds",
        "review_queue_seconds",
        "retry_explicit_rejection",
        "daily_evaluation_limit",
        "daily_reply_limit",
    ),
}
KEY_GROUP = {key: group for group, keys in GROUPS.items() for key in keys}


# Channel fallback is deliberately outside the 0.9.0 migration.
MIGRATED_KEYS = set(KEY_GROUP) - {"allow_channel_fallback"}


def migrate_groups(raw):
    changed = False
    advanced = raw.setdefault("advanced", {})
    for key in MIGRATED_KEYS:
        section = advanced.setdefault(KEY_GROUP[key], {})
        if key in raw:
            if section.get(key) is None:
                section[key] = deepcopy(raw[key])
            del raw[key]
            changed = True
    return changed


RELAY_MODES = ("typesafe", "nanbei", "custom")
RELAY_CONFIG_VERSION = 1


def migrate_relay(raw):
    """0.10.1: one selector (relay_mode) for the three Jev-format services.

    Old ``relay_mode=jev`` (or absent) takes the old ``primary_channel``; the
    removed ``astrbot`` provider and OpenAI ``custom`` relay fall back to Nanbei.
    """
    if raw.get("_relay_config_version", 0) >= RELAY_CONFIG_VERSION:
        return False
    primary = raw.get("primary_channel")
    if primary in ("mindshub", "mindshub_air"):
        primary = "typesafe" if (raw.get("typesafe") or {}).get("keys") else "nanbei"
    if primary not in ("typesafe", "nanbei"):
        primary = "nanbei"
    mode = raw.get("relay_mode")
    if mode in (None, "", "jev"):
        new = primary
    elif mode in ("typesafe", "nanbei"):
        new = mode
    else:
        new = "nanbei"
        logger.warning(
            "[JevActive] relay_mode=%s is no longer supported (Jev judgment only); "
            "switched to 南北绿豆. Re-select the judgment service if needed.",
            str(mode)[:20],
        )
    relay = raw.get("relay")
    if isinstance(relay, dict) and any(relay.get(k) for k in ("base_url", "api_key")):
        # The old relay section was OpenAI chat/completions; never send Jev
        # systemone requests to it. The pre-migration file backup keeps it.
        relay.update(base_url="", api_key="", model="jev-latest")
        logger.warning(
            "[JevActive] cleared the old OpenAI-format custom relay settings; "
            "custom relays must now speak the Jev systemone protocol."
        )
    raw["relay_mode"] = new
    raw.pop("primary_channel", None)
    raw["_relay_config_version"] = RELAY_CONFIG_VERSION
    return True


def migrate_config_file(path):
    """Run before AstrBotConfig removes keys absent from the new schema."""
    path = Path(path)
    if not path.is_file():
        return
    original = path.read_bytes()
    raw = json.loads(original.decode("utf-8-sig"))
    changed = migrate_groups(raw)
    if not migrate_relay(raw) and not changed:
        return
    backup = path.with_name(
        path.name + datetime.now().strftime(".%Y%m%d-%H%M%S-%f.bak")
    )
    with os.fdopen(
        os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
    ) as handle:
        handle.write(original)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".jev-migrate-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(raw, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ConfigView(MutableMapping):
    """Keep one effective value; null grouped entries inherit legacy values once."""

    def __init__(self, raw):
        self.raw = raw
        advanced = raw.get("advanced")
        changed = False
        changed = migrate_relay(raw) or changed
        for obsolete in ("mindshub", "mindshub_air"):
            if obsolete in raw:
                del raw[obsolete]
                changed = True
        if isinstance(advanced, dict):
            for group, keys in GROUPS.items():
                section = advanced.setdefault(group, {})
                for key in keys:
                    if section.get(key) is None and key in raw:
                        section[key] = deepcopy(raw[key])
                        changed = True
        if raw.get("_product_config_version", 0) < 1:
            # Retire the old onboarding trap and artificial consumption caps.
            raw["dry_run"] = False
            raw["allow_channel_fallback"] = False
            for key in ("daily_evaluation_limit", "daily_reply_limit"):
                raw[key] = 0
            if isinstance(advanced, dict):
                network = advanced.setdefault("network", {})
                network.update(
                    allow_channel_fallback=False,
                    daily_evaluation_limit=0,
                    daily_reply_limit=0,
                )
            for kind in ("nanbei", "typesafe"):
                channel = raw.get(kind)
                if isinstance(channel, dict):
                    channel.update(
                        quota_rpm=0, key_rpm=0, daily_request_limit=0, concurrency=8
                    )
            raw["_product_config_version"] = 1
            changed = True
        changed = migrate_groups(raw) or changed
        if changed:
            save = getattr(raw, "save_config", None)
            if callable(save):
                save()

    def __getitem__(self, key):
        group = KEY_GROUP.get(key)
        if group:
            section = self.raw.get("advanced", {}).get(group, {})
            if section.get(key) is not None:
                return section[key]
            if key in MIGRATED_KEYS:
                raise KeyError(key)
        return self.raw[key]

    def __setitem__(self, key, value):
        group = KEY_GROUP.get(key)
        if group and "advanced" in self.raw:
            self.raw["advanced"].setdefault(group, {})[key] = value
        else:
            self.raw[key] = value

    def __delitem__(self, key):
        group = KEY_GROUP.get(key)
        if group:
            self.raw.get("advanced", {}).get(group, {}).pop(key, None)
        if key in MIGRATED_KEYS:
            return
        del self.raw[key]

    def __iter__(self):
        keys = dict.fromkeys(self.raw)
        for group, section in self.raw.get("advanced", {}).items():
            if group in GROUPS and isinstance(section, dict):
                keys.update({key: None for key in section if key in GROUPS[group]})
        return iter(keys)

    def __len__(self):
        return sum(1 for _ in self)

    def save_config(self):
        save = getattr(self.raw, "save_config", None)
        if callable(save):
            return save()
