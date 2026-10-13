"""Read-only adapters for installed Stars. Do not invoke their event hooks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter

from .policy import pack_state


def instance(context, event, name):
    allowed = getattr(event, "plugins_name", None)
    for metadata in context.get_all_stars():
        if not getattr(metadata, "activated", True):
            continue
        module = str(getattr(metadata, "module_path", ""))
        actual = str(getattr(metadata, "name", ""))
        root_name = str(getattr(metadata, "root_dir_name", ""))
        if name not in (
            actual,
            root_name,
            module.split(".")[-1],
        ) and name not in module.split("."):
            continue
        if allowed is not None and "*" not in allowed and actual not in allowed:
            continue
        return getattr(metadata, "star_cls", None)
    return None


def conflicts(context, event):
    found = []
    for name in ("astrbot_plugin_jev_gate", "astrbot_plugin_wakepro"):
        plugin = instance(context, event, name)
        if plugin is None:
            continue
        if name.endswith("jev_gate"):
            cfg = getattr(plugin, "config", {})
            sessions = cfg.get("enabled_sessions", [])
            disabled = cfg.get("disabled_sessions", [])
            enabled = (
                event.unified_msg_origin not in disabled
                if cfg.get("session_mode") == "blacklist"
                else ("*" in sessions or event.unified_msg_origin in sessions)
            )
            if cfg.get("dry_run", True) or not enabled:
                continue
        else:
            cfg = getattr(plugin, "cfg", None)
            pipeline = getattr(cfg, "pipeline", None)
            if pipeline is not None:
                steps = getattr(pipeline, "steps", [])
                if not any(str(x).startswith(("wake", "debounce")) for x in steps):
                    continue
        found.append(name)
    config = context.get_config(umo=event.unified_msg_origin)
    if (
        config.get("provider_ltm_settings", {})
        .get("active_reply", {})
        .get("enable", False)
    ):
        found.append("astrbot_builtin_active_reply")
    return found


def native_busy(event):
    """Do not feed a separately admitted turn into Core's active follow-up."""
    from astrbot.core.pipeline.process_stage import follow_up

    runner = follow_up._ACTIVE_AGENT_RUNNERS.get(event.unified_msg_origin)
    if runner is None or runner.done():
        return False
    owner = getattr(
        getattr(getattr(runner, "run_context", None), "context", None), "event", None
    )
    return bool(
        owner is not None
        and owner is not event
        and not owner.get_extra("agent_stop_requested")
        and not owner.is_stopped()
    )


def fingerprint(conversation):
    return hashlib.sha256(
        str(getattr(conversation, "history", "")).encode()
    ).hexdigest()


def latest_reply(conversation):
    """Expose the newest completed reply even before public history catches up."""
    history = getattr(conversation, "history", [])
    try:
        history = json.loads(history) if isinstance(history, str) else history
    except (ValueError, TypeError):
        return ""
    if not isinstance(history, list) or not history:
        return ""
    last = history[-1]
    if (
        not isinstance(last, dict)
        or last.get("role") != "assistant"
        or last.get("tool_calls")
    ):
        return ""
    content = last.get("content", "")
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "output_text")
        )
    return content if isinstance(content, str) else ""


async def conversation(context, event):
    manager = context.conversation_manager
    cid = await manager.get_curr_conversation_id(event.unified_msg_origin)
    conv = (
        await manager.get_conversation(event.unified_msg_origin, cid) if cid else None
    )
    return str(cid or ""), conv


async def build_snapshot(context, event, room, target, config):
    cid, conv = await conversation(context, event)
    cfg = context.get_config(umo=event.unified_msg_origin)
    _, persona, _, _ = await context.persona_manager.resolve_selected_persona(
        umo=event.unified_msg_origin,
        conversation_persona_id=getattr(conv, "persona_id", None),
        platform_name=event.get_platform_name(),
        provider_settings=cfg.get("provider_settings", {}),
    )
    persona_text = str(
        config.get("persona_override") or (persona or {}).get("prompt") or ""
    )
    if not persona_text:
        persona_text = (
            "A quiet group participant; do not assume interests or relationships."
        )
    rows = list(room.rows)
    if room.cid and cid != room.cid:
        rows = rows[-1:]
    ca = instance(context, event, "astrbot_plugin_context_aware")
    window = max(4, min(100, int(config.get("context_message_limit", 40))))
    used_ca = False
    ca_status = "not_enabled"
    if ca and callable(getattr(ca, "get_recent_messages", None)):
        try:
            records = ca.get_recent_messages(event.unified_msg_origin, count=window)
            ca_status = "empty"
            if isinstance(records, list) and records:
                native_rows = rows
                rows = [dict(r) for r in records if isinstance(r, dict)]

                def identity(r):
                    return (
                        r.get("content"),
                        r.get("sender_name"),
                        bool(r.get("is_bot")),
                    )

                native_counts = Counter(identity(r) for r in native_rows)
                public_counts = Counter(identity(r) for r in rows)
                for record in rows:
                    match = next(
                        (
                            r
                            for r in reversed(native_rows)
                            if identity(r) == identity(record)
                            and native_counts[identity(r)] == 1
                            and public_counts[identity(r)] == 1
                        ),
                        None,
                    )
                    if match:
                        record.update(
                            {
                                k: match[k]
                                for k in ("sender_id", "message_id", "reply_to")
                                if k in match
                            }
                        )
                used_ca = bool(rows)
                ca_status = "available"
        except Exception:
            ca_status = "unavailable"  # Never call a private store to recover.
    if not used_ca and conv:
        try:
            history = getattr(conv, "history", [])
            history = json.loads(history) if isinstance(history, str) else history
            restored = []
            for item in (history or [])[-window:]:
                if not isinstance(item, dict) or item.get("role") not in (
                    "user",
                    "assistant",
                ):
                    continue
                content = item.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        str(p.get("text", ""))
                        for p in content
                        if isinstance(p, dict)
                        and p.get("type") in ("text", "input_text", "output_text")
                    )
                if isinstance(content, str) and content.strip():
                    restored.append(
                        {
                            "sender_name": "Bot"
                            if item["role"] == "assistant"
                            else "Native conversation participant (identity unknown)",
                            "content": content,
                            "is_bot": item["role"] == "assistant",
                            "source": "native_history",
                        }
                    )
            existing = {(r.get("is_bot"), r.get("content")) for r in rows}
            rows = [
                r for r in restored if (r["is_bot"], r["content"]) not in existing
            ] + rows
        except (ValueError, TypeError):
            pass
    source_ids = set(
        target.get("source_message_ids") or [target.get("message_id", "")]
    ) - {""}
    rows = [
        r
        for r in rows
        if not r.get("message_id") or r.get("message_id") not in source_ids
    ]
    if (
        rows
        and not rows[-1].get("message_id")
        and rows[-1].get("content") == target.get("content")
        and rows[-1].get("sender_name") == target.get("sender")
    ):
        rows = rows[:-1]
    life = {}
    if config.get("read_life_schedule", True):
        scheduler = instance(context, event, "astrbot_plugin_life_scheduler")
        if scheduler and callable(getattr(scheduler, "get_life_context", None)):
            try:
                life = await asyncio.wait_for(
                    scheduler.get_life_context(allow_generate=False), 1
                )
            except Exception:
                pass
    activity = {
        "last_delivered_reply": room.last_text,
        "last_reply_to": room.last_sender,
        "bot_id": str(event.get_self_id()),
        "target_sender_id": str(event.get_sender_id()),
        "observed_at_unix": time.time(),
        "last_reply_seconds_ago": round(time.monotonic() - room.last_sent, 1)
        if room.last_sent
        else None,
        "history_source": "context_aware" if used_ca else "native_history_and_local",
        "context_aware_status": ca_status,
        "generation": room.generation,
    }
    if room.cid and cid != room.cid:
        activity["last_delivered_reply"] = ""
        activity["last_reply_to"] = ""
    return (
        pack_state(
            persona_text,
            rows,
            target,
            activity,
            life,
            max_bytes=int(config.get("request_byte_budget", 192000)),
            config=config,
        ),
        cid,
        fingerprint(conv),
    )
