"""Synthetic storage faults and platform-send audit regressions."""

import json
import sqlite3
import time
from unittest.mock import AsyncMock

import pytest

from ..state import scope_id
from . import test_v0101_explicit_wake as wake
from .test_fx4 import payload
from .test_v0102_wake_warnings import adapter_event, delivery_hooks, env, pipeline

__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("failure", ["writer", "counter", "quiet", "entry"])
async def test_storage_failure_reply(env, monkeypatch, kind, failure):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    writer = None
    if failure == "writer":
        path = env.plugin.ledger.db.execute("PRAGMA database_list").fetchone()[2]
        writer = sqlite3.connect(path)
        writer.execute("BEGIN IMMEDIATE")
    elif failure in ("counter", "quiet"):
        monkeypatch.setattr(
            env.plugin.ledger,
            "abump" if failure == "counter" else "acooldown",
            AsyncMock(side_effect=sqlite3.OperationalError("synthetic failure")),
        )
    else:
        monkeypatch.setattr(
            env.plugin, "_evaluate", AsyncMock(side_effect=RuntimeError("synthetic"))
        )
    try:
        _, responses = await pipeline(env, event, adapter=True)
    finally:
        if writer:
            writer.rollback()
            writer.close()
    assert bot.transport.await_count == (0 if failure == "entry" else 1), payload(bot)
    assert len(responses) <= 1
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
    await env.plugin._context_failure_reply(event)
    assert bot.transport.await_count == (0 if failure == "entry" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
async def test_quiet_read_failure_preserves_rule(env, monkeypatch, kind):
    wake.configure(env)
    event, bot = adapter_event(kind)
    await env.plugin.ledger.acooldown(
        "room_quiet:" + scope_id(event.unified_msg_origin), time.time() + 60
    )
    monkeypatch.setattr(
        env.plugin.ledger,
        "acooldown",
        AsyncMock(side_effect=sqlite3.OperationalError("synthetic")),
    )
    await pipeline(env, event, adapter=True)
    assert bot.transport.await_count == 0


@pytest.mark.asyncio
async def test_each_platform_send_has_persistable_evidence(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")

    async def emit(ev):
        for text in ("synthetic one", "synthetic two"):
            await ev.bot.call_action(
                "send_msg",
                message=[{"type": "text", "data": {"text": text}}],
                group_id=987654321,
            )

    await pipeline(env, event, "[PASS]", adapter=True, native_send=emit)
    assert bot.transport.await_count == 1
    rows = list(env.plugin.audit_queue)
    for reason in ("send_attempt", "send_completed"):
        details = [d for _, r, d in rows if r == reason]
        assert [d["send_sequence"] for d in details] == [1]
        assert all(
            d["ingress_id"] == scope_id(str(event.message_obj.message_id))
            for d in details
        )
        assert all(
            d["sender_hash"] == scope_id(str(event.get_sender_id())) for d in details
        )
        if reason == "send_completed":
            assert [d["receipt_id"] for d in details] == ["42"]
        for detail in details:
            env.plugin.ledger.record(event.unified_msg_origin, reason, detail)
        stored = env.plugin.ledger.db.execute(
            "SELECT details FROM audit WHERE reason=? ORDER BY id", (reason,)
        ).fetchall()
        assert [json.loads(row[0])["send_sequence"] for row in stored] == [1]
    assert env.plugin.metrics.replies == 1


@pytest.mark.asyncio
async def test_unreadable_quiet_policy_does_not_send(env, monkeypatch):
    wake.configure(env)
    event, bot = adapter_event("at")
    for method in ("acooldown", "read_quiet"):
        monkeypatch.setattr(
            env.plugin.ledger,
            method,
            AsyncMock(side_effect=sqlite3.OperationalError("synthetic")),
        )
    with pytest.raises(sqlite3.OperationalError):
        await pipeline(env, event, adapter=True)
    assert bot.transport.await_count == 0
    assert event.is_stopped()
    assert any(
        reason == "quiet_policy_unavailable" for _, reason, _ in env.plugin.audit_queue
    )


@pytest.mark.asyncio
async def test_core_send_audits_actual_platform_receipt(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await pipeline(env, event, adapter=True)
    completed = [d for _, r, d in env.plugin.audit_queue if r == "send_completed"]
    assert bot.transport.await_count == len(completed) == 1
    assert completed[0]["receipt_id"] == "42"
