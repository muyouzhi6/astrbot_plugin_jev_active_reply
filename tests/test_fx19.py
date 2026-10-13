"""Release-19 regressions using Core stages and a mocked OneBot transport."""

import asyncio
import json

import pytest

from .. import bridge, silence
from ..main import MARK
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import (
    adapter_event,
    delivery_hooks,
    env,
    pipeline,
    platform_payload,
)

__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("flag", ["agent_stop_requested", "_suanle_silent_requested"])
async def test_external_stop_during_generation_still_delivers(env, kind, flag):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        pipeline(
            env,
            event,
            "Synthetic generated answer.",
            adapter=True,
            generation_gate=(entered, release),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 4)
        event.set_extra(flag, True)
        event.stop_event()
    finally:
        release.set()
        await task
    assert "Synthetic generated answer." in platform_payload(bot)
    assert event.get_extra(MARK)["sent_parts"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
async def test_external_stop_before_completion_preserves_ready_result(env, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    event.set_result(event.plain_result("Synthetic ready answer."))
    event.set_extra("agent_stop_requested", True)
    event.stop_event()
    await env.plugin._complete_output(event)
    assert "Synthetic ready answer." in platform_payload(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
@pytest.mark.parametrize("phase", ["candidate", "judgment"])
async def test_external_stop_during_admission_closes_required_turn(
    env, monkeypatch, kind, phase
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    if phase == "candidate":

        def busy(_):
            entered.set()
            return not release.is_set()

        monkeypatch.setattr(bridge, "native_busy", busy)
    else:
        original = env.plugin.client.evaluate

        async def evaluate(*args):
            entered.set()
            await release.wait()
            return await original(*args)

        monkeypatch.setattr(env.plugin.client, "evaluate", evaluate)
    task = asyncio.create_task(pipeline(env, event, adapter=True))
    try:
        await asyncio.wait_for(entered.wait(), 4)
        event.set_extra("agent_stop_requested", True)
        event.stop_event()
        release.set()
        await asyncio.wait_for(task, 4)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert bot.transport.await_count == 1
    assert event.get_extra("_jev_send_sequence") == 1
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
    if phase == "candidate":
        if env.plugin.audit_task:
            await env.plugin.audit_task
        with env.plugin.ledger.lock:
            records = env.plugin.ledger.db.execute(
                "SELECT reason,details FROM audit WHERE reason IN ('send_attempt','send_completed') ORDER BY id"
            ).fetchall()
        assert [reason for reason, _ in records] == ["send_attempt", "send_completed"]
        attempt, complete = [json.loads(details) for _, details in records]
        assert attempt["turn_id"] == complete["turn_id"]
        assert attempt["ingress_id"] and attempt["sender_hash"]
        assert attempt["send_sequence"] == 1
        assert complete["delivery_state"] == "delivered"
        assert complete["receipt_id"] == "42"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "reply", "private"])
@pytest.mark.parametrize("filter_kind", ["repeated", "keyword"])
async def test_explicit_request_bypasses_autonomous_text_filters(
    env, kind, filter_kind
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    room = env.plugin.rooms.get(event.unified_msg_origin)
    if filter_kind == "repeated":
        room.last_text = event.get_message_str()
    else:
        env.cfg["blocked_keywords"] = ["上下文"]
    await pipeline(env, event, "Synthetic direct answer.", adapter=True)
    assert "Synthetic direct answer." in platform_payload(bot)


@pytest.mark.asyncio
async def test_external_stop_does_not_override_explicit_recall(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    event.set_result(event.plain_result("Withdrawn answer."))
    event.set_extra(silence.RECALL, True)
    event.set_extra("agent_stop_requested", True)
    event.stop_event()
    await env.plugin._complete_output(event)
    assert bot.transport.await_count == 0


@pytest.mark.asyncio
async def test_native_transport_exception_has_uncertain_terminal_receipt(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    request = await event.bot.send_msg(group_id=987654321, message="Synthetic body")
    bot.transport.side_effect = TimeoutError("synthetic receipt loss")
    await env.plugin._complete_output(event)
    assert bot.transport.await_count == 1
    assert request.status == "uncertain"
    assert request.receipt is None
    assert event.get_extra("_jev_delivery_state") == "uncertain"
    await env.plugin._complete_output(event)
    assert bot.transport.await_count == 1
    if env.plugin.audit_task:
        await env.plugin.audit_task
    with env.plugin.ledger.lock:
        rows = env.plugin.ledger.db.execute(
            "SELECT reason,details FROM audit WHERE reason IN ('send_failed','send_completed')"
        ).fetchall()
    assert [reason for reason, _ in rows] == ["send_failed"]
    assert json.loads(rows[0][1])["delivery_state"] == "uncertain"


@pytest.mark.asyncio
async def test_send_audit_correlates_entry_turn_and_receipt(env, monkeypatch):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    records = []
    original = env.plugin.ledger.arecord

    async def capture(*args):
        records.append(args)
        return await original(*args)

    monkeypatch.setattr(env.plugin.ledger, "arecord", capture)
    await pipeline(env, event, "Synthetic audited answer.", adapter=True)
    assert "Synthetic audited answer." in platform_payload(bot)
    if env.plugin.audit_task:
        await env.plugin.audit_task
    attempt = next(
        details for _, reason, details in records if reason == "send_attempt"
    )
    completed = next(
        details for _, reason, details in records if reason == "send_completed"
    )
    for details in (attempt, completed):
        assert details["turn_id"] == event.get_extra(MARK)["ticket"]
        assert details["ingress_id"] and details["sender_hash"]
        assert details["wake_kind"] == "explicit_at"
        assert details["sender_is_admin"] is False
        assert details["send_sequence"] == 1
    assert completed["receipt_id"] == "42"
    with env.plugin.ledger.lock:
        row = env.plugin.ledger.db.execute(
            "SELECT details FROM audit WHERE reason='send_completed' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    saved = json.loads(row[0])
    assert saved["turn_id"] == attempt["turn_id"]
    assert saved["wake_kind"] == "explicit_at"
    assert saved["sender_is_admin"] is False
    assert saved["delivery_state"] == "delivered"
    assert saved["receipt_id"] == "42"


@pytest.mark.asyncio
async def test_default_takeover_off_keeps_native_wake_path(env):
    wake.configure(env)
    env.cfg["takeover_wake_messages"] = False
    event, bot = adapter_event("at")
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    assert not event.get_extra(MARK)
    assert not bot.transport.await_count
    assert not event.is_stopped()


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_kind", ["repeated", "keyword"])
async def test_default_takeover_off_preserves_text_filter(env, filter_kind):
    wake.configure(env)
    env.cfg["takeover_wake_messages"] = False
    event, bot = adapter_event("at")
    if filter_kind == "repeated":
        env.plugin.rooms.get(
            event.unified_msg_origin
        ).last_text = event.get_message_str()
    else:
        env.cfg["blocked_keywords"] = ["上下文"]
    await env.plugin.on_input(event)
    assert event.is_stopped()
    assert not event.get_extra(MARK)
    assert bot.transport.await_count == 0


@pytest.mark.asyncio
async def test_merged_required_candidate_has_one_owner(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    env.cfg["burst_merge_enabled"] = True
    first, first_bot = adapter_event("at")
    second, second_bot = adapter_event("at")
    second.message_obj.message_id = "synthetic-second"
    entered, release = asyncio.Event(), asyncio.Event()
    response = env.plugin.client.evaluate.return_value

    async def evaluate(*args):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return response

    env.plugin.client.evaluate.side_effect = evaluate
    older = asyncio.create_task(pipeline(env, first, adapter=True))
    newer = None
    try:
        await asyncio.wait_for(entered.wait(), 4)
        newer = asyncio.create_task(pipeline(env, second, adapter=True))
        await asyncio.wait_for(newer, 4)
        await asyncio.gather(older, return_exceptions=True)
    finally:
        release.set()
        if not older.done():
            older.cancel()
        await asyncio.gather(older, return_exceptions=True)
        if newer is not None and not newer.done():
            newer.cancel()
            await asyncio.gather(newer, return_exceptions=True)
    assert first_bot.transport.await_count == 0
    assert second_bot.transport.await_count == 1
    assert first.get_extra("_jev_admission_cancelled")
