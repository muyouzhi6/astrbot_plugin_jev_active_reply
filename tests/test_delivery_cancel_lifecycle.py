"""Cancellation must finish the admitted turn without sending success hooks."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot.api.message_components import Plain
from astrbot.core.pipeline.respond import stage as core_respond
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.star.star import star_map
from astrbot.core.star.star_handler import (
    EventType,
    StarHandlerMetadata,
    star_handlers_registry,
)

from .. import delivery, silence
from ..main import MARK
from . import test_v0101_explicit_wake as wake
from .test_delivery_cancel import prepared_reply, respond_stage, watch_core
from .test_fx18_scheduler import run_scheduler
from .test_v0102_wake_warnings import adapter_event, delivery_hooks

env = wake.env
__all__ = ["delivery_hooks"]


def assert_released(plugin, event, lease):
    meta = event.get_extra(MARK)
    room = plugin.rooms.get(event.unified_msg_origin)
    assert meta.get("finished"), "Cancelled turn was not finished"
    assert not room.pending and not room.busy(time.monotonic())
    assert meta["ticket"] not in plugin.pending_events
    assert "lease" not in meta and lease.cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["stop", "recall", "silence", "context_read_race"])
async def test_scheduler_cancel_releases_turn_and_admits_followup(
    env, monkeypatch, reason
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("natural")
    original_respond = RespondStage.process
    original_read = env.plugin._read_conversation
    hook = AsyncMock(wraps=core_respond.call_event_hook)
    monkeypatch.setattr(core_respond, "call_event_hook", hook)
    errors = []
    monkeypatch.setattr(
        core_respond.logger, "error", lambda *args, **kwargs: errors.append(kwargs)
    )
    snapshots = []

    async def intercepted(stage, ev):
        meta = ev.get_extra(MARK, {})
        if ev is event and meta.get("output_ready") and ev.get_result():
            metrics = env.plugin.metrics
            snapshots.append(
                (meta["lease"], (metrics.replies, metrics.failures, metrics.cancelled))
            )
            if reason == "stop":
                ev.stop_event()
            elif reason == "recall":
                ev.set_extra(silence.RECALL, True)
            elif reason == "silence":
                ev.set_extra(silence.SILENT, True)
            else:
                async def raced(e):
                    value = await original_read(e)
                    e.stop_event()
                    return value

                monkeypatch.setattr(env.plugin, "_read_conversation", raced)
        return await original_respond(stage, ev)

    monkeypatch.setattr(RespondStage, "process", intercepted)
    await run_scheduler(env, event, monkeypatch)

    assert len(snapshots) == 1
    lease, counters = snapshots[0]
    assert bot.transport.await_count == 0
    assert event.get_extra("_jev_delivery_state") == "cancelled"
    assert event.get_result() is None
    assert not errors
    hook.assert_not_awaited()
    metrics = env.plugin.metrics
    assert (metrics.replies, metrics.failures, metrics.cancelled) == counters
    assert await env.plugin.ledger.acount(event.unified_msg_origin, "sent") == 0
    assert_released(env.plugin, event, lease)

    monkeypatch.setattr(env.plugin, "_read_conversation", original_read)
    following, following_bot = adapter_event("at")
    following.message_obj.message_id = "followup-after-cancel"
    await env.plugin.on_input(following)
    await asyncio.wait_for(env.plugin.on_message(following), timeout=1)
    next_meta = following.get_extra(MARK)
    assert next_meta and next_meta["ticket"] != event.get_extra(MARK)["ticket"]
    room = env.plugin.rooms.get(following.unified_msg_origin)
    assert room.pending == next_meta["ticket"]

    # Repeated cancellation and a stale timer cannot release the new owner.
    delivery._finish_cancel(event, "autonomous_delivery_cancelled")
    env.plugin._expire(event)
    assert room.pending == next_meta["ticket"]
    assert next_meta["ticket"] in env.plugin.pending_events
    assert not next_meta["lease"].cancelled()

    following.set_result(following.plain_result("Followup answer"))
    await (await respond_stage()).process(following)
    await env.plugin._complete_output(following)
    assert following_bot.transport.await_count == 1
    assert not room.pending and next_meta["finished"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", [0, 1, 2])
async def test_split_cancel_releases_turn_before_returning(env, monkeypatch, cancel_at):
    event, bot = await prepared_reply(env)
    lease = event.get_extra(MARK)["lease"]
    after_sent = AsyncMock(wraps=core_respond.call_event_hook)
    monkeypatch.setattr(core_respond, "call_event_hook", after_sent)

    class SplitStep:
        def __init__(self):
            self.plugin_config = SimpleNamespace(context=SimpleNamespace())

        async def handle(self, ctx):
            for index in range(3):
                if index == cancel_at:
                    ctx.event.set_extra(silence.RECALL, True)
                await self.plugin_config.context.send_message(
                    ctx.event.unified_msg_origin,
                    ctx.event.plain_result(f"Segment {index}"),
                )
            raise AssertionError("Later segment reached")

    step = SplitStep()
    adapter = delivery.SplitAdapters(MARK)
    adapter.attach(SimpleNamespace(pipeline=SimpleNamespace(_steps=[step])))
    try:
        result = await step.handle(
            SimpleNamespace(event=event, chain=event.get_result().chain)
        )
    finally:
        adapter.restore()

    assert result.abort and bot.transport.await_count == cancel_at
    assert event.get_extra("_jev_delivery_state") == (
        "completed" if cancel_at else "cancelled"
    )
    assert event.get_extra(MARK).get("sent_parts", 0) == cancel_at
    assert_released(env.plugin, event, lease)
    await (await respond_stage()).process(event)
    after_sent.assert_not_awaited()
    assert bot.transport.await_count == cancel_at
    assert await env.plugin.ledger.acount(
        event.unified_msg_origin, "sent"
    ) == bool(cancel_at)
    assert env.plugin.metrics.replies == bool(cancel_at)
    assert env.plugin.metrics.failures == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", [0, -200000])
async def test_split_cancel_skips_core_after_sent_and_admits_followup(
    env, monkeypatch, delivery_hooks, priority
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("natural")
    after_sent = AsyncMock(wraps=core_respond.call_event_hook)
    monkeypatch.setattr(core_respond, "call_event_hook", after_sent)
    errors, snapshots = [], []
    monkeypatch.setattr(
        core_respond.logger, "error", lambda *args, **kwargs: errors.append(kwargs)
    )

    class SplitStep:
        def __init__(self):
            self.plugin_config = SimpleNamespace(context=SimpleNamespace())

        async def handle(self, ctx):
            ctx.event.set_extra(silence.RECALL, True)
            await self.plugin_config.context.send_message(
                ctx.event.unified_msg_origin,
                ctx.event.plain_result("Cancelled segment"),
            )
            raise AssertionError("Later segment reached")

    step = SplitStep()
    adapter = delivery.SplitAdapters(MARK)
    adapter.attach(SimpleNamespace(pipeline=SimpleNamespace(_steps=[step])))

    async def split_hook(ev):
        if ev is event:
            metrics = env.plugin.metrics
            snapshots.append(
                (
                    ev.get_extra(MARK)["lease"],
                    (metrics.replies, metrics.failures, metrics.cancelled),
                )
            )
            result = await step.handle(
                SimpleNamespace(event=ev, chain=ev.get_result().chain)
            )
            assert result.abort

    module = "lifecycle_splitter"
    monkeypatch.setitem(
        star_map,
        module,
        SimpleNamespace(name=module, activated=True, reserved=False),
    )
    star_handlers_registry.append(
        StarHandlerMetadata(
            EventType.OnDecoratingResultEvent,
            module + "_split",
            "split",
            module,
            split_hook,
            [],
            extras_configs={"priority": priority},
        )
    )
    try:
        await run_scheduler(env, event, monkeypatch)
    finally:
        adapter.restore()

    room = env.plugin.rooms.get(event.unified_msg_origin)
    cancel_hook_calls = after_sent.await_count
    has_backoff = room.delivery_backoff_until > time.monotonic()
    assert bot.transport.await_count == 0
    assert event.get_extra("_jev_delivery_state") == "cancelled"
    assert len(snapshots) == 1
    lease, counters = snapshots[0]
    assert_released(env.plugin, event, lease)
    metrics = env.plugin.metrics
    assert (metrics.replies, metrics.failures, metrics.cancelled) == counters
    assert await env.plugin.ledger.acount(event.unified_msg_origin, "sent") == 0

    room.evaluated_at -= 60
    following, following_bot = adapter_event("natural")
    following.message_obj.message_id = "followup-after-split-cancel"
    env.plugin.client.evaluate.reset_mock()
    await run_scheduler(env, following, monkeypatch)

    assert (
        cancel_hook_calls,
        has_backoff,
        bool(following.get_extra(MARK)),
        following_bot.transport.await_count,
    ) == (
        0,
        False,
        True,
        1,
    )
    assert event.get_result() is None
    assert not room.pending
    assert env.plugin.client.evaluate.await_count > 0
    assert following.get_extra(MARK)["finished"]
    assert not errors


@pytest.mark.asyncio
async def test_mid_segment_stop_releases_turn_and_preserves_receipt(env, monkeypatch):
    event, bot = await prepared_reply(env)
    lease = event.get_extra(MARK)["lease"]
    event.set_result(event.chain_result([Plain(f"Segment {i}") for i in range(3)]))
    previous_send = event.send
    attempts = 0

    async def guarded_send(chain):
        nonlocal attempts
        attempts += 1
        result = await previous_send(chain)
        event.stop_event()
        return result

    event.send = guarded_send
    stage = await respond_stage()
    monkeypatch.setattr(stage, "is_seg_reply_required", lambda ev: True)
    monkeypatch.setattr(stage, "_calc_comp_interval", AsyncMock(return_value=0))
    errors, hook = watch_core(monkeypatch)
    await stage.process(event)

    assert attempts == 2 and bot.transport.await_count == 1
    assert event.get_extra("_jev_delivery_state") == "completed"
    assert event.get_extra(MARK)["sent_parts"] == 1
    assert await env.plugin.ledger.acount(event.unified_msg_origin, "sent") == 1
    assert not errors
    hook.assert_not_awaited()
    assert_released(env.plugin, event, lease)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
async def test_cancel_finishes_queued_native_receipts_without_late_send(
    env, monkeypatch, kind
):
    event, bot = await prepared_reply(env, kind)
    lease = event.get_extra(MARK)["lease"]
    target = (
        {"group_id": int(event.get_group_id())}
        if kind == "at"
        else {"user_id": int(event.get_sender_id())}
    )
    receipt = await event.bot.send_msg(
        **target, message=[{"type": "text", "data": {"text": "Queued tool output"}}]
    )
    assert receipt.status == "queued"
    event.set_extra(silence.RECALL, True)
    errors, hook = watch_core(monkeypatch)
    await (await respond_stage()).process(event)

    assert receipt.status == "cancelled" and receipt.receipt is None
    assert event.get_extra("_jev_delivery_state") == "cancelled"
    assert_released(env.plugin, event, lease)
    await env.plugin._complete_output(event)
    env.plugin._expire(event)
    assert bot.transport.await_count == 0
    assert not errors
    hook.assert_not_awaited()
