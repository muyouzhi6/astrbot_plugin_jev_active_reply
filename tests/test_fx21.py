"""Release-21 regressions for a redelivered explicit request."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from .. import bridge, silence
from ..main import MARK
from . import test_v0101_explicit_wake as wake
from .test_fx18_scheduler import run_scheduler
from .test_v0102_wake_warnings import (
    adapter_event,
    delivery_hooks,
    env,
    pipeline,
    platform_payload,
)

__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
async def test_sequential_redelivery_has_one_platform_reply(env, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    first, first_bot = adapter_event(kind)
    replay, replay_bot = adapter_event(kind)
    assert first.message_obj.message_id == replay.message_obj.message_id

    await pipeline(env, first, "Synthetic normal answer.", adapter=True)
    await pipeline(env, replay, "This replay must not be generated.", adapter=True)

    assert "Synthetic normal answer." in platform_payload(first_bot)
    assert replay_bot.transport.await_count == 0
    assert replay.is_stopped() and replay.get_extra(MARK) is None
    assert first.get_extra("_jev_send_sequence") == 1
    assert not env.plugin.rooms.get(first.unified_msg_origin).pending


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
async def test_concurrent_redelivery_does_not_race_accepted_reply(env, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    first, first_bot = adapter_event(kind)
    replay, replay_bot = adapter_event(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    first_task = asyncio.create_task(
        pipeline(
            env,
            first,
            "Synthetic normal answer.",
            adapter=True,
            generation_gate=(entered, release),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 4)
        await asyncio.wait_for(
            pipeline(env, replay, "Replay must not be generated.", adapter=True), 4
        )
    finally:
        release.set()
        await asyncio.wait_for(first_task, 4)

    assert "Synthetic normal answer." in platform_payload(first_bot)
    assert replay_bot.transport.await_count == 0
    assert replay.is_stopped() and replay.get_extra(MARK) is None
    assert first.get_extra("_jev_send_sequence") == 1
    assert not env.plugin.rooms.get(first.unified_msg_origin).pending


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
async def test_scheduler_redelivery_does_not_start_second_generation(
    env, monkeypatch, kind
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    first, first_bot = adapter_event(kind)
    replay, replay_bot = adapter_event(kind)

    await run_scheduler(env, first, monkeypatch)
    await run_scheduler(env, replay, monkeypatch)

    assert "Ready model answer." in platform_payload(first_bot)
    assert replay_bot.transport.await_count == 0
    assert replay.is_stopped() and replay.get_extra(MARK) is None


@pytest.mark.asyncio
async def test_cancelled_admission_is_not_reclassified_as_failure(env, monkeypatch):
    wake.configure(env)
    monkeypatch.setattr(
        bridge, "build_snapshot", AsyncMock(side_effect=asyncio.CancelledError())
    )
    event, bot = adapter_event("at")

    await env.plugin.on_input(event)
    with pytest.raises(asyncio.CancelledError):
        await env.plugin.on_message(event)

    assert event.is_stopped()
    assert event.get_extra("_jev_admission_cancelled")
    assert not event.get_extra("_jev_admission_failed")
    assert not event.get_extra(MARK)
    assert bot.transport.await_count == 0
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("role", ["member", "admin"])
@pytest.mark.parametrize(
    "mode",
    [
        "external_media_stop",
        "history_advanced",
        "candidate_timeout",
        "lease_expired",
        "dispatched_empty",
    ],
)
async def test_online_wake_miss_modes_have_one_terminal_send(
    env, monkeypatch, kind, role, mode
):
    wake.configure(env)
    wake.judge_result(env, "silence")
    event, bot = adapter_event(kind)
    event.role = role
    if mode == "candidate_timeout":
        monkeypatch.setattr(bridge, "native_busy", lambda _: True)
        monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.02)
        await pipeline(env, event, "[PASS]。", adapter=True)
    else:
        if mode == "history_advanced":
            evaluate = bridge.build_snapshot

            async def advance(*args):
                answer = await evaluate(*args)
                env.history = '[{"role":"assistant","content":"Synthetic change"}]'
                env.plugin.rooms.get(event.unified_msg_origin).generation += 2
                return answer

            monkeypatch.setattr(bridge, "build_snapshot", advance)
        entered, release = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(
            pipeline(
                env,
                event,
                "" if mode == "dispatched_empty" else "[不回复]。",
                adapter=True,
                generation_gate=(entered, release),
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 4)
            if mode == "external_media_stop":
                target = (
                    {"user_id": 123456789}
                    if kind == "private"
                    else {"group_id": 987654321}
                )
                queued = await event.bot.send_msg(
                    **target, message="[CQ:image,file=base64://c3ludGhldGlj]"
                )
                assert queued.status == "queued"
                event.set_extra("agent_stop_requested", True)
                event.stop_event()
            elif mode == "lease_expired":
                env.plugin._expire(event)
                await asyncio.sleep(0.05)
        finally:
            release.set()
            await asyncio.wait_for(task, 4)

    body = platform_payload(bot)
    assert '"type": "image"' in body if mode == "external_media_stop" else body
    assert silence.FALLBACK_REPLY in body or '"type": "image"' in body
    assert event.get_extra("_jev_send_sequence") == 1
    assert event.get_extra("_jev_delivery_state") == "completed"
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
