"""Capacity regressions with synthetic inputs only."""

import asyncio

import pytest

from .. import bridge
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
@pytest.mark.parametrize("body", ["[PASS]。", "[不回复]", ""])
async def test_capacity_exhaustion_closes_required_delivery(env, kind, body):
    wake.configure(env)
    for index in range(env.plugin.rooms.capacity):
        env.plugin.rooms.get(f"synthetic:{index}").claim(f"owner:{index}", 0)
    event, bot = adapter_event(kind)
    await pipeline(env, event, body, adapter=True)
    bot.transport.assert_not_awaited()
    assert not event.get_extra("_jev_send_sequence")
    assert event.is_stopped()
    assert len(env.plugin.rooms.rooms) == env.plugin.rooms.capacity
    await env.plugin.on_message(event)
    await env.plugin._context_failure_reply(event)
    assert bot.transport.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize(
    "mode",
    [
        "external_media_stop",
        "superseded",
        "candidate_timeout",
        "lease_timeout",
        "empty_completion",
    ],
)
async def test_online_miss_modes_close_once(env, monkeypatch, kind, mode):
    wake.configure(env)
    wake.judge_result(env, "silence")
    event, bot = adapter_event(kind)
    event.role = "member"
    if mode == "candidate_timeout":
        monkeypatch.setattr(bridge, "native_busy", lambda _: True)
        monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.02)
        await pipeline(env, event, "[PASS]。", adapter=True)
    else:
        if mode == "superseded":
            original = bridge.build_snapshot

            async def advance(*args):
                answer = await original(*args)
                env.history = '[{"role":"assistant","content":"changed"}]'
                env.plugin.rooms.get(event.unified_msg_origin).generation += 1
                return answer

            monkeypatch.setattr(bridge, "build_snapshot", advance)
        entered, release = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(
            pipeline(
                env,
                event,
                "" if mode == "empty_completion" else "[不回复]",
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
                assert queued.get("message_id") is None
                event.set_extra("agent_stop_requested", True)
                event.stop_event()
            if mode == "lease_timeout":
                env.plugin._expire(event)
                await asyncio.sleep(0.05)
        finally:
            release.set()
            await task
    value = platform_payload(bot)
    assert '"type": "image"' in value if mode == "external_media_stop" else value
    assert event.get_extra("_jev_send_sequence") == 1
    assert event.get_extra("_jev_delivery_state") == "completed"
    if mode == "external_media_stop":
        assert queued.status == "delivered"
        assert queued["message_id"] == 42
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending


@pytest.mark.asyncio
async def test_capacity_takeover_off_preserves_native_path(env):
    wake.configure(env)
    env.cfg["takeover_wake_messages"] = False
    for index in range(env.plugin.rooms.capacity):
        env.plugin.rooms.get(f"synthetic:{index}").claim(f"owner:{index}", 0)
    event, bot = adapter_event("at")
    await pipeline(env, event, "Synthetic native answer.", adapter=True)
    assert "Synthetic native answer." in platform_payload(bot)
    assert not event.get_extra("_jev_takeover")
