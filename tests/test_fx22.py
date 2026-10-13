"""Release-22 required wake outcomes at the mocked OneBot transport."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from astrbot.core import html_renderer
from astrbot.api.message_components import Image, Plain

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

__all__ = ["delivery_hooks", "env"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
async def test_required_candidate_timeout_finishes_with_normal_reply(
    env, monkeypatch, kind
):
    wake.configure(env)
    event, bot = adapter_event(kind)
    monkeypatch.setattr(bridge, "native_busy", lambda _: True)
    monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.02)

    await pipeline(env, event, "[PASS]。", adapter=True)

    payload = platform_payload(bot)
    assert silence.FALLBACK_REPLY in payload
    assert event.get_extra("_jev_send_sequence") == 1
    assert not event.get_extra("_jev_admission_failed")
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
async def test_required_lease_expiry_flushes_buffered_normal_body(
    env, monkeypatch, kind
):
    wake.configure(env)
    event, bot = adapter_event(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        pipeline(
            env, event, "[PASS]。", adapter=True, generation_gate=(entered, release)
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 4)
        target = (
            {"user_id": 123456789} if kind == "private" else {"group_id": 987654321}
        )
        receipt = await event.bot.send_msg(**target, message="Already prepared answer.")
        assert receipt.status == "queued"
        assert event.get_extra("_jev_buffered_content")()
        env.plugin._expire(event)
        await asyncio.sleep(0.05)
    finally:
        release.set()
        await asyncio.wait_for(task, 4)

    payload = platform_payload(bot)
    assert "Already prepared answer." in payload
    assert silence.FALLBACK_REPLY not in payload
    assert event.get_extra("_jev_send_sequence") == 1
    assert event.get_extra("_jev_delivery_state") == "completed"
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "reply"])
@pytest.mark.parametrize("text", ["[PASS]。", "[不回复]"])
async def test_native_wake_marker_is_never_a_platform_payload(env, kind, text):
    wake.configure(env)
    env.cfg["takeover_wake_messages"] = False
    event, bot = adapter_event(kind)

    await pipeline(env, event, text, adapter=True)

    assert event.get_extra(MARK) is None
    assert silence.FALLBACK_REPLY in platform_payload(bot)


@pytest.mark.asyncio
async def test_native_wake_retains_normal_body_without_takeover(env):
    wake.configure(env)
    env.cfg["takeover_wake_messages"] = False
    event, bot = adapter_event("at")

    await pipeline(env, event, "Ordinary native answer.", adapter=True)

    assert event.get_extra(MARK) is None
    assert "Ordinary native answer." in platform_payload(bot)


@pytest.mark.asyncio
async def test_native_late_marker_after_body_is_suppressed(env):
    wake.configure(env)
    env.cfg["takeover_wake_messages"] = False
    event, bot = adapter_event("at")

    async def emit(ev):
        await ev.send(ev.plain_result("Native body."))
        await ev.send(ev.plain_result("[PASS]。"))

    await pipeline(env, event, adapter=True, native_send=emit)

    assert event.get_extra(MARK) is None
    assert "Native body." in platform_payload(bot)
    assert silence.FALLBACK_REPLY not in platform_payload(bot)


@pytest.mark.asyncio
async def test_native_multiple_normal_sends_keep_their_original_order(env):
    wake.configure(env)
    env.cfg["takeover_wake_messages"] = False
    event, bot = adapter_event("at")

    async def emit(ev):
        for text in ("First native body.", "Second native body.", "[PASS]。"):
            await ev.send(ev.plain_result(text))

    await pipeline(env, event, adapter=True, native_send=emit)

    assert bot.transport.await_count == 2
    for call, expected in zip(
        bot.transport.await_args_list, ("First native body.", "Second native body.")
    ):
        assert call.kwargs["message"] == [{"type": "text", "data": {"text": expected}}]
    assert event.get_extra(MARK) is None


@pytest.mark.asyncio
async def test_native_wake_late_decoration_marker_is_cleaned(env):
    wake.configure(env)
    env.cfg["takeover_wake_messages"] = False
    event, bot = adapter_event("at")

    await pipeline(
        env, event, "Unmodified earlier draft.", decoration="[PASS]。", adapter=True
    )

    assert silence.FALLBACK_REPLY in platform_payload(bot)


@pytest.mark.asyncio
async def test_waiting_timeout_does_not_release_other_request(env, monkeypatch):
    wake.configure(env)
    event, bot = adapter_event("at")
    room = env.plugin.rooms.get(event.unified_msg_origin)
    room.claim("other-request", 0)
    monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.02)

    await pipeline(env, event, "Unreachable late answer.", adapter=True)

    assert silence.FALLBACK_REPLY in platform_payload(bot)
    assert room.pending == "other-request"


@pytest.mark.asyncio
async def test_native_command_is_not_taken_over_by_output_guard(env):
    wake.configure(env)
    env.cfg["takeover_wake_messages"] = False
    event, bot = adapter_event("at")
    event.set_extra("handlers_parsed_params", {"synthetic_command": {}})

    await pipeline(env, event, "Command result.", adapter=True)

    assert event.get_extra("_jev_native_reply_guard") is None
    assert "Command result." in platform_payload(bot)


@pytest.mark.asyncio
async def test_native_split_marker_uses_one_normal_reply(env):
    wake.configure(env)
    env.cfg["takeover_wake_messages"] = False
    event, bot = adapter_event("at")

    await pipeline(env, event, parts=[Plain("[PA"), Plain("SS]。")], adapter=True)

    assert silence.FALLBACK_REPLY in platform_payload(bot)


@pytest.mark.asyncio
async def test_native_reasoning_is_clean_before_t2i(env, monkeypatch):
    wake.configure(env)
    env.cfg["takeover_wake_messages"] = False
    event, bot = adapter_event("at")
    render = AsyncMock(return_value="https://example.invalid/render.png")
    monkeypatch.setattr(html_renderer, "render_t2i", render)
    monkeypatch.setattr(Image, "convert_to_base64", AsyncMock(return_value="aW1hZ2U="))

    await pipeline(
        env,
        event,
        text="Normal body. " * 150,
        reasoning="[PASS]Reasoned normally.[不回复]",
        display_reasoning=True,
        t2i=True,
        adapter=True,
    )

    render.assert_awaited_once()
    rendered = render.await_args.args[0]
    assert "Reasoned normally." in rendered
    assert "[PASS]" not in rendered and "[不回复]" not in rendered
    platform_payload(bot)
