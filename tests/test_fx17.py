"""Release-17 regressions for complete required delivery."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from astrbot.api.message_components import Image, Plain
from astrbot.core.tools.message_tools import SendMessageToUserTool

from ..main import MARK
from .. import bridge, silence
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
@pytest.mark.parametrize("tool_type", ["plain", "image"])
async def test_tool_does_not_drop_distinct_followup_body(
    env, monkeypatch, kind, tool_type
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    monkeypatch.setattr(
        Image, "convert_to_base64", AsyncMock(return_value="c3ludGhldGlj")
    )
    entered, release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        pipeline(
            env,
            event,
            adapter=True,
            parts=[Plain("First explanation."), Plain("Second explanation.")],
            generation_gate=(entered, release),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 4)
        result = await SendMessageToUserTool().call(
            NS(context=NS(event=event, context=env.ctx)),
            messages=[
                {"type": "plain", "text": "Tool introduction."}
                if tool_type == "plain"
                else {"type": "image", "url": "https://example.invalid/synthetic.png"}
            ],
        )
        assert "Message sent" in result
    finally:
        release.set()
        await task
    body = platform_payload(bot)
    assert body.count("First explanation.") == 1
    assert body.count("Second explanation.") == 1
    assert ("Tool introduction." if tool_type == "plain" else '"type": "image"') in body


@pytest.mark.asyncio
async def test_tool_exact_text_echo_is_not_repeated(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    entered, release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        pipeline(
            env,
            event,
            "Exact answer.",
            adapter=True,
            generation_gate=(entered, release),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 4)
        result = await SendMessageToUserTool().call(
            NS(context=NS(event=event, context=env.ctx)),
            messages=[{"type": "plain", "text": "Exact answer."}],
        )
        assert "Message sent" in result
    finally:
        release.set()
        await task
    assert platform_payload(bot).count("Exact answer.") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
@pytest.mark.parametrize("native_first", [False, True])
async def test_presend_media_error_still_delivers_body(
    env, monkeypatch, kind, native_first
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    monkeypatch.setattr(
        Image,
        "convert_to_base64",
        AsyncMock(side_effect=FileNotFoundError("synthetic missing media")),
    )
    event, bot = adapter_event(kind)

    async def emit(ev):
        if native_first:
            target = (
                {"user_id": 123456789} if kind == "private" else {"group_id": 987654321}
            )
            await ev.bot.send_msg(**target, message="First preserved body.")
        await ev.send(
            ev.chain_result(
                [
                    Plain("Preserved body."),
                    Image.fromURL("https://example.invalid/missing.png"),
                ]
            )
        )

    await pipeline(env, event, adapter=True, native_send=emit)
    body = platform_payload(bot)
    assert "Preserved body." in body
    if native_first:
        assert "First preserved body." in body
    assert event.get_extra(MARK)["finished"]


@pytest.mark.asyncio
async def test_presend_media_error_without_text_uses_fallback(env, monkeypatch):
    wake.configure(env)
    wake.judge_result(env, "reply")
    monkeypatch.setattr(
        Image,
        "convert_to_base64",
        AsyncMock(side_effect=FileNotFoundError("synthetic missing media")),
    )
    event, bot = adapter_event("at")

    async def emit(ev):
        await ev.send(
            ev.chain_result([Image.fromURL("https://example.invalid/missing.png")])
        )

    await pipeline(env, event, adapter=True, native_send=emit)
    assert silence.FALLBACK_REPLY in platform_payload(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize(
    "mode", ["external_stop", "superseded", "wait", "lease", "dispatched_empty"]
)
async def test_online_required_lifecycle(env, monkeypatch, kind, mode):
    wake.configure(env)
    wake.judge_result(env, "silence")
    event, bot = adapter_event(kind)
    assert not event.is_admin()
    if mode == "wait":
        monkeypatch.setattr(bridge, "native_busy", lambda ev: True)
        monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.02)
        await pipeline(env, event, "[PASS]。", adapter=True)
    else:
        if mode == "superseded":
            original = env.plugin.client.evaluate

            async def evaluate(*args):
                result = await original(*args)
                env.history = '[{"role":"assistant","content":"synthetic new history"}]'
                env.plugin.rooms.get(event.unified_msg_origin).generation += 1
                return result

            env.plugin.client.evaluate = evaluate
        entered, release = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(
            pipeline(
                env,
                event,
                "" if mode == "dispatched_empty" else "[不回复]",
                adapter=True,
                generation_gate=(entered, release),
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 4)
            if mode == "external_stop":
                target = (
                    {"user_id": 123456789}
                    if kind == "private"
                    else {"group_id": 987654321}
                )
                await event.bot.send_msg(
                    **target, message="[CQ:image,file=base64://c3ludGhldGlj]"
                )
                event.stop_event()
            if mode == "lease":
                env.plugin._expire(event)
                await asyncio.sleep(0.05)
        finally:
            release.set()
            await task
    body = platform_payload(bot)
    assert body and not env.plugin.rooms.get(event.unified_msg_origin).pending
    if mode == "external_stop":
        assert '"type": "image"' in body
        assert silence.FALLBACK_REPLY not in body
