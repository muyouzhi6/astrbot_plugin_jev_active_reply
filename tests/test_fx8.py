"""Eighth-review regressions using synthetic content and real Core hooks."""

import asyncio
import wave
from types import SimpleNamespace as NS

import pytest
from astrbot.api.message_components import At, Plain, Record
from astrbot.core.astr_agent_hooks import MainAgentHooks
from astrbot.core.star.star import star_map
from astrbot.core.star.star_handler import (
    EventType,
    StarHandlerMetadata,
    star_handlers_registry,
)

from .. import bridge
from ..main import MARK
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import adapter_event, delivery_hooks, env, pipeline
from .test_fx4 import payload

__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
async def test_pending_plain_stop_before_response(env, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        pipeline(env, event, adapter=True, generation_gate=(entered, release))
    )
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await event.send(event.plain_result("Synthetic early answer"))
        event.stop_event()
        release.set()
        await task
        assert bot.transport.await_count == 1, payload(bot)
        assert "Synthetic early answer" in payload(bot)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


def register(monkeypatch, name, handler, priority):
    monkeypatch.setitem(star_map, name, NS(name=name, activated=True, reserved=False))
    star_handlers_registry.append(
        StarHandlerMetadata(
            EventType.OnLLMResponseEvent,
            name,
            name,
            name,
            handler,
            [],
            extras_configs={"priority": priority},
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("output", ["plain", "media", "none"])
@pytest.mark.parametrize("priority", [100, 0])
async def test_real_response_stop_hooks(env, monkeypatch, kind, output, priority):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    original = env.plugin.on_response
    register(monkeypatch, "jev_response", original, 80)

    async def stealer(ev, response):
        if output == "plain":
            await ev.send(ev.plain_result("Synthetic external answer"))
        elif output == "media":
            target = (
                {"user_id": 123456789} if kind == "private" else {"group_id": 987654321}
            )
            await ev.bot.send_msg(
                message=[{"type": "image", "data": {"file": "base64://c3ludGhldGlj"}}],
                **target,
            )
        ev.stop_event()

    async def stop(ev, response):
        ev.stop_event()

    register(monkeypatch, "synthetic_stealer", stealer, priority)
    register(monkeypatch, "synthetic_recall_cancel", stop, priority - 1)
    register(monkeypatch, "synthetic_growth_memory", stop, priority - 2)

    async def real_hooks(ev, response):
        await MainAgentHooks().on_agent_done(NS(context=NS(event=ev)), response)

    monkeypatch.setattr(env.plugin, "on_response", real_hooks)
    await pipeline(env, event, adapter=True)
    assert bot.transport.await_count == 1, payload(bot)
    if output == "plain":
        assert "Synthetic external answer" in payload(bot)
    assert event.get_extra(MARK)["finished"]
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending


@pytest.mark.asyncio
async def test_real_reasoning_cache(env, monkeypatch):
    from astrbot.core import html_renderer
    from astrbot.api.message_components import Image
    from unittest.mock import AsyncMock

    wake.configure(env)
    wake.judge_result(env, "reply")
    event, _ = adapter_event("at")
    register(monkeypatch, "jev_response", env.plugin.on_response, 80)

    async def real_hooks(ev, response):
        await MainAgentHooks().on_agent_done(NS(context=NS(event=ev)), response)
        assert "[PASS]" not in ev.get_extra("_llm_reasoning_content", "")
        assert "[不回复]" not in ev.get_extra("_llm_reasoning_content", "")

    monkeypatch.setattr(env.plugin, "on_response", real_hooks)
    monkeypatch.setattr(Image, "convert_to_base64", AsyncMock(return_value="aW1hZ2U="))
    render = AsyncMock(return_value="https://example.invalid/render.png")
    monkeypatch.setattr(html_renderer, "render_t2i", render)
    await pipeline(
        env,
        event,
        text="Synthetic answer. " * 100,
        reasoning="[PASS]Synthetic reasoning[不回复]",
        adapter=True,
        display_reasoning=True,
        t2i=True,
    )
    render.assert_awaited_once()
    assert "[PASS]" not in render.await_args.args[0]
    assert "[不回复]" not in render.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("reverse", [False, True])
async def test_record_punctuation(env, tmp_path, kind, reverse):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(1600))
    parts = [Plain("。"), Record.fromFileSystem(str(path))]
    await pipeline(env, event, adapter=True, parts=parts[::-1] if reverse else parts)
    assert bot.transport.await_count == 1, payload(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("takeover", [False, True])
@pytest.mark.parametrize("conflict", [False, True])
@pytest.mark.parametrize("bare", [False, True])
async def test_bare_at_and_conflict(env, monkeypatch, takeover, conflict, bare):
    wake.configure(env)
    env.cfg["takeover_wake_messages"] = takeover
    wake.judge_result(env, "silence")
    event, bot = adapter_event("at")
    if bare:
        event.message_str = ""
        event.message_obj.message = [At(qq=event.get_self_id())]
    if conflict:
        monkeypatch.setattr(bridge, "conflicts", lambda *a: ["synthetic_waker"])
    await pipeline(env, event, adapter=True)
    assert bot.transport.await_count == 1, payload(bot)
