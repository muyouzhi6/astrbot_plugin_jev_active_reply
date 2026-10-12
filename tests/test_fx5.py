"""Fifth-review synthetic regressions with mock OneBot transport."""

import asyncio

import pytest
from astrbot.api.message_components import Plain
from astrbot.core.pipeline.respond.stage import RespondStage

from ..main import MARK
from .. import silence
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import adapter_event, delivery_hooks, env, pipeline
from .test_fx4 import payload

__all__ = ["env", "delivery_hooks"]


def texts(bot):
    return [
        c.kwargs["message"]
        if isinstance(c.kwargs["message"], str)
        else "".join(p["data"].get("text", "") for p in c.kwargs["message"])
        for c in bot.transport.await_args_list
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("text", ["Normal answer", "[PASS]", "", "[不回复]"])
async def test_successful_media_then_model_output(env, kind, text):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        pipeline(
            env, event, text=text, adapter=True, generation_gate=(entered, release)
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 3)
        message = [{"type": "image", "data": {"file": "base64://c3ludGhldGlj"}}]
        if kind == "private":
            await event.bot.send_private_msg(user_id=123456789, message=message)
        else:
            await event.bot.send_group_msg(group_id=987654321, message=message)
        release.set()
        await task
        assert bot.transport.await_count == 1
        assert "image" in payload(bot)
        assert silence.FALLBACK_REPLY not in payload(bot)
        assert "[PASS]" not in payload(bot) and "[不回复]" not in payload(bot)
        if text == "Normal answer":
            assert text in payload(bot)
        assert event.get_extra(MARK)["finished"]
        assert not event.get_extra(MARK)["rejected"]
        assert not env.plugin.rooms.get(event.unified_msg_origin).pending
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("body,tail", [("Second answer", "。"), ("func(", ")")])
async def test_plain_tail_preserved_with_body(env, kind, native, body, tail):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)

    async def emit(ev):
        for text in ["First answer", body, tail]:
            if not native:
                await ev.send(ev.plain_result(text))
            elif kind == "private":
                await ev.bot.send_private_msg(user_id=123456789, message=text)
            else:
                await ev.bot.send_group_msg(group_id=987654321, message=text)

    await pipeline(
        env, event, adapter=True, native_send=emit, decoration_priority=-200000
    )
    expected = ["First answer" + body + tail]
    assert texts(bot) == expected
    assert event.get_extra(MARK)["finished"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
async def test_core_segmented_tail(env, monkeypatch, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    original = RespondStage.initialize

    async def initialize(stage, ctx):
        await original(stage, ctx)
        stage.enable_seg = True
        stage.only_llm_result = False
        stage.interval_method = "random"
        stage.interval = [0, 0]

    monkeypatch.setattr(RespondStage, "initialize", initialize)
    await pipeline(
        env,
        event,
        adapter=True,
        parts=[Plain("First answer"), Plain("Second answer"), Plain("。")],
    )
    assert texts(bot) == ["First answerSecond answer。"]
