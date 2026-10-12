"""Release-review regressions through Core and mocked OneBot transport."""

import json
from unittest.mock import AsyncMock
from types import SimpleNamespace as NS
import wave

import pytest
from astrbot.api.message_components import Node, Plain
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
from astrbot.core.star.session_llm_manager import SessionServiceManager
from astrbot.core.message.message_event_result import ResultContentType

from .. import silence
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
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("priority", [0, -200000])
@pytest.mark.parametrize("order", ["first", "middle", "last", "only"])
async def test_silent_fragments_wait_for_complete_output(
    env, kind, native, priority, order
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    texts = {
        "first": ["[PASS]。", "Answer"],
        "middle": ["Answer", "[PASS]。", "Conclusion"],
        "last": ["Answer", "[PASS]。"],
        "only": ["[PASS]。", "[不回复]！"],
    }[order]

    async def emit(ev):
        for text in texts:
            if native:
                if ev.is_private_chat():
                    await ev.bot.send_private_msg(user_id=123456789, message=text)
                else:
                    await ev.bot.send_group_msg(group_id=987654321, message=text)
            else:
                await ev.send(ev.plain_result(text))

    await pipeline(
        env, event, adapter=True, native_send=emit, decoration_priority=priority
    )
    encoded = json.dumps([c.kwargs for c in bot.transport.await_args_list])
    assert "[PASS]" not in encoded
    if order == "only":
        assert silence.FALLBACK_REPLY in platform_payload(bot)
    else:
        assert "Answer" in encoded
        assert silence.FALLBACK_REPLY not in json.loads(json.dumps(encoded))
        assert bot.transport.await_count == (2 if order == "middle" else 1)
        if order == "middle":
            assert "Conclusion" in encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
async def test_forward_is_planned_before_core_segmentation(env, monkeypatch, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    original = RespondStage.initialize

    async def initialize(stage, ctx):
        await original(stage, ctx)
        stage.enable_seg = True
        stage.only_llm_result = False
        stage.interval_method = "random"
        stage.interval = [0, 0]

    monkeypatch.setattr(RespondStage, "initialize", initialize)
    event, bot = adapter_event(kind)
    await pipeline(
        env,
        event,
        adapter=True,
        parts=[
            Plain("Introduction"),
            Node(uin="123", name="Bot", content=[Plain("Answer")]),
            Plain("Conclusion"),
        ],
    )
    payload = platform_payload(bot)
    assert payload.index("Introduction") < payload.index("Answer")
    assert payload.index("Answer") < payload.index("Conclusion")


@pytest.mark.asyncio
async def test_regular_punctuation_components_survive(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await pipeline(
        env,
        event,
        adapter=True,
        parts=[Plain("Call f"), Plain("("), Plain("x"), Plain(")")],
    )
    payload = bot.transport.await_args.kwargs["message"]
    assert "".join(p["data"]["text"] for p in payload) == "Call f(x)"


@pytest.mark.asyncio
@pytest.mark.parametrize("dual", [False, True])
async def test_core_tts_preserves_all_segments(env, monkeypatch, tmp_path, dual):
    monkeypatch.setattr(
        SessionServiceManager,
        "should_process_tts_request",
        AsyncMock(return_value=True),
    )
    wake.configure(env)
    wake.judge_result(env, "reply")
    path = tmp_path / "synthetic.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(1600))
    provider = NS(get_audio=AsyncMock(return_value=str(path)))
    original = ResultDecorateStage.initialize

    async def initialize(stage, ctx):
        ctx.astrbot_config["provider_tts_settings"].update(
            enable=True,
            dual_output=dual,
            use_file_service=False,
            trigger_probability=1.0,
        )
        ctx.plugin_manager.context.get_using_tts_provider = lambda *a: provider
        ctx.plugin_manager.context.get_using_tts_provider_async = AsyncMock(
            return_value=provider
        )
        await original(stage, ctx)

    monkeypatch.setattr(ResultDecorateStage, "initialize", initialize)
    event, bot = adapter_event("at")
    original_set = event.set_result

    def set_result(result):
        if result is not None:
            result.use_tts = True
            result.set_result_content_type(ResultContentType.LLM_RESULT)
        return original_set(result)

    monkeypatch.setattr(event, "set_result", set_result)
    await pipeline(
        env,
        event,
        adapter=True,
        parts=[Plain("First answer"), Plain("Second answer[PASS]。")],
    )
    assert [c.args[0] for c in provider.get_audio.await_args_list] == [
        "First answer",
        "Second answer。",
    ]
    calls = bot.transport.await_args_list
    assert len(calls) == (3 if dual else 2)
    assert all(c.kwargs["message"][0]["type"] == "record" for c in calls[:2])
    if dual:
        assert [p["data"]["text"] for p in calls[2].kwargs["message"]] == [
            "First answer",
            "Second answer。",
        ]
