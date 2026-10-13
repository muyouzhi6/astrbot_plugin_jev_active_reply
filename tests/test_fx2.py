"""Synthetic regressions for review 2 and observed lifecycle failures."""

import asyncio
import json

import pytest
from astrbot.api.message_components import Node, Plain
from astrbot.core.pipeline.respond.stage import RespondStage

from .. import bridge
from ..delivery import DeliveryDeclined
from ..main import MARK
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import adapter_event, delivery_hooks, env, pipeline

__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("texts", [["[PASS]。"], ["Answer", "[PA", "SS]"]])
async def test_late_consumed_and_split_marker(env, native, texts):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")

    async def emit(ev):
        for text in texts:
            if native:
                await ev.bot.send_group_msg(group_id=987654321, message=text)
            else:
                await ev.send(ev.plain_result(text))

    await pipeline(
        env, event, adapter=True, native_send=emit, decoration_priority=-200000
    )
    assert bot.transport.await_count == 1
    payload = json.dumps(bot.transport.await_args.kwargs, ensure_ascii=False)
    assert "[PA" not in payload and "SS]" not in payload
    assert event.get_extra(MARK)["finished"]
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending


@pytest.mark.asyncio
async def test_candidate_timeout_is_silent(env, monkeypatch):
    wake.configure(env)
    event, bot = adapter_event("at")
    monkeypatch.setattr(bridge, "native_busy", lambda ev: True)
    monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.02)
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    assert bot.transport.await_count == 0
    assert event.is_stopped()


@pytest.mark.asyncio
async def test_stopped_generation_completes_once(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    entered, release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        pipeline(env, event, adapter=True, generation_gate=(entered, release))
    )
    await entered.wait()
    event.stop_event()
    release.set()
    await task
    assert bot.transport.await_count == 1


@pytest.mark.asyncio
async def test_lease_rejects_late_send(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    env.plugin._expire(event)
    await asyncio.sleep(0.05)
    with pytest.raises(DeliveryDeclined):
        await event.send(event.plain_result("Late answer"))
    assert bot.transport.await_count == 0


@pytest.mark.asyncio
async def test_segmented_punctuation(env, monkeypatch):
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
    event, bot = adapter_event("at")
    await pipeline(
        env,
        event,
        adapter=True,
        parts=[Plain("Call f"), Plain("("), Plain("x"), Plain(")")],
    )
    assert (
        "".join(
            p["data"]["text"]
            for c in bot.transport.await_args_list
            for p in c.kwargs["message"]
        )
        == "Call f(x)"
    )


@pytest.mark.asyncio
async def test_marker_punctuation_before_node(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await pipeline(
        env,
        event,
        adapter=True,
        parts=[
            Plain("[PASS]"),
            Plain("。"),
            Node(uin="123", name="Bot", content=[Plain("Answer")]),
        ],
    )
    assert '"text": "。"' not in json.dumps(
        bot.transport.await_args.kwargs, ensure_ascii=False
    )


@pytest.mark.asyncio
async def test_history_advance_during_judge(env, monkeypatch):
    wake.configure(env)
    wake.judge_result(env, "reply")
    original = bridge.build_snapshot

    async def judge(*args):
        result = await original(*args)
        monkeypatch.setattr(bridge, "fingerprint", lambda conv: "advanced")
        return result

    monkeypatch.setattr(bridge, "build_snapshot", judge)
    event, bot = adapter_event("at")
    await pipeline(env, event, adapter=True)
    assert bot.transport.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["early_media", "lease"])
async def test_dispatched_terminal(env, mode):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    assert event.get_extra(MARK)
    if mode == "early_media":
        await event.bot.send_group_msg(
            group_id=987654321,
            message=[{"type": "image", "data": {"file": "base64://c3ludGhldGlj"}}],
        )
        event.stop_event()
        await env.plugin.after_sent(event)
        assert bot.transport.await_count == 0
        await event.get_extra("_jev_complete_output")()
    else:
        env.plugin._expire(event)
        await asyncio.sleep(0.05)
    assert bot.transport.await_count == (0 if mode == "lease" else 1)
    assert event.get_extra(MARK)["finished"]
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
