"""Explicit wakes survive generation markers and concurrent room updates."""

import asyncio
from copy import copy, deepcopy
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from astrbot.api.message_components import Image, Plain
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.star.star import star_map
from astrbot.core.star.star_handler import (
    EventType,
    StarHandlerMetadata,
    star_handlers_registry,
)

from .. import bridge, main, silence
from ..main import MARK
from . import test_integration as base
from . import test_v0101_explicit_wake as wake

env = wake.env


@pytest.fixture(autouse=True)
def delivery_hooks(env, monkeypatch):
    """Let Core sort and run the actual registered hooks around decorators."""
    handlers = []
    for name in ("before_output", "after_output_decoration", "after_sent"):
        func = getattr(type(env.plugin), name)
        metadata = copy(
            star_handlers_registry.get_handler_by_full_name(
                func.__module__ + "_" + name
            )
        )
        metadata.handler = getattr(env.plugin, name)
        handlers.append(metadata)
        monkeypatch.setitem(
            star_map,
            metadata.handler_module_path,
            NS(name="jev-test", activated=True, reserved=False),
        )
    monkeypatch.setattr(star_handlers_registry, "_handlers", [])
    monkeypatch.setattr(star_handlers_registry, "star_handlers_map", {})
    for handler in handlers:
        star_handlers_registry.append(handler)

    for priority in (200000, 0, -200000):

        async def decorate(ev, priority=priority):
            if ev.get_extra("_test_decoration_priority") != priority:
                return
            value = ev.get_extra("_test_decoration")
            if value is None:
                return
            result = (
                ev.chain_result(value)
                if isinstance(value, list)
                else ev.plain_result(value)
            )
            if ev.get_extra("_test_send_in_decorator"):
                await ev.send(result)
                ev.set_result(ev.chain_result([]))
            else:
                ev.set_result(result)

        module = f"test_decorator_{priority}"
        monkeypatch.setitem(
            star_map, module, NS(name=module, activated=True, reserved=False)
        )
        star_handlers_registry.append(
            StarHandlerMetadata(
                EventType.OnDecoratingResultEvent,
                module + "_decorate",
                "decorate",
                module,
                decorate,
                [],
                extras_configs={"priority": priority},
            )
        )


async def pipeline(
    env,
    event,
    text="模型上下文的回答",
    *,
    decoration=None,
    media=False,
    decoration_priority=0,
    send_in_decorator=False,
    parts=None,
    reasoning=None,
    generation_gate=None,
):
    """Use Core's native tail, decoration ordering and final respond stage."""
    send = AsyncMock()
    event.send = send
    event.set_extra("_test_decoration", decoration)
    event.set_extra("_test_decoration_priority", decoration_priority)
    event.set_extra("_test_send_in_decorator", send_in_decorator)
    responses = []
    config = deepcopy(DEFAULT_CONFIG)
    config["platform_settings"].update(
        reply_prefix="",
        reply_with_mention=False,
        reply_with_quote=False,
        forward_threshold=10000,
    )
    config["platform_settings"]["segmented_reply"]["enable"] = False
    config["provider_tts_settings"]["enable"] = False
    config["content_safety"]["also_use_in_response"] = False
    config["provider_settings"]["display_reasoning_text"] = False
    config["t2i"] = False
    context = NS(
        astrbot_config=config,
        plugin_manager=NS(
            context=NS(
                get_using_tts_provider=lambda *a: None,
                get_using_tts_provider_async=AsyncMock(return_value=None),
            )
        ),
    )
    decorate_stage, respond_stage = ResultDecorateStage(), RespondStage()
    await decorate_stage.initialize(context)
    await respond_stage.initialize(context)

    async def handlers(ev):
        await env.plugin.capture_ignored(ev)
        if not ev.is_stopped():
            await env.plugin.on_input(ev)
        if not ev.is_stopped():
            await env.plugin.on_message(ev)
        if False:
            yield

    async def generate(ev):
        ev.call_llm = True
        if generation_gate:
            entered, release = generation_gate
            entered.set()
            await release.wait()
        response = LLMResponse(
            role="assistant", completion_text=text, reasoning_content=reasoning
        )
        if parts is not None:
            response.result_chain = ev.chain_result(parts)
        elif text:
            response.result_chain = ev.plain_result(text)
        if media:
            response.result_chain.chain.append(
                Image.fromURL("https://example.invalid/a.png")
            )
        responses.append(response)
        await env.plugin.on_response(ev, response)
        if not ev.is_stopped():
            ev.set_result(
                ev.chain_result(response.result_chain.chain)
                if response.result_chain is not None
                else ev.plain_result(response.completion_text)
            )
            async for _ in decorate_stage.process(ev):
                pass
            if not ev.is_stopped():
                await respond_stage.process(ev)
        yield

    stage = ProcessStage()
    stage.ctx = NS(astrbot_config=env.settings)
    stage.star_request_sub_stage = NS(process=handlers)
    stage.agent_sub_stage = NS(process=generate)
    event.set_extra("activated_handlers", [object()])
    async for _ in stage.process(event):
        pass
    return send, responses


async def assert_one_reply(env, event, send, total=1):
    send.assert_awaited_once()
    delivered = send.await_args.args[0].get_plain_text().strip()
    assert delivered and "[PASS]" not in delivered
    assert "[pass]" not in delivered and "[不回复]" not in delivered
    assert any(char.isalnum() for char in delivered)
    assert not event.is_stopped()
    assert event.get_extra(MARK)["committed"]
    assert event.get_extra(MARK)["sent_parts"] == 1
    assert env.plugin.metrics.replies == total
    assert await env.plugin.ledger.acount(event.unified_msg_origin, "sent") == total
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
    return delivered


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", wake.KINDS)
@pytest.mark.parametrize("stage", ["response", "decoration"])
@pytest.mark.parametrize(
    "text",
    [
        "[PASS]",
        "[pass]",
        "[不回复]",
        "  [PASS]。！ ",
        "[PASS][不回复]",
        "[PASS]模型上下文的回答[不回复]",
    ],
)
async def test_explicit_marker_delivers_once(env, kind, stage, text):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event = wake.request_event(kind)
    send, responses = await pipeline(
        env,
        event,
        text if stage == "response" else "模型上下文的回答",
        decoration=text if stage == "decoration" else None,
    )
    delivered = await assert_one_reply(env, event, send)
    assert len(responses) == 1
    if "模型上下文的回答" in text:
        assert delivered == "模型上下文的回答"
    if stage == "response":
        assert responses[0].completion_text == delivered
        assert responses[0].result_chain.get_plain_text() == delivered
    env.plugin.client.evaluate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", wake.KINDS)
async def test_marker_cleanup_preserves_nontext_parts(env, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event = wake.request_event(kind)
    send, _ = await pipeline(env, event, "[PASS]模型上下文的回答", media=True)
    assert await assert_one_reply(env, event, send) == "模型上下文的回答"
    assert sum(isinstance(part, Image) for part in send.await_args.args[0].chain) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", wake.KINDS)
@pytest.mark.parametrize("send_in_decorator", [False, True])
async def test_final_sender_repairs_after_decoration_hooks(
    env, kind, send_in_decorator
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event = wake.request_event(kind)
    send, _ = await pipeline(
        env,
        event,
        decoration="[PASS]。",
        decoration_priority=0 if send_in_decorator else -200000,
        send_in_decorator=send_in_decorator,
    )
    assert await assert_one_reply(env, event, send) == silence.FALLBACK_REPLY


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", wake.KINDS)
@pytest.mark.parametrize("parts", [None, [], [Plain("")]])
async def test_reasoning_without_content_still_delivers(env, kind, parts):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event = wake.request_event(kind)
    send, responses = await pipeline(
        env, event, "", reasoning="需要理解问题再回答", parts=deepcopy(parts)
    )
    assert await assert_one_reply(env, event, send) == silence.FALLBACK_REPLY
    assert len(responses) == 1
    assert responses[0].completion_text == silence.FALLBACK_REPLY


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["response", "decoration", "final_send"])
async def test_cleanup_keeps_each_plain_and_image_in_order(env, stage):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event = wake.request_event("at")
    parts = [
        Plain("图前说明[PA"),
        Image.fromURL("https://example.invalid/a.png"),
        Plain("SS]图后解释"),
        Plain("，保留标点。"),
        Image.fromURL("https://example.invalid/b.png"),
        Plain("尾注[不回复]"),
    ]
    expected = list(parts)
    send, responses = await pipeline(
        env,
        event,
        parts=parts if stage == "response" else None,
        decoration=parts if stage != "response" else None,
        decoration_priority=-200000 if stage == "final_send" else 0,
    )
    await assert_one_reply(env, event, send)
    chain = send.await_args.args[0].chain
    assert len(chain) == len(expected)
    assert all(actual is original for actual, original in zip(chain, expected))
    assert [part.text for part in chain if isinstance(part, Plain)] == [
        "图前说明",
        "图后解释",
        "，保留标点。",
        "尾注",
    ]
    if stage == "response":
        assert responses[0].result_chain.chain == chain


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["response", "decoration", "final_send"])
@pytest.mark.parametrize(
    "texts",
    [
        ["[PASS]", "。"],
        ["[PA", "SS]", "！"],
        ["[不", "回复]", "……"],
        ["[pa", "ss]", "?!"],
        ["[PASS][不", "回复][pass]", "；"],
        ["。", "！"],
    ],
)
async def test_split_markers_and_punctuation_fall_back(env, stage, texts):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event = wake.request_event("at")
    parts = [Plain(text) for text in texts]
    send, _ = await pipeline(
        env,
        event,
        parts=parts if stage == "response" else None,
        decoration=parts if stage != "response" else None,
        decoration_priority=-200000 if stage == "final_send" else 0,
    )
    assert await assert_one_reply(env, event, send) == silence.FALLBACK_REPLY


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [*wake.KINDS, "natural"])
async def test_slow_generation_queues_required_wakes_past_candidate_deadline(
    env, monkeypatch, kind
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    first = wake.request_event("private" if kind == "private" else "at", mid="first")
    second = (
        base.event(mid="second", text="模型上下文怎么处理")
        if kind == "natural"
        else wake.request_event(kind, mid="second")
    )
    room = env.plugin.rooms.get(first.unified_msg_origin)
    entered, release, waited = asyncio.Event(), asyncio.Event(), asyncio.Event()
    # Advance only the plugin clock after the waiter has set its default 15s
    # deadline, leaving asyncio's own clock and test timeouts untouched.
    offset = 0
    monkeypatch.setattr(
        main,
        "time",
        NS(
            monotonic=lambda: time.monotonic() + offset,
            time=time.time,
        ),
    )
    original_wait = asyncio.Event.wait

    async def wait(signal):
        nonlocal offset
        if signal is room.changed:
            offset = 16
            waited.set()
        return await original_wait(signal)

    monkeypatch.setattr(asyncio.Event, "wait", wait)
    one = asyncio.create_task(pipeline(env, first, generation_gate=(entered, release)))
    two = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        two = asyncio.create_task(pipeline(env, second))
        await asyncio.wait_for(waited.wait(), 2)
        await asyncio.sleep(0.6)
        if kind != "natural":
            assert not two.done(), "explicit wake was dropped at candidate deadline"
            assert not second.is_stopped()
        release.set()
        first_send, _ = await asyncio.wait_for(one, 3)
        second_send, responses = await asyncio.wait_for(two, 3)
        first_send.assert_awaited_once()
        if kind == "natural":
            second_send.assert_not_awaited()
            assert not responses
            assert env.plugin.metrics.replies == 1
        else:
            await assert_one_reply(env, second, second_send, total=2)
            assert len(responses) == 1
            assert env.plugin.client.evaluate.await_count == 2
    finally:
        release.set()
        for task in (one, two):
            if task and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (one, two) if task), return_exceptions=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["response", "decoration"])
@pytest.mark.parametrize("text", ["[PASS]", "[pass]", "[不回复]", "  [PASS]。！ "])
async def test_natural_marker_still_silent(env, stage, text):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event = base.event(text="模型上下文怎么处理")
    send, _ = await pipeline(
        env,
        event,
        text if stage == "response" else "模型上下文的回答",
        decoration=text if stage == "decoration" else None,
        # Preserve the existing pre-review silence behavior for natural turns.
        decoration_priority=200000,
    )
    send.assert_not_awaited()
    assert event.is_stopped() and event.get_extra(MARK)["rejected"]
    assert env.plugin.metrics.replies == 0
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [*wake.KINDS, "natural"])
@pytest.mark.parametrize("phase", ["snapshot", "judge"])
async def test_room_update_during_admission_keeps_explicit_reply(
    env, monkeypatch, kind, phase
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    env.cfg.update(burst_merge_enabled=True, min_reply_interval=0)
    event = (
        base.event(mid="natural", text="模型上下文怎么处理")
        if kind == "natural"
        else wake.request_event(kind)
    )
    room = env.plugin.rooms.get(event.unified_msg_origin)
    entered, release, observed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    snapshot = bridge.build_snapshot
    original_observe = room.observe

    def observe(row, mid):
        result = original_observe(row, mid)
        if mid == "other-message":
            observed.set()
        return result

    async def pause():
        entered.set()
        await release.wait()

    async def build_snapshot(context, ev, *args):
        if ev is event and phase == "snapshot":
            await pause()
        return await snapshot(context, ev, *args)

    async def judge(*args):
        if args[0]["target_message"]["sender_id"] == event.get_sender_id():
            if phase == "judge":
                await pause()
            return env.plugin.client.evaluate.return_value
        return {**env.response, "values": {"reply_now": 0}}

    monkeypatch.setattr(room, "observe", observe)
    monkeypatch.setattr(bridge, "build_snapshot", build_snapshot)
    env.plugin.client.evaluate.side_effect = judge
    task = asyncio.create_task(pipeline(env, event))
    other_task = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        generation = room.generation
        if kind == "private":
            # Private sessions have no other group sender; exercise a context
            # update in that session, with group isolation checked separately.
            room.observe({"sender_id": "other", "content": "新上下文"}, "other-message")
        else:
            other = base.event(
                mid="other-message", sender="other", text="模型上下文的新问题"
            )
            other_task = asyncio.create_task(pipeline(env, other))
        await asyncio.wait_for(observed.wait(), 2)
        assert room.generation == generation + 1
        room.evaluated_at -= 2
        release.set()
        send, responses = await asyncio.wait_for(task, 3)
        if kind == "natural":
            send.assert_not_awaited()
            assert not responses and not event.get_extra(MARK)
        else:
            await assert_one_reply(env, event, send)
            assert len(responses) == 1
        if other_task:
            other_send, _ = await asyncio.wait_for(other_task, 3)
            other_send.assert_not_awaited()
    finally:
        release.set()
        for pending in (task, other_task):
            if pending and not pending.done():
                pending.cancel()
        await asyncio.gather(
            *(t for t in (task, other_task) if t), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_private_judge_unaffected_by_other_group(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    private = wake.request_event("private")
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def judge(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
            return env.plugin.client.evaluate.return_value
        return {**env.response, "values": {"reply_now": 0}}

    env.plugin.client.evaluate.side_effect = judge
    task = asyncio.create_task(pipeline(env, private))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        group = base.event(sender="other", text="模型上下文的新问题")
        send, _ = await pipeline(env, group)
        send.assert_not_awaited()
        release.set()
        send, _ = await asyncio.wait_for(task, 3)
        await assert_one_reply(env, private, send)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
