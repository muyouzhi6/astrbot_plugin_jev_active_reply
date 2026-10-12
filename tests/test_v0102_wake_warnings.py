"""Explicit wakes survive generation markers and concurrent room updates."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from astrbot.api.message_components import Image
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.provider.entities import LLMResponse

from .. import bridge
from ..main import MARK
from . import test_integration as base
from . import test_v0101_explicit_wake as wake

env = wake.env


async def pipeline(
    env, event, text="模型上下文的回答", *, decoration=None, media=False
):
    """Use Core's native tail selection and the real review/delivery hooks."""
    send = AsyncMock()
    event.send = send
    responses = []

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
        response = LLMResponse(role="assistant", completion_text=text)
        response.result_chain = ev.plain_result(text)
        if media:
            response.result_chain.chain.append(
                Image.fromURL("https://example.invalid/a.png")
            )
        responses.append(response)
        await env.plugin.on_response(ev, response)
        if not ev.is_stopped():
            ev.set_result(
                response.result_chain or ev.plain_result(response.completion_text)
            )
            if decoration is not None:
                ev.set_result(ev.plain_result(decoration))
            await env.plugin.before_output(ev)
            await env.plugin.after_output_decoration(ev)
            if not ev.is_stopped() and ev.get_result().chain:
                await ev.send(ev.get_result())
                await env.plugin.after_sent(ev)
        yield

    stage = ProcessStage()
    stage.ctx = NS(astrbot_config=env.settings)
    stage.star_request_sub_stage = NS(process=handlers)
    stage.agent_sub_stage = NS(process=generate)
    event.set_extra("activated_handlers", [object()])
    async for _ in stage.process(event):
        pass
    return send, responses


async def assert_one_reply(env, event, send):
    send.assert_awaited_once()
    delivered = send.await_args.args[0].get_plain_text().strip()
    assert delivered and "[PASS]" not in delivered
    assert "[pass]" not in delivered and "[不回复]" not in delivered
    assert not event.is_stopped()
    assert event.get_extra(MARK)["committed"]
    assert event.get_extra(MARK)["sent_parts"] == 1
    assert env.plugin.metrics.replies == 1
    assert await env.plugin.ledger.acount(event.unified_msg_origin, "sent") == 1
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
