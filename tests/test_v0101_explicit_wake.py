"""Explicit requests must survive Jev silence decisions on both Core versions."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from astrbot.api.message_components import Plain, Reply
from astrbot.core.agent.tool import ToolSet
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from .. import bridge, silence
from ..client import DecisionError
from ..main import MARK
from . import test_integration as base
from . import test_v020 as previous
from .test_v0100_takeover import incoming, pipeline

env = previous.env
KINDS = ["at", "name", "prefix", "reply", "private"]


def configure(env):
    env.cfg.update(
        takeover_wake_messages=True,
        enable_private_chat=True,
        wake_names=["吱吱"],
        decision_mode="natural",
        silence={
            "enable": True,
            "tool_enable": True,
            "normal_replies": True,
            "private_replies": True,
        },
    )
    env.settings["admins_id"] = ["someone_else"]
    env.settings["wake_prefix"] = ["/"]


def request_event(kind, mid="explicit"):
    ev = incoming(kind, mid=mid, sender="123456789")
    if kind == "name":
        ev.message_str = "吱吱，模型上下文怎么处理"
        ev.message_obj.message = [Plain(ev.message_str)]
    elif kind == "reply":
        ev.message_obj.message.insert(0, Reply(id="bot-message", sender_id="bot"))
    assert not ev.is_admin()
    return ev


def judge_result(env, outcome):
    if outcome in {"transport", "timeout", "exception"}:
        env.plugin.client.evaluate.side_effect = {
            "transport": DecisionError("transport_uncertain"),
            "timeout": TimeoutError(),
            "exception": RuntimeError("synthetic failure"),
        }[outcome]
    else:
        env.plugin.client.evaluate.return_value = {
            **env.response,
            "values": dict(
                reply_now=0 if outcome == "silence" else 1,
                addressed=0,
                continuation=0,
                worthwhile=0,
                intrusive=1,
            ),
        }
        if outcome == "malformed":
            env.plugin.client.evaluate.return_value = {}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "outcome", ["reply", "silence", "transport", "timeout", "exception", "malformed"]
)
async def test_non_admin_explicit_request_always_delivers(env, kind, outcome):
    configure(env)
    judge_result(env, outcome)
    ev, sent = request_event(kind), []
    await pipeline(env, ev, sent)
    assert sent == [ev]
    assert await env.plugin.ledger.acount(ev.unified_msg_origin, "sent") == 1
    assert not ev.is_stopped()
    assert ev.get_extra(MARK)["committed"]
    assert env.plugin.metrics.replies == 1
    env.plugin.client.evaluate.assert_not_awaited()
    req = ev.get_extra("_test_generation_request")
    assert req.func_tool.names() == ["web_search"]
    guidance = "\n".join(part.text for part in req.extra_user_content_parts)
    assert "keep_silent" not in guidance and "[PASS]" not in guidance
    assert "请给出正常回复" in guidance
    assert not ev.get_extra(silence.SILENT)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("admitted", [False, True])
async def test_silence_tool_removed_and_direct_call_refused(env, kind, admitted):
    configure(env)
    judge_result(env, "reply")
    ev = request_event(kind)
    if admitted:
        await env.plugin.on_input(ev)
        await env.plugin.on_message(ev)
        assert ev.get_extra(MARK)
    tools = ToolSet()
    tools.tools = [NS(name="keep_silent"), NS(name="web_search")]
    req = ProviderRequest(func_tool=tools, conversation=NS(cid=env.cid))
    await env.plugin.prepare_silence(ev, req)
    await env.plugin.on_request(ev, req)
    assert req.func_tool.names() == ["web_search"]
    assert tools.names() == ["keep_silent", "web_search"]
    assert isinstance(await env.plugin.keep_silent(ev, "保持安静不打扰"), str)
    assert not ev.is_stopped() and not ev.get_extra(silence.SILENT)
    if admitted:
        assert not ev.get_extra(MARK)["rejected"]
        assert env.plugin.rooms.get(ev.unified_msg_origin).pending


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("review", ["always", "adaptive"])
async def test_judge_cannot_veto_explicit_draft_after_new_chatter(env, kind, review):
    configure(env)
    judge_result(env, "reply")
    ev = request_event(kind)
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    room.observe({"sender_id": "other", "content": "模型上下文的新问题"}, "new-message")
    env.cfg["second_review"] = review
    env.plugin.client.evaluate.reset_mock()
    env.plugin.client.evaluate.side_effect = DecisionError("transport_uncertain")
    reply = LLMResponse(role="assistant", completion_text="模型上下文的回答")
    room.last_text = reply.completion_text
    await env.plugin.on_response(ev, reply)
    assert reply.completion_text == "模型上下文的回答"
    ev.set_result(ev.plain_result(reply.completion_text))
    await env.plugin.before_output(ev)
    assert await env.plugin._permit(ev)
    assert ev.get_extra(MARK)["reviewed"] and not ev.is_stopped()
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["silence", "transport"])
async def test_classic_judge_also_cannot_drop_explicit_wake(env, outcome):
    configure(env)
    env.cfg["decision_mode"] = "classic"
    judge_result(env, outcome)
    ev, sent = request_event("at"), []
    await pipeline(env, ev, sent)
    assert sent == [ev] and not ev.is_stopped()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_unavailable_jev_snapshot_still_delivers(env, monkeypatch, kind):
    configure(env)
    monkeypatch.setattr(bridge, "build_snapshot", AsyncMock(side_effect=TimeoutError()))
    ev, sent = request_event(kind), []
    await pipeline(env, ev, sent)
    assert sent == [ev] and not ev.is_stopped()
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_snapshot_does_not_revive_cancelled_turn(env, monkeypatch):
    configure(env)
    monkeypatch.setattr(
        bridge, "build_snapshot", AsyncMock(side_effect=asyncio.CancelledError())
    )
    ev = request_event("at")
    await env.plugin.on_input(ev)
    with pytest.raises(asyncio.CancelledError):
        await env.plugin.on_message(ev)
    assert ev.is_stopped() and not ev.get_extra(MARK)
    assert not env.plugin.rooms.get(ev.unified_msg_origin).pending


@pytest.mark.asyncio
async def test_autonomous_budgets_do_not_drop_explicit_wake(env):
    configure(env)
    judge_result(env, "silence")
    env.cfg.update(daily_evaluation_limit=1, daily_reply_limit=1)
    ev, sent = request_event("at"), []
    await env.plugin.ledger.abump(ev.unified_msg_origin, "evaluations")
    await env.plugin.ledger.abump(ev.unified_msg_origin, "sent")
    await pipeline(env, ev, sent)
    assert sent == [ev]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["reply", "silence", "transport"])
async def test_natural_group_chatter_keeps_silence_behavior(env, outcome):
    configure(env)
    judge_result(env, outcome)
    ev = base.event(text="讨论模型上下文")
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    env.plugin.client.evaluate.assert_awaited_once()
    if outcome != "reply":
        assert not ev.get_extra(MARK) and not ev.is_at_or_wake_command
        return
    assert ev.get_extra(MARK)
    tools = ToolSet()
    tools.tools = [NS(name="keep_silent")]
    req = ProviderRequest(func_tool=tools)
    await env.plugin.prepare_silence(ev, req)
    assert req.func_tool.names() == ["keep_silent"]
    assert await env.plugin.keep_silent(ev) is None
    assert ev.is_stopped() and ev.get_extra(silence.SILENT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope", ["must_reply_uid", "must_reply_gid", "must_reply_umo"]
)
async def test_configured_must_reply_is_not_vetoed(env, scope):
    configure(env)
    judge_result(env, "silence")
    ev = base.event()
    env.cfg["silence"][scope] = [
        {
            "must_reply_uid": ev.get_sender_id(),
            "must_reply_gid": ev.get_group_id(),
            "must_reply_umo": ev.unified_msg_origin,
        }[scope]
    ]
    sent = []
    await pipeline(env, ev, sent)
    assert sent == [ev]
    assert (
        "keep_silent" not in ev.get_extra("_test_generation_request").func_tool.names()
    )


@pytest.mark.asyncio
async def test_merged_followup_inherits_required_reply(env):
    configure(env)
    env.cfg["burst_merge_enabled"] = True
    judge_result(env, "silence")
    one = request_event("at")
    await env.plugin.on_input(one)
    await env.plugin.on_message(one)
    two = base.event(
        mid="followup", sender=one.get_sender_id(), text="补充模型上下文问题"
    )
    sent = []
    await pipeline(env, two, sent)
    assert one.is_stopped()
    assert sent == [two]
    assert two.get_extra("_jev_merged_message_ids") == ["explicit", "followup"]
    assert two.get_extra("_test_generation_request").func_tool.names() == ["web_search"]


@pytest.mark.asyncio
async def test_native_tool_loop_continues_after_refused_silence(env):
    from astrbot.core.agent.runners.tool_loop_agent_runner import (
        AgentState,
        ToolLoopAgentRunner,
    )

    configure(env)
    ev = request_event("at")
    tools = ToolSet()
    tools.tools = [
        NS(
            name="keep_silent",
            handler=env.plugin.keep_silent,
            parameters={"properties": {}},
        )
    ]
    runner = ToolLoopAgentRunner.__new__(ToolLoopAgentRunner)
    runner.req = ProviderRequest(func_tool=tools)
    runner.tool_schema_mode = "full"
    runner.run_context = NS(context=NS(event=ev))
    runner.stats = NS(end_time=0)
    runner._state = AgentState.RUNNING
    runner._pending_follow_ups = []
    runner._abort_signal = asyncio.Event()
    runner._track_tool_call_streak = lambda *a: 1
    runner._build_repeated_tool_call_guidance = lambda *a: ""
    runner.agent_hooks = NS(on_tool_start=AsyncMock(), on_tool_end=AsyncMock())

    async def execute(**kwargs):
        yield await env.plugin.keep_silent(ev)

    runner.tool_executor = NS(execute=execute)
    response = LLMResponse(
        role="assistant",
        tools_call_name=["keep_silent"],
        tools_call_args=[{}],
        tools_call_ids=["explicit-silent-attempt"],
    )
    assert [row async for row in runner._handle_function_tools(runner.req, response)]
    assert runner._state == AgentState.RUNNING
    assert not ev.is_stopped() and not ev.get_extra(silence.SILENT)
