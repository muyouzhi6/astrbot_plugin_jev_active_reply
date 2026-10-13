"""Admission refreshes a completed native reply without losing follow-ups."""

import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from astrbot.core.provider.entities import LLMResponse

from .. import bridge
from ..main import MARK
from . import test_integration as base
from .test_v020 import env

__all__ = ["env"]


def configure(env):
    env.cfg.update(decision_mode="natural", second_review="adaptive")


def reasons(env):
    return [reason for _, reason, _ in env.plugin.audit_queue]


@pytest.mark.asyncio
@pytest.mark.parametrize("score,expected", [(0.85, True), (0.2, False)])
async def test_native_history_advance_refreshes_before_admission(env, score, expected):
    configure(env)
    states = []

    async def judge(state, questions):
        states.append(state)
        if len(states) == 1:
            env.history = json.dumps(
                [{"role": "assistant", "content": "The previous reply just finished"}]
            )
            return {"values": {"reply_now": 0.85}}
        return {"values": {"reply_now": score}}

    env.plugin.client.evaluate = AsyncMock(side_effect=judge)
    ev = base.event(text="Actually, I meant the other one")
    await env.plugin.on_message(ev)

    assert len(states) == 2
    assert "The previous reply just finished" in json.dumps(states[1])
    assert states[1]["target_message"]["content"] == ev.message_str
    assert bool(ev.get_extra(MARK)) is expected
    assert ev.is_at_or_wake_command is expected
    if expected:
        assert reasons(env).count("dispatched") == 1
        assert ev.get_extra(MARK)["state"] == states[1]
    else:
        assert not env.plugin.rooms.get(ev.unified_msg_origin).pending
        assert "dispatched" not in reasons(env)


@pytest.mark.asyncio
async def test_history_refresh_is_bounded_when_native_history_keeps_changing(env):
    configure(env)
    calls = 0

    async def judge(*args):
        nonlocal calls
        calls += 1
        env.history = json.dumps([{"role": "assistant", "content": str(calls)}])
        return {"values": {"reply_now": 0.9}}

    env.plugin.client.evaluate = AsyncMock(side_effect=judge)
    ev = base.event()
    await env.plugin.on_message(ev)
    assert calls == 2
    assert ev.get_extra(MARK) is None
    assert not env.plugin.rooms.get(ev.unified_msg_origin).pending
    assert not ev.is_at_or_wake_command


@pytest.mark.asyncio
async def test_real_conversation_switch_never_refreshes_old_target(env):
    configure(env)

    async def judge(*args):
        env.cid = "different-conversation"
        return {"values": {"reply_now": 0.9}}

    env.plugin.client.evaluate = AsyncMock(side_effect=judge)
    ev = base.event()
    await env.plugin.on_message(ev)
    assert env.plugin.client.evaluate.await_count == 1
    assert ev.get_extra(MARK) is None
    assert not env.plugin.rooms.get(ev.unified_msg_origin).pending


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["switch", "cancel", "new_message"])
async def test_refresh_rechecks_cancellation_and_ownership(env, change):
    configure(env)
    ev = base.event()
    calls = 0

    async def judge(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            env.history = '[{"role":"assistant","content":"new reply"}]'
        elif change == "switch":
            env.cid = "different-conversation"
        elif change == "cancel":
            ev.set_extra("agent_stop_requested", True)
        else:
            env.plugin.rooms.get(ev.unified_msg_origin).observe(
                {"sender_id": "human", "content": "Never mind"}, "newer"
            )
        return {"values": {"reply_now": 0.9}}

    env.plugin.client.evaluate = AsyncMock(side_effect=judge)
    await env.plugin.on_message(ev)
    assert calls == 2
    assert ev.get_extra(MARK) is None
    assert not env.plugin.rooms.get(ev.unified_msg_origin).pending


@pytest.mark.asyncio
async def test_switch_during_refreshed_snapshot_never_retargets(env, monkeypatch):
    configure(env)
    original = bridge.build_snapshot
    calls = 0

    async def snapshot(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            env.cid = "different-conversation"
        return await original(*args)

    async def judge(*args):
        env.history = '[{"role":"assistant","content":"new reply"}]'
        return {"values": {"reply_now": 0.9}}

    monkeypatch.setattr(bridge, "build_snapshot", snapshot)
    env.plugin.client.evaluate = AsyncMock(side_effect=judge)
    ev = base.event()
    await env.plugin.on_message(ev)
    assert calls == 2
    assert env.plugin.client.evaluate.await_count == 1
    assert ev.get_extra(MARK) is None


@pytest.mark.asyncio
async def test_unchanged_history_needs_only_one_judgment(env):
    configure(env)
    env.plugin.client.evaluate = AsyncMock(return_value={"values": {"reply_now": 0.9}})
    await env.admit()
    assert env.plugin.client.evaluate.await_count == 1


@pytest.mark.asyncio
async def test_refresh_exposes_latest_reply_when_public_history_is_stale(env):
    configure(env)
    env.stars.append(
        NS(
            name="astrbot_plugin_context_aware",
            star_cls=NS(
                get_recent_messages=lambda *_args, **_kwargs: [
                    {"sender_name": "Bot", "content": "Older reply", "is_bot": True}
                ]
            ),
        )
    )
    calls = 0

    async def judge(state, _questions):
        nonlocal calls
        calls += 1
        if calls == 1:
            env.history = '[{"role":"assistant","content":"Latest completed reply"}]'
        else:
            assert (
                state["activity"]["latest_completed_reply"] == "Latest completed reply"
            )
            assert state["activity"]["history_refreshed"]
            assert "already handles" in state["activity"]["refresh_reason"]
        return {"values": {"reply_now": 0.9}}

    env.plugin.client.evaluate = AsyncMock(side_effect=judge)
    await env.admit()
    assert calls == 2


@pytest.mark.asyncio
async def test_refreshed_candidate_delivers_once_and_releases_room(env):
    configure(env)
    calls = 0

    async def judge(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            env.history = '[{"role":"assistant","content":"Earlier reply"}]'
        return {"values": {"reply_now": 0.9}}

    env.plugin.client.evaluate = AsyncMock(side_effect=judge)
    ev = base.event()
    transport = AsyncMock(return_value={"message_id": 42})
    ev.send = transport
    await env.admit(ev)
    response = LLMResponse(role="assistant", completion_text="A relevant follow-up")
    await env.plugin.on_response(ev, response)
    ev.set_result(ev.plain_result(response.completion_text))
    await env.plugin.before_output(ev)
    await ev.send(ev.get_result())
    await env.plugin.after_sent(ev)
    assert transport.await_count == 1
    assert not env.plugin.rooms.get(ev.unified_msg_origin).pending
    assert ev.get_extra(MARK)["finished"]
    assert env.plugin.ledger.count(ev.unified_msg_origin, "sent") == 1


@pytest.mark.parametrize(
    "history,expected",
    [
        ("broken", ""),
        ("null", ""),
        ('[{"role":"user","content":"latest question"}]', ""),
        ('[{"role":"assistant","tool_calls":[{}],"content":"thinking"}]', ""),
        ('[{"role":"assistant","content":[{"type":"text","text":"reply"}]}]', "reply"),
    ],
)
def test_latest_reply_uses_only_completed_assistant_text(history, expected):
    assert bridge.latest_reply(NS(history=history)) == expected
