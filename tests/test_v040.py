"""Integrated silence is a terminal decision, not a third paid judge."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from astrbot.core.agent.tool import ToolSet
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.platform.message_type import MessageType

from .. import silence
from ..main import MARK, JevActiveReply
from . import test_integration as base
from . import test_v020 as previous

env = previous.env


@pytest.mark.asyncio
async def test_autonomous_silence_releases_lease_without_review_or_later_expiry(env):
    ev = await env.admit()
    meta = ev.get_extra(MARK)
    lease = meta["lease"]
    env.plugin.client.evaluate.reset_mock()
    assert await env.plugin.keep_silent(ev, "sensitive reason not to log") is None
    assert ev.is_stopped() and meta["finished"] and meta["rejected"]
    assert lease.cancelled()
    assert not env.plugin.pending_events
    assert not env.plugin.rooms.get(ev.unified_msg_origin).pending
    resp = LLMResponse(
        role="assistant", completion_text="should not leak", reasoning_content="private"
    )
    await env.plugin.on_response(ev, resp)
    env.plugin._expire(ev)
    await env.plugin.audit_task
    assert resp.completion_text == resp.reasoning_content == ""
    env.plugin.client.evaluate.assert_not_awaited()
    reasons = [r[0] for r in await env.plugin.ledger.arecent(ev.unified_msg_origin)]
    assert reasons.count("model_keep_silent") == 1
    assert "request_lease_expired" not in reasons


@pytest.mark.asyncio
async def test_normal_silence_switch_preserves_explicit_and_private_replies(env):
    ev = base.event()
    assert await env.plugin.keep_silent(ev) is None
    assert ev.is_stopped()
    explicit = base.event()
    explicit.is_at_or_wake_command = True
    assert isinstance(await env.plugin.keep_silent(explicit), str)
    assert not explicit.is_stopped()
    private = base.event()
    private.message_obj.type = MessageType.FRIEND_MESSAGE
    assert private.is_private_chat()
    env.cfg["silence"] = {"private_replies": False}
    assert isinstance(await env.plugin.keep_silent(private), str)
    assert not private.is_stopped()
    env.cfg["silence"]["private_replies"] = True
    assert isinstance(await env.plugin.keep_silent(private), str)
    assert not private.is_stopped()
    env.cfg["silence"]["normal_replies"] = False
    normal = base.event()
    assert isinstance(await env.plugin.keep_silent(normal), str)
    assert not normal.is_stopped()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        {"tool_enable": False},
        {"must_reply_uid": ["human"]},
        {"sessions": []},
        {"enable": False},
    ],
)
async def test_disallowed_silence_removes_tool_from_request_only(env, settings):
    env.cfg["silence"] = settings
    ev = base.event()
    tools = ToolSet()
    tools.tools = [NS(name="keep_silent"), NS(name="search")]
    req = ProviderRequest(func_tool=tools)
    await env.plugin.prepare_silence(ev, req)
    # Disabled scope leaves other plugin configuration entirely untouched.
    if settings.get("enable", True) and settings.get("sessions", ["*"]):
        assert req.func_tool.names() == ["search"]
        assert tools.names() == ["keep_silent", "search"]
    assert isinstance(await env.plugin.keep_silent(ev), str)
    assert not ev.is_stopped()


@pytest.mark.asyncio
async def test_single_policy_normal_and_no_duplicate_autonomous_policy(env):
    tools = ToolSet()
    tools.tools = [NS(name="keep_silent")]
    ev, req = base.event(), ProviderRequest(func_tool=tools)
    await env.plugin.prepare_silence(ev, req)
    await env.plugin.prepare_silence(ev, req)
    assert len(req.extra_user_content_parts) == 1
    auto = await env.admit(base.event(mid="auto"))
    req2 = ProviderRequest(func_tool=tools)
    await env.plugin.prepare_silence(auto, req2)
    assert not req2.extra_user_content_parts


@pytest.mark.asyncio
async def test_ignored_user_never_wakes_but_is_background_and_recall_passes(env):
    env.cfg["silence"] = {"ignored_users": ["human"]}
    ev = base.event(mid="ignored-1", text="background about the topic")
    ev.is_at_or_wake_command = True
    await env.plugin.capture_ignored(ev)
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    assert ev.is_stopped()
    env.plugin.client.evaluate.assert_not_awaited()
    other, req = base.event(sender="other"), ProviderRequest()
    await env.plugin.prepare_silence(other, req)
    assert "background about the topic" in req.extra_user_content_parts[0].text
    recall = base.event()
    recall.message_obj.raw_message = {
        "post_type": "notice",
        "notice_type": "group_recall",
        "message_id": "ignored-1",
    }
    await env.plugin.capture_ignored(recall)
    await env.plugin.on_input(recall)
    await env.plugin.on_message(recall)
    assert not recall.is_stopped()
    assert not env.plugin.ignored_background.text(ev.unified_msg_origin, {})


@pytest.mark.parametrize(
    "settings,admins",
    [({"must_reply_uid": ["human"]}, []), ({"protected_admins": True}, ["human"])],
)
def test_ignore_protection(settings, admins):
    assert not silence.ignored(
        base.event(), {"ignored_users": ["human"], **settings}, admins
    )


def test_group_specific_ignore_does_not_cross_rooms():
    cfg = {"ignored_group_users": ["one:human"]}
    assert silence.ignored(base.event(group="one"), cfg)
    assert not silence.ignored(base.event(group="two"), cfg)


@pytest.mark.asyncio
async def test_dry_run_does_not_ignore_or_silence(env):
    env.cfg.update(dry_run=True, silence={"ignored_users": ["human"]})
    ev = base.event()
    await env.plugin.capture_ignored(ev)
    assert not ev.is_stopped()
    assert isinstance(await env.plugin.keep_silent(ev), str)
    assert not ev.get_extra(silence.SILENT)


def test_background_cache_limits_ttl_and_room_isolation():
    now = [100.0]
    cache = silence.IgnoredBackground(clock=lambda: now[0])
    for n in range(270):
        cache.record(base.event(group=str(n), text="字" * 20000), {})
    assert len(cache.rooms) <= 256 and cache.bytes <= 2097152
    text = cache.text(base.event(group="269").unified_msg_origin, {})
    assert "BACKGROUND_TRUNCATED" in text
    assert not cache.text("another-room", {})
    now[0] += 61
    assert not cache.text(
        base.event(group="269").unified_msg_origin, {"blocked_context_ttl_seconds": 60}
    )
    cache.clear()
    assert cache.bytes == 0


@pytest.mark.asyncio
async def test_followups_release_exact_runner_without_consuming_or_touching_other_event(
    monkeypatch,
):
    from astrbot.core.pipeline.process_stage import follow_up
    from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner

    ev = base.event()
    ticket = NS(consumed=False, resolved=asyncio.Event())
    runner = ToolLoopAgentRunner.__new__(ToolLoopAgentRunner)
    runner.run_context = NS(context=NS(event=ev))
    runner._pending_follow_ups = [ticket]
    monkeypatch.setitem(follow_up._ACTIVE_AGENT_RUNNERS, ev.unified_msg_origin, runner)
    assert silence.release_follow_ups(base.event(mid="different")) == 0
    assert not ticket.resolved.is_set()
    assert silence.release_follow_ups(ev) == 1
    assert ticket.resolved.is_set() and not ticket.consumed
    assert not runner._pending_follow_ups


@pytest.mark.asyncio
async def test_final_response_and_decorated_segments_cannot_revive_silence(env):
    ev = base.event()
    await env.plugin.keep_silent(ev)
    resp = LLMResponse(role="assistant", completion_text="interruption message")
    await env.plugin.silence_agent_done(ev, None, resp)
    assert resp.completion_text == ""
    for hook in (env.plugin.before_output, env.plugin.after_output_decoration):
        ev.set_result(ev.plain_result("stale fragment"))
        await hook(ev)
        assert not ev.get_result().chain


@pytest.mark.asyncio
async def test_ignore_commands_and_protected_admins(env):
    env.settings["admins_id"] = ["boss"]
    ev = base.event(sender="boss")
    assert [r async for r in env.plugin.ignore_user(ev, "someone")]
    assert env.cfg["silence"]["ignored_users"] == ["someone"]
    assert [r async for r in env.plugin.ignore_user(ev, "boss")]
    assert "boss" not in env.cfg["silence"]["ignored_users"]
    assert [r async for r in env.plugin.unignore_user(ev, "someone")]
    assert not env.cfg["silence"]["ignored_users"]


def test_ignore_commands_have_native_admin_permission():
    from astrbot.core.star.star_handler import star_handlers_registry
    from astrbot.core.star.filter.permission import PermissionTypeFilter

    for name in ("ignore_user", "unignore_user", "ignored_status"):
        fn = getattr(JevActiveReply, name)
        md = star_handlers_registry.get_handler_by_full_name(
            fn.__module__ + "_" + fn.__name__
        )
        guard = next(f for f in md.event_filters if isinstance(f, PermissionTypeFilter))
        assert not guard.filter(base.event(), {})


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected", [True, False])
async def test_summary_cancel_only_suppressed_for_retired_turn(env, rejected):
    ev = await env.admit()
    meta = ev.get_extra(MARK)
    meta["generation_evidence"] = {"text": "bounded auxiliary context"}
    entered = asyncio.Event()
    env.ctx.get_current_chat_provider_id = AsyncMock(return_value="mock")
    env.ctx.get_provider_by_id = lambda pid: NS(
        provider_config={"max_context_tokens": 100000}
    )

    async def generate(**kwargs):
        entered.set()
        await asyncio.sleep(60)

    env.ctx.llm_generate = generate
    task = asyncio.create_task(env.plugin._summarize_overflow(ev, meta))
    await entered.wait()
    if rejected:
        env.plugin._reject(ev, None, "burst_replaced_before_delivery")
    task.cancel()
    if rejected:
        assert await task is False
    else:
        with pytest.raises(asyncio.CancelledError):
            await task
    assert task not in env.plugin.tasks


@pytest.mark.asyncio
async def test_native_tool_loop_transitions_done_on_silence_without_provider_call(env):
    from astrbot.core.agent.runners.tool_loop_agent_runner import (
        ToolLoopAgentRunner,
        AgentState,
    )

    ev = await env.admit()
    tool = NS(
        name="keep_silent",
        handler=env.plugin.keep_silent,
        parameters={"properties": {"reason": {"type": "string"}}},
    )
    tools = ToolSet()
    tools.tools = [tool]
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
        tools_call_ids=["silent-1"],
    )
    result = [row async for row in runner._handle_function_tools(runner.req, response)]
    assert result
    assert runner._state == AgentState.DONE
    assert not env.plugin.pending_events and ev.is_stopped()
