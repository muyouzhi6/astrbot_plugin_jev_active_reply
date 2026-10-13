"""Standalone waking and silence regression; no external network."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from astrbot.api.message_components import At, Plain, Reply

from ..admission import input_decision
from .. import bridge
from ..main import MARK
from ..state import RoomBook
from . import test_integration as base
from . import test_v020 as previous_tests

env = previous_tests.env


@pytest.mark.parametrize(
    "text,expected",
    [
        ("吱吱，你在吗", "direct"),
        ("吱吱帮我看看", "direct"),
        ("刚才吱吱说过这事", "pass"),
        ("吱吱声有点吵", "pass"),
        ("讨论模型上下文的实现", "pass"),
        ("吱吱，别回复了", "quiet"),
        ("吱吱，别回复那个人，先帮我看看", "direct"),
        ("他让我闭嘴", "pass"),
        ("吱吱，可以说话了", "resume"),
    ],
)
def test_address_and_quiet_are_context_sensitive(text, expected):
    assert (
        input_decision(base.event(text=text), {"wake_names": ["吱吱"]})[0] == expected
    )


def test_admin_short_name_and_other_person_routing():
    cfg = {"wake_names": ["吱吱"], "admin_wake_names": ["吱"]}
    ev = base.event(text="吱，过来")
    assert input_decision(ev, cfg)[0] == "pass"
    ev.role = "admin"
    assert input_decision(ev, cfg)[0] == "direct"
    ev = base.event(parts=[At(qq="other"), Plain("吱吱，来看看")])
    assert input_decision(ev, cfg)[0] == "skip"
    ev.message_obj.message.insert(0, At(qq="bot"))
    assert input_decision(ev, cfg)[0] == "direct"


@pytest.mark.parametrize(
    "sender,cfg,text,previous,reason",
    [
        ("human", {"blocked_targets": ["group"]}, "test", "", "blocked_target"),
        ("robot", {"other_bot_ids": ["robot"]}, "test", "", "known_bot"),
        ("2854196301", {}, "test", "", "official_bot"),
        ("human", {"blocked_keywords": ["广告"]}, "广告来了", "", "blocked_keyword"),
        ("human", {}, "你说得对！", "你说得对。", "repeated_bot_message"),
        (
            "human",
            {"blocked_commands": ["reset"], "block_builtin_commands": True},
            "/reset",
            "",
            "blocked_command",
        ),
    ],
)
def test_filters_and_exemptions(sender, cfg, text, previous, reason):
    ev = base.event(sender=sender, text=text)
    assert input_decision(ev, cfg, previous) == ("block", reason)
    if reason != "blocked_target":
        assert (
            input_decision(ev, {**cfg, "guard_exempt_targets": [sender]}, previous)[0]
            != "block"
        )


@pytest.mark.asyncio
async def test_reply_to_bot_wakes_without_keys(env):
    ev = base.event(parts=[Reply(id="old", sender_id="bot"), Plain("然后呢")])
    await env.plugin.on_input(ev)
    assert ev.is_at_or_wake_command
    await env.plugin.on_message(ev)
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_dry_run_changes_no_input_state(env):
    env.cfg.update(dry_run=True, blocked_keywords=["test"], wake_names=["吱吱"])
    for text in ("test", "吱吱，别回复", "吱吱，你好"):
        ev = base.event(text=text)
        await env.plugin.on_input(ev)
        assert not ev.is_stopped()
        assert not ev.is_at_or_wake_command
        assert ev.get_extra("_jev_input_action") is None


@pytest.mark.asyncio
async def test_excluded_autonomous_room_keeps_direct_wakes(env):
    env.cfg.update(disabled_sessions=["group"], wake_names=["吱吱"])
    ev = base.event(text="吱吱，你好")
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    assert not ev.is_at_or_wake_command
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_quiet_only_current_sender_cancels_pending_and_explicit_can_resume(env):
    env.cfg["wake_names"] = ["吱吱"]
    pending = await env.admit()
    quiet = base.event(mid="q", text="吱吱，闭嘴")
    await env.plugin.on_input(quiet)
    assert pending.get_extra(MARK)["rejected"]
    assert quiet.is_stopped()
    ordinary = base.event(mid="later")
    other = base.event(mid="other", sender="another")
    await env.plugin.on_input(ordinary)
    await env.plugin.on_input(other)
    assert ordinary.is_stopped()
    assert not other.is_stopped()
    direct = base.event(mid="direct", text="吱吱，再聊聊")
    await env.plugin.on_input(direct)
    assert direct.is_at_or_wake_command
    assert not direct.is_stopped()


@pytest.mark.asyncio
async def test_quiet_invalidates_inflight_judgment(env):
    env.cfg["wake_names"] = ["吱吱"]
    started, release = asyncio.Event(), asyncio.Event()

    async def judge(*args):
        started.set()
        await release.wait()
        return env.response

    env.plugin.client.evaluate = AsyncMock(side_effect=judge)
    ev = base.event()
    task = asyncio.create_task(env.plugin.on_message(ev))
    await started.wait()
    await env.plugin.on_input(base.event(mid="q", text="吱吱，别回复"))
    release.set()
    await task
    assert not ev.get_extra(MARK)


@pytest.mark.asyncio
async def test_room_quiet_persists_and_management_can_resume(env):
    ev = base.event(text="/jev静音 10")
    result = [item async for item in env.plugin.quiet_room(ev, 10)]
    assert result
    env.plugin.rooms = RoomBook()
    directed = base.event(parts=[At(qq="bot"), Plain("来聊聊")])
    await env.plugin.on_input(directed)
    assert directed.is_stopped()
    management = base.event(text="/jev开口")
    await env.plugin.on_input(management)
    assert not management.is_stopped()
    assert [item async for item in env.plugin.resume_room(management)]
    direct = base.event(parts=[At(qq="bot"), Plain("来聊聊")])
    await env.plugin.on_input(direct)
    assert direct.is_at_or_wake_command and not direct.is_stopped()


@pytest.mark.asyncio
async def test_registered_commands_survive_quiet_and_unknown_prefix_is_not_topic(env):
    ev = base.event(text="/help")
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    room.quiet_until = time.monotonic() + 60
    ev.set_extra("handlers_parsed_params", {"help": {}})
    await env.plugin.on_input(ev)
    assert not ev.is_stopped()
    unknown = base.event(text="/unknown")
    await env.plugin.on_message(unknown)
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_filter_exempt_can_directly_call_during_room_quiet(env):
    env.cfg["guard_exempt_targets"] = ["human"]
    ev = base.event(parts=[At(qq="bot"), Plain("来聊聊")])
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    room.quiet_loaded = True
    room.quiet_until = time.monotonic() + 60
    await env.plugin.on_input(ev)
    assert ev.is_at_or_wake_command and not ev.is_stopped()


def test_punctuation_only_is_not_duplicate():
    assert input_decision(base.event(text="?"), {}, "!")[0] == "pass"


@pytest.mark.asyncio
async def test_reply_wake_switch_clears_core_mark_but_keeps_at_and_prefix(env):
    env.cfg["reply_to_bot_wakes"] = False
    ev = base.event(parts=[Reply(id="old", sender_id="bot"), Plain("转述一下")])
    ev.is_at_or_wake_command = True
    await env.plugin.on_input(ev)
    assert ev.is_at_or_wake_command
    assert not ev.is_stopped()
    direct = base.event(
        parts=[Reply(id="old", sender_id="bot"), At(qq="bot"), Plain("转述一下")]
    )
    await env.plugin.on_input(direct)
    assert direct.is_at_or_wake_command
    env.settings["wake_prefix"] = ["/"]
    prefixed = base.event(parts=[Reply(id="old", sender_id="bot"), Plain("/转述一下")])
    prefixed.is_at_or_wake_command = True
    await env.plugin.on_input(prefixed)
    assert prefixed.is_at_or_wake_command


@pytest.mark.asyncio
async def test_direct_cooldown_is_per_sender_and_optional(env):
    env.cfg["wake_names"] = ["吱吱"]
    one, two = base.event(text="吱吱，你好"), base.event(mid="two", text="吱吱，继续")
    await env.plugin.on_input(one)
    await env.plugin.on_input(two)
    assert one.is_at_or_wake_command and two.is_stopped()
    other = base.event(text="吱吱，你好", sender="another")
    await env.plugin.on_input(other)
    assert other.is_at_or_wake_command
    env.cfg["direct_wake_interval"] = 0
    third = base.event(mid="three", text="吱吱，继续")
    await env.plugin.on_input(third)
    assert third.is_at_or_wake_command and not third.is_stopped()


@pytest.mark.asyncio
async def test_on_message_also_loads_persistent_room_silence(env):
    ev = base.event()
    assert [item async for item in env.plugin.quiet_room(ev, 10)]
    env.plugin.rooms = RoomBook()
    await env.plugin.on_message(base.event(mid="new"))
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_ambiguous_contextaware_records_do_not_invent_message_ids(env):
    from types import SimpleNamespace as NS

    rows = [{"sender_name": "same", "content": "好", "is_bot": False}] * 2
    env.stars.append(
        NS(
            name="astrbot_plugin_context_aware",
            activated=True,
            star_cls=NS(get_recent_messages=lambda *a, **k: rows),
        )
    )
    ev = base.event(text="新话题")
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    room.rows.extend(
        [{**r, "message_id": str(i), "sender_id": "human"} for i, r in enumerate(rows)]
    )
    state, _, _ = await bridge.build_snapshot(
        env.ctx, ev, room, env.plugin._outline(ev), env.cfg
    )
    assert state["activity"]["context_aware_status"] == "available"
    assert all("message_id" not in r for r in state["recent_conversation"])


@pytest.mark.asyncio
async def test_native_prefix_resumes_personal_quiet_but_not_group_quiet(env):
    ev = base.event(text="/再聊聊")
    ev.is_at_or_wake_command = True
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    room.quiet_loaded = True
    room.quiet_senders["human"] = time.monotonic() + 60
    await env.plugin.on_input(ev)
    assert not ev.is_stopped() and "human" not in room.quiet_senders
    env.cfg["reply_to_bot_wakes"] = False
    env.settings["wake_prefix"] = ["/"]
    room.quiet_until = time.monotonic() + 60
    quoted = base.event(parts=[Reply(id="old", sender_id="bot"), Plain("/再聊聊")])
    quoted.is_at_or_wake_command = True
    await env.plugin.on_input(quoted)
    assert quoted.is_stopped()


@pytest.mark.asyncio
async def test_non_autonomous_room_still_records_sent_reply_for_repeat_filter(env):
    from astrbot.core.provider.entities import LLMResponse

    env.cfg["disabled_sessions"] = ["group"]
    ev = base.event()
    await env.plugin.on_input(ev)
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="正常回复")
    )
    ev._has_send_oper = True
    await env.plugin.after_sent(ev)
    assert env.plugin.rooms.get(ev.unified_msg_origin).last_text == ""
    repeated = base.event(mid="repeat", text="正常回复")
    await env.plugin.on_input(repeated)
    assert not repeated.is_stopped()


@pytest.mark.asyncio
async def test_nickname_runs_native_process_stage_once_without_jev(env):
    from types import SimpleNamespace as NS
    from astrbot.core.pipeline.process_stage.stage import ProcessStage

    env.cfg["wake_names"] = ["吱吱"]
    ev = base.event(text="吱吱，你好")
    ev.set_extra("activated_handlers", [object()])

    async def handlers(event):
        await env.plugin.on_input(event)
        await env.plugin.on_message(event)
        yield None

    calls = []

    async def agent(event):
        calls.append(event)
        yield None

    stage = ProcessStage()
    stage.ctx = NS(astrbot_config=env.settings)
    stage.star_request_sub_stage = NS(process=handlers)
    stage.agent_sub_stage = NS(process=agent)
    async for _ in stage.process(ev):
        pass
    assert calls == [ev]
    env.plugin.client.evaluate.assert_not_awaited()


def test_management_handlers_reject_nonadmins_via_native_filter():
    from ..main import JevActiveReply
    from astrbot.core.star.star_handler import star_handlers_registry
    from astrbot.core.star.filter.permission import PermissionTypeFilter

    for name in ("quiet_room", "resume_room", "recover", "status"):
        fn = getattr(JevActiveReply, name)
        md = star_handlers_registry.get_handler_by_full_name(
            fn.__module__ + "_" + fn.__name__
        )
        permission = next(
            f for f in md.event_filters if isinstance(f, PermissionTypeFilter)
        )
        ev = base.event()
        assert not permission.filter(ev, {})
        ev.role = "admin"
        assert permission.filter(ev, {})
