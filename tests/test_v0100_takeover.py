"""Wake/private ownership against real Core stages; synthetic messages only."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import At, Plain, Image
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.agent.tool import ToolSet

from ..main import MARK
from ..client import DecisionError
from . import test_integration as base
from . import test_v020 as previous

env = previous.env


def incoming(kind, mid="one", text="模型上下文怎么处理", sender="human"):
    ev = base.event(mid=mid, text=text, sender=sender)
    if kind == "private":
        ev.message_obj.type = MessageType.FRIEND_MESSAGE
        ev.message_obj.group_id = ""
        ev = AstrMessageEvent(
            text, ev.message_obj, PlatformMetadata("aiocqhttp", "test", "test"), sender
        )
        ev.is_at_or_wake_command = True
    elif kind == "at":
        ev.message_obj.message.insert(0, At(qq="bot"))
        ev.is_at_or_wake_command = True
    elif kind == "prefix":
        # Core has stripped the wake prefix from message_str, preserving parts.
        ev.message_obj.message = [Plain("/" + text)]
        ev.is_at_or_wake_command = True
    elif kind == "name":
        ev.message_str = "测试助手，" + text
        ev.message_obj.message = [Plain(ev.message_str)]
    ev.stop_event = Mock(wraps=ev.stop_event)
    return ev


async def pipeline(env, ev, sent, entered=None, release=None):
    """Real ProcessStage and Jev hooks with synthetic handler/provider stages."""
    ev.send = AsyncMock()

    async def handlers(event):
        await env.plugin.capture_ignored(event)
        if not event.is_stopped():
            await env.plugin.on_input(event)
        if not event.is_stopped():
            await env.plugin.on_message(event)
        if False:
            yield

    async def generate(event):
        event.call_llm = True
        tools = ToolSet()
        tools.tools = [NS(name="keep_silent"), NS(name="web_search")]
        request = ProviderRequest(
            prompt=event.get_message_str(),
            conversation=NS(cid=env.cid),
            func_tool=tools,
        )
        await env.plugin.prepare_silence(event, request)
        await env.plugin.on_request(event, request)
        event.set_extra("_test_generation_request", request)
        if entered:
            entered.set()
            await release.wait()
        response = LLMResponse(role="assistant", completion_text="这是合成的回复")
        if event.get_extra(MARK):
            await env.plugin.on_response(event, response)
        event.set_result(
            event.plain_result(response.completion_text)
        ) if not event.is_stopped() else None
        if event.get_extra(MARK):
            await env.plugin.before_output(event)
        if not event.is_stopped():
            await event.send(event.get_result())
            sent.append(event)
        yield

    stage = ProcessStage()
    stage.ctx = NS(astrbot_config=env.settings)
    stage.star_request_sub_stage = NS(process=handlers)
    stage.agent_sub_stage = NS(process=generate)
    ev.set_extra("activated_handlers", [object()])
    async for _ in stage.process(ev):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "prefix", "name", "private"])
async def test_switches_off_preserve_native_reply(env, kind):
    env.cfg["wake_names"] = ["测试助手"]
    ev = incoming(kind)
    sent = []
    await pipeline(env, ev, sent)
    assert sent == [ev]
    env.plugin.client.evaluate.assert_not_awaited()
    assert not ev.get_extra(MARK)
    ev.stop_event.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "prefix", "name", "private"])
async def test_takeover_admits_one_native_generation(env, kind):
    env.cfg.update(
        takeover_wake_messages=True, enable_private_chat=True, wake_names=["测试助手"]
    )
    sent = []
    ev = incoming(kind)
    await pipeline(env, ev, sent)
    assert sent == [ev]
    assert ev.get_extra(MARK) and ev.get_extra("_jev_takeover")
    env.plugin.client.evaluate.assert_not_awaited()
    assert env.plugin.metrics.replies == 1
    assert await env.plugin.ledger.acount(ev.unified_msg_origin, "sent") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "prefix", "name", "private"])
@pytest.mark.parametrize("failure", ["silence", "transport"])
async def test_explicit_wake_replies_once_despite_silence_or_judge_failure(
    env, kind, failure
):
    env.cfg.update(
        takeover_wake_messages=True, enable_private_chat=True, wake_names=["测试助手"]
    )
    if failure == "silence":
        env.plugin.client.evaluate.return_value = {
            **env.response,
            "values": dict(addressed=0, continuation=0, worthwhile=0, intrusive=1),
        }
    else:
        env.plugin.client.evaluate.side_effect = DecisionError("transport_uncertain")
    ev, sent = incoming(kind), []
    await pipeline(env, ev, sent)
    assert sent == [ev] and not ev.is_stopped()
    ev.stop_event.assert_not_called()
    assert ev.get_result().get_plain_text() == "这是合成的回复"
    assert ev.get_extra(MARK)["committed"] and ev.get_extra("_jev_takeover")
    assert ev.get_extra("_test_generation_request").func_tool.names() == ["web_search"]
    env.plugin.client.evaluate.assert_not_awaited()
    assert env.plugin.metrics.replies == 1
    assert await env.plugin.ledger.acount(ev.unified_msg_origin, "sent") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "prefix", "private"])
@pytest.mark.parametrize("phase", ["judge", "generate"])
async def test_concurrent_burst_replaces_old_turn_and_delivers_once(
    env, monkeypatch, kind, phase
):
    env.cfg.update(
        takeover_wake_messages=True,
        enable_private_chat=True,
        burst_merge_enabled=True,
        evaluation_interval=1,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    from .. import bridge

    original = bridge.build_snapshot

    async def judge(*args):
        if not entered.is_set() and phase == "judge":
            entered.set()
            await release.wait()
        return await original(*args)

    monkeypatch.setattr(bridge, "build_snapshot", judge)
    one = incoming(kind, "first", "看看模型上下文")
    two = incoming(kind, "second", "补充模型上下文的问题")
    sent = []
    task = asyncio.create_task(
        pipeline(env, one, sent, entered if phase == "generate" else None, release)
    )
    await asyncio.wait_for(entered.wait(), 2)
    await pipeline(env, two, sent)
    release.set()
    await asyncio.gather(task, return_exceptions=True)
    assert sent == [two]
    assert two.get_extra("_jev_merged_message_ids") == ["first", "second"]
    assert one.is_stopped()
    one.stop_event.assert_called()
    assert env.plugin.metrics.replies == 1
    assert not env.plugin.client.waiters


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["sender", "umo", "wildcard", "missing", "denied"])
async def test_private_allow_and_deny_scope(env, scope):
    env.cfg["enable_private_chat"] = True
    ev = incoming("private")
    target = {
        "sender": "human",
        "umo": ev.unified_msg_origin,
        "wildcard": "*",
        "missing": "another",
        "denied": "*",
    }[scope]
    env.cfg["enabled_sessions"] = [target]
    env.cfg["disabled_sessions"] = ["human"] if scope == "denied" else []
    sent = []
    await pipeline(env, ev, sent)
    assert bool(ev.get_extra(MARK)) == (scope != "denied")
    assert sent == [ev]
    if scope in {"missing", "denied"}:
        env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", ["blocked_targets", "ignored_users"])
async def test_private_blacklisted_user_is_not_judged_or_sent(env, blocked):
    env.cfg["enable_private_chat"] = True
    if blocked == "ignored_users":
        env.cfg["silence"] = {"ignored_users": ["human"]}
    else:
        env.cfg[blocked] = ["human"]
    ev, sent = incoming("private"), []
    await pipeline(env, ev, sent)
    assert sent == [] and ev.is_stopped()
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["prefix", "private"])
@pytest.mark.parametrize("command", ["plugin.command", "astrbot.builtin.command"])
async def test_registered_command_is_never_swallowed(env, kind, command):
    env.cfg.update(
        takeover_wake_messages=True,
        enable_private_chat=True,
        block_prefixed_commands=True,
        block_builtin_commands=True,
        blocked_commands=["help"],
        blocked_keywords=["help"],
        silence={"ignored_users": ["human"]},
    )
    ev = incoming(kind, text="help")
    ev.set_extra("handlers_parsed_params", {command: {}})
    for handler in [
        env.plugin.capture_ignored,
        env.plugin.on_input,
        env.plugin.on_message,
    ]:
        await handler(ev)
    assert not ev.is_stopped() and not ev.get_extra(MARK)
    ev.stop_event.assert_not_called()
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_takeover_keeps_images_and_other_at_in_explicit_request(env):
    env.cfg["takeover_wake_messages"] = True
    ev = incoming("at")
    ev.message_obj.message += [
        At(qq="other"),
        Image.fromURL("https://example.invalid/photo.png"),
    ]
    before = list(ev.get_messages())
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    assert ev.get_extra(MARK)
    assert ev.get_messages() == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owned", ["provider_request", "call_llm", "_has_send_oper", "stopped"]
)
async def test_existing_owner_is_not_taken_over(env, owned):
    env.cfg["takeover_wake_messages"] = True
    ev = incoming("at")
    if owned == "provider_request":
        ev.set_extra(owned, object())
    elif owned == "stopped":
        ev.stop_event()
    else:
        setattr(ev, owned, True)
    await env.plugin.on_message(ev)
    assert not ev.get_extra("_jev_takeover")
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_dry_run_does_not_take_native_wake(env):
    env.cfg.update(dry_run=True, takeover_wake_messages=True, enable_private_chat=True)
    ev, sent = incoming("private"), []
    await pipeline(env, ev, sent)
    assert sent == [ev] and not ev.get_extra(MARK)
    ev.stop_event.assert_not_called()


@pytest.mark.asyncio
async def test_reentry_does_not_clear_an_admitted_wake(env):
    env.cfg["takeover_wake_messages"] = True
    ev = incoming("at")
    await env.plugin.on_message(ev)
    ticket = ev.get_extra(MARK)["ticket"]
    await env.plugin.on_message(ev)
    assert ev.is_at_or_wake_command and ev.get_extra(MARK)["ticket"] == ticket
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["prefix", "private"])
@pytest.mark.parametrize(
    "module", ["test_plugin", "astrbot.builtin_stars.builtin_commands.main"]
)
async def test_real_core_waking_and_handler_stages_preserve_parameterless_command(
    env, monkeypatch, kind, module
):
    from copy import copy
    from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
    from astrbot.core.pipeline.process_stage.method.star_request import (
        StarRequestSubStage,
    )
    from astrbot.core.star.filter.command import CommandFilter
    from astrbot.core.star.star_handler import star_handlers_registry
    from astrbot.core.star.star import star_map
    from astrbot.core.star.session_plugin_manager import SessionPluginManager
    from ..main import JevActiveReply

    env.cfg.update(
        takeover_wake_messages=True,
        enable_private_chat=True,
        block_prefixed_commands=True,
    )
    env.settings.update(wake_prefix=["/"], admins_id=[], platform_settings={})
    handlers = []
    for name in ["capture_ignored", "on_input", "on_message"]:
        func = getattr(JevActiveReply, name)
        md = copy(
            star_handlers_registry.get_handler_by_full_name(
                func.__module__ + "_" + name
            )
        )
        md.handler = getattr(env.plugin, name)
        handlers.append(md)
        monkeypatch.setitem(star_map, md.handler_module_path, NS(name="jev-test"))
    called = []

    async def command(event):
        called.append(event)
        event._has_send_oper = True

    command_filter = CommandFilter("help", alias={"帮助"})
    command_filter.handler_params = {}
    handlers.append(
        NS(
            handler=command,
            handler_name="help",
            handler_full_name=module + "_help",
            handler_module_path=module,
            event_filters=[command_filter],
            extras_configs={"priority": -2000},
        )
    )
    # The command deliberately runs after Jev; exemption cannot rely on priority.
    monkeypatch.setitem(star_map, module, NS(name="command-test"))
    monkeypatch.setattr(
        star_handlers_registry, "get_handlers_by_event_type", lambda *a, **k: handlers
    )
    monkeypatch.setattr(
        SessionPluginManager,
        "filter_handlers_by_session",
        AsyncMock(side_effect=lambda ev, hs: hs),
    )
    ev = incoming(kind, text="/帮助")
    ev.message_obj.message = [Plain("/帮助")]
    ev.message_str = "/帮助"
    waking = WakingCheckStage()
    await waking.initialize(
        NS(astrbot_config=env.settings, db_helper=AsyncMock(), astrbot_config_id="test")
    )
    await waking.process(ev)
    assert ev.get_extra("handlers_parsed_params") == {module + "_help": {}}
    assert len(ev.get_extra("activated_handlers")) == 4
    stage = ProcessStage()
    stage.ctx = NS(astrbot_config=env.settings)
    stage.star_request_sub_stage = StarRequestSubStage()
    native_calls = []

    async def agent(event):
        native_calls.append(event)
        yield

    stage.agent_sub_stage = NS(process=agent)
    async for _ in stage.process(ev):
        pass
    assert called == [ev] and not native_calls
    ev.stop_event.assert_not_called()
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
@pytest.mark.parametrize(
    "reason", ["sent", "different_sender", "window", "disabled", "unrelated"]
)
async def test_new_burst_modes_keep_existing_separation(env, kind, reason):
    import time

    env.cfg.update(
        takeover_wake_messages=True, enable_private_chat=True, burst_merge_enabled=True
    )
    one = incoming(kind, "first", "讨论模型上下文")
    room = env.plugin.rooms.get(one.unified_msg_origin)
    env.plugin._coalesce(one, room, env.cid)
    if reason == "sent":
        one.get_extra("_jev_burst_frame").committed = True
    if reason == "window":
        one.get_extra("_jev_burst_frame").last = time.monotonic() - 20
    if reason == "disabled":
        env.cfg["burst_merge_enabled"] = False
    text = "今天晴天" if reason == "unrelated" else "补充模型上下文"
    sender = "other" if reason == "different_sender" else "human"
    two = incoming(kind, "second", text, sender)
    env.plugin._coalesce(two, room, env.cid)
    assert not two.get_extra("_jev_merged_message_ids")


@pytest.mark.asyncio
async def test_native_wake_and_ordinary_followup_share_burst(env):
    env.cfg.update(takeover_wake_messages=True, burst_merge_enabled=True)
    one = incoming("at", "first", "模型上下文问题")
    await env.plugin.on_input(one)
    await env.plugin.on_message(one)
    two = base.event(mid="second", text="补充模型上下文问题")
    await env.plugin.on_input(two)
    await env.plugin.on_message(two)
    assert two.get_extra("_jev_merged_message_ids") == ["first", "second"]
    assert one.is_stopped() and two.get_extra(MARK)
    assert two.get_extra("_jev_takeover")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
async def test_takeover_snapshot_and_native_request_preserve_persona(env, kind):
    from astrbot.core.provider.entities import ProviderRequest
    from astrbot.core.agent.message import TextPart

    env.cfg.update(takeover_wake_messages=True, enable_private_chat=True)
    ev = incoming(kind)
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    state = ev.get_extra(MARK)["state"]
    env.plugin.client.evaluate.assert_not_awaited()
    assert state["activity"]["interaction"] == (
        "private_chat" if kind == "private" else "explicit_wake"
    )
    req = ProviderRequest(
        prompt="合成正文", system_prompt="原有人设内容", conversation=NS(cid=env.cid)
    )
    scene = TextPart(
        text="<conversation_scene><instruction>原有场景</instruction></conversation_scene>"
    )
    req.extra_user_content_parts = [scene]
    await env.plugin.on_request(ev, req)
    assert req.system_prompt == "原有人设内容"
    assert (
        scene.text
        == "<conversation_scene><instruction>原有场景</instruction></conversation_scene>"
    )
    assert any("用户呼叫或私聊" in p.text for p in req.extra_user_content_parts)
