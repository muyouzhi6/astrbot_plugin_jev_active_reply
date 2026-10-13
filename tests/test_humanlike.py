"""Humanlike replies through Core and OneBot, without external requests."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)
from astrbot.core.provider.entities import LLMResponse

from .. import bridge, main, silence
from ..delivery import EventBotProxy, is_message_action
from ..main import MARK
from ..settings import ConfigView, DEFAULT_SCOPE_GROUPS, migrate_config_file
from . import test_integration as base
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
@pytest.mark.parametrize("kind", ["at", "natural"])
@pytest.mark.parametrize("route", ["method", "call_action"])
@pytest.mark.parametrize(
    "action", ["send_like", "set_group_ban", "group_poke", "friend_poke", "send_poke"]
)
async def test_nonmessage_tool_does_not_interrupt_text(env, kind, route, action):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        pipeline(
            env,
            event,
            "正常回答",
            adapter=True,
            generation_gate=(entered, release),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert not event.get_extra(MARK)["reviewed"]
        params = dict(user_id=123456789)
        if route == "method":
            result = await getattr(event.bot, action)(**params)
        else:
            result = await event.bot.call_action(action, **params)
        assert result == {"message_id": 42}
        assert not event.is_stopped()
        bot.transport.assert_awaited_once_with(action, **params)
        bot.transport.reset_mock()
    finally:
        release.set()
        await asyncio.wait_for(task, 3)
    assert "正常回答" in platform_payload(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    [
        "send_msg",
        "send_group_msg",
        "send_forward_msg",
        "send_group_forward_msg",
        "send_private_msg",
    ],
)
@pytest.mark.parametrize("kind", ["natural", "at"])
async def test_declined_message_tool_continues_with_normal_text(env, action, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        pipeline(
            env,
            event,
            "聊聊你刚才说的事。",
            adapter=True,
            generation_gate=(entered, release),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 3)
        # Foreign target must not escape, nor abort the model's normal answer.
        result = await event.bot.call_action(action, group_id=111, message="不应发送")
        assert result.status == "cancelled"
        assert not event.is_stopped()
        bot.transport.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait_for(task, 3)
    payload = platform_payload(bot)
    assert "聊聊你刚才说的事。" in payload
    assert "不应发送" not in payload and "失败" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "reply", "private"])
@pytest.mark.parametrize("takeover", [False, True])
async def test_explicit_reply_skips_judge_even_quiet(env, kind, takeover):
    wake.configure(env)
    wake.judge_result(env, "silence")
    env.cfg.update(
        style="quiet",
        address_threshold=1,
        takeover_wake_messages=takeover,
        reply_to_bot_wakes=False,
    )
    event, bot = adapter_event(kind)
    await pipeline(env, event, "正常回答", adapter=True)
    assert "正常回答" in platform_payload(bot)
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["whitelist_users", "whitelist_groups"])
async def test_direct_whitelist_skips_judge(env, key):
    wake.configure(env)
    env.cfg.update(takeover_wake_messages=False, style="quiet")
    wake.judge_result(env, "silence")
    event, bot = adapter_event("natural")
    env.cfg[key] = [
        event.get_sender_id() if key == "whitelist_users" else event.get_group_id()
    ]
    await pipeline(env, event, "正常回答", adapter=True)
    assert "正常回答" in platform_payload(bot)
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["whitelist", "blacklist"])
@pytest.mark.parametrize("listed", [False, True])
async def test_scope_controls_whitelisted_group_work(env, mode, listed):
    wake.configure(env)
    event, bot = adapter_event("natural")
    env.cfg.update(
        group_scope_mode=mode,
        scope_groups=[event.get_group_id()] if listed else [],
        whitelist_groups=[event.get_group_id()],
    )
    await pipeline(env, event, "正常回答", adapter=True)
    enabled = listed if mode == "whitelist" else not listed
    assert env.plugin._in_scope(event) is enabled
    if enabled:
        assert "正常回答" in platform_payload(bot)
    else:
        bot.transport.assert_not_awaited()
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_private_scope_and_switch(env):
    wake.configure(env)
    event = wake.request_event("private")
    env.cfg.update(scope_groups=[], group_scope_mode="whitelist")
    assert env.plugin._in_scope(event)
    env.cfg["enable_private_chat"] = False
    assert not env.plugin._in_scope(event)


@pytest.mark.parametrize("legacy", [[], ["123", 456], ["test:GroupMessage:123"], ["*"]])
def test_legacy_scope_migration(legacy):
    raw = {"enabled_sessions": legacy}
    view = ConfigView(raw)
    assert view["scope_groups"] == [str(value) for value in legacy]
    assert view["group_scope_mode"] == "whitelist"
    raw["scope_groups"] = ["999"]
    assert ConfigView(raw)["scope_groups"] == ["999"]


def test_file_migration_precedes_schema_defaults_and_preserves_bom(tmp_path):
    path = tmp_path / "config.json"
    original = b'\xef\xbb\xbf{\r\n  "enabled_sessions": ["123"]\r\n}'
    path.write_bytes(original)
    migrate_config_file(path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in path.read_bytes()
    assert json.loads(path.read_text(encoding="utf-8-sig"))["scope_groups"] == ["123"]
    assert next(tmp_path.glob("*.bak")).read_bytes() == original
    before = path.read_bytes()
    migrate_config_file(path)
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_default_scope(env):
    del env.cfg["scope_groups"]
    for gid in DEFAULT_SCOPE_GROUPS:
        assert env.plugin._in_scope(base.event(group=gid))
    assert not env.plugin._in_scope(base.event(group="other"))


@pytest.mark.asyncio
async def test_whitelist_does_not_override_ignored_user(env):
    wake.configure(env)
    event, bot = adapter_event("at")
    env.cfg["whitelist_users"] = [event.get_sender_id()]
    env.cfg["silence"]["ignored_users"] = [event.get_sender_id()]
    await pipeline(env, event, adapter=True)
    bot.transport.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["context", "admission", "capacity", "generation", "timeout"]
)
async def test_internal_failure_is_silent_and_info_logged(env, monkeypatch, failure):
    wake.configure(env)
    event, bot = adapter_event("at")
    info = Mock()
    monkeypatch.setattr(main.logger, "info", info)
    if failure == "context":
        monkeypatch.setattr(
            bridge, "conversation", AsyncMock(side_effect=RuntimeError("private error"))
        )
    elif failure == "admission":
        monkeypatch.setattr(
            env.plugin,
            "_on_message",
            AsyncMock(side_effect=RuntimeError("private error")),
        )
    elif failure == "capacity":
        monkeypatch.setattr(env.plugin.rooms, "get", lambda _: None)
    elif failure == "timeout":
        monkeypatch.setattr(bridge, "native_busy", lambda _: True)
        monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.01)
    if failure == "generation":
        await env.plugin.on_input(event)
        await env.plugin.on_message(event)
        await InternalAgentSubStage._send_llm_error_message(
            object(), event, "private error"
        )
        await env.plugin._complete_output(event)
    else:
        await pipeline(env, event, adapter=True)
    bot.transport.assert_not_awaited()
    logs = [args[0] % args[1:] for args, _ in info.call_args_list]
    assert any(
        f"session={event.unified_msg_origin}" in line
        and f"sender={event.get_sender_id()}" in line
        and f"message_id={event.message_obj.message_id}" in line
        and "reason=" in line
        for line in logs
    )
    assert all("private error" not in line for line in logs)


@pytest.mark.asyncio
async def test_silence_decision_logs_reason_at_info(env, monkeypatch):
    wake.configure(env)
    wake.judge_result(env, "silence")
    info = Mock()
    monkeypatch.setattr(main.logger, "info", info)
    event = base.event(mid="silent-message")
    await env.plugin.on_message(event)
    assert any(
        args[1] == "natural_silence"
        and args[2:]
        == (event.unified_msg_origin, event.get_sender_id(), "silent-message")
        for args, _ in info.call_args_list
    )


@pytest.mark.asyncio
async def test_empty_text_after_declined_tool_gets_plain_reply(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("natural")
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    await event.bot.call_action("send_msg", group_id=987654321, message="premature")
    response = LLMResponse(role="assistant", completion_text="")
    await env.plugin.on_response(event, response)
    event.set_result(event.plain_result(response.completion_text))
    await env.plugin._complete_output(event)
    payload = platform_payload(bot)
    assert "premature" not in payload and "失败" not in payload
    assert silence.FALLBACK_REPLY in payload


@pytest.mark.parametrize(
    "action",
    [
        "send_msg",
        "send_group_msg_async",
        "send_private_msg_rate_limited",
        "send_forward_msg",
        "send_group_forward_msg",
        "send_private_forward_msg",
        "forward_group_single_msg",
        "forward_friend_single_msg",
        "send_group_ai_record",
    ],
)
def test_message_api_classification(action):
    assert is_message_action(action)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    [
        "send_like",
        "set_group_ban",
        "group_poke",
        "friend_poke",
        "send_poke",
        "send_group_sign",
        "get_msg",
        "send_packet",
        "unknown_api",
    ],
)
async def test_nonmessage_api_never_consults_permit(action):
    bot = Mock()
    operation = AsyncMock(return_value={"ok": True})
    setattr(bot, action, operation)
    bot.call_action = AsyncMock(return_value={"ok": True})
    permit = AsyncMock(side_effect=AssertionError("must not gate nonmessage API"))
    proxy = EventBotProxy(bot, permit)
    assert await getattr(proxy, action)(user_id=123) == {"ok": True}
    assert await proxy.call_action(action, user_id=123) == {"ok": True}
    permit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "reply"])
async def test_explicit_wake_not_dropped_by_cooldown(env, kind):
    wake.configure(env)
    env.cfg.update(takeover_wake_messages=False, direct_wake_interval=10)
    first, _ = adapter_event(kind)
    second, bot = adapter_event(kind)
    second.message_obj.message_id = "second-request"
    await env.plugin.on_input(first)
    await pipeline(env, second, "接着说。", adapter=True)
    assert "接着说。" in platform_payload(bot)


@pytest.mark.asyncio
async def test_declined_tool_info_contains_reason_and_ingress(env, monkeypatch):
    wake.configure(env)
    event, bot = adapter_event("at")
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    info = Mock()
    monkeypatch.setattr(main.logger, "info", info)
    result = await event.bot.send_group_msg(group_id=111, message="wrong room")
    assert result.status == "cancelled"
    assert any(
        args[1:]
        == (
            "autonomous_cross_session_send",
            event.unified_msg_origin,
            event.get_sender_id(),
            str(event.message_obj.message_id),
        )
        for args, _ in info.call_args_list
    )
    event.set_result(event.plain_result("正常文字"))
    await env.plugin._complete_output(event)
    assert "正常文字" in platform_payload(bot)
