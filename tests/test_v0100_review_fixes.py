"""Release-review regressions using synthetic messages and native Core APIs."""

import asyncio
import time
from copy import copy, deepcopy
from itertools import product
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from astrbot.api.message_components import Image
from astrbot.core.astr_agent_run_util import run_agent
from astrbot.core import astr_main_agent
from astrbot.core.astr_main_agent import MainAgentBuildConfig
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse

from ..main import MARK
from ..state import RoomBook, scope_id
from .test_v0100_takeover import incoming, pipeline
from . import test_v020 as previous
from .test_integration import collect_request

env = previous.env

SETTINGS = ("streaming_response", "show_tool_use_status", "show_tool_call_result")
COMBINATIONS = list(product((False, True), repeat=len(SETTINGS)))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "prefix", "name", "private"])
@pytest.mark.parametrize("takeover", [False, True])
@pytest.mark.parametrize("values", COMBINATIONS)
async def test_runtime_settings_preserve_one_reply_and_native_tools(
    env, kind, takeover, values
):
    env.cfg.update(
        takeover_wake_messages=takeover,
        enable_private_chat=takeover,
        wake_names=["测试助手"],
    )
    env.settings["provider_settings"].update(zip(SETTINGS, values))
    original = deepcopy(env.settings)
    ev, sent = incoming(kind), []
    await pipeline(env, ev, sent)
    assert sent == [ev]
    request = ev.get_extra("_test_generation_request")
    assert "web_search" in [tool.name for tool in request.func_tool.tools]
    assert env.settings == original
    if takeover:
        env.plugin.client.evaluate.assert_not_awaited()
        assert ev.get_extra(MARK)["committed"]
        assert ev.get_extra("enable_streaming") is False
        assert await env.plugin.ledger.acount(ev.unified_msg_origin, "sent") == 1
    else:
        env.plugin.client.evaluate.assert_not_awaited()
        assert ev.get_extra("enable_streaming") is None
    ev.stop_event.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
@pytest.mark.parametrize("show_result", [False, True])
async def test_native_tool_status_does_not_abort_or_count_as_delivery(
    env, kind, show_result
):
    env.cfg.update(takeover_wake_messages=True, enable_private_chat=True)
    ev = incoming(kind)
    transport = AsyncMock()
    ev.send = transport
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    meta = ev.get_extra(MARK)
    assert meta
    finished = False

    async def steps():
        nonlocal finished
        yield NS(type="tool_call", data={"chain": MessageChain().message("search")})
        yield NS(
            type="tool_call_result",
            data={"chain": MessageChain().message("synthetic search result")},
        )
        assert not ev.is_stopped()
        assert not meta.get("committed")
        transport.assert_not_awaited()
        response = LLMResponse(role="assistant", completion_text="合成的检索回答")
        await env.plugin.on_response(ev, response)
        finished = True
        yield NS(
            type="llm_result",
            data={"chain": MessageChain().message(response.completion_text)},
        )

    runner = NS(
        run_context=NS(context=NS(event=ev)),
        streaming=False,
        done=lambda: finished,
        request_stop=lambda: None,
        step=steps,
    )
    async for _ in run_agent(
        runner, show_tool_use=True, show_tool_call_result=show_result
    ):
        await env.plugin.before_output(ev)
        if not ev.is_stopped():
            await ev.send(ev.get_result())
    await env.plugin.after_sent(ev)
    await ev.get_extra("_jev_complete_output")()
    assert not ev.is_stopped()
    transport.assert_awaited_once()
    assert transport.await_args.args[0].get_plain_text() == "合成的检索回答"
    assert meta["sent_parts"] == env.plugin.metrics.replies == 1


@pytest.mark.asyncio
async def test_takeover_disables_event_stream_override(env):
    env.cfg["takeover_wake_messages"] = True
    ev = incoming("at")
    ev.set_extra("enable_streaming", True)
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    assert ev.get_extra(MARK)
    assert ev.get_extra("enable_streaming") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["sender", "umo", "wildcard"])
async def test_private_quiet_persists_blocks_and_resumes(env, scope):
    env.cfg["enable_private_chat"] = True
    quiet = incoming("private", text="jev静音 10")
    env.cfg["enabled_sessions"] = [
        {"sender": "human", "umo": quiet.unified_msg_origin, "wildcard": "*"}[scope]
    ]
    assert [result async for result in env.plugin.quiet_room(quiet, 10)]
    room = env.plugin.rooms.get(quiet.unified_msg_origin)
    assert room.quiet_until > time.monotonic()
    key = "room_quiet:" + scope_id(quiet.unified_msg_origin)
    assert await env.plugin.ledger.acooldown(key) > time.time()
    env.plugin.rooms = RoomBook()
    blocked, sent = incoming("private", "blocked"), []
    await pipeline(env, blocked, sent)
    assert blocked.is_stopped() and not sent
    env.plugin.client.evaluate.assert_not_awaited()
    resume = incoming("private", "resume", "jev开口")
    await env.plugin.on_input(resume)
    assert not resume.is_stopped()
    assert [result async for result in env.plugin.resume_room(resume)]
    assert await env.plugin.ledger.acooldown(key) == 0
    reopened = incoming("private", "reopened")
    await pipeline(env, reopened, sent)
    assert sent == [reopened]
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["quiet_room", "resume_room"])
@pytest.mark.parametrize(
    "reason", ["disabled", "private_off", "dry_run", "unlisted", "denied"]
)
async def test_private_management_respects_enablement_and_scope(env, operation, reason):
    env.cfg["enable_private_chat"] = True
    if reason in {"disabled", "private_off", "dry_run"}:
        env.cfg[
            {
                "disabled": "enabled",
                "private_off": "enable_private_chat",
                "dry_run": "dry_run",
            }[reason]
        ] = reason == "dry_run"
    elif reason == "unlisted":
        env.cfg["enabled_sessions"] = ["another"]
    else:
        env.cfg["disabled_sessions"] = ["human"]
    ev = incoming("private", text="jev静音" if operation == "quiet_room" else "jev开口")
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    before = time.monotonic() + 600 if operation == "resume_room" else 0
    room.quiet_until = before
    assert [result async for result in getattr(env.plugin, operation)(ev)]
    if reason == "unlisted":
        assert (room.quiet_until > time.monotonic()) == (operation == "quiet_room")
    else:
        assert room.quiet_until == before


@pytest.mark.asyncio
async def test_private_quiet_cancels_pending_generation(env):
    env.cfg["enable_private_chat"] = True
    ev = incoming("private")
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    assert ev.get_extra(MARK)
    quiet = incoming("private", "quiet", "jev静音 10")
    assert [result async for result in env.plugin.quiet_room(quiet, 10)]
    assert ev.is_stopped() and ev.get_extra(MARK)["rejected"]
    assert not env.plugin.pending_events


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "prefix", "name", "private"])
@pytest.mark.parametrize(
    ("configured_prefix", "native_prefix", "bot_prefix"),
    [("!", "!", []), ("/!", "!", ["/"])],
)
async def test_takeover_bypasses_provider_wake_prefix_after_jev_admission(
    env, kind, configured_prefix, native_prefix, bot_prefix
):
    """An admitted takeover must reach Core even when provider wake_prefix is set."""
    env.cfg.update(
        takeover_wake_messages=True,
        enable_private_chat=True,
        wake_names=["测试助手"],
    )
    env.settings["wake_prefix"] = bot_prefix
    env.settings["provider_settings"]["wake_prefix"] = configured_prefix
    original_settings = deepcopy(env.settings)
    ev = incoming(kind)
    original = ev.get_message_str()
    transport = AsyncMock()
    ev.send = transport
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    assert ev.get_extra("provider_request") is None
    request, _ = await collect_request(
        ev,
        env.ctx,
        MainAgentBuildConfig(
            tool_call_timeout=30,
            provider_wake_prefix=native_prefix,
            provider_settings=env.settings["provider_settings"],
        ),
    )
    assert request is not None
    assert request.prompt == original
    assert request.conversation.cid == env.cid
    assert ev.get_extra("provider_request") is request
    await env.plugin.restore_takeover_message(ev, request)
    await env.plugin.on_request(ev, request)
    assert ev.get_message_str() == original
    response = LLMResponse(role="assistant", completion_text="合成的回答")
    await env.plugin.on_response(ev, response)
    ev.set_result(ev.plain_result(response.completion_text))
    await env.plugin.before_output(ev)
    await ev.send(ev.get_result())
    await env.plugin.after_sent(ev)
    await ev.get_extra("_jev_complete_output")()
    transport.assert_awaited_once()
    assert env.plugin.metrics.replies == 1
    assert env.settings == original_settings
    assert ev.get_extra(MARK) and ev.get_extra("_jev_takeover")
    assert "unsupported_runtime_settings" not in [
        reason for _, reason, _ in env.plugin.audit_queue
    ]
    assert env.plugin.client.evaluate.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
async def test_provider_prefix_respects_disabled_takeover_and_allows_explicit_wake(
    env, kind
):
    env.settings["provider_settings"]["wake_prefix"] = "!"
    ev = incoming(kind)
    original = ev.get_message_str()
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    config = MainAgentBuildConfig(
        tool_call_timeout=30,
        provider_wake_prefix="!",
        provider_settings=env.settings["provider_settings"],
    )
    if hasattr(astr_main_agent, "collect_initial_request"):
        request, _ = await collect_request(ev, env.ctx, config)
        assert request is None
    else:
        assert (
            await astr_main_agent.build_main_agent(
                event=ev, plugin_context=env.ctx, config=config, provider=object()
            )
            is None
        )
    assert ev.get_message_str() == original
    env.plugin.client.evaluate.assert_not_awaited()

    env.cfg.update(takeover_wake_messages=True, enable_private_chat=True)
    env.plugin.client.evaluate.return_value = {
        **env.response,
        "values": dict(addressed=0, continuation=0, worthwhile=0, intrusive=1),
    }
    explicit = incoming(kind, mid="explicit")
    transport = AsyncMock()
    explicit.send = transport
    await env.plugin.on_input(explicit)
    await env.plugin.on_message(explicit)
    assert not explicit.is_stopped()
    assert explicit.get_extra(MARK) and explicit.get_extra("_jev_takeover")
    request, _ = await collect_request(explicit, env.ctx, config)
    assert request is not None and request.prompt == original
    await env.plugin.restore_takeover_message(explicit, request)
    await env.plugin.on_request(explicit, request)
    assert explicit.get_message_str() == original
    response = LLMResponse(role="assistant", completion_text="合成的回答")
    await env.plugin.on_response(explicit, response)
    explicit.set_result(explicit.plain_result(response.completion_text))
    await env.plugin.before_output(explicit)
    if not explicit.is_stopped():
        await explicit.send(explicit.get_result())
    await env.plugin.after_sent(explicit)
    await explicit.get_extra("_jev_complete_output")()
    transport.assert_awaited_once()
    assert transport.await_args.args[0].get_plain_text() == "合成的回答"
    explicit.stop_event.assert_not_called()
    env.plugin.client.evaluate.assert_not_awaited()
    assert env.plugin.metrics.replies == 1
    assert await env.plugin.ledger.acount(explicit.unified_msg_origin, "sent") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
async def test_existing_provider_prefix_is_not_doubled(env, kind):
    env.cfg.update(takeover_wake_messages=True, enable_private_chat=True)
    env.settings["provider_settings"]["wake_prefix"] = "!"
    ev = incoming(kind, text="!模型上下文怎么处理")
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    request, _ = await collect_request(
        ev,
        env.ctx,
        MainAgentBuildConfig(
            tool_call_timeout=30,
            provider_wake_prefix="!",
            provider_settings=env.settings["provider_settings"],
        ),
    )
    assert request.prompt == "模型上下文怎么处理"
    assert ev.get_message_str() == "!模型上下文怎么处理"
    assert "native_prefix_original" not in ev.get_extra(MARK)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["rejected", "expired", "silent"])
async def test_prefix_shim_restores_event_when_generation_is_rejected(env, reason):
    env.cfg["takeover_wake_messages"] = True
    env.settings["provider_settings"]["wake_prefix"] = "!"
    ev = incoming("at")
    original = ev.get_message_str()
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    assert ev.get_message_str() == "!" + original
    if reason == "expired":
        env.plugin._expire(ev)
        await asyncio.sleep(0.05)
    elif reason == "silent":
        env.plugin._finish_silent(ev)
    else:
        env.plugin._reject(ev, None, "cancelled_before_generation")
    assert ev.get_message_str() == original
    assert ev.is_stopped()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
@pytest.mark.parametrize("phase", ["judge", "generate"])
async def test_provider_prefix_burst_keeps_one_native_reply(
    env, monkeypatch, kind, phase
):
    env.cfg.update(
        takeover_wake_messages=True,
        enable_private_chat=True,
        burst_merge_enabled=True,
        evaluation_interval=1,
    )
    env.settings["provider_settings"]["wake_prefix"] = "!"
    entered, release = asyncio.Event(), asyncio.Event()
    first = incoming(kind, "first", "看看模型上下文")
    second = incoming(kind, "second", "补充模型上下文的问题")
    sent, prompts = [], []
    from .. import bridge

    snapshot = bridge.build_snapshot

    async def judge(*args):
        if phase == "judge" and not entered.is_set():
            entered.set()
            await release.wait()
        return await snapshot(*args)

    monkeypatch.setattr(bridge, "build_snapshot", judge)

    async def native_reply(event):
        event.send = AsyncMock()
        await env.plugin.on_input(event)
        await env.plugin.on_message(event)
        if event.is_stopped():
            return
        request, _ = await collect_request(
            event,
            env.ctx,
            MainAgentBuildConfig(tool_call_timeout=30, provider_wake_prefix="!"),
        )
        assert request is not None
        await env.plugin.restore_takeover_message(event, request)
        await env.plugin.on_request(event, request)
        assert event.message_str == request.prompt
        if phase == "generate" and event is first:
            entered.set()
            await release.wait()
        response = LLMResponse(role="assistant", completion_text="合成的合并回答")
        await env.plugin.on_response(event, response)
        event.set_result(event.plain_result(response.completion_text))
        await env.plugin.before_output(event)
        if not event.is_stopped():
            await event.send(event.get_result())
            await env.plugin.after_sent(event)
            await event.get_extra("_jev_complete_output")()
            sent.append(event)
            prompts.append(request.prompt)

    task = asyncio.create_task(native_reply(first))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await native_reply(second)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert sent == [second]
    assert first.is_stopped()
    assert second.get_extra("_jev_merged_message_ids") == ["first", "second"]
    assert "[1] 看看模型上下文" in prompts[0]
    assert "[2] 补充模型上下文的问题" in prompts[0]
    assert not prompts[0].startswith("!")
    assert env.plugin.metrics.replies == 1


@pytest.mark.asyncio
async def test_provider_prefix_native_collector_keeps_images_and_restores_via_hook(
    env, monkeypatch
):
    from astrbot.core.pipeline.context import call_event_hook
    from astrbot.core.star.star import star_map
    from astrbot.core.star.star_handler import EventType, star_handlers_registry
    from ..main import JevActiveReply

    env.cfg["takeover_wake_messages"] = True
    env.settings["provider_settings"]["wake_prefix"] = "!"
    ev = incoming("at")
    original = ev.get_message_str()
    attachment = Image.fromURL("https://example.invalid/synthetic.png")
    ev.message_obj.message.append(attachment)
    parts = list(ev.get_messages())
    monkeypatch.setattr(
        Image, "convert_to_file_path", AsyncMock(return_value="/test/synthetic.png")
    )
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    request, _ = await collect_request(
        ev,
        env.ctx,
        MainAgentBuildConfig(tool_call_timeout=30, provider_wake_prefix="!"),
    )
    assert request.prompt == original
    assert request.image_urls == ["/test/synthetic.png"]
    assert request.conversation.cid == env.cid
    assert ev.get_messages() == parts
    handler = JevActiveReply.restore_takeover_message
    metadata = copy(
        star_handlers_registry.get_handler_by_full_name(
            handler.__module__ + "_restore_takeover_message"
        )
    )
    assert metadata.event_type == EventType.OnLLMRequestEvent
    assert metadata.extras_configs["priority"] == 1000000
    metadata.handler = env.plugin.restore_takeover_message
    monkeypatch.setitem(star_map, metadata.handler_module_path, NS(name="jev-test"))
    monkeypatch.setattr(
        star_handlers_registry, "get_handlers_by_event_type", lambda *a, **k: [metadata]
    )
    assert not await call_event_hook(ev, EventType.OnLLMRequestEvent, request)
    assert ev.get_message_str() == original
    assert request.prompt == original
