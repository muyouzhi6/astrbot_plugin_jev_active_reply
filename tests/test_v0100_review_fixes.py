"""Release-review regressions using synthetic messages and native Core APIs."""

import time
from copy import deepcopy
from itertools import product
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from astrbot.core.astr_agent_run_util import run_agent
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse

from ..main import MARK
from ..state import RoomBook, scope_id
from .test_v0100_takeover import incoming, pipeline
from . import test_v020 as previous

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
        env.plugin.client.evaluate.assert_awaited_once()
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
    env.plugin.client.evaluate.assert_awaited_once()


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
