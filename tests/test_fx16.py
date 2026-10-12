"""Release-16 regressions through Core stages and the mocked OneBot transport."""

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from astrbot.api.message_components import Image
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.agent.tool import ToolSet
from astrbot.core.star.star import star_map
from astrbot.core.star.star_handler import (
    EventType,
    StarHandlerMetadata,
    star_handlers_registry,
)
from astrbot.core.tools.message_tools import SendMessageToUserTool

from .. import silence
from ..main import MARK
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
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("answer", ["正常模型结果", "[PASS]。"])
@pytest.mark.parametrize("tool_text", ["工具生成的正文", "[不回复]。"])
async def test_builtin_send_tool_joins_required_outbound(env, kind, answer, tool_text):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)

    async def send_message(session, chain):
        target = (
            {"user_id": 123456789} if kind == "private" else {"group_id": 987654321}
        )
        await bot.transport(
            "send_msg",
            **target,
            message=[{"type": "text", "data": {"text": chain.get_plain_text()}}],
        )
        return True

    env.ctx.send_message = AsyncMock(side_effect=send_message)
    entered, release = asyncio.Event(), asyncio.Event()
    running = asyncio.create_task(
        pipeline(env, event, answer, adapter=True, generation_gate=(entered, release))
    )
    try:
        await asyncio.wait_for(entered.wait(), 4)
        tool = SendMessageToUserTool()
        request = ProviderRequest(
            func_tool=ToolSet(tools=[tool]), conversation=NS(cid=env.cid)
        )
        await env.plugin.on_request(event, request)
        assert "send_message_to_user" in request.func_tool.names()
        result = await request.func_tool.get_tool("send_message_to_user").call(
            NS(context=NS(event=event, context=env.ctx)),
            messages=[{"type": "plain", "text": tool_text}],
        )
        assert "Message sent" in result
    finally:
        release.set()
        await running

    encoded = platform_payload(bot)
    assert env.ctx.send_message.await_count == 0
    if tool_text == "[不回复]。":
        assert (answer if answer != "[PASS]。" else silence.FALLBACK_REPLY) in encoded
    else:
        assert tool_text in encoded
        if answer not in {"[PASS]。", "[不回复]。"}:
            assert answer in encoded
    assert event.get_extra(MARK)["sent_parts"] == 1


@pytest.mark.asyncio
async def test_stop_only_decorator_preserves_core_body(env, monkeypatch):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")

    async def stop_only(ev):
        ev.stop_event()

    module = "fx16_stop_only"
    monkeypatch.setitem(
        star_map, module, NS(name=module, activated=True, reserved=False)
    )
    star_handlers_registry.append(
        StarHandlerMetadata(
            EventType.OnDecoratingResultEvent,
            module,
            module,
            module,
            stop_only,
            [],
            extras_configs={"priority": -50000},
        )
    )
    await pipeline(env, event, "合成正文第一段。合成正文第二段。", adapter=True)
    encoded = platform_payload(bot)
    assert "合成正文第一段" in encoded and "合成正文第二段" in encoded
    assert silence.FALLBACK_REPLY not in encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
async def test_queued_native_receipt_has_mapping_view(env, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    target = {"user_id": 123456789} if kind == "private" else {"group_id": 987654321}
    result = await event.bot.send_msg(message="合成正文", **target)
    assert isinstance(result, Mapping)
    assert result.get("message_id") is None  # queued; transport has not run
    await env.plugin._complete_output(event)
    platform_payload(bot)
    assert result.status == "delivered"
    assert result["message_id"] == result.get("message_id") == 42
    assert dict(result) == {"message_id": 42}


@pytest.mark.asyncio
async def test_tool_scope_leaves_default_off_path_untouched(env):
    event, bot = adapter_event("at")
    assert not event.get_extra(MARK)
    delivered = AsyncMock(return_value=True)
    env.ctx.send_message = delivered
    tool = SendMessageToUserTool()
    result = await tool.call(
        NS(context=NS(event=event, context=env.ctx)),
        messages=[{"type": "plain", "text": "合成普通工具发送"}],
    )
    assert "Message sent" in result
    delivered.assert_awaited_once()
    assert bot.transport.await_count == 0


@pytest.mark.asyncio
async def test_tool_media_survives_stop_chain(env, monkeypatch):
    wake.configure(env)
    wake.judge_result(env, "silence")
    monkeypatch.setattr(Image, "convert_to_base64", AsyncMock(return_value="aW1hZ2U="))
    event, bot = adapter_event("at")
    entered, release = asyncio.Event(), asyncio.Event()
    running = asyncio.create_task(
        pipeline(
            env,
            event,
            "[PASS]。",
            adapter=True,
            generation_gate=(entered, release),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 4)
        tool = SendMessageToUserTool()
        request = ProviderRequest(
            func_tool=ToolSet(tools=[tool]), conversation=NS(cid=env.cid)
        )
        await env.plugin.on_request(event, request)
        result = await request.func_tool.get_tool("send_message_to_user").call(
            NS(context=NS(event=event, context=env.ctx)),
            messages=[{"type": "image", "url": "https://example.invalid/one.png"}],
        )
        assert "Message sent" in result
        event.stop_event()
    finally:
        release.set()
        await running
    assert '"type": "image"' in platform_payload(bot)
