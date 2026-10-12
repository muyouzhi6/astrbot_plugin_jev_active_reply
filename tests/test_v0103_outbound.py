"""Regressions exercise real Core decoration and OneBot serialization."""

import pytest
from unittest.mock import AsyncMock

from astrbot.api.message_components import Node, Nodes, Plain
from astrbot.core import html_renderer

from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import (
    adapter_event,
    delivery_hooks,
    env,
    pipeline,
    platform_payload,
)

__all__ = ["delivery_hooks", "env"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_mixed_forward_is_one_platform_send(env, kind, wrapped):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    node = Node(uin="123", name="Bot", content=[Plain("Synthetic answer")])
    await pipeline(
        env,
        event,
        adapter=True,
        parts=[
            Plain("Synthetic introduction"),
            Nodes([node]) if wrapped else node,
            Plain("Synthetic conclusion"),
        ],
    )
    payload = platform_payload(bot)
    assert payload.index("Synthetic introduction") < payload.index("Synthetic answer")
    assert payload.index("Synthetic answer") < payload.index("Synthetic conclusion")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("admin", [False, True])
@pytest.mark.parametrize("outcome", ["reply", "silence"])
@pytest.mark.parametrize("text", ["[PASS]", "[PASS]。", "[不回复]", "完整回答"])
async def test_explicit_platform_send_matrix(env, kind, admin, outcome, text):
    wake.configure(env)
    wake.judge_result(env, outcome)
    event, bot = adapter_event(kind)
    event.role = "admin" if admin else "member"
    assert event.is_admin() is admin
    await pipeline(env, event, text=text, adapter=True)
    payload = platform_payload(bot)
    assert "完整回答" in payload or "我在，请再说一下你的问题。" in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("positional", [False, True])
async def test_native_marker_after_body_does_not_send_fallback(env, positional):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")

    async def send(ev):
        for text in ("完整回答", "[PASS]。"):
            if positional:
                await ev.bot.send(ev.message_obj.raw_message, text)
            else:
                await ev.bot.send_group_msg(group_id=987654321, message=text)

    await pipeline(env, event, adapter=True, native_send=send)
    assert "完整回答" in platform_payload(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("punctuation", ["[PASS]。", "[不回复]！"])
async def test_marker_punctuation_before_forward_is_not_a_separate_send(
    env, punctuation
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await pipeline(
        env,
        event,
        adapter=True,
        parts=[
            Plain(punctuation),
            Node(uin="123", name="Bot", content=[Plain("完整回答")]),
        ],
    )
    assert "完整回答" in platform_payload(bot)


@pytest.mark.asyncio
async def test_reasoning_is_clean_before_t2i(env, monkeypatch):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    render = AsyncMock(return_value="https://example.invalid/render.png")
    from astrbot.api.message_components import Image

    monkeypatch.setattr(Image, "convert_to_base64", AsyncMock(return_value="aW1hZ2U="))
    monkeypatch.setattr(html_renderer, "render_t2i", render)
    await pipeline(
        env,
        event,
        text="完整回答。" * 100,
        reasoning="[PASS]分析问题[不回复]",
        display_reasoning=True,
        t2i=True,
        adapter=True,
    )
    render.assert_awaited_once()
    rendered = render.await_args.args[0]
    assert "分析问题" in rendered
    assert "[PASS]" not in rendered and "[不回复]" not in rendered
    platform_payload(bot)
