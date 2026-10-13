"""Native CQ semantics and synthetic production lifecycle regressions."""

import asyncio
import json

import pytest

from .. import bridge
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import adapter_event, delivery_hooks, env, pipeline

__all__ = ["env", "delivery_hooks"]


def payload(bot):
    assert bot.transport.await_count == 1
    params = bot.transport.await_args.kwargs
    value = params.get("messages", params.get("message"))
    assert value
    assert "[PASS]" not in json.dumps(value)
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("escape", [None, False, True])
@pytest.mark.parametrize("media", ["image", "record"])
async def test_cq_string_media(env, kind, escape, media):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    cq = f"[CQ:{media},file=base64://c3ludGhldGlj]"

    async def emit(ev):
        target = (
            {"user_id": 123456789} if kind == "private" else {"group_id": 987654321}
        )
        if escape is not None:
            target["auto_escape"] = escape
        await ev.bot.send_msg(**target, message=cq)

    await pipeline(env, event, adapter=True, native_send=emit)
    value = payload(bot)
    expected = (
        {"type": "text", "data": {"text": cq}}
        if escape
        else {"type": media, "data": {"file": "base64://c3ludGhldGlj"}}
    )
    assert value == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("escape", [None, False, True])
async def test_cq_escaped_text_and_media_parameters(env, escape):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    text = "&#91;CQ:image,file=literal&#93;&amp;"
    cq = "[CQ:image,file=synthetic&#44;name&amp;value]"

    async def emit(ev):
        options = {} if escape is None else {"auto_escape": escape}
        await ev.bot.send_msg(group_id=987654321, message=text + cq, **options)

    await pipeline(env, event, adapter=True, native_send=emit)
    assert payload(bot) == (
        [{"type": "text", "data": {"text": text + cq}}]
        if escape
        else [
            {"type": "text", "data": {"text": "[CQ:image,file=literal]&"}},
            {"type": "image", "data": {"file": "synthetic,name&value"}},
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("mode", ["history_stop", "lease", "wait", "multi"])
async def test_log_lifecycle_combinations(env, monkeypatch, kind, mode):
    wake.configure(env)
    wake.judge_result(env, "silence")
    event, bot = adapter_event(kind)
    if mode == "wait":
        monkeypatch.setattr(bridge, "native_busy", lambda ev: True)
        monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.015)
        await pipeline(env, event, "[PASS]。", adapter=True)
    elif mode == "multi":

        async def emit(ev):
            for text in ["First body", "[PA", "SS]。", "Second body"]:
                await ev.send(ev.plain_result(text))

        await pipeline(env, event, adapter=True, native_send=emit)
    else:
        original = bridge.build_snapshot

        async def evaluate(*args):
            result = await original(*args)
            env.history = '[{"role":"assistant","content":"synthetic advance"}]'
            env.plugin.rooms.get(event.unified_msg_origin).generation += 1
            return result

        monkeypatch.setattr(bridge, "build_snapshot", evaluate)
        entered, release = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(
            pipeline(
                env, event, "[不回复]", adapter=True, generation_gate=(entered, release)
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if mode == "lease":
                env.plugin._expire(event)
                await asyncio.sleep(0.03)
            else:
                target = (
                    {"user_id": 123456789}
                    if kind == "private"
                    else {"group_id": 987654321}
                )
                await event.bot.send_msg(
                    **target,
                    message=[
                        {"type": "image", "data": {"file": "base64://c3ludGhldGlj"}}
                    ],
                )
                event.stop_event()
        finally:
            release.set()
            await task
    if mode in {"wait", "lease"}:
        bot.transport.assert_not_awaited()
        assert not env.plugin.rooms.get(event.unified_msg_origin).pending
        return
    value = json.dumps(payload(bot), ensure_ascii=False)
    assert "[不回复]" not in value
    if mode == "multi":
        assert "First body" in value and "Second body" in value
    if mode == "history_stop":
        assert "image" in value
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
