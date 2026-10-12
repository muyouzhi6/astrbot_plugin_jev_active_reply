"""Strict event-level delivery; synthetic payloads only."""

import asyncio

import pytest

from . import test_v0101_explicit_wake as wake
from .test_fx4 import payload
from .test_v0102_wake_warnings import adapter_event, delivery_hooks, env, pipeline

__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
async def test_queue_snapshots_and_late_send_rejected(env):
    from ..delivery import DeliveryDeclined

    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    chain = event.plain_result("Original")
    await event.send(chain)
    chain.chain.clear()
    await env.plugin._complete_output(event)
    with pytest.raises(DeliveryDeclined):
        await event.send(event.plain_result("Late"))
    await env.plugin._complete_output(event)
    assert bot.transport.await_count == 1
    assert "Original" in payload(bot) and "Late" not in payload(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "parts",
    [
        ["First", "Second", "。"],
        ["First", "[PA", "SS]", "Second"],
        ["First", "[PASS]。", "Second"],
    ],
)
async def test_fragments_single_submission(env, kind, native, parts):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)

    async def emit(ev):
        for text in parts:
            if native:
                target = (
                    {"user_id": 123456789}
                    if kind == "private"
                    else {"group_id": 987654321}
                )
                await ev.bot.send_msg(message=text, **target)
            else:
                await ev.send(ev.plain_result(text))

    await pipeline(env, event, adapter=True, native_send=emit)
    encoded = payload(bot)
    assert encoded.count("First") == encoded.count("Second") == 1
    assert "[PASS]" not in encoded and "[PA" not in encoded
    assert bot.transport.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("finish_while_waiting", [False, True])
async def test_concurrent_submission_and_completion(
    env, monkeypatch, kind, finish_while_waiting
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    original = env.plugin._permit
    entered, release = asyncio.Event(), asyncio.Event()
    paused_task = None

    async def permit(ev):
        if asyncio.current_task() is paused_task:
            entered.set()
            await release.wait()
        return await original(ev)

    monkeypatch.setattr(env.plugin, "_permit", permit)
    await event.send(event.plain_result("Alpha"))
    paused_task = asyncio.create_task(event.send(event.plain_result("Bravo")))
    finishing = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await event.send(event.plain_result("Charlie"))
        if finish_while_waiting:
            finishing = asyncio.create_task(env.plugin._complete_output(event))
            await asyncio.sleep(0.01)
            assert not finishing.done()
        release.set()
        await paused_task
        if finishing:
            await finishing
        else:
            await env.plugin._complete_output(event)
    finally:
        release.set()
        await asyncio.gather(
            paused_task, *([finishing] if finishing else []), return_exceptions=True
        )
    encoded = payload(bot)
    assert all(encoded.count(s) == 1 for s in ("Alpha", "Bravo", "Charlie"))
    assert encoded.index("Alpha") < encoded.index("Bravo") < encoded.index("Charlie")
    assert bot.transport.await_count == 1
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
