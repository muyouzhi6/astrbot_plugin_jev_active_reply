"""Seventh-review regressions; synthetic input and mocked platform transport."""

import asyncio

import pytest

from .. import bridge, silence
from ..main import MARK
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import adapter_event, delivery_hooks, env, pipeline
from .test_fx5 import texts

__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("fail_at", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("persistent", [False, True])
async def test_later_conversation_failure(env, monkeypatch, kind, fail_at, persistent):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    original = bridge.conversation
    calls = 0

    async def read(*args):
        nonlocal calls
        calls += 1
        if calls == fail_at or (persistent and calls >= fail_at):
            raise TimeoutError("synthetic later read failure")
        return await original(*args)

    monkeypatch.setattr(bridge, "conversation", read)
    await pipeline(env, event, adapter=True)
    assert bot.transport.await_count == 1
    assert any(texts(bot))
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
    assert not event.get_extra(MARK, {}).get("lease")
    if persistent:
        assert texts(bot) == [silence.FALLBACK_REPLY]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("fail_at", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("cancel", ["switch", "recall"])
async def test_recovered_read_preserves_cancellation(
    env, monkeypatch, kind, fail_at, cancel
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    original = bridge.conversation
    calls = 0

    async def read(*args):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            if cancel == "recall":
                event.set_extra(silence.RECALL, True)
            raise TimeoutError("synthetic read failure before cancellation")
        cid, conv = await original(*args)
        return (
            "different-conversation" if cancel == "switch" and calls > fail_at else cid
        ), conv

    monkeypatch.setattr(bridge, "conversation", read)
    await pipeline(env, event, adapter=True)
    assert bot.transport.await_count == 0
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("payload", ["。", [{"type": "text", "data": {"text": "。"}}]])
async def test_native_punctuation_not_standalone(env, kind, payload):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    receipts = []

    async def emit(ev):
        target = (
            {"user_id": 123456789} if kind == "private" else {"group_id": 987654321}
        )
        for message in ["Synthetic answer", payload]:
            receipts.append(await ev.bot.send_msg(message=message, **target))

    await pipeline(env, event, adapter=True, native_send=emit)
    assert texts(bot) == ["Synthetic answer。"]
    assert all(
        r.status == "delivered" and r.receipt == {"message_id": 42} for r in receipts
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("mode", ["stop", "media_stop", "history", "wait", "lease"])
async def test_online_lifecycle(env, monkeypatch, kind, mode):
    wake.configure(env)
    wake.judge_result(env, "silence")
    event, bot = adapter_event(kind)
    if mode == "wait":
        monkeypatch.setattr(bridge, "native_busy", lambda ev: True)
        monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.015)
        await pipeline(env, event, adapter=True)
    else:
        if mode == "history":
            original = env.plugin.client.evaluate

            async def judge(*args):
                response = await original(*args)
                env.plugin.rooms.get(event.unified_msg_origin).generation += 1
                monkeypatch.setattr(bridge, "fingerprint", lambda conv: "new-history")
                return response

            monkeypatch.setattr(env.plugin.client, "evaluate", judge)
        entered, release = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(
            pipeline(
                env,
                event,
                text="[PASS]。",
                adapter=True,
                generation_gate=(entered, release),
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if mode == "media_stop":
                target = (
                    {"user_id": 123456789}
                    if kind == "private"
                    else {"group_id": 987654321}
                )
                await event.bot.send_msg(
                    message=[
                        {"type": "image", "data": {"file": "base64://c3ludGhldGlj"}}
                    ],
                    **target,
                )
            if mode in ("stop", "media_stop"):
                event.stop_event()
            if mode == "lease":
                env.plugin._expire(event)
                await asyncio.sleep(0.03)
            release.set()
            await asyncio.wait_for(task, 3)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert bot.transport.await_count == 1
    assert "[PASS]" not in str(bot.transport.await_args_list)
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
