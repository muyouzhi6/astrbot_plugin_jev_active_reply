import asyncio

import pytest

from .. import bridge
from ..main import MARK
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import adapter_event, delivery_hooks, env, pipeline
from .test_fx4 import payload

__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("persistent", [False, True])
async def test_ingress_conversation_failure(env, monkeypatch, kind, persistent):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    original = bridge.conversation
    calls = 0

    async def transient(*args):
        nonlocal calls
        calls += 1
        if persistent or calls == 1:
            raise TimeoutError("synthetic conversation read timeout")
        return await original(*args)

    monkeypatch.setattr(bridge, "conversation", transient)
    await pipeline(env, event, adapter=True)
    assert bot.transport.await_count == 0, (
        event.is_stopped(),
        event.get_extra(MARK),
        payload(bot),
    )
    assert event.is_stopped()
    assert "我在，请再说一下你的问题。" not in payload(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("priority", [200000, 0, -200000])
async def test_decorator_stop_without_delivery(env, kind, priority):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)

    async def stop(ev):
        ev.stop_event()

    await pipeline(
        env, event, adapter=True, native_send=stop, decoration_priority=priority
    )
    assert bot.transport.await_count == 1, payload(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("text", ["", "[PASS]", "[不回复]"])
async def test_inflight_media_completion_no_duplicate(env, kind, text):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    sending, delivered = asyncio.Event(), asyncio.Event()

    async def transport(*args, **kwargs):
        sending.set()
        await delivered.wait()
        return {"message_id": 42}

    bot.transport.side_effect = transport
    task = asyncio.create_task(
        pipeline(
            env, event, text=text, adapter=True, generation_gate=(entered, release)
        )
    )
    send_task = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        target = (
            {"user_id": 123456789} if kind == "private" else {"group_id": 987654321}
        )
        send_task = asyncio.create_task(
            event.bot.send_msg(
                message=[{"type": "image", "data": {"file": "base64://c3ludGhldGlj"}}],
                **target,
            )
        )
        await send_task
        release.set()
        await asyncio.wait_for(sending.wait(), 3)
        await asyncio.sleep(0.05)
        delivered.set()
        await asyncio.gather(task, send_task)
        assert bot.transport.await_count == 1, payload(bot)
        assert event.get_extra(MARK)["sent_parts"] == 1
        assert event.get_extra(MARK)["finished"]
        assert not env.plugin.rooms.get(event.unified_msg_origin).pending
    finally:
        release.set()
        delivered.set()
        await asyncio.gather(
            task, *([send_task] if send_task else []), return_exceptions=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
async def test_native_send_receipt_contract(env, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    target = {"user_id": 123456789} if kind == "private" else {"group_id": 987654321}
    receipt = await event.bot.send_msg(message="Synthetic answer", **target)
    await env.plugin._complete_output(event)
    assert receipt.status == "delivered"
    assert receipt.receipt == {"message_id": 42}
    assert bot.transport.await_count == 1
    assert any(
        r == "send_completed" and d.get("receipt_id") == "42"
        for _, r, d in env.plugin.audit_queue
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("fail", [False, True])
async def test_completion_waits_for_transport(env, monkeypatch, expired, fail):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    entered, release = asyncio.Event(), asyncio.Event()
    reasons = []
    original_note = env.plugin._note

    def note(ev, reason, details=None):
        reasons.append(reason)
        return original_note(ev, reason, details)

    monkeypatch.setattr(env.plugin, "_note", note)

    async def transport(*args, **kwargs):
        entered.set()
        await release.wait()
        if fail:
            raise TimeoutError("synthetic transport timeout")
        return {"message_id": 42}

    bot.transport.side_effect = transport
    sending = asyncio.create_task(
        event.bot.send_msg(
            group_id=987654321,
            message=[{"type": "image", "data": {"file": "base64://c3ludGhldGlj"}}],
        )
    )
    finishing = None
    try:
        await sending
        finishing = asyncio.create_task(
            env.plugin._complete_output(event, expired=expired)
        )
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.sleep(0.02)
        assert not finishing.done()
        assert not event.get_extra(MARK).get("finished")
        release.set()
        results = await asyncio.gather(sending, finishing, return_exceptions=True)
        assert results[0].status == ("uncertain" if fail else "delivered")
        assert results[1] is None
        assert bot.transport.await_count == 1
        assert event.get_extra(MARK)["finished"]
        assert bool(event.get_extra(MARK).get("sent_parts")) == (not fail)
        assert ("send_failed" in reasons) == fail
        if expired:
            assert "request_lease_expired" in reasons
        assert not env.plugin.rooms.get(event.unified_msg_origin).pending
    finally:
        release.set()
        await asyncio.gather(
            sending, *([finishing] if finishing else []), return_exceptions=True
        )
