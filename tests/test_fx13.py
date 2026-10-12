import asyncio

import pytest

from .. import bridge
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import env, delivery_hooks, adapter_event, pipeline


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize(
    "failure", ["lost_receipt", "timeout", "error", "cancel", "success"]
)
async def test_candidate_timeout_has_single_transport_terminal(
    env, monkeypatch, kind, failure
):
    wake.configure(env)
    wake.judge_result(env, "silence")
    event, bot = adapter_event(kind)
    monkeypatch.setattr(bridge, "native_busy", lambda ev: True)
    monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.01)
    accepted = []

    async def transport(*args, **kwargs):
        if failure == "lost_receipt":
            accepted.append(kwargs)
            if len(accepted) == 1:
                raise TimeoutError("synthetic lost receipt")
        elif failure == "timeout":
            raise TimeoutError("synthetic timeout")
        elif failure == "error":
            raise RuntimeError("synthetic transport error")
        elif failure == "cancel":
            raise asyncio.CancelledError
        return {"message_id": 43}

    bot.transport.side_effect = transport
    try:
        await pipeline(env, event, adapter=True)
    except (TimeoutError, RuntimeError, asyncio.CancelledError):
        pass
    # Completion may be invoked again by Core even after admission unwinds.
    await env.plugin._context_failure_reply(event)
    records = [(reason, data) for _, reason, data in env.plugin.audit_queue]
    assert bot.transport.await_count == 1
    assert event.get_extra("_jev_send_sequence") == 1
    assert event.is_stopped()
    assert not event.get_extra("_jev_candidate_running")
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
    attempts = [
        data["send_sequence"] for reason, data in records if reason == "send_attempt"
    ]
    assert attempts == [1]
    terminal = "send_completed" if failure == "success" else "send_failed"
    assert sum(reason == terminal for reason, _ in records) == 1
    if failure == "lost_receipt":
        assert len(accepted) == 1
    assert "[PASS]" not in str(bot.transport.await_args_list)


__all__ = ["env", "delivery_hooks"]
