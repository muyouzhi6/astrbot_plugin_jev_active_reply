"""Cancellation handling at the real Core respond-stage boundary."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot.api.message_components import Plain, Record
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.pipeline.respond import stage as core_respond
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.provider.entities import LLMResponse

from .. import delivery
from ..main import MARK
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import adapter_event

env = wake.env


async def prepared_reply(env, kind="natural", original_send=None):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    if original_send is not None:
        event.send = original_send
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    assert event.get_extra(MARK)
    answer = LLMResponse(role="assistant", completion_text="Synthetic answer")
    answer.result_chain = event.plain_result(answer.completion_text)
    await env.plugin.on_response(event, answer)
    event.set_result(event.plain_result(answer.completion_text))
    await env.plugin.before_output(event)
    assert event.get_result().chain
    return event, bot


async def respond_stage():
    config = deepcopy(DEFAULT_CONFIG)
    config["platform_settings"]["segmented_reply"]["enable"] = False
    stage = RespondStage()
    await stage.initialize(SimpleNamespace(astrbot_config=config))
    return stage


def watch_core(monkeypatch):
    errors = []
    after_sent = AsyncMock(return_value=False)
    monkeypatch.setattr(core_respond, "call_event_hook", after_sent)
    monkeypatch.setattr(
        core_respond.logger,
        "error",
        lambda message, **kwargs: errors.append((message, kwargs)),
    )
    return errors, after_sent


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["natural", "at", "private"])
async def test_core_respond_cancel_skips_error_and_after_sent(env, monkeypatch, kind):
    event, bot = await prepared_reply(env, kind)
    lease = event.get_extra(MARK)["lease"]
    previous_send = event.send
    wrapper_errors = []

    async def guarded_send(chain):
        try:
            return await previous_send(chain)
        except Exception as exc:
            wrapper_errors.append(exc)
            raise

    event.send = guarded_send
    env.cid = "conversation-changed-after-decoration"
    errors, after_sent = watch_core(monkeypatch)
    info_logs = []
    monkeypatch.setattr(
        delivery.logger,
        "info",
        lambda message, *args, **kwargs: info_logs.append((message, args)),
    )
    metrics = env.plugin.metrics
    counters = metrics.replies, metrics.failures, metrics.cancelled

    stage = await respond_stage()
    await stage.process(event)

    assert not errors and not wrapper_errors
    assert bot.transport.await_count == 0
    assert event.get_extra(MARK)["rejected"]
    assert event.get_extra(MARK)["finished"]
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
    assert event.get_extra(MARK)["ticket"] not in env.plugin.pending_events
    assert "lease" not in event.get_extra(MARK) and lease.cancelled()
    assert event.get_extra("_jev_delivery_state") == "cancelled"
    assert event.get_result() is None
    assert (metrics.replies, metrics.failures, metrics.cancelled) == counters
    after_sent.assert_not_awaited()
    cancellations = [
        args for message, args in info_logs if "delivery cancelled reason=" in message
    ]
    assert cancellations == [("autonomous_delivery_cancelled", event.unified_msg_origin)]


@pytest.mark.asyncio
async def test_need_separately_stops_after_mid_send_cancellation(env, monkeypatch):
    platform_send = AsyncMock(return_value={"message_id": 42})
    event, bot = await prepared_reply(env, original_send=platform_send)
    event.set_result(
        event.chain_result(
            [
                Record.fromBase64("c3ludGhldGlj"),
                Record.fromBase64("c3ludGhldGlj"),
                Plain("Remaining text"),
            ]
        )
    )
    errors, after_sent = watch_core(monkeypatch)
    previous_send = event.send
    attempts = 0

    async def guarded_send(chain):
        nonlocal attempts
        attempts += 1
        result = await previous_send(chain)
        if attempts == 1:
            env.cid = "conversation-changed-after-first-part"
        return result

    event.send = guarded_send
    await (await respond_stage()).process(event)

    assert attempts == 2
    assert platform_send.await_count == 1
    assert bot.transport.await_count == 0
    assert not errors
    assert event.get_extra(MARK)["rejected"]
    assert env.plugin.metrics.failures == 0
    after_sent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cross_session", [False, True])
async def test_split_adapter_skips_other_plugins_exception_error_handler(
    env, cross_session
):
    event, bot = await prepared_reply(env)
    env.cid = "conversation-changed-before-split"
    splitter_errors = []

    class SplitStep:
        def __init__(self):
            self.plugin_config = SimpleNamespace(context=SimpleNamespace())

        async def handle(self, ctx):
            try:
                await self.plugin_config.context.send_message(
                    "another-session" if cross_session else ctx.event.unified_msg_origin,
                    ctx.event.plain_result("First segment"),
                )
            except Exception as exc:
                splitter_errors.append(exc)
            ctx.chain[:] = [Plain("Remaining segment")]
            return SimpleNamespace(ok=True, abort=False)

    step = SplitStep()
    adapter = delivery.SplitAdapters(MARK)
    adapter.attach(SimpleNamespace(pipeline=SimpleNamespace(_steps=[step])))
    ctx = SimpleNamespace(event=event, chain=event.get_result().chain)
    try:
        result = await step.handle(ctx)
    finally:
        adapter.restore()

    assert result.abort
    assert not splitter_errors
    assert not ctx.chain
    assert bot.transport.await_count == 0
    assert event.get_extra("_jev_delivery_state") == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
async def test_required_reply_still_delivers_once(env, monkeypatch, kind):
    event, bot = await prepared_reply(env, kind)
    errors, after_sent = watch_core(monkeypatch)
    previous_send = event.send
    attempts = 0

    async def guarded_send(chain):
        nonlocal attempts
        attempts += 1
        return await previous_send(chain)

    event.send = guarded_send
    await (await respond_stage()).process(event)
    await env.plugin._complete_output(event)

    assert attempts == 1
    assert bot.transport.await_count == 1
    assert event.get_extra("_jev_delivery_state") == "completed"
    assert event.get_extra(MARK)["sent_parts"] == 1
    assert not errors
    after_sent.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_send_failure_still_reaches_core_error_log(env, monkeypatch):
    event, bot = await prepared_reply(env)
    errors, after_sent = watch_core(monkeypatch)
    bot.transport.side_effect = RuntimeError("synthetic transport failure")

    await (await respond_stage()).process(event)

    assert len(errors) == 1
    assert "synthetic transport failure" in errors[0][0]
    assert errors[0][1]["exc_info"] is True
    assert bot.transport.await_count == 1
    after_sent.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_send_failure_still_raises_outside_respond(env):
    event, bot = await prepared_reply(env)
    bot.transport.side_effect = RuntimeError("synthetic transport failure")

    with pytest.raises(RuntimeError, match="synthetic transport failure"):
        await event.send(event.plain_result("Synthetic answer"))
