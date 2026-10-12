from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from .test_v0102_wake_warnings import env, delivery_hooks, adapter_event, pipeline
from . import test_v0101_explicit_wake as wake
from .test_fx4 import payload
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.pipeline.scheduler import PipelineScheduler
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
from astrbot.core.provider.entities import LLMResponse


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.parametrize("kind", ["at", "name", "private"])
async def test_multiple_core_yields_keep_all_body(env, kind, tool):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    config = deepcopy(DEFAULT_CONFIG)
    config["platform_settings"].update(
        reply_prefix="", reply_with_mention=False, reply_with_quote=False
    )
    config["platform_settings"]["segmented_reply"]["enable"] = False
    config["provider_tts_settings"]["enable"] = False
    config["content_safety"]["also_use_in_response"] = False
    config["t2i"] = False
    ctx = NS(
        astrbot_config=config,
        plugin_manager=NS(
            context=NS(
                get_using_tts_provider=lambda *a: None,
                get_using_tts_provider_async=AsyncMock(return_value=None),
            )
        ),
    )
    decorate, respond = ResultDecorateStage(), RespondStage()
    await decorate.initialize(ctx)
    await respond.initialize(ctx)

    async def handlers(ev):
        await env.plugin.on_input(ev)
        await env.plugin.on_message(ev)
        if False:
            yield

    async def generate(ev):
        ev.call_llm = True
        for text in ["First substantive body", "Second substantive body"]:
            response = LLMResponse(role="assistant", completion_text=text)
            await env.plugin.on_response(ev, response)
            ev.set_result(
                ev.chain_result(response.result_chain.chain)
                if response.result_chain is not None
                else ev.plain_result(response.completion_text)
            )
            yield
            if tool and text.startswith("First"):
                target = (
                    {"user_id": 123456789}
                    if kind == "private"
                    else {"group_id": 987654321}
                )
                await ev.bot.send_msg(message="Tool body", **target)
        response = LLMResponse(role="assistant", completion_text="[PASS]。")
        await env.plugin.on_response(ev, response)
        ev.set_result(ev.plain_result(response.completion_text))
        yield

    stage = ProcessStage()
    stage.ctx = NS(astrbot_config=env.settings)
    stage.star_request_sub_stage = NS(process=handlers)
    stage.agent_sub_stage = NS(process=generate)
    scheduler = PipelineScheduler.__new__(PipelineScheduler)
    scheduler.stages = [stage, decorate, respond]
    event.set_extra("activated_handlers", [object()])
    await scheduler._process_stages(event)
    encoded = payload(bot)
    print("MULTI_YIELD", kind, bot.transport.await_count, encoded)
    assert bot.transport.await_count == 1
    assert "First substantive body" in encoded
    assert "Second substantive body" in encoded
    assert "[PASS]" not in encoded
    if tool:
        assert "Tool body" in encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["private", "group"])
@pytest.mark.parametrize("kind", ["at", "name", "private"])
async def test_foreign_target_not_redirected_to_current_room(env, kind, target):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)

    async def emit(ev):
        from ..delivery import DeliveryDeclined

        with pytest.raises(DeliveryDeclined):
            await getattr(ev.bot, f"send_{target}_msg")(
                **{f"{'user' if target == 'private' else 'group'}_id": 777777777},
                message="Foreign recipient body",
            )

    await pipeline(env, event, adapter=True, native_send=emit)
    calls = bot.transport.await_args_list
    print("FOREIGN_TARGET", kind, payload(bot))
    assert not any(
        "Foreign recipient body" in str(call.kwargs)
        and call.kwargs.get("user_id") != 777777777
        for call in calls
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("mode", ["stop", "media_stop", "superseded", "lease", "wait"])
async def test_lifecycle_independent(env, monkeypatch, kind, mode):
    import asyncio
    from .. import bridge

    wake.configure(env)
    wake.judge_result(env, "silence")
    event, bot = adapter_event(kind)
    if mode == "wait":
        monkeypatch.setattr(bridge, "native_busy", lambda ev: True)
        monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.02)
        await pipeline(env, event, "[不回复]", adapter=True)
    else:
        if mode == "superseded":
            original = env.plugin.client.evaluate

            async def evaluate(*args):
                value = await original(*args)
                env.plugin.rooms.get(event.unified_msg_origin).generation += 1
                monkeypatch.setattr(bridge, "fingerprint", lambda conv: "new-history")
                return value

            env.plugin.client.evaluate = evaluate
        entered, release = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(
            pipeline(
                env, event, "[PASS]。", adapter=True, generation_gate=(entered, release)
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if mode == "media_stop":
                await event.send(
                    event.chain_result(
                        [
                            __import__(
                                "astrbot.api.message_components", fromlist=["Image"]
                            ).Image.fromBase64("c3ludGhldGlj")
                        ]
                    )
                )
                event.stop_event()
            elif mode == "stop":
                event.stop_event()
            elif mode == "lease":
                env.plugin._expire(event)
                await asyncio.sleep(0.05)
        finally:
            release.set()
            await task
    encoded = payload(bot)
    assert bot.transport.await_count == 1, encoded
    assert "[PASS]" not in encoded and "[不回复]" not in encoded
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending


__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
async def test_native_queue_receipt_is_explicit(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    receipt = await event.bot.send_group_msg(
        group_id=987654321, message="Synthetic body"
    )
    assert receipt.status == "queued"
    assert receipt.receipt is None
    assert bot.transport.await_count == 0
    await env.plugin._complete_output(event)
    assert receipt.status == "delivered"
    assert receipt.receipt == {"message_id": 42}
