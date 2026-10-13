"""Independent release probes; synthetic payloads and mock transport only."""

import asyncio
import json
import pytest
from .. import bridge, silence
from ..main import MARK
from ..tests import test_v0101_explicit_wake as wake
from ..tests.test_v0102_wake_warnings import (
    adapter_event,
    delivery_hooks,
    env,
    pipeline,
)

from astrbot.api.message_components import Plain, Image, Node
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.pipeline.scheduler import PipelineScheduler
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
from astrbot.core.provider.entities import LLMResponse

__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
async def test_three_fragments_keep_media(env, nested, mixed):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")

    async def emit(ev):
        await ev.send(ev.plain_result("[PA"))
        parts = [Plain("S"), Image.fromBase64("c3ludGhldGlj")]
        if nested:
            parts = [Node(uin="123", name="Bot", content=parts)]
        await ev.send(ev.chain_result(parts))
        if mixed:
            await ev.bot.send_group_msg(group_id=987654321, message="S]")
        else:
            await ev.send(ev.plain_result("S]"))

    await pipeline(
        env, event, adapter=True, native_send=emit, decoration_priority=-200000
    )
    encoded = payload(bot)
    assert bot.transport.await_count == 1, encoded
    assert "image" in encoded
    assert "[PA" not in encoded and "S]" not in encoded
    assert silence.FALLBACK_REPLY not in encoded


def payload(bot):
    return json.dumps(
        [c.kwargs for c in bot.transport.await_args_list], ensure_ascii=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", [200000, 0, -200000])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("text", ["[PASS]。", "Use ["])
async def test_consumed_output_finishes(env, priority, native, text):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")

    async def emit(ev):
        if native:
            await ev.bot.send_group_msg(group_id=987654321, message=text)
        else:
            await ev.send(ev.plain_result(text))

    await pipeline(
        env, event, adapter=True, native_send=emit, decoration_priority=priority
    )
    assert bot.transport.await_count == 1, payload(bot)
    assert ("Use [" if text == "Use [" else silence.FALLBACK_REPLY) in payload(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize(
    "mode", ["stop", "early_media_stop", "superseded", "lease", "wait"]
)
async def test_online_lifecycle(env, monkeypatch, kind, mode):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    if mode == "wait":
        monkeypatch.setattr(bridge, "native_busy", lambda ev: True)
        monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.02)
        await pipeline(env, event, adapter=True)
    else:
        entered, release = asyncio.Event(), asyncio.Event()
        if mode == "superseded":
            original = bridge.build_snapshot

            async def judge(*args):
                result = await original(*args)
                env.plugin.rooms.get(event.unified_msg_origin).generation += 1
                monkeypatch.setattr(bridge, "fingerprint", lambda conv: "advanced")
                return result

            monkeypatch.setattr(bridge, "build_snapshot", judge)
        task = asyncio.create_task(
            pipeline(env, event, adapter=True, generation_gate=(entered, release))
        )
        await asyncio.wait_for(entered.wait(), 3)
        if mode == "early_media_stop":
            await event.bot.send_msg(
                **(
                    {"user_id": 123456789}
                    if kind == "private"
                    else {"group_id": 987654321}
                ),
                message=[{"type": "image", "data": {"file": "base64://c3ludGhldGlj"}}],
            )
            event.stop_event()
        elif mode == "stop":
            event.stop_event()
        elif mode == "lease":
            env.plugin._expire(event)
            await asyncio.sleep(0.05)
        release.set()
        await task
    assert bot.transport.await_count == 1, payload(bot)
    assert "[PASS]" not in payload(bot)
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_marker_split_with_media_continuation(env, native):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")

    async def emit(ev):
        if native:
            await ev.bot.send_group_msg(group_id=987654321, message="[PA")
            await ev.bot.send_group_msg(
                group_id=987654321,
                message=[
                    {"type": "text", "data": {"text": "SS]"}},
                    {"type": "image", "data": {"file": "base64://c3ludGhldGlj"}},
                ],
            )
        else:
            await ev.send(ev.plain_result("[PA"))
            await ev.send(
                ev.chain_result([Plain("SS]"), Image.fromBase64("c3ludGhldGlj")])
            )

    await pipeline(
        env, event, adapter=True, native_send=emit, decoration_priority=-200000
    )
    assert bot.transport.await_count == 1, payload(bot)
    assert "[PA" not in payload(bot) and "SS]" not in payload(bot), payload(bot)


@pytest.mark.asyncio
async def test_punctuation_tail_is_not_standalone(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")

    async def emit(ev):
        for text in ["Answer", "[PASS]", "Next", "。"]:
            await ev.send(ev.plain_result(text))

    await pipeline(
        env, event, adapter=True, native_send=emit, decoration_priority=-200000
    )
    texts = [
        "".join(p["data"]["text"] for p in c.kwargs["message"])
        for c in bot.transport.await_args_list
    ]
    assert "。" not in texts, texts


@pytest.mark.asyncio
async def test_pending_marker_keeps_node_media(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")

    async def emit(ev):
        await ev.send(ev.plain_result("[PA"))
        await ev.send(
            ev.chain_result(
                [
                    Node(
                        uin="123",
                        name="Bot",
                        content=[Plain("SS]"), Image.fromBase64("c3ludGhldGlj")],
                    )
                ]
            )
        )

    await pipeline(
        env, event, adapter=True, native_send=emit, decoration_priority=-200000
    )
    assert "image" in payload(bot), payload(bot)
    assert silence.FALLBACK_REPLY not in payload(bot), payload(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("admin", [False, True])
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize(
    "mode", ["normal", "stop", "early_media_stop", "empty", "command", "takeover_off"]
)
async def test_real_scheduler(env, kind, mode, admin):
    wake.configure(env)
    wake.judge_result(env, "silence")
    if mode == "takeover_off":
        env.cfg["takeover_wake_messages"] = False
        if kind == "private":
            env.cfg["enable_private_chat"] = False
    event, bot = adapter_event(kind)
    event.role = "admin" if admin else "member"
    if mode == "command":
        event.set_extra("handlers_parsed_params", {"synthetic_command": {}})
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
        await env.plugin.capture_ignored(ev)
        await env.plugin.on_input(ev)
        await env.plugin.on_message(ev)
        if mode == "command":
            ev.set_result(ev.plain_result("Command handled"))
            yield

    async def generate(ev):
        ev.call_llm = True
        if mode == "early_media_stop":
            await ev.bot.send_msg(
                **(
                    {"user_id": 123456789}
                    if kind == "private"
                    else {"group_id": 987654321}
                ),
                message=[{"type": "image", "data": {"file": "base64://c3ludGhldGlj"}}],
            )
            ev.stop_event()
            return
        if mode == "stop":
            ev.stop_event()
            return
        if mode == "empty":
            return
        response = LLMResponse(
            role="assistant",
            completion_text="[PASS]" if mode == "normal" else "Normal answer",
        )
        await env.plugin.on_response(ev, response)
        ev.set_result(
            ev.chain_result(response.result_chain.chain)
            if response.result_chain
            else ev.plain_result(response.completion_text)
        )
        yield

    stage = ProcessStage()
    stage.ctx = NS(astrbot_config=env.settings)
    stage.star_request_sub_stage = NS(process=handlers)
    stage.agent_sub_stage = NS(process=generate)
    scheduler = PipelineScheduler.__new__(PipelineScheduler)
    scheduler.stages = [stage, decorate, respond]
    event.set_extra("activated_handlers", [object()])
    await scheduler._process_stages(event)
    await asyncio.sleep(0.05)
    assert bot.transport.await_count == 1, payload(bot)
    assert "[PASS]" not in payload(bot), payload(bot)
    if mode == "command":
        assert "Command handled" in payload(bot) and not event.get_extra(MARK)
    if mode == "takeover_off":
        assert "Normal answer" in payload(bot) and not event.get_extra(MARK)
