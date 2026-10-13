"""Independent release-18 entry probes; all platform operations are mocked."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest


from astrbot.api.message_components import Image, Plain
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
from astrbot.core.tools.message_tools import SendMessageToUserTool
from astrbot.core.star.star import star_map
from astrbot.core.star.star_handler import (
    EventType,
    StarHandlerMetadata,
    star_handlers_registry,
)

from .. import bridge
from ..main import MARK
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import (
    adapter_event,
    delivery_hooks,
    env,
    pipeline,
)


def delivered(bot):
    assert bot.transport.await_count == 1, bot.transport.call_args_list
    kwargs = bot.transport.await_args.kwargs
    payload = kwargs.get("message", kwargs.get("messages"))
    encoded = json.dumps(payload, ensure_ascii=False)
    assert payload and not any(
        marker in encoded for marker in ("[PASS]", "[pass]", "[不回复]")
    ), encoded
    assert encoded != "。" and encoded != '"。"'
    return encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize(
    "mode", ["stop_media", "history_advance", "wait", "lease", "empty"]
)
async def test_online_lifecycle_one_platform_send(env, monkeypatch, kind, mode):
    wake.configure(env)
    wake.judge_result(env, "silence")
    event, bot = adapter_event(kind)
    assert not event.is_admin()
    if mode == "wait":
        monkeypatch.setattr(bridge, "native_busy", lambda _: True)
        monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.02)
        await pipeline(env, event, "[PASS]。", adapter=True)
    else:
        if mode == "history_advance":
            evaluator = bridge.build_snapshot

            async def advanced(*args):
                result = await evaluator(*args)
                env.history = '[{"role":"assistant","content":"synthetic advance"}]'
                env.plugin.rooms.get(event.unified_msg_origin).generation += 1
                return result

            monkeypatch.setattr(bridge, "build_snapshot", advanced)
        entered, release = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(
            pipeline(
                env,
                event,
                "" if mode == "empty" else "[不回复]",
                adapter=True,
                generation_gate=(entered, release),
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 4)
            if mode == "stop_media":
                target = (
                    {"user_id": 123456789}
                    if kind == "private"
                    else {"group_id": 987654321}
                )
                await event.bot.send_msg(
                    **target, message="[CQ:image,file=base64://c3ludGhldGlj]"
                )
                event.stop_event()
            elif mode == "lease":
                env.plugin._expire(event)
                await asyncio.sleep(0.05)
        finally:
            release.set()
            await task
    if mode in {"wait", "lease"}:
        bot.transport.assert_not_awaited()
        assert not env.plugin.rooms.get(event.unified_msg_origin).pending
        return
    body = delivered(bot)
    meta = event.get_extra(MARK)
    assert meta is None or meta.get("sent_parts", 0) == 1
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
    if mode == "stop_media":
        assert '"type": "image"' in body


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
async def test_stop_after_media_keeps_distinct_ready_body(env, monkeypatch, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)

    async def media_then_stop(ev):
        target = (
            {"user_id": 123456789} if kind == "private" else {"group_id": 987654321}
        )
        await ev.bot.send_msg(**target, message="[CQ:image,file=base64://c3ludGhldGlj]")
        # Simulates a competing decorator that stops propagation without
        # clearing the model's distinct, already available body.
        ev.stop_event()

    module = "fx18_preserve_result"
    monkeypatch.setitem(
        star_map, module, NS(name=module, activated=True, reserved=False)
    )
    star_handlers_registry.append(
        StarHandlerMetadata(
            EventType.OnDecoratingResultEvent,
            module,
            module,
            module,
            media_then_stop,
            [],
            extras_configs={"priority": 0},
        )
    )
    await pipeline(
        env,
        event,
        "Distinct explanation after the media.",
        adapter=True,
    )
    body = delivered(bot)
    assert '"type": "image"' in body
    assert "Distinct explanation after the media." in body, body


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
async def test_tool_image_and_core_body_survive_external_stop(env, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        pipeline(
            env,
            event,
            "Distinct generated answer.",
            adapter=True,
            generation_gate=(entered, release),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 4)
        result = await SendMessageToUserTool().call(
            NS(context=NS(event=event, context=env.ctx)),
            messages=[{"type": "plain", "text": "Tool introduction."}],
        )
        assert "Message sent" in result
        event.stop_event()
    finally:
        release.set()
        await task
    body = delivered(bot)
    assert "Tool introduction." in body
    assert "Distinct generated answer." in body, body


@pytest.mark.asyncio
async def test_segmented_reply_finishes_after_lease(env, monkeypatch):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    monkeypatch.setattr(env.plugin, "_lifetime", lambda: 0.04)
    decorate_init = ResultDecorateStage.initialize
    respond_init = RespondStage.initialize

    async def decorate(stage, context):
        await decorate_init(stage, context)
        stage.enable_segmented_reply = True
        stage.only_llm_result = False

    async def respond(stage, context):
        await respond_init(stage, context)
        stage.enable_seg = True
        stage.only_llm_result = False

    async def delay(stage, comp):
        return 0.03

    monkeypatch.setattr(ResultDecorateStage, "initialize", decorate)
    monkeypatch.setattr(RespondStage, "initialize", respond)
    monkeypatch.setattr(RespondStage, "_calc_comp_interval", delay)
    await pipeline(
        env,
        event,
        adapter=True,
        parts=[
            Plain("First normal part."),
            Plain("Second normal part."),
            Plain("Third normal part."),
        ],
    )
    body = delivered(bot)
    assert "First normal part." in body
    assert "Second normal part." in body and "Third normal part." in body, body


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
async def test_disabled_takeover_keeps_native_send(env, kind):
    event, bot = adapter_event(kind)
    await pipeline(env, event, "Normal native reply.", adapter=True)
    assert event.get_extra(MARK) is None
    assert "Normal native reply." in delivered(bot)


@pytest.mark.asyncio
async def test_native_marker_and_text_chain_share_single_delivery(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")

    async def interleaved(ev):
        await ev.bot.send_msg(group_id=987654321, message="[PA")
        await ev.send(
            ev.chain_result([Plain("SS]Intro. "), Image.fromBase64("c3ludGhldGlj")])
        )
        await ev.bot.send_msg(group_id=987654321, message="Detail. [不回复]。")

    await pipeline(
        env, event, "Final normal text.", adapter=True, native_send=interleaved
    )
    body = delivered(bot)
    assert "Intro." in body and "Detail." in body
    assert '"type": "image"' in body
    payload = bot.transport.await_args.kwargs["message"]
    assert not any(
        part.get("type") == "text" and part["data"]["text"].strip() == "。"
        for part in payload
    )


@pytest.mark.asyncio
async def test_normal_multi_part_without_stop(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await pipeline(
        env,
        event,
        adapter=True,
        parts=[Plain("Part one."), Plain("Part two.")],
    )
    body = delivered(bot)
    assert "Part one." in body and "Part two." in body


@pytest.mark.asyncio
async def test_registered_command_not_owned(env):
    wake.configure(env)
    event, _ = adapter_event("at")
    event.set_extra("handlers_parsed_params", {"synthetic.command": {}})
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    assert event.get_extra(MARK) is None
    assert not event.is_stopped()


__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_completion_failure_records_transport_phase(env, monkeypatch, started):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    request = await event.bot.send_msg(group_id=987654321, message="Synthetic answer.")
    assert request.status == "queued"
    reasons = []
    monkeypatch.setattr(
        env.plugin, "_note", lambda ev, reason, details=None: reasons.append(reason)
    )

    async def fail(*args, **kwargs):
        raise TimeoutError("synthetic failure")

    if started:
        bot.transport.side_effect = fail
    else:
        monkeypatch.setattr(env.plugin, "_permit", fail)
    await env.plugin._complete_output(event)
    assert bot.transport.await_count == int(started)
    assert (
        "output_completion_uncertain"
        if started
        else "output_completion_failed_before_send"
    ) in reasons
    assert request.status == ("uncertain" if started else "failed")
    assert event.get_extra(MARK)["finished"]
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
    await env.plugin._complete_output(event)
    assert bot.transport.await_count == int(started)
