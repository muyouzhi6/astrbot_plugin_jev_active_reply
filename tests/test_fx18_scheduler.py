"""Release-18 probes through the actual scheduler and StarRequestSubStage."""

import base64
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest


from astrbot.api.message_components import Image, Plain
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.pipeline.process_stage.method.star_request import StarRequestSubStage
from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.pipeline.scheduler import PipelineScheduler
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.star.star import star_map
from astrbot.core.star.star_handler import (
    EventType,
    StarHandlerMetadata,
    star_handlers_registry,
)

from ..main import MARK
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import env, delivery_hooks, adapter_event
from .test_fx18_entry import delivered


async def run_scheduler(env, event, monkeypatch, text="Ready model answer."):
    config = deepcopy(DEFAULT_CONFIG)
    config["platform_settings"].update(
        reply_prefix="", reply_with_mention=False, reply_with_quote=False
    )
    config["platform_settings"]["segmented_reply"]["enable"] = False
    config["provider_tts_settings"]["enable"] = False
    config["content_safety"]["also_use_in_response"] = False
    config["t2i"] = False
    config["admins_id"] = ["not-the-test-user"]
    config["wake_prefix"] = ["/"]
    env.settings = config
    context = NS(
        astrbot_config=config,
        plugin_manager=NS(
            context=NS(
                get_using_tts_provider=lambda *args: None,
                get_using_tts_provider_async=AsyncMock(return_value=None),
            )
        ),
    )
    handlers = []
    for name in ("capture_ignored", "on_input", "on_message"):
        module = "rv18_ingress"
        monkeypatch.setitem(
            star_map, module, NS(name=module, activated=True, reserved=False)
        )
        handlers.append(
            StarHandlerMetadata(
                EventType.AdapterMessageEvent,
                module + name,
                name,
                module,
                getattr(env.plugin, name),
                [],
            )
        )
    event.set_extra("activated_handlers", handlers)

    async def generate(ev):
        ev.call_llm = True
        request = ProviderRequest(prompt=ev.message_str, conversation=NS(cid=env.cid))
        await env.plugin.on_request(ev, request)
        response = LLMResponse(role="assistant", completion_text=text)
        response.result_chain = ev.chain_result([Plain(text)])
        await env.plugin.on_response(ev, response)
        if not ev.is_stopped():
            ev.set_result(ev.chain_result(response.result_chain.chain))
            yield

    process, decorate, respond = ProcessStage(), ResultDecorateStage(), RespondStage()
    process.ctx = context
    process.star_request_sub_stage = StarRequestSubStage()
    await process.star_request_sub_stage.initialize(context)
    process.agent_sub_stage = NS(process=generate)
    await decorate.initialize(context)
    await respond.initialize(context)
    scheduler = PipelineScheduler.__new__(PipelineScheduler)
    scheduler.ctx = context
    scheduler.stages = [process, decorate, respond]
    await scheduler.execute(event)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("stop", [False, True])
async def test_scheduler_completion_precedes_temp_cleanup(
    env, monkeypatch, tmp_path, kind, stop
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    path = tmp_path / "synthetic.png"
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jOZkAAAAASUVORK5CYII="
        )
    )
    event.track_temporary_local_file(str(path))

    async def consume(ev):
        if ev.get_extra("rv18_consumed"):
            return
        ev.set_extra("rv18_consumed", True)
        await ev.send(ev.chain_result([Image.fromFileSystem(str(path))]))
        ev.set_result(ev.chain_result([]))
        if stop:
            ev.stop_event()

    module = "rv18_real_decorator"
    monkeypatch.setitem(
        star_map, module, NS(name=module, activated=True, reserved=False)
    )
    star_handlers_registry.append(
        StarHandlerMetadata(
            EventType.OnDecoratingResultEvent,
            module,
            module,
            module,
            consume,
            [],
            extras_configs={"priority": -50000},
        )
    )
    await run_scheduler(env, event, monkeypatch)
    assert event.get_extra(MARK)["finished"]
    assert not path.exists()
    body = delivered(bot)
    assert '"type": "image"' in body, body


__all__ = ["env", "delivery_hooks"]
