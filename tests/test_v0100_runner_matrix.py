"""Takeover vs third-party agent runners (Dify/Coze) on real Core stages, both config shapes."""

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from . import test_integration as base
from .test_v0100_takeover import incoming
from ..main import MARK, JevActiveReply as Main
from astrbot import __version__
from astrbot.core import astr_agent_hooks
from astrbot.core.star.star_handler import EventType
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.pipeline.process_stage.method.agent_request import (
    AgentRequestSubStage,
)
from astrbot.core.pipeline.process_stage.method.agent_sub_stages import third_party
from astrbot.core.star.session_llm_manager import SessionServiceManager
from astrbot.api.star import Context
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.message.message_event_result import MessageChain


@pytest_asyncio.fixture
async def env():
    value = base.IntegrationTests()
    await value.asyncSetUp()
    try:
        yield value
    finally:
        await value.asyncTearDown()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "private"])
@pytest.mark.parametrize("runner_type", ["dify", "coze"])
@pytest.mark.parametrize("takeover", [False, True])
async def test_native_third_party_approved_wake_keeps_one_reply(
    env, monkeypatch, kind, runner_type, takeover
):
    env.cfg.update(takeover_wake_messages=takeover, enable_private_chat=takeover)
    env.settings["wake_prefix"] = []
    env.settings["provider_settings"].update(
        wake_prefix="",
        agent_runner_type=runner_type,
        unsupported_streaming_strategy="turn_off",
        dify_agent_runner_provider_id="synthetic-runner",
        coze_agent_runner_provider_id="synthetic-runner",
    )
    if __version__ == "4.26.8":
        env.settings.pop("agent_runner", None)
        monkeypatch.setattr(
            third_party, "astrbot_config", {"provider": [{"id": "synthetic-runner"}]}
        )
    else:
        env.settings["agent_runner"] = {"runner_type": runner_type, "config": {}}
    resets = []

    class SyntheticRunner:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

        async def reset(self, **kwargs):
            resets.append(kwargs["request"])
            self.hooks = kwargs["agent_hooks"]
            self.run_context = kwargs["run_context"]
            self.response = LLMResponse(
                role="assistant",
                completion_text="合成回答",
                result_chain=MessageChain().message("合成回答"),
            )

        async def step_until_done(self, **kwargs):
            await self.hooks.on_agent_done(self.run_context, self.response)
            yield NS(type="llm_result", data={"chain": self.response.result_chain})

        def get_final_llm_resp(self):
            return self.response

        async def close(self):
            pass

    monkeypatch.setattr(third_party, "DifyAgentRunner", SyntheticRunner)
    monkeypatch.setattr(third_party, "CozeAgentRunner", SyntheticRunner)
    monkeypatch.setattr(third_party.Metric, "upload", AsyncMock())
    monkeypatch.setattr(
        SessionServiceManager,
        "should_process_llm_request",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        third_party.ThirdPartyAgentSubStage,
        "_resolve_persona_custom_error_message",
        AsyncMock(return_value=None),
    )

    async def native_hooks(event, event_type, request):
        await env.plugin.restore_takeover_message(event, request)
        await env.plugin.prepare_silence(event, request)
        await env.plugin.on_request(event, request)
        return event.is_stopped()

    async def response_hooks(event, event_type, *args):
        if event_type == EventType.OnLLMResponseEvent:
            await env.plugin.on_response(event, args[0])
        elif event_type == EventType.OnAgentDoneEvent:
            await env.plugin.silence_agent_done(event, *args)
        return event.is_stopped()

    monkeypatch.setattr(astr_agent_hooks, "call_event_hook", response_hooks)
    monkeypatch.setattr(third_party, "call_event_hook", native_hooks)
    native_context = Context.__new__(Context)
    native_context.__dict__.update(vars(env.ctx))
    ctx = NS(astrbot_config=env.settings, plugin_manager=NS(context=native_context))
    agent = AgentRequestSubStage()
    await agent.initialize(ctx)
    assert isinstance(agent.agent_sub_stage, third_party.ThirdPartyAgentSubStage)

    async def handlers(event):
        await env.plugin.capture_ignored(event)
        if not event.is_stopped():
            await env.plugin.on_input(event)
        if not event.is_stopped():
            await env.plugin.on_message(event)
        if False:
            yield

    stage = ProcessStage()
    stage.ctx = ctx
    stage.star_request_sub_stage = NS(process=handlers)
    stage.agent_sub_stage = agent
    ev = incoming(kind)
    transport = AsyncMock()
    ev.send = transport
    ev.set_extra("activated_handlers", [object()])
    async for _ in stage.process(ev):
        if ev.get_result() and not ev.is_stopped():
            await env.plugin.before_output(ev)
            if not ev.is_stopped():
                await ev.send(ev.get_result())
            ev.clear_result()
    await asyncio.sleep(0)
    record = dict(
        version=__version__,
        kind=kind,
        runner=runner_type,
        takeover=takeover,
        judge_calls=env.plugin.client.evaluate.await_count,
        native_runner_calls=len(resets),
        sends=transport.await_count,
        stopped=ev.is_stopped(),
        marked=bool(ev.get_extra(MARK)),
        unsupported=any(
            reason == "unsupported_runtime_settings"
            for _, reason, _ in env.plugin.audit_queue
        ),
        conversation_rejected=any(
            reason == "conversation_changed" for _, reason, _ in env.plugin.audit_queue
        ),
    )
    print(json.dumps(record))
    assert record["sends"] == 1, record
    assert record["native_runner_calls"] == 1, record


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({"agent_runner": {"runner_type": "dify"}}, "dify"),
        ({"provider_settings": {"agent_runner_type": "coze"}}, "coze"),
        (
            {
                "agent_runner": {"runner_type": "local"},
                "provider_settings": {"agent_runner_type": "dify"},
            },
            "local",
        ),
        (
            {"agent_runner": {}, "provider_settings": {"agent_runner_type": "dify"}},
            "dify",
        ),
        ({}, "local"),
        (None, "local"),
    ],
)
def test_runner_type_reads_both_config_shapes(settings, expected):
    assert Main._runner_type(settings) == expected
