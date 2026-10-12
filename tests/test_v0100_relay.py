"""OpenAI-compatible judging, safe errors and legacy channel isolation."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from astrbot.core.provider.entities import LLMResponse

from ..client import Adapter, DecisionError, JevClient, normalize_relay_url
from ..policy import QUESTIONS, REVIEW_QUESTIONS
from ..settings import ConfigView
from ..telemetry import Metrics
from . import test_v020 as previous
from . import test_integration as base
from .test_client import answer

env = previous.env


def config(**extra):
    return {
        "relay_mode": "custom",
        "relay": {
            "base_url": "https://relay.example",
            "api_key": "synthetic-secret",
            "model": "test-model",
            "concurrency": 2,
        },
        **extra,
    }


def completion(questions=QUESTIONS):
    return {
        "choices": [
            {"message": {"content": json.dumps({name: 0.9 for name in questions})}}
        ]
    }


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "/",
        "/v1",
        "/v1/",
        "/v1/chat/completions",
        "/v1/chat/completions/",
        "/chat/completions/",
    ],
)
@pytest.mark.parametrize(
    "base", ["https://relay.example", "https://relay.example:8443/proxy"]
)
def test_base_url_normalization(base, suffix):
    assert (
        normalize_relay_url(" " + base + suffix + " ") == base + "/v1/chat/completions"
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "relay.example",
        "http://relay.example",
        "ftp://relay.example",
        "https://u:secret@relay.example",
        "https://relay.example?key=secret",
        "https://relay.example/#secret",
        "https://[",
        "https://relay.example:bad",
        "https://relay.example:0",
        "https://relay. example",
    ],
)
def test_invalid_base_url_has_safe_error(value):
    with pytest.raises(ValueError) as error:
        normalize_relay_url(value)
    assert "secret" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("questions", [QUESTIONS, REVIEW_QUESTIONS])
async def test_custom_wire_and_review_share_pool(questions):
    wire = AsyncMock(return_value=(200, {}, completion(questions)))
    client = JevClient(config(), transport=wire)
    result = await client.evaluate(
        {"candidate_reply": "合成草稿"} if questions == REVIEW_QUESTIONS else {},
        questions,
    )
    endpoint, key, payload = wire.await_args.args
    assert endpoint == "https://relay.example/v1/chat/completions"
    assert key == "synthetic-secret"
    assert payload["model"] == "test-model" and payload["response_format"] == {
        "type": "json_object"
    }
    assert (
        json.loads(payload["messages"][0]["content"].split("\n", 1)[1])["questions"]
        == questions
    )
    assert result["channel"] == "relay" and result["values"] == dict.fromkeys(
        questions, 0.9
    )
    assert "synthetic-secret" not in repr(result)
    assert client.pools["relay"].active == 0
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected",
    [
        (401, "auth_disabled"),
        (403, "auth_disabled"),
        (402, "funding_cooldown"),
        (429, "rate_limited"),
        (529, "overloaded"),
        (500, "upstream_error"),
        (404, "configuration_error"),
        (302, "http_error"),
    ],
)
async def test_relay_error_classes_and_safe_logs(status, expected, caplog):
    transport = AsyncMock(
        return_value=(
            status,
            {"retry-after": "9"},
            {"error": {"message": "synthetic-secret"}},
        )
    )
    client = JevClient(config(), transport=transport)
    try:
        with pytest.raises(DecisionError) as error:
            await client.evaluate({}, QUESTIONS)
        assert error.value.code == expected and error.value.status == status
        assert "synthetic-secret" not in str(error.value) + caplog.text + repr(
            client.pools["relay"].keys
        )
        transport.assert_awaited_once()
        assert client.pools["relay"].active == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_timeout_never_replayed_and_releases_slot():
    wire = AsyncMock(side_effect=asyncio.TimeoutError("synthetic-secret"))
    client = JevClient(config(retry_explicit_rejection=True), transport=wire)
    with pytest.raises(DecisionError, match="transport_uncertain"):
        await client.evaluate({}, QUESTIONS)
    wire.assert_awaited_once()
    assert client.pools["relay"].active == 0
    await client.close()


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"choices": []},
        {"choices": [{"message": {"content": "not json synthetic-secret"}}]},
        {"choices": [{"message": {"content": "[]"}}]},
    ],
)
def test_malformed_completion_is_safe(body):
    with pytest.raises(DecisionError) as error:
        Adapter.create("relay", config()["relay"]).decode(body, QUESTIONS)
    assert "synthetic-secret" not in str(error.value)


@pytest.mark.asyncio
async def test_custom_falls_back_only_when_enabled_with_own_channel_keys():
    cfg = config(
        allow_channel_fallback=True,
        retry_explicit_rejection=True,
        primary_channel="nanbei",
        nanbei={"keys": ["site-test"]},
        typesafe={"keys": ["official-test"]},
    )
    wire = AsyncMock(side_effect=[(401, {}, {}), (200, {}, answer())])
    client = JevClient(cfg, transport=wire)
    result = await client.evaluate({}, QUESTIONS)
    assert client.order == ["relay", "nanbei", "typesafe"]
    assert result["channel"] == "nanbei"
    assert [call.args[1] for call in wire.await_args_list] == [
        "synthetic-secret",
        "site-test",
    ]
    await client.close()
    isolated = JevClient(config(nanbei={"keys": ["site-test"]}))
    assert isolated.order == ["relay"] and list(isolated.pools) == ["relay"]
    await isolated.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [True, False])
async def test_legacy_configuration_without_new_fields_keeps_pool_behavior(fallback):
    raw = {
        "_product_config_version": 1,
        "primary_channel": "typesafe",
        "allow_channel_fallback": fallback,
        "typesafe": {"keys": ["official-test"]},
        "nanbei": {"keys": ["site-test"]},
        "style": "quiet",
        "enabled_sessions": ["group"],
    }
    before = deepcopy(raw)
    view = ConfigView(raw)
    client = JevClient(view)
    assert client.order == (["typesafe", "nanbei"] if fallback else ["typesafe"])
    assert not view.get("takeover_wake_messages", False) and not view.get(
        "enable_private_chat", False
    )
    for key in before:
        assert view[key] == before[key]
    await client.close()


@pytest.mark.asyncio
async def test_real_http_relay_non_json_auth_and_no_key_logging(caplog):
    seen = []
    behavior = {"status": 200, "json": True}

    async def handler(request):
        seen.append(
            (request.path, request.headers.get("Authorization"), await request.json())
        )
        if not behavior["json"]:
            return web.Response(text="synthetic-secret", status=behavior["status"])
        return web.json_response(completion(), status=behavior["status"])

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        for status, is_json, expected in [
            (200, True, None),
            (200, False, "invalid_response"),
            (401, False, "auth_disabled"),
            (429, False, "rate_limited"),
        ]:
            behavior.update(status=status, json=is_json)
            client = JevClient(config())
            pool = client.pools["relay"]
            pool.adapter = replace(
                pool.adapter, endpoint=f"http://127.0.0.1:{port}/v1/chat/completions"
            )
            try:
                if expected:
                    with pytest.raises(DecisionError, match=expected):
                        await client.evaluate({}, QUESTIONS)
                else:
                    await client.evaluate({}, QUESTIONS)
            finally:
                await client.close()
        assert len(seen) == 4
        assert all(
            path == "/v1/chat/completions" and key == "Bearer synthetic-secret"
            for path, key, _ in seen
        )
        assert "synthetic-secret" not in caplog.text
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_astrbot_provider_selection_and_timeout(env):
    env.cfg["relay_mode"] = "astrbot"
    env.ctx.get_current_chat_provider_id = AsyncMock(return_value="session-provider")
    env.ctx.llm_generate = AsyncMock(
        return_value=LLMResponse(
            role="assistant", completion_text=json.dumps(dict.fromkeys(QUESTIONS, 0.8))
        )
    )
    ev = base.event()
    result = await env.plugin._judge(ev, {}, QUESTIONS)
    assert result["channel"] == "astrbot"
    env.ctx.get_current_chat_provider_id.assert_awaited_once_with(
        umo=ev.unified_msg_origin
    )
    assert (
        env.ctx.llm_generate.await_args.kwargs["chat_provider_id"] == "session-provider"
    )
    assert env.ctx.llm_generate.await_args.kwargs["tools"] is None
    env.plugin.client.evaluate.assert_not_awaited()
    env.ctx.get_current_chat_provider_id.side_effect = asyncio.TimeoutError()
    with pytest.raises(DecisionError, match="transport_uncertain"):
        await env.plugin._judge(ev, {}, QUESTIONS)
    assert env.plugin.client.provider_waiters == 0
    assert env.plugin.client.provider_slots._value == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected",
    [
        (401, "auth_disabled"),
        (429, "rate_limited"),
        (500, "upstream_error"),
        (None, "provider_error"),
    ],
)
async def test_astrbot_provider_classifies_errors_without_exception_text(
    status, expected
):
    error = RuntimeError("synthetic-secret")
    error.status_code = status
    context = NS(
        get_current_chat_provider_id=AsyncMock(return_value="test"),
        llm_generate=AsyncMock(side_effect=error),
    )
    client = JevClient({"relay_mode": "astrbot"})
    with pytest.raises(DecisionError, match=expected) as captured:
        await client.evaluate_provider(
            {}, QUESTIONS, context, "test:FriendMessage:human"
        )
    assert "synthetic-secret" not in str(captured.value)
    assert not client.pools
    await client.close()


def test_new_config_defaults_and_secret_schema():
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    assert schema["takeover_wake_messages"]["default"] is False
    assert schema["enable_private_chat"]["default"] is False
    assert schema["relay_mode"]["default"] == "jev"
    fields = schema["relay"]["items"]
    assert {k: v["default"] for k, v in fields.items()} == {
        "base_url": "",
        "api_key": "",
        "model": "",
        "concurrency": 4,
    }
    assert fields["api_key"]["secret"] is True and fields["api_key"]["type"] == "string"


def test_cancelled_calls_are_separate_from_errors_and_every_hour_resets():
    metrics = Metrics()
    metrics.decision(10011, "transport_uncertain")
    metrics.decision(700, "cancelled")
    metrics.outcomes["natural_silence"] = 173
    assert metrics.failures == 1 and metrics.cancelled == 1
    assert metrics.errors == {"transport_uncertain": 1}
    assert metrics.take() == (2, 0, 10011, 1)
    assert not metrics.errors and not metrics.outcomes and metrics.cancelled == 0


@pytest.mark.asyncio
async def test_hourly_summary_reports_and_resets_failure_reasons(env, caplog):
    env.plugin.metrics.decision(10011, "transport_uncertain")
    env.plugin.metrics.decision(500, "cancelled")
    env.plugin._note(base.event(), "natural_silence")
    env.plugin._summary_tick()
    assert 'failure_reasons={"transport_uncertain": 1}' in caplog.text
    assert (
        "cancelled=1" in caplog.text
        and 'outcomes={"natural_silence": 1}' in caplog.text
    )
    env.plugin.summary_timer.cancel()
    caplog.clear()
    env.plugin._summary_tick()
    assert "failure_reasons={}" in caplog.text and "cancelled=0" in caplog.text


@pytest.mark.asyncio
async def test_custom_concurrency_queue_cancellation_and_review_priority():
    entered = asyncio.Event()
    finish = asyncio.Event()
    order = []

    async def wire(endpoint, key, payload):
        data = json.loads(payload["messages"][0]["content"].split("\n", 1)[1])
        order.append("review" if "candidate_reply" in data["state"] else "gate")
        entered.set()
        await finish.wait()
        return 200, {}, completion(data["questions"])

    cfg = config(decision_queue_seconds=2, review_queue_seconds=2)
    cfg["relay"]["concurrency"] = 1
    client = JevClient(cfg, transport=wire)
    first = asyncio.create_task(client.evaluate({}, QUESTIONS))
    await entered.wait()
    gate = asyncio.create_task(client.evaluate({}, QUESTIONS))
    review = asyncio.create_task(
        client.evaluate({"candidate_reply": "synthetic"}, REVIEW_QUESTIONS)
    )
    await asyncio.sleep(0.01)
    assert client.pools["relay"].active == 1
    finish.set()
    await asyncio.gather(first, gate, review)
    assert order == ["gate", "review", "gate"]
    assert client.pools["relay"].active == 0 and not client.waiters
    entered.clear()
    finish.clear()
    task = asyncio.create_task(client.evaluate({}, QUESTIONS))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.pools["relay"].active == 0 and not client.waiters
    await client.close()


@pytest.mark.asyncio
async def test_provider_real_context_api_and_capacity_queue():
    from astrbot.core.star.context import Context
    from astrbot.core.provider.provider import Provider
    from unittest.mock import Mock

    provider = Mock(spec=Provider)
    provider.text_chat = AsyncMock(
        return_value=LLMResponse(
            role="assistant", completion_text=json.dumps(dict.fromkeys(QUESTIONS, 0.8))
        )
    )
    ctx = NS(
        provider_manager=NS(get_provider_by_id=AsyncMock(return_value=provider)),
        get_current_chat_provider_id=AsyncMock(return_value="selected"),
    )
    ctx.llm_generate = Context.llm_generate.__get__(ctx)
    client = JevClient({"relay_mode": "astrbot", "decision_queue_seconds": 0})
    result = await client.evaluate_provider(
        {}, QUESTIONS, ctx, "test:FriendMessage:human"
    )
    assert result["values"] == dict.fromkeys(QUESTIONS, 0.8)
    assert provider.text_chat.await_args.kwargs["func_tool"] is None
    for _ in range(4):
        await client.provider_slots.acquire()
    with pytest.raises(DecisionError, match="no_capacity"):
        await client.evaluate_provider({}, QUESTIONS, ctx, "test:FriendMessage:human")
    for _ in range(4):
        client.provider_slots.release()
    assert client.provider_waiters == 0
    await client.close()
    with pytest.raises(DecisionError, match="client_closed"):
        await client.evaluate_provider({}, QUESTIONS, ctx, "test:FriendMessage:human")


@pytest.mark.asyncio
async def test_provider_cancellation_releases_capacity():
    entered = asyncio.Event()

    async def generate(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    ctx = NS(
        get_current_chat_provider_id=AsyncMock(return_value="selected"),
        llm_generate=generate,
    )
    client = JevClient({"relay_mode": "astrbot"})
    task = asyncio.create_task(client.evaluate_provider({}, QUESTIONS, ctx, "session"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.provider_slots._value == 4 and client.provider_waiters == 0
    await client.close()


@pytest.mark.parametrize(
    "value", [True, False, -0.1, 1.1, float("nan"), float("inf"), None, "0.8"]
)
def test_relay_rejects_non_probability_values(value):
    body = completion()
    body["choices"][0]["message"]["content"] = json.dumps(
        dict.fromkeys(QUESTIONS, value)
    )
    with pytest.raises(DecisionError, match="invalid_probability"):
        Adapter.create("relay", config()["relay"]).decode(body, QUESTIONS)


@pytest.mark.asyncio
async def test_unexpected_relay_exception_is_redacted_and_never_retried(caplog):
    wire = AsyncMock(side_effect=RuntimeError("synthetic-secret"))
    client = JevClient(config(retry_explicit_rejection=True), transport=wire)
    try:
        with pytest.raises(DecisionError, match="transport_uncertain") as error:
            await client.evaluate({}, QUESTIONS)
        assert "synthetic-secret" not in str(error.value) + caplog.text
        wire.assert_awaited_once()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_relay_budget_rejects_before_network():
    wire = AsyncMock()
    client = JevClient(config(request_byte_budget=8000), transport=wire)
    try:
        with pytest.raises(DecisionError):
            await client.evaluate(
                {"target_message": {"content": "测试" * 100000}}, QUESTIONS
            )
        wire.assert_not_awaited()
    finally:
        await client.close()
