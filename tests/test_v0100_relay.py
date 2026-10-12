"""0.10.1: Jev-only judgment (official / Nanbei / custom Jev-format relay)."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from ..client import Adapter, DecisionError, JevClient, normalize_relay_url
from ..policy import QUESTIONS, REVIEW_QUESTIONS
from ..settings import ConfigView, migrate_config_file
from ..telemetry import Metrics
from . import test_v020 as previous
from . import test_integration as base
from .test_client import answer

env = previous.env
ROOT = Path(__file__).parents[1]


def config(**extra):
    return {
        "relay_mode": "custom",
        "relay": {
            "base_url": "https://relay.example",
            "api_key": "synthetic-secret",
            "model": "",
            "concurrency": 2,
        },
        **extra,
    }


def jev_answer(questions=QUESTIONS, value=0.9):
    return {
        "answers": {
            name: {"type": q["type"], "noul": value} for name, q in questions.items()
        },
        "model": "jev-latest",
    }


@pytest.mark.parametrize(
    "suffix", ["", "/", "/v1", "/v1/", "/v1/systemone", "/v1/systemone/", "/systemone"]
)
@pytest.mark.parametrize(
    "base", ["https://relay.example", "https://relay.example:8443/proxy"]
)
def test_base_url_normalization(base, suffix):
    assert normalize_relay_url(" " + base + suffix + " ") == base + "/v1/systemone"


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
async def test_custom_speaks_jev_systemone_protocol(questions):
    wire = AsyncMock(return_value=(200, {}, jev_answer(questions)))
    client = JevClient(config(), transport=wire)
    state = {"candidate_reply": "合成草稿"} if questions == REVIEW_QUESTIONS else {}
    result = await client.evaluate(state, questions)
    endpoint, key, payload = wire.await_args.args
    assert endpoint == "https://relay.example/v1/systemone"
    assert key == "synthetic-secret"
    # Same wire shape as the official adapters: no OpenAI messages.
    assert set(payload) == {"model", "state", "questions"}
    assert payload["model"] == "jev-latest" and payload["questions"] == questions
    assert payload["state"].items() >= state.items()
    assert result["channel"] == "custom"
    assert result["values"] == dict.fromkeys(questions, 0.9)
    assert "synthetic-secret" not in repr(result)
    assert client.pools["custom"].active == 0
    await client.close()


def test_custom_wire_matches_official_adapter():
    official = Adapter.create("typesafe", {})
    custom = Adapter.create(
        "custom", {"base_url": "https://relay.example", "model": "m"}
    )
    assert official.encode({"a": 1}, QUESTIONS) == {
        **custom.encode({"a": 1}, QUESTIONS),
        "model": "jev-latest",
    }
    assert (
        custom.decode(jev_answer(), QUESTIONS)["values"]
        == official.decode(jev_answer(), QUESTIONS)["values"]
    )


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
async def test_custom_error_classes_and_safe_logs(status, expected, caplog):
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
            client.pools["custom"].keys
        )
        transport.assert_awaited_once()
        assert client.pools["custom"].active == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_timeout_never_replayed_and_releases_slot():
    wire = AsyncMock(side_effect=asyncio.TimeoutError("synthetic-secret"))
    client = JevClient(config(retry_explicit_rejection=True), transport=wire)
    with pytest.raises(DecisionError, match="transport_uncertain"):
        await client.evaluate({}, QUESTIONS)
    wire.assert_awaited_once()
    assert client.pools["custom"].active == 0
    await client.close()


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"choices": [{"message": {"content": "{}"}}]},
        {"answers": []},
        {"answers": {}},
    ],
)
def test_non_jev_response_is_rejected_safely(body):
    with pytest.raises(DecisionError):
        Adapter.create("custom", config()["relay"]).decode(body, QUESTIONS)


@pytest.mark.asyncio
@pytest.mark.parametrize("primary", ["typesafe", "nanbei", "custom"])
async def test_selected_service_primary_others_only_as_configured_fallback(primary):
    full = config(
        relay_mode=primary,
        allow_channel_fallback=True,
        nanbei={"keys": ["site-test"]},
        typesafe={"keys": ["apikey_test"]},
    )
    client = JevClient(full)
    assert client.order[0] == primary
    assert sorted(client.order) == ["custom", "nanbei", "typesafe"]
    await client.close()
    # Without fallback only the selected service is used.
    alone = JevClient({**full, "allow_channel_fallback": False})
    assert alone.order == [primary]
    await alone.close()
    # Fallback never adds services that have no credentials.
    sparse = JevClient(
        {"relay_mode": primary, "allow_channel_fallback": True, **{primary: {}}}
    )
    assert sparse.order == [primary]
    await sparse.close()


@pytest.mark.asyncio
async def test_custom_falls_back_to_nanbei_on_explicit_rejection():
    cfg = config(
        allow_channel_fallback=True,
        retry_explicit_rejection=True,
        nanbei={"keys": ["site-test"]},
    )
    wire = AsyncMock(side_effect=[(401, {}, {}), (200, {}, answer())])
    client = JevClient(cfg, transport=wire)
    result = await client.evaluate({}, QUESTIONS)
    assert client.order == ["custom", "nanbei"]
    assert result["channel"] == "nanbei"
    assert [c.args[1] for c in wire.await_args_list] == [
        "synthetic-secret",
        "site-test",
    ]
    await client.close()


@pytest.mark.asyncio
async def test_unconfigured_or_invalid_custom_never_breaks_loading():
    empty = JevClient({"relay_mode": "custom"})
    assert empty.order == ["custom"] and not empty.pools
    assert not await empty.review_ready()
    with pytest.raises(DecisionError, match="no_capacity"):
        await empty.evaluate({}, QUESTIONS)
    await empty.close()
    bad = JevClient(
        {"relay_mode": "custom", "relay": {"base_url": "http://x", "api_key": "k"}}
    )
    assert bad.config_errors == ["custom"] and not bad.pools
    await bad.close()


@pytest.mark.parametrize("mode", ["astrbot", "jev", "openai", 3])
def test_removed_modes_rejected_by_client_unless_migrated(mode):
    if mode == "jev":
        client = JevClient({"relay_mode": "jev", "primary_channel": "typesafe"})
        assert client.order == ["typesafe"]
        return
    with pytest.raises(ValueError):
        JevClient({"relay_mode": mode})


def test_no_provider_or_openai_path_remains():
    source = (ROOT / "client.py").read_text() + (ROOT / "main.py").read_text()
    # llm_generate remains only for the pre-existing overflow summary, not judging.
    for gone in (
        "evaluate_provider",
        "chat/completions",
        '"astrbot"',
        "provider_slots",
    ):
        assert gone not in source


@pytest.mark.parametrize(
    "old,expected,cleared",
    [
        ({"relay_mode": "jev", "primary_channel": "typesafe"}, "typesafe", False),
        ({"relay_mode": "jev", "primary_channel": "nanbei"}, "nanbei", False),
        ({"primary_channel": "typesafe"}, "typesafe", False),
        ({}, "nanbei", False),
        ({"relay_mode": "jev", "primary_channel": "mindshub"}, "nanbei", False),
        ({"relay_mode": "astrbot", "primary_channel": "typesafe"}, "nanbei", False),
        (
            {
                "relay_mode": "custom",
                "primary_channel": "typesafe",
                "relay": {
                    "base_url": "https://openai.example/v1",
                    "api_key": "sk-old",
                    "model": "gpt-x",
                    "concurrency": 3,
                },
            },
            "nanbei",
            True,
        ),
    ],
)
def test_migration_to_single_selector(old, expected, cleared, caplog):
    raw = deepcopy(old)
    view = ConfigView(raw)
    assert raw["relay_mode"] == view["relay_mode"] == expected
    assert "primary_channel" not in raw and raw["_relay_config_version"] == 1
    if old.get("relay_mode") in ("astrbot", "custom"):
        assert "no longer supported" in caplog.text
    if cleared:
        assert raw["relay"] == {
            "base_url": "",
            "api_key": "",
            "model": "jev-latest",
            "concurrency": 3,
        }
        assert "sk-old" not in caplog.text
    JevClient(view)


def test_migration_runs_once_and_keeps_new_custom_choice():
    raw = {
        "_relay_config_version": 1,
        "relay_mode": "custom",
        "relay": {"base_url": "https://jev-relay.example", "api_key": "k"},
    }
    before = deepcopy(raw)
    view = ConfigView(raw)
    assert {k: raw[k] for k in before} == before
    assert view["relay_mode"] == "custom"


def test_file_migration_runs_before_schema_strips_primary_channel(tmp_path):
    path = tmp_path / "jev_config.json"
    path.write_text(
        "\ufeff" + json.dumps({"relay_mode": "jev", "primary_channel": "typesafe"}),
        encoding="utf-8",
    )
    migrate_config_file(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    assert data["relay_mode"] == "typesafe" and "primary_channel" not in data
    assert list(tmp_path.glob("jev_config.json.*.bak"))
    stat = path.stat()
    migrate_config_file(path)
    assert path.stat().st_mtime_ns == stat.st_mtime_ns


@pytest.mark.asyncio
async def test_real_http_custom_relay_non_json_auth_and_no_key_logging(caplog):
    seen = []
    behavior = {"status": 200, "json": True}

    async def handler(request):
        seen.append(
            (request.path, request.headers.get("Authorization"), await request.json())
        )
        if not behavior["json"]:
            return web.Response(text="synthetic-secret", status=behavior["status"])
        return web.json_response(jev_answer(), status=behavior["status"])

    app = web.Application()
    app.router.add_post("/v1/systemone", handler)
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
            pool = client.pools["custom"]
            pool.adapter = replace(
                pool.adapter, endpoint=f"http://127.0.0.1:{port}/v1/systemone"
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
            path == "/v1/systemone"
            and key == "Bearer synthetic-secret"
            and set(body) == {"model", "state", "questions"}
            for path, key, body in seen
        )
        assert "synthetic-secret" not in caplog.text
    finally:
        await runner.cleanup()


def test_new_config_defaults_and_secret_schema():
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8-sig"))
    assert schema["takeover_wake_messages"]["default"] is False
    assert schema["enable_private_chat"]["default"] is False
    assert schema["relay_mode"]["default"] == "nanbei"
    assert schema["relay_mode"]["options"] == ["typesafe", "nanbei", "custom"]
    assert "primary_channel" not in schema
    assert schema["_relay_config_version"]["invisible"] is True
    fields = schema["relay"]["items"]
    assert {k: v["default"] for k, v in fields.items()} == {
        "base_url": "",
        "api_key": "",
        "model": "jev-latest",
        "concurrency": 4,
    }
    assert fields["api_key"]["secret"] is True and fields["api_key"]["type"] == "string"
    text = json.dumps(schema, ensure_ascii=False)
    assert "OpenAI 中转" not in text and "AstrBot 当前提供商" not in text


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
        order.append("review" if "candidate_reply" in payload["state"] else "gate")
        entered.set()
        await finish.wait()
        return 200, {}, jev_answer(payload["questions"])

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
    assert client.pools["custom"].active == 1
    finish.set()
    await asyncio.gather(first, gate, review)
    assert order == ["gate", "review", "gate"]
    assert client.pools["custom"].active == 0 and not client.waiters
    entered.clear()
    finish.clear()
    task = asyncio.create_task(client.evaluate({}, QUESTIONS))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.pools["custom"].active == 0 and not client.waiters
    await client.close()


@pytest.mark.parametrize(
    "value", [True, False, -0.1, 1.1, float("nan"), float("inf"), None, "0.8"]
)
def test_custom_rejects_non_probability_values(value):
    body = jev_answer()
    for answer_ in body["answers"].values():
        answer_["noul"] = value
    with pytest.raises(DecisionError, match="invalid_probability"):
        Adapter.create("custom", config()["relay"]).decode(body, QUESTIONS)


@pytest.mark.asyncio
async def test_unexpected_custom_exception_is_redacted_and_never_retried(caplog):
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
async def test_custom_budget_rejects_before_network():
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


@pytest.mark.asyncio
async def test_status_names_selected_service_without_keys(env):
    env.cfg["relay_mode"] = "nanbei"
    ev = base.event()
    lines = [r async for r in env.plugin.status(ev)]
    text = str(lines)
    assert "判断服务：南北绿豆站" in text and "AstrBot" not in text
