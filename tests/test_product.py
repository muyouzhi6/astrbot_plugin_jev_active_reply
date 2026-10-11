"""Product defaults, channel isolation, migration, and bounded unlimited RPM."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ..client import JevClient, KeyPool
from ..settings import ConfigView


@pytest.mark.parametrize("kind", ["nanbei", "typesafe"])
def test_dashboard_only_selected_channel_needs_configuration(kind):
    schema = json.loads(
        (Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    visible = [
        name
        for name in ("nanbei", "typesafe")
        if schema[name]["condition"] == {"primary_channel": kind}
    ]
    assert visible == [kind]
    fields = schema[kind]["items"]
    assert [name for name, field in fields.items() if not field.get("invisible")] == [
        "keys"
    ]
    assert schema["primary_channel"]["labels"] == ["南北绿豆站（推荐）", "Jev 官方"]
    assert schema["dry_run"]["invisible"] and schema["dry_run"]["default"] is False


def test_legacy_migration_removes_caps_and_preserves_conversation_settings():
    raw = {
        "primary_channel": "nanbei",
        "dry_run": True,
        "enabled_sessions": ["group"],
        "burst_window_seconds": 15,
        "behavior_guidance": "keep it short",
        "style": "quiet",
        "nanbei": {
            "keys": ["sk-test-only"],
            "quota_rpm": 55,
            "key_rpm": 55,
            "daily_request_limit": 80,
        },
        "typesafe": {
            "keys": ["apikey_test_only"],
            "quota_rpm": 55,
            "daily_request_limit": 500,
        },
        "silence": {"ignored_users": ["example"]},
        "advanced": {
            "network": {"daily_evaluation_limit": 50, "allow_channel_fallback": True},
            "judgment": {"second_review": "always"},
        },
    }
    before = deepcopy(raw)
    config = ConfigView(raw)
    for name in (
        "enabled_sessions",
        "burst_window_seconds",
        "behavior_guidance",
        "style",
        "silence",
    ):
        assert config[name] == before[name]
    assert config["second_review"] == "always"
    assert not config["dry_run"] and not config["allow_channel_fallback"]
    assert config["daily_evaluation_limit"] == config["daily_reply_limit"] == 0
    for kind in ("nanbei", "typesafe"):
        assert config[kind]["keys"] == before[kind]["keys"]
        assert (
            config[kind]["quota_rpm"]
            == config[kind]["key_rpm"]
            == config[kind]["daily_request_limit"]
            == 0
        )
    once = deepcopy(raw)
    ConfigView(raw)
    assert raw == once


def test_patch_removes_only_legacy_duplicates_and_preserves_effective_values():
    fixtures = Path(__file__).parent / "fixtures"
    fixture = json.loads((fixtures / "legacy-config.json").read_text(encoding="utf-8"))
    source = fixture["source"]
    patch = json.loads((fixtures / "config-patch.json").read_text(encoding="utf-8"))
    patched = deepcopy(source)
    for key in patch["delete_top_level"]:
        patched.pop(key, None)
    for path, value in patch["set"].items():
        current = patched
        segments = path.split(".")
        for segment in segments[:-1]:
            current = current.setdefault(segment, {})
        current[segments[-1]] = value
    before, after = ConfigView(deepcopy(source)), ConfigView(patched)
    excluded = {"advanced", "interest_topics"}
    first = {k: before[k] for k in before if k not in excluded}
    second = {k: after[k] for k in after if k not in excluded}
    # Frozen expected values were evaluated by the pre-0.9.0 parser.
    assert first == second == fixture["effective"]
    assert set(patch["set"]) == {"advanced.context.interest_topics"}
    assert set(patch["delete_top_level"]) == {
        "timeout_seconds",
        "second_review",
        "image_decision_mode",
        "context_message_limit",
        "natural_reply_threshold",
        "reply_to_bot_wakes",
        "min_reply_interval",
    }
    assert before["interest_topics"] != after["interest_topics"]


def test_inactive_channel_bad_configuration_cannot_break_selected_channel():
    config = ConfigView(
        {
            "primary_channel": "nanbei",
            "nanbei": {"keys": ["sk-example"]},
            "typesafe": {"keys": ["sk-wrong-provider"], "endpoint": "not-a-url"},
        }
    )
    client = JevClient(config)
    assert list(client.pools) == ["nanbei"]
    assert config["typesafe"]["keys"] == ["sk-wrong-provider"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["nanbei", "typesafe"])
async def test_default_rpm_is_unlimited_with_bounded_observation_history(kind):
    pool = KeyPool(kind, {"keys": ["test-only"]}, clock=lambda: 100.0)
    assert pool.rpm == pool.key_rpm == pool.daily_limit == 0
    for _ in range(10050):
        slot = await pool.acquire()
        assert slot is not None
        await pool.release(slot)
    assert len(pool.requests) == len(pool.keys[0].requests) == 10000
    assert pool.active == pool.keys[0].active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["nanbei", "typesafe"])
async def test_single_key_serves_parallel_requests_without_serial_bottleneck(kind):
    pool = KeyPool(kind, {"keys": ["test-only"]})
    slots = await asyncio.gather(*(pool.acquire() for _ in range(8)))
    assert all(slot is not None for slot in slots)
    assert pool.active == pool.keys[0].active == 8
    assert await pool.acquire() is None  # Global in-flight memory protection remains.
    await asyncio.gather(*(pool.release(slot) for slot in slots))
    assert pool.active == pool.keys[0].active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,secret", [("nanbei", "sk-test"), ("typesafe", "apikey_test")]
)
async def test_only_selected_channel_is_called_after_switch(kind, secret):
    calls = []

    async def transport(endpoint, key, payload):
        calls.append((endpoint, key))
        return 200, {}, {"answers": {"ok": {"type": "noul", "noul": 0.8}}}

    config = ConfigView(
        {
            "primary_channel": kind,
            "nanbei": {"keys": ["sk-test"]},
            "typesafe": {"keys": ["apikey_test"]},
        }
    )
    client = JevClient(config, transport=transport)
    try:
        await client.evaluate(
            {"text": "example"},
            {"ok": {"type": "noul", "instructions": "Is it appropriate?"}},
        )
        assert len(calls) == 1 and calls[0][1] == secret
        assert ("hajimi.165201.xyz" in calls[0][0]) == (kind == "nanbei")
    finally:
        await client.close()
