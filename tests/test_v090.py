"""0.9.0 configuration, filtering, draft races and durable statistics."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from unittest.mock import AsyncMock

import pytest
from astrbot.api.message_components import At, Plain, Image
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.provider.entities import LLMResponse

from ..admission import local_prefilter
from ..client import DecisionError
from ..main import MARK
from ..settings import ConfigView, migrate_config_file
from ..state import Ledger, Room, audit_preview
from ..telemetry import Metrics
from . import test_integration as base
from . import test_v020 as previous

env = previous.env


@pytest.mark.parametrize(
    "text", ["哈哈", "6", "?", "嗯", "ok", "OK!", "哈哈哈哈", "🙂" * 10000, "👩‍💻🌈"]
)
def test_prefilter_extremes_and_disable(text):
    ev = base.event(text=text)
    assert local_prefilter(ev, {})
    assert local_prefilter(ev, {"local_prefilter_enabled": False}) is None
    ev.message_obj.message.insert(0, At(qq="bot"))
    assert local_prefilter(ev, {}) is None


@pytest.mark.parametrize(
    "text", ["", "42", "什么", "你好", "https://example.invalid", "计算机" * 100000]
)
def test_meaningful_or_empty_is_not_low_information(text):
    assert local_prefilter(base.event(text=text), {}) is None


def test_named_wakes_photo_and_stickers():
    assert (
        local_prefilter(base.event(text="吱吱哈哈"), {"wake_names": ["吱吱"]}) is None
    )
    image = Image.fromURL("https://example.invalid/photo.png")
    assert local_prefilter(base.event(text="", parts=[image]), {}) is None
    sticker = type("Sticker", (), {"type": "sticker"})()
    assert (
        local_prefilter(base.event(text="[贴纸]", parts=[sticker]), {})
        == "local_prefilter_sticker"
    )
    assert (
        local_prefilter(
            base.event(
                text="这个贴纸的角色叫什么",
                parts=[Plain("这个贴纸的角色叫什么"), sticker],
            ),
            {},
        )
        is None
    )


def test_recent_topic_ttl_and_information_change():
    ev = base.event(text="AstrBot 插件报错复现步骤。")
    old = "AstrBot 插件报错复现步骤"
    assert local_prefilter(ev, {}, old, 100, 119) == "local_prefilter_recent_topic"
    assert local_prefilter(ev, {}, old, 100, 120) is None
    assert local_prefilter(ev, {"prefilter_cooldown_seconds": 0}, old, 100, 101) is None
    assert (
        local_prefilter(
            base.event(text="AstrBot 插件报错已经修好，原因是网络断开"),
            {},
            old,
            100,
            101,
        )
        is None
    )


def test_legacy_file_before_core_cleanup_and_idempotence(tmp_path):
    raw = {
        "_product_config_version": 1,
        "second_review": "always",
        "timeout_seconds": 8,
        "natural_reply_threshold": 0.5,
        "reply_to_bot_wakes": True,
        "advanced": {
            "network": {"timeout_seconds": 10},
            "guards": {"reply_to_bot_wakes": False},
            "judgment": {"natural_reply_threshold": 0.6},
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8-sig")
    migrate_config_file(path)
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    cfg = ConfigView(AstrBotConfig(str(path), schema=schema))
    assert cfg["second_review"] == "always"
    assert cfg["timeout_seconds"] == 10
    assert cfg["natural_reply_threshold"] == 0.6
    assert cfg["reply_to_bot_wakes"] is False
    assert "second_review" not in cfg.raw
    cfg.raw["second_review"] = "off"
    assert cfg["second_review"] == "always"
    cfg.raw.pop("second_review")
    migrate_config_file(path)
    backups = list(tmp_path.glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].stat().st_mode & 0o777 == 0o600
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(backups[0].read_text(encoding="utf-8-sig")) == raw


def test_false_zero_empty_and_channel_settings_survive():
    source = {
        "_product_config_version": 1,
        "timeout_seconds": 8,
        "reply_to_bot_wakes": True,
        "allow_channel_fallback": True,
        "advanced": {
            "guards": {"reply_to_bot_wakes": False},
            "network": {"allow_channel_fallback": False},
        },
        "evaluation_interval": 0,
        "interest_topics": [],
        "nanbei": {"keys": []},
    }
    original = deepcopy(source)
    config = ConfigView(source)
    assert config["evaluation_interval"] == 0
    assert config["interest_topics"] == []
    assert config["reply_to_bot_wakes"] is False
    assert config.raw["allow_channel_fallback"] == original["allow_channel_fallback"]
    assert config.raw["advanced"]["network"]["allow_channel_fallback"] is False
    assert config["nanbei"] == original["nanbei"]
    saved = deepcopy(source)
    ConfigView(source)
    assert source == saved


def test_new_defaults(tmp_path):
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    config = ConfigView(AstrBotConfig(str(tmp_path / "new.json"), schema=schema))
    assert config["evaluation_interval"] == config["burst_window_seconds"] == 10
    assert config["local_prefilter_enabled"]
    assert config["prefilter_cooldown_seconds"] == 20


@pytest.mark.asyncio
async def test_declined_topic_skips_second_call_and_explicit_wake_survives(env):
    env.cfg.update(decision_mode="natural", second_review="adaptive")
    env.plugin.client.evaluate = AsyncMock(return_value={"values": {"reply_now": 0.1}})
    await env.plugin.on_message(base.event(text="AstrBot插件报错的复现步骤"))
    await env.plugin.on_message(base.event(mid="2", text="AstrBot插件报错的复现步骤。"))
    assert env.plugin.client.evaluate.await_count == 1
    ev = base.event(mid="3", text="嗯", parts=[At(qq="bot"), Plain("嗯")])
    await env.plugin.on_input(ev)
    await env.plugin.on_message(ev)
    assert ev.is_at_or_wake_command and not ev.is_stopped()


@pytest.mark.asyncio
@pytest.mark.parametrize("sender", ["human", "other"])
async def test_unrelated_message_keeps_generated_draft_through_delivery(env, sender):
    env.cfg.update(
        second_review="adaptive", burst_merge_enabled=True, candidate_wait_seconds=1
    )
    ev = await env.admit(base.event(text="AstrBot插件报错需要排查"))
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="检查插件配置")
    )
    calls = env.plugin.client.evaluate.await_count
    other = base.event(mid="unrelated", sender=sender, text="今天晚饭吃面条")
    # Full ingress executes concurrently while the approved draft is delivered.
    task = asyncio.create_task(env.plugin.on_message(other))
    for _ in range(30):
        if room.generation > ev.get_extra(MARK)["generation"]:
            break
        await asyncio.sleep(0.001)
    ev.set_result(ev.plain_result("检查插件配置"))
    await env.plugin.before_output(ev)
    assert not ev.get_extra(MARK)["rejected"]
    assert not other.get_extra("_jev_merged_message_ids")
    assert env.plugin.client.evaluate.await_count == calls
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_related_review_then_unrelated_and_related_delivery_races(env):
    env.cfg["second_review"] = "adaptive"
    ev = await env.admit(base.event(text="AstrBot插件报错"))
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    room.observe({"sender_id": "other", "content": "AstrBot插件报错已解决"}, "new")

    async def review(*args):
        room.observe({"sender_id": "stranger", "content": "晚饭吃面条"}, "noise")
        return {"values": {"still_fits": 0.9}}

    env.plugin.client.evaluate = AsyncMock(side_effect=review)
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="插件配置已确认")
    )
    assert ev.get_extra(MARK)["reviewed"]
    ev.set_result(ev.plain_result("插件配置已确认"))
    await env.plugin.before_output(ev)
    assert not ev.get_extra(MARK)["rejected"]
    room.observe({"sender_id": "other", "content": "AstrBot配置又有报错"}, "related")
    await env.plugin.before_output(ev)
    assert ev.get_extra(MARK)["rejected"]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "中" * 100000, "🙂" * 10000])
async def test_extreme_input_makes_no_judgment(env, text):
    await env.plugin.on_message(base.event(text=text))
    env.plugin.client.evaluate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "score,preview",
    [(0.619, False), (0.62, True), (0.7, True), (0.78, True), (0.781, False)],
)
async def test_audit_preview_uses_effective_quiet_threshold(env, score, preview):
    env.cfg.update(decision_mode="natural", natural_reply_threshold=0.6, style="quiet")
    env.plugin.client.evaluate = AsyncMock(
        return_value={"values": {"reply_now": score}}
    )
    text = "检查插件配置和复现步骤" * 10
    await env.plugin.on_message(base.event(text=text))
    await env.plugin.audit_task
    row = env.plugin.ledger.db.execute(
        "SELECT details FROM audit WHERE reason IN ('natural_reply','natural_silence')"
    ).fetchone()
    details = json.loads(row[0])
    assert ("trigger_preview" in details) is preview
    if preview:
        assert details["trigger_preview"] == text[:50]


def test_preview_redacts_before_truncation():
    text = "token=" + "x" * 100
    assert "x" not in audit_preview(text)
    assert len(audit_preview("字" * 100)) == 50


def test_metrics_p95_failures_and_reset():
    metrics = Metrics()
    for n in range(1, 101):
        metrics.decision(n)
    metrics.replies = 2
    assert metrics.take() == (100, 2, 95, 0)
    assert metrics.take() == (0, 0, 0, 0)
    assert not metrics.decision(10, "transport_uncertain")
    assert not metrics.decision(10, "transport_uncertain")
    assert metrics.decision(10, "transport_uncertain")
    assert not metrics.decision(10, "transport_uncertain")
    assert metrics.take() == (4, 0, 10, 4)
    metrics.decision(1)
    assert metrics.transport_streak == 0


@pytest.mark.asyncio
async def test_judge_warnings_hourly_idle_timer_and_shutdown(env, caplog):
    ev = base.event()
    env.plugin.client.evaluate = AsyncMock(
        side_effect=DecisionError("transport_uncertain")
    )
    for _ in range(4):
        with pytest.raises(DecisionError):
            await env.plugin._judge(ev, {}, {})
    assert sum("three consecutive" in r.message for r in caplog.records) == 1
    await env.plugin.initialize()
    timer = env.plugin.summary_timer
    assert 3599 < timer.when() - asyncio.get_running_loop().time() <= 3600
    timer.cancel()
    env.plugin._summary_tick()
    assert any(
        "decisions=4 replies=0" in r.message and "failures=4" in r.message
        for r in caplog.records
    )
    timer = env.plugin.summary_timer
    await env.plugin.terminate()
    assert timer.cancelled()


def test_old_database_backfill_is_once_and_daily_stats_outlive_audit(tmp_path):
    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE audit(id INTEGER PRIMARY KEY, ts REAL, scope TEXT, reason TEXT, details TEXT)"
    )
    db.executemany(
        "INSERT INTO audit(ts,scope,reason,details) VALUES(?,?,?,?)",
        [(946684800, "old-scope", "natural_silence", "{}") for _ in range(5000)],
    )
    db.commit()
    db.close()
    ledger = Ledger(path)
    assert ledger.db.execute("SELECT SUM(n) FROM daily_summary").fetchone()[0] == 5000
    ledger.record("new-room", "natural_reply")
    assert ledger.db.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 5000
    assert ledger.db.execute("SELECT SUM(n) FROM daily_summary").fetchone()[0] == 5001
    ledger.close()
    ledger = Ledger(path)
    assert ledger.db.execute("SELECT SUM(n) FROM daily_summary").fetchone()[0] == 5001
    assert (
        ledger.db.execute(
            "SELECT n FROM daily_summary WHERE day='2000-01-01'"
        ).fetchone()[0]
        == 5000
    )
    ledger.close()


@pytest.mark.asyncio
async def test_concurrent_daily_records_are_not_lost(tmp_path):
    ledger = Ledger(tmp_path / "concurrent.sqlite3")
    try:
        await asyncio.gather(
            *(ledger.arecord("room", "jev_evaluation") for _ in range(100))
        )
        assert ledger.db.execute("SELECT n FROM daily_summary").fetchone()[0] == 100
    finally:
        await ledger.aclose()


def test_room_topic_history_bound():
    room = Room()
    for n in range(1000):
        room.observe({"content": "message", "sender_id": "a"}, str(n))
    assert len(room.rows) == 64
    assert room.rows[-1]["generation"] == 1000


@pytest.mark.parametrize(
    "prefix",
    ["Authorization: Bearer ", "key=", "api_key=", "Cookie: session=", "Bearer "],
)
def test_preview_never_keeps_credential_value(prefix):
    assert "z" not in audit_preview(prefix + "z" * 80)


@pytest.mark.asyncio
async def test_classic_preview_tracks_join_and_style_adjustment(env):
    env.cfg.update(decision_mode="classic", style="quiet", join_threshold_offset=0.02)
    assert env.plugin._boundary_decision({"worthwhile": 0.8}, 0)
    assert not env.plugin._boundary_decision({"worthwhile": 0.79}, 0)
    assert env.plugin._boundary_decision({"worthwhile": 0.98}, 2)
    assert not env.plugin._boundary_decision({"worthwhile": 0.69}, 0)


@pytest.mark.asyncio
async def test_blank_unrelated_photo_does_not_replace_existing_draft(env):
    env.cfg["burst_merge_enabled"] = True
    ev = await env.admit(base.event(text="AstrBot插件设置"))
    photo = base.event(
        mid="photo", text="", parts=[Image.fromURL("https://example.invalid/new.png")]
    )
    env.plugin._coalesce(photo, env.plugin.rooms.get(ev.unified_msg_origin), env.cid)
    assert not ev.get_extra(MARK)["rejected"]
    assert not photo.get_extra("_jev_merged_message_ids")
