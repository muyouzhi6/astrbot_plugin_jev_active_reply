"""Fast judgment paths and dashboard migration, no external requests."""

import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from astrbot.api.message_components import Image, Plain
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from ..client import KeyPool
from ..main import MARK
from ..policy import natural_decide, questions_for
from ..settings import ConfigView, GROUPS, migrate_config_file
from . import test_integration as base
from . import test_v020 as previous

env = previous.env


def test_one_social_question_instead_of_four_independent_vetoes():
    questions = questions_for({"decision_mode": "natural"})
    assert set(questions) == {"reply_now"}
    assert natural_decide({"reply_now": 0.65}).speak
    assert not natural_decide({"reply_now": 0.35}).speak
    assert not natural_decide({}).speak
    assert natural_decide({"reply_now": 0.55}, "balanced").speak
    assert not natural_decide({"reply_now": 0.55}, "quiet").speak
    assert set(questions_for({"decision_mode": "classic"})) == {
        "addressed",
        "continuation",
        "worthwhile",
        "intrusive",
    }


@pytest.mark.asyncio
async def test_natural_gate_has_only_one_call_before_main_generation(env):
    env.cfg.update(decision_mode="natural", second_review="adaptive")
    env.plugin.client.evaluate = AsyncMock(return_value={"values": {"reply_now": 0.7}})
    ev = await env.admit()
    assert ev.get_extra(MARK)["mode"] == "natural"
    assert set(env.plugin.client.evaluate.await_args.args[1]) == {"reply_now"}
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="接上这句话")
    )
    assert env.plugin.client.evaluate.await_count == 1
    assert ev.get_extra(MARK)["reviewed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("visual", [False, True])
async def test_adaptive_no_summary_or_review_even_with_full_large_context(env, visual):
    env.cfg["second_review"] = "adaptive"
    parts = [Plain("接着聊")]
    if visual:
        parts.append(Image.fromURL("https://example.invalid/image.png"))
    ev = await env.admit(base.event(parts=parts))
    env.plugin.client.evaluate.reset_mock()
    env.ctx.llm_generate = AsyncMock()
    original = "memory context " * 40000
    req = ProviderRequest(
        prompt="原始问题", system_prompt=original, conversation=NS(cid=env.cid)
    )
    await env.plugin.on_request(ev, req)
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="自然回应")
    )
    ev.set_result(ev.plain_result("自然回应"))
    await env.plugin.before_output(ev)
    assert not ev.is_stopped()
    assert await env.plugin._permit(ev)
    assert not ev.get_extra(MARK).get("evidence_overflow")
    assert req.system_prompt == original
    env.plugin.client.evaluate.assert_not_awaited()
    env.ctx.llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_adaptive_changed_scene_one_freshness_check_without_summary(env):
    env.cfg["second_review"] = "adaptive"
    ev = await env.admit()
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    room.observe({"content": "another interjection"}, "later")
    env.plugin.client.evaluate = AsyncMock(return_value={"values": {"still_fits": 0.9}})
    env.ctx.llm_generate = AsyncMock()
    ev.get_extra(MARK)["evidence_overflow"] = True
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="仍然相关的回应")
    )
    env.plugin.client.evaluate.assert_awaited_once()
    state, questions = env.plugin.client.evaluate.await_args.args
    assert set(questions) == {"still_fits"}
    assert "generation_evidence" not in state
    env.ctx.llm_generate.assert_not_awaited()
    assert ev.get_extra(MARK)["approved_generation"] == room.generation


@pytest.mark.asyncio
@pytest.mark.parametrize("fits", [0.1, 0.9])
async def test_adaptive_new_message_during_freshness_never_approves_stale_snapshot(
    env, fits
):
    env.cfg["second_review"] = "adaptive"
    ev = await env.admit()
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    room.observe({"content": "new"}, "later")

    async def check(*args):
        room.observe({"content": "changed again"}, "latest")
        return {"values": {"still_fits": fits}}

    env.plugin.client.evaluate = AsyncMock(side_effect=check)
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="旧草稿")
    )
    assert ev.is_stopped()
    env.plugin.client.evaluate.assert_awaited_once()


@pytest.mark.asyncio
async def test_adaptive_after_approval_change_still_blocked_at_send(env):
    env.cfg["second_review"] = "adaptive"
    ev = await env.admit()
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="待发")
    )
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    room.observe({"content": "new"}, "later")
    ev.set_result(ev.plain_result("待发"))
    await env.plugin.before_output(ev)
    assert ev.is_stopped()


@pytest.mark.asyncio
async def test_freshness_negative_is_not_ignored(env):
    env.cfg["second_review"] = "adaptive"
    ev = await env.admit()
    env.plugin.rooms.get(ev.unified_msg_origin).observe(
        {"content": "withdrawn"}, "later"
    )
    env.plugin.client.evaluate = AsyncMock(return_value={"values": {"still_fits": 0.1}})
    response = LLMResponse(role="assistant", completion_text="obsolete")
    await env.plugin.on_response(ev, response)
    assert ev.is_stopped()
    assert not response.completion_text


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["duplicate", "silence", "cid", "expired"])
async def test_adaptive_retains_hard_guards(env, case):
    env.cfg["second_review"] = "adaptive"
    ev = await env.admit()
    if case == "duplicate":
        env.plugin.rooms.get(ev.unified_msg_origin).last_text = "草稿"
    elif case == "silence":
        env.plugin._finish_silent(ev)
    elif case == "cid":
        env.cid = "different"
    else:
        ev.get_extra(MARK)["created"] -= 200
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="草稿")
    )
    ev.set_result(ev.plain_result("草稿"))
    await env.plugin.before_output(ev)
    assert ev.is_stopped()


@pytest.mark.asyncio
async def test_adaptive_can_still_silence_without_extra_model_call(env):
    env.cfg["second_review"] = "adaptive"
    ev = await env.admit()
    env.plugin.client.evaluate.reset_mock()
    env.ctx.llm_generate = AsyncMock()
    env.plugin._finish_silent(ev)
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="[PASS]")
    )
    env.plugin.client.evaluate.assert_not_awaited()
    env.ctx.llm_generate.assert_not_awaited()
    assert ev.is_stopped()


def test_real_core_schema_roundtrip_preserves_old_values_and_new_ui_edits(tmp_path):
    schema = json.loads(
        (Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    path = tmp_path / "config.json"
    old = {
        "second_review": "always",
        "evaluation_interval": 7,
        "blocked_keywords": [],
        "silence": {"ignored_users": ["test-user"]},
        "typesafe": {"keys": ["test-key"], "daily_request_limit": 0},
        "style": "quiet",
    }
    path.write_text(json.dumps(old), encoding="utf-8")
    migrate_config_file(path)
    raw = AstrBotConfig(str(path), schema=schema)
    config = ConfigView(raw)
    for key in ("second_review", "evaluation_interval", "style", "blocked_keywords"):
        assert config[key] == old[key]
    assert config["typesafe"]["keys"] == ["test-key"]
    assert config["silence"]["ignored_users"] == ["test-user"]
    raw["advanced"]["judgment"]["second_review"] = "adaptive"
    raw["advanced"]["timing"]["evaluation_interval"] = 1
    raw.save_config()
    again = ConfigView(AstrBotConfig(str(path), schema=schema))
    assert again["second_review"] == "adaptive"
    assert again["evaluation_interval"] == 1
    assert dict(again)["second_review"] == "adaptive"
    assert len(again) == len(dict(again))
    for group, keys in GROUPS.items():
        assert all(again.raw["advanced"][group][key] is not None for key in keys)
    assert len([field for field in schema.values() if not field.get("invisible")]) == 15
    assert schema["advanced"]["condition"] == {"show_advanced": True}
    assert schema["dry_run"]["invisible"] and not again["dry_run"]


def test_new_install_defaults_fast_without_local_daily_cap(tmp_path):
    schema = json.loads(
        (Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    config = ConfigView(AstrBotConfig(str(tmp_path / "new.json"), schema=schema))
    assert config["decision_mode"] == "natural"
    assert config["second_review"] == "adaptive"
    assert config["image_decision_mode"] == "text_gate"
    for kind in ("typesafe", "nanbei"):
        assert config[kind]["daily_request_limit"] == 0
        assert KeyPool(kind, {"keys": ["test-key"]}).daily_limit == 0


@pytest.mark.asyncio
async def test_no_daily_cap_does_not_reset_or_skip_accounting():
    pool = KeyPool("typesafe", {"keys": ["test-key"], "daily_request_limit": 0})
    pool.ledger = NS(acount=AsyncMock(return_value=500), abump=AsyncMock())
    key = await pool.acquire()
    assert key is not None
    pool.ledger.abump.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("score, admitted", [(0.1, False), (0.8, True)])
async def test_image_is_screened_before_main_generation_with_original_preserved(
    env, score, admitted
):
    env.cfg.update(
        decision_mode="natural",
        image_decision_mode="text_gate",
        second_review="adaptive",
    )
    env.plugin.client.evaluate = AsyncMock(
        return_value={"values": {"reply_now": score}}
    )
    image = Image.fromURL("https://example.invalid/original.png")
    ev = base.event(text="再看这张", parts=[Plain("再看这张"), image])
    await env.plugin.on_message(ev)
    env.plugin.client.evaluate.assert_awaited_once()
    state, questions = env.plugin.client.evaluate.await_args.args
    assert "Image" in state["target_message"]["actual_media"]
    assert "no pixels" in questions["reply_now"]["instructions"]
    assert bool(ev.get_extra(MARK)) == admitted
    assert ev.is_at_or_wake_command == admitted
    assert ev.get_messages()[1] is image


@pytest.mark.asyncio
async def test_explicit_image_request_does_not_wait_for_jev_gate(env):
    env.cfg.update(
        decision_mode="natural",
        image_decision_mode="text_gate",
        second_review="adaptive",
    )
    ev = base.event(parts=[Image.fromURL("https://example.invalid/original.png")])
    ev.is_at_or_wake_command = True
    await env.plugin.on_message(ev)
    env.plugin.client.evaluate.assert_not_awaited()
    assert not ev.is_stopped()
