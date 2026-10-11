"""Regression tests for 0.2: no network, no real credentials or QQ sends."""

import asyncio
import json
import sqlite3
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from astrbot.api.message_components import Image, Plain
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from ..client import Adapter, JevClient, KeyPool
from ..delivery import DeliveryDeclined, SplitAdapters
from ..main import MARK
from ..policy import QUESTIONS, is_refusal, pack_state, questions_for
from ..state import Ledger
from . import test_integration as base


@pytest_asyncio.fixture
async def env():
    fixture = base.IntegrationTests()
    await fixture.asyncSetUp()
    try:
        yield fixture
    finally:
        await fixture.asyncTearDown()


def test_quality_preserves_every_selected_character_under_budget():
    marker = "CORE_PERSONA_AND_TOPIC_DETAIL"
    persona = "p" * 1800 + marker + "q" * 1800
    content = "m" * 700 + marker + "n" * 700
    state = pack_state(persona, [{"content": content}], {"content": content}, {})
    assert state["bot_persona"] == persona
    assert state["recent_conversation"][0]["content"] == content
    assert state["target_message"]["content"] == content
    assert state["truncated"] is False


@pytest.mark.asyncio
async def test_failed_quota_write_never_reserves_a_slot():
    class LedgerFailure:
        def count(self, *args):
            return 0

        def bump(self, *args):
            raise sqlite3.OperationalError("locked")

    pool = KeyPool("nanbei", {"keys": ["fake-a", "fake-b"]})
    pool.ledger = LedgerFailure()
    for _ in range(4):
        with pytest.raises(sqlite3.OperationalError):
            await pool.acquire()
        assert pool.active == 0
        assert all(k.active == 0 for k in pool.keys)
    pool.ledger = None
    assert await pool.acquire() is not None


@pytest.mark.asyncio
async def test_sqlite_contention_does_not_block_loop(tmp_path):
    path = tmp_path / "ledger.db"
    ledger = Ledger(path)
    other = sqlite3.connect(path)
    other.execute("BEGIN IMMEDIATE")
    task = asyncio.create_task(ledger.arecord("room", "probe"))
    start = time.monotonic()
    await asyncio.sleep(0.03)
    assert time.monotonic() - start < 0.2
    other.rollback()
    other.close()
    await task
    await ledger.aclose()


@pytest.mark.asyncio
async def test_latest_candidate_waits_then_is_evaluated(env):
    entered, release = asyncio.Event(), asyncio.Event()
    env.cfg["evaluation_interval"] = 1

    async def judge(*args):
        entered.set()
        await release.wait()
        return env.response

    env.plugin.client.evaluate = AsyncMock(side_effect=judge)
    older = base.event(mid="1", sender="first")
    newer = base.event(mid="2", sender="second")
    one = asyncio.create_task(env.plugin.on_message(older))
    await entered.wait()
    two = asyncio.create_task(env.plugin.on_message(newer))
    await asyncio.sleep(0.02)
    env.plugin.rooms.get(older.unified_msg_origin).evaluated_at -= 2
    release.set()
    await asyncio.gather(one, two)
    assert older.get_extra(MARK) is None
    assert newer.get_extra(MARK) is not None
    assert env.plugin.client.evaluate.await_count == 2


@pytest.mark.asyncio
async def test_burst_replaces_unreleased_generation_and_keeps_images(env):
    env.cfg["burst_merge_enabled"] = True
    first = base.event(
        mid="one",
        text="看看这张图",
        parts=[Plain("看看这张图"), Image.fromURL("https://example.invalid/a.png")],
    )
    await env.admit(first)
    second = base.event(
        mid="two",
        text="再看右下角",
        parts=[Plain("再看右下角"), Image.fromURL("https://example.invalid/b.png")],
    )
    await env.plugin.on_message(second)
    assert first.is_stopped()
    assert first.get_extra(MARK)["rejected"]
    assert second.get_extra(MARK) is not None
    assert second.get_extra("_jev_merged_message_ids") == ["one", "two"]
    assert "看看这张图" in second.get_message_str()
    assert "再看右下角" in second.get_message_str()
    assert sum(isinstance(p, Image) for p in second.get_messages()) == 2
    old_response = LLMResponse(role="assistant", completion_text="obsolete answer")
    await env.plugin.on_response(first, old_response)
    assert old_response.completion_text == ""


@pytest.mark.asyncio
async def test_burst_cancels_exact_old_gate_task_not_latest(env):
    env.cfg["burst_merge_enabled"] = True
    entered = asyncio.Event()
    calls = 0

    async def judge(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await asyncio.sleep(60)
        return env.response

    env.plugin.client.evaluate = AsyncMock(side_effect=judge)
    first, second = (
        base.event(mid="a", text="第一句"),
        base.event(mid="b", text="第一句的补充"),
    )
    task = asyncio.create_task(env.plugin.on_message(first))
    await entered.wait()
    await env.plugin.on_message(second)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert second.get_extra(MARK) is not None
    assert calls == 2


@pytest.mark.asyncio
async def test_delivered_reply_is_not_merged_again(env):
    env.cfg["burst_merge_enabled"] = True
    first = await env.admit(base.event(mid="a", text="第一句"))
    first.get_extra(MARK)["sent_parts"] = 1
    first.get_extra(MARK)["committed"] = True
    env.plugin._release(first)
    env.plugin.rooms.get(first.unified_msg_origin).evaluated_at -= 5
    second = base.event(mid="b", text="第二句")
    await env.plugin.on_message(second)
    assert not second.get_extra("_jev_merged_message_ids")
    assert not first.get_extra(MARK)["rejected"]
    assert second.get_message_str() == "第二句"


@pytest.mark.asyncio
async def test_explicit_and_other_speaker_messages_are_not_merged(env):
    env.cfg["burst_merge_enabled"] = True
    first = await env.admit(base.event(mid="a"))
    explicit = base.event(mid="b")
    explicit.is_at_or_wake_command = True
    await env.plugin.on_message(explicit)
    assert not explicit.get_extra(MARK)
    assert not explicit.get_extra("_jev_merged_message_ids")
    env.plugin._release(first)
    env.plugin.rooms.get(first.unified_msg_origin).evaluated_at -= 5
    other = base.event(mid="c", sender="other person")
    await env.plugin.on_message(other)
    assert not other.get_extra("_jev_merged_message_ids")


@pytest.mark.asyncio
async def test_changed_conversation_blocks_final_delivery(env):
    ev = await env.admit()
    response = LLMResponse(role="assistant", completion_text="draft")
    await env.plugin.on_response(ev, response)
    env.cid = "another-conversation"
    ev.set_result(ev.plain_result("draft"))
    await env.plugin.before_output(ev)
    assert ev.is_stopped()
    assert ev.get_extra(MARK)["rejected"]


@pytest.mark.asyncio
async def test_core_technical_error_is_suppressed(env):
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
        InternalAgentSubStage,
    )

    ev = base.event()
    original_send = AsyncMock()
    ev.send = original_send
    await env.admit(ev)
    with pytest.raises(DeliveryDeclined):
        await InternalAgentSubStage._send_llm_error_message(
            object(), ev, "technical failure"
        )
    original_send.assert_not_awaited()
    assert not env.plugin.rooms.get(ev.unified_msg_origin).pending


@pytest.mark.asyncio
async def test_empty_output_releases_ticket_without_timeout(env):
    ev = await env.admit()
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="draft")
    )
    ev.set_result(ev.make_result())
    await env.plugin.before_output(ev)
    assert not env.plugin.rooms.get(ev.unified_msg_origin).pending
    assert not env.plugin.pending_events


@pytest.mark.asyncio
async def test_native_history_is_a_real_fallback(env):
    env.history = json.dumps([{"role": "assistant", "content": "EARLIER_NATIVE_TOPIC"}])
    await env.admit()
    state = env.plugin.client.evaluate.await_args.args[0]
    assert "EARLIER_NATIVE_TOPIC" in str(state)


@pytest.mark.asyncio
async def test_shared_behavior_and_scene_uncertainty_are_consistent(env):
    env.cfg["behavior_guidance"] = "自然接梗，不主动长篇说教"
    ev = await env.admit()
    from astrbot.core.agent.message import TextPart

    req = ProviderRequest(
        prompt="hi", conversation=NS(cid=env.cid), system_prompt="Original persona"
    )
    req.extra_user_content_parts.append(
        TextPart(
            text="<conversation_scene><current_message><talking_to>群里所有人</talking_to></current_message><instruction>不是在问你</instruction></conversation_scene>"
        )
    )
    await env.plugin.on_request(ev, req)
    assert "不是在问你" not in req.extra_user_content_parts[0].text
    assert env.cfg["behavior_guidance"] in str(req.extra_user_content_parts)
    assert req.system_prompt == "Original persona"


@pytest.mark.asyncio
async def test_review_waits_for_capacity_instead_of_immediately_dropping():
    async def transport(endpoint, key, payload):
        return (
            200,
            {},
            {
                "answers": {
                    n: {"type": "noul", "noul": 0.9} for n in payload["questions"]
                }
            },
        )

    client = JevClient(
        {"nanbei": {"keys": ["fake-key"], "concurrency": 1}}, transport=transport
    )
    pool = client.pools["nanbei"]
    slot = await pool.acquire()
    task = asyncio.create_task(client.evaluate({"candidate_reply": "draft"}, QUESTIONS))
    await asyncio.sleep(0.02)
    assert not task.done()
    await pool.release(slot)
    assert (await task)["channel"] == "nanbei"
    assert pool.active == 0
    await client.close()


@pytest.mark.asyncio
async def test_explicitly_different_quota_groups_do_not_share_cooldown():
    client = JevClient(
        {
            "nanbei": {
                "keys": ["fake-a", "fake-b"],
                "quota_group_ids": ["org-a", "org-b"],
            }
        }
    )
    pools = [p for p in client.pools.values() if p.adapter.kind == "nanbei"]
    first = await pools[0].acquire()
    await pools[0].release(first, cooldown=60, shared=True)
    assert await pools[0].acquire() is None
    second = await pools[1].acquire()
    assert second is not None
    await pools[1].release(second)
    await client.close()


@pytest.mark.asyncio
async def test_scoped_split_adapter_leaves_unmarked_events_unchanged():
    class SplitStep:
        def __init__(self):
            self.plugin_config = NS(context=NS(send_message=AsyncMock()))

        async def handle(self, ctx):
            await self.plugin_config.context.send_message(
                ctx.event.unified_msg_origin, "segment"
            )
            return True

    step = SplitStep()
    original = step.handle
    adapter = SplitAdapters(MARK)
    adapter.attach(NS(pipeline=NS(_steps=[step])))
    marked, plain = base.event(), base.event(mid="plain")
    marked.set_extra(MARK, {})
    marked.get_extra(MARK)["ticket"] = "x"
    marked.send = AsyncMock()
    await step.handle(NS(event=marked, chain=[]))
    marked.send.assert_awaited_once_with("segment")
    await step.handle(NS(event=plain, chain=[]))
    step.plugin_config.context.send_message.assert_awaited_once()
    adapter.restore()
    assert step.handle == original


@pytest.mark.asyncio
async def test_burst_window_and_conversation_boundaries(env):
    env.cfg["burst_merge_enabled"] = True
    first = await env.admit(base.event(mid="one"))
    room = env.plugin.rooms.get(first.unified_msg_origin)
    env.plugin._release(first)
    first.get_extra("_jev_burst_frame").last -= 10
    room.evaluated_at -= 10
    second = base.event(mid="two")
    await env.plugin.on_message(second)
    assert not second.get_extra("_jev_merged_message_ids")
    env.plugin._release(second)
    room.evaluated_at -= 10
    env.cid = "new-conversation"
    third = base.event(mid="three")
    await env.plugin.on_message(third)
    assert not third.get_extra("_jev_merged_message_ids")


@pytest.mark.asyncio
async def test_shadow_mode_does_not_mutate_messages(env):
    env.cfg.update(dry_run=True, burst_merge_enabled=True, evaluation_interval=1)
    first = base.event(mid="one", text="first")
    await env.plugin.on_message(first)
    room = env.plugin.rooms.get(first.unified_msg_origin)
    room.evaluated_at -= 2
    second = base.event(mid="two", text="second")
    await env.plugin.on_message(second)
    assert first.get_message_str() == "first"
    assert second.get_message_str() == "second"
    assert not first.is_stopped() and not second.is_stopped()
    assert not second.get_extra(MARK)
    assert env.plugin.ledger.count(second.unified_msg_origin, "evaluations") == 0
    assert env.plugin.ledger.count(second.unified_msg_origin, "shadow_evaluations") == 2


@pytest.mark.asyncio
async def test_nested_event_and_bot_send_count_one_logical_reply(env):
    from astrbot.api.event import MessageChain

    ev = base.event()
    bot = NS(send=AsyncMock(return_value={"message_id": 123}))
    ev.bot = bot

    async def adapter_send(chain):
        await ev.bot.send({}, chain)
        ev._has_send_oper = True

    ev.send = adapter_send
    await env.admit(ev)
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="hello")
    )
    ev.set_result(ev.plain_result("hello"))
    await env.plugin.before_output(ev)
    await ev.send(MessageChain([Plain("first part")]))
    await ev.send(MessageChain([Plain("second part")]))
    await env.plugin.after_sent(ev)
    assert bot.send.await_count == 2
    assert ev.get_extra(MARK)["sent_parts"] == 2
    assert env.plugin.ledger.count(ev.unified_msg_origin, "sent") == 1
    assert not env.plugin.pending_events


@pytest.mark.asyncio
async def test_review_result_is_invalidated_by_new_message(env):
    ev = await env.admit()
    env.cfg["second_review"] = "always"
    entered, finish = asyncio.Event(), asyncio.Event()

    async def review(*args):
        entered.set()
        await finish.wait()
        return {"values": {"appropriate": 0.99, "redundant": 0.01}}

    env.plugin.client.evaluate = AsyncMock(side_effect=review)
    draft = LLMResponse(role="assistant", completion_text="outdated reply")
    task = asyncio.create_task(env.plugin.on_response(ev, draft))
    await entered.wait()
    env.plugin.rooms.get(ev.unified_msg_origin).observe(
        {"content": "solved already"}, "new"
    )
    finish.set()
    await task
    assert ev.is_stopped()
    assert draft.completion_text == ""


@pytest.mark.asyncio
async def test_picture_does_not_start_generation_with_no_judgment_keys(env):
    env.plugin.client = JevClient({"nanbei": {"keys": []}})
    ev = base.event(parts=[Image.fromURL("https://example.invalid/a.png")])
    await env.plugin.on_message(ev)
    assert not ev.get_extra(MARK)
    assert not ev.is_at_or_wake_command


def test_edge_denial_does_not_permanently_disable_valid_key():
    code, duration, shared = Adapter.create("nanbei", {}).classify(
        403, None, {"server": "cloudflare"}
    )
    assert code == "edge_rejected"
    assert duration == 60 and shared


@pytest.mark.asyncio
async def test_confirmed_auth_rejection_can_use_explicitly_enabled_backup():
    calls = []

    async def transport(endpoint, key, payload):
        calls.append(endpoint)
        if "hajimi.165201.xyz" in endpoint:
            return 401, {}, {"error": {"code": "invalid_api_key"}}
        return (
            200,
            {},
            {
                "answers": {
                    n: {"type": "noul", "noul": 0.9} for n in payload["questions"]
                }
            },
        )

    client = JevClient(
        {
            "retry_explicit_rejection": True,
            "allow_channel_fallback": True,
            "nanbei": {"keys": ["fake-nanbei"]},
            "typesafe": {"keys": ["fake-typesafe"]},
        },
        transport=transport,
    )
    result = await client.evaluate({}, QUESTIONS)
    assert result["channel"] == "typesafe"
    assert len(calls) == 2
    assert all(pool.active == 0 for pool in client.pools.values())
    await client.close()


@pytest.mark.asyncio
async def test_repeated_requests_leave_no_leases_waiters_or_unbounded_windows():
    seen = []

    async def transport(endpoint, key, payload):
        seen.append(key)
        await asyncio.sleep(0)
        return (
            200,
            {},
            {
                "answers": {
                    n: {"type": "noul", "noul": 0.9} for n in payload["questions"]
                }
            },
        )

    keys = [f"fake-key-{i}" for i in range(4)]
    client = JevClient(
        {
            "nanbei": {
                "keys": keys,
                "quota_rpm": 10000,
                "key_rpm": 10000,
                "concurrency": 4,
                "daily_request_limit": 0,
            }
        },
        transport=transport,
    )
    for _ in range(25):
        await asyncio.gather(*(client.evaluate({}, QUESTIONS) for _ in range(20)))
    pool = client.pools["nanbei"]
    assert len(seen) == 500
    assert [seen.count(key) for key in keys] == [125] * 4
    assert pool.active == 0 and all(k.active == 0 for k in pool.keys)
    assert not client.waiters
    assert len(pool.requests) == 500
    await client.close()


@pytest.mark.asyncio
async def test_lifetime_expiry_releases_own_event(env):
    ev = await env.admit()
    env.plugin._expire(ev)
    assert ev.is_stopped()
    assert not env.plugin.pending_events
    assert not env.plugin.rooms.get(ev.unified_msg_origin).pending
    assert ev.get_extra(MARK)["rejected"]


@pytest.mark.asyncio
async def test_burst_does_not_replace_a_part_already_being_sent(env):
    env.cfg["burst_merge_enabled"] = True
    first = await env.admit(base.event(mid="first"))
    first.get_extra(MARK)["send_depth"] = 1
    room = env.plugin.rooms.get(first.unified_msg_origin)
    env.plugin._release(first)
    room.evaluated_at -= 10
    second = base.event(mid="second", text="new information")
    await env.plugin.on_message(second)
    assert not second.get_extra("_jev_merged_message_ids")
    assert not first.get_extra(MARK)["rejected"]


@pytest.mark.asyncio
async def test_overflow_summarizes_only_auxiliary_evidence_not_original_request(env):
    env.cfg["second_review"] = "always"
    env.ctx.get_current_chat_provider_id = AsyncMock(return_value="main-provider")
    env.ctx.get_provider_by_id = lambda _: NS(
        provider_config={"max_context_tokens": 200000}
    )
    env.ctx.llm_generate = AsyncMock(
        return_value=LLMResponse(
            role="assistant",
            completion_text="事实摘要：此人喜欢游戏；没有要求长篇建议。",
        )
    )
    ev = await env.admit()
    original = "reference memory " * 20000
    req = ProviderRequest(
        prompt="原始消息", system_prompt=original, conversation=NS(cid=env.cid)
    )
    await env.plugin.on_request(ev, req)
    env.ctx.llm_generate.assert_not_awaited()
    env.plugin.client.evaluate = AsyncMock(
        return_value={"values": {"appropriate": 0.9, "redundant": 0.1}}
    )
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="自然接话草稿")
    )
    env.ctx.llm_generate.assert_awaited_once()
    assert req.system_prompt == original
    assert req.prompt == "原始消息"
    evidence = ev.get_extra(MARK)["generation_evidence"]
    assert evidence["source"].startswith("lossy_digest")
    assert evidence["original_bytes"] > 100000
    assert ev.get_extra(MARK)["state"]["bot_persona"] == "安静但喜欢一起聊游戏的朋友"
    assert not ev.is_stopped()


@pytest.mark.asyncio
async def test_under_budget_never_calls_auxiliary_summarizer(env):
    env.cfg["second_review"] = "always"
    env.ctx.llm_generate = AsyncMock()
    ev = await env.admit()
    req = ProviderRequest(
        prompt="原消息", system_prompt="完整人设", conversation=NS(cid=env.cid)
    )
    await env.plugin.on_request(ev, req)
    env.ctx.llm_generate.assert_not_called()
    assert (
        ev.get_extra(MARK)["generation_evidence"]["effective_system_prompt"]
        == "完整人设"
    )


@pytest.mark.asyncio
async def test_unavailable_summary_does_not_silently_clip_evidence(env):
    env.cfg["second_review"] = "always"
    env.ctx.get_current_chat_provider_id = AsyncMock(return_value="unknown")
    env.ctx.get_provider_by_id = lambda _: NS(provider_config={"max_context_tokens": 0})
    env.ctx.llm_generate = AsyncMock()
    ev = await env.admit()
    req = ProviderRequest(
        prompt="原消息", system_prompt="memory " * 50000, conversation=NS(cid=env.cid)
    )
    await env.plugin.on_request(ev, req)
    env.ctx.llm_generate.assert_not_called()
    assert not ev.is_stopped()
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="待审草稿")
    )
    assert ev.is_stopped()
    assert req.system_prompt == "memory " * 50000


@pytest.mark.asyncio
async def test_channel_cooldown_survives_reload_and_can_be_reset(tmp_path):
    ledger = Ledger(tmp_path / "health.db")
    first = JevClient({"nanbei": {"keys": ["fake-a"]}})
    for pool in first.pools.values():
        pool.ledger = ledger
    pool = first.pools["nanbei"]
    key = await pool.acquire()
    await pool.release(key, 3600, True)
    await first.close()
    second = JevClient({"nanbei": {"keys": ["fake-a"]}})
    for pool in second.pools.values():
        pool.ledger = ledger
    assert not await second.review_ready()
    await second.reset_cooldowns()
    assert await second.review_ready()
    await second.close()
    await ledger.aclose()


@pytest.mark.asyncio
async def test_unavailable_jev_still_wakes_by_name_without_wakepro(env):
    env.plugin.client = JevClient({"nanbei": {"keys": []}})
    env.cfg["wake_names"] = ["吱吱"]
    ev = base.event(text="吱吱，你在吗")
    await env.plugin.on_input(ev)
    assert ev.is_at_or_wake_command
    assert not hasattr(env.plugin, "wake_adapters")


def test_legacy_inspired_question_override_is_typed_and_has_safe_fallback():
    q = questions_for(
        {
            "question_overrides": '{"worthwhile":{"instructions":"Natural reactions are welcome", "type":"score"}}'
        }
    )
    assert q["worthwhile"]["type"] == "noul"
    assert "Natural reactions" in q["worthwhile"]["instructions"]
    assert questions_for({"question_overrides": "bad-json"}) == questions_for({})


def test_refusal_marker_tolerates_punctuation_without_fuzzy_matching():
    assert is_refusal(" [不回复]。 ")
    assert is_refusal("[PASS]!")
    assert not is_refusal("[PASS]，喵")
    assert not is_refusal("[笑死]")


@pytest.mark.asyncio
async def test_target_is_not_duplicated_in_gate_history(env):
    ev = await env.admit(base.event(mid="target", text="UNIQUE_CURRENT_MESSAGE"))
    state = ev.get_extra(MARK)["state"]
    assert state["target_message"]["content"] == "UNIQUE_CURRENT_MESSAGE"
    assert all(
        row.get("content") != "UNIQUE_CURRENT_MESSAGE"
        for row in state["recent_conversation"]
    )
