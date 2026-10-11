"""Adversarial conversational scheduling, no network or platform messages."""

import asyncio
import gc
import weakref
import time
from unittest.mock import AsyncMock

import pytest
from astrbot.core.provider.entities import LLMResponse
from astrbot.api.message_components import Image, Plain

from ..delivery import DeliveryDeclined
from ..main import MARK
from .. import burst
from . import test_integration as base
from . import test_v020 as previous

env = previous.env


@pytest.mark.asyncio
async def test_interleaved_other_speaker_does_not_erase_same_sender_burst(env):
    env.cfg["burst_merge_enabled"] = True
    room = env.plugin.rooms.get(base.event().unified_msg_origin)
    one = base.event(mid="a1", sender="alice", text="我遇到一个问题")
    other = base.event(mid="b1", sender="bob", text="路过")
    two = base.event(mid="a2", sender="alice", text="就是右下角这块")
    env.plugin._coalesce(one, room, env.cid)
    env.plugin._coalesce(other, room, env.cid)
    env.plugin._coalesce(two, room, env.cid)
    assert two.get_extra("_jev_merged_message_ids") == ["a1", "a2"]
    assert "路过" not in two.get_message_str()


@pytest.mark.asyncio
async def test_delivered_burst_stays_closed_after_event_garbage_collection(env):
    env.cfg["burst_merge_enabled"] = True
    ev = await env.admit(base.event(mid="a1", text="已经答过的问题"))
    ev.get_extra(MARK)["draft"] = "已经发送的回复"
    await env.plugin._accepted(ev)
    env.plugin._release(ev)
    ref = weakref.ref(ev)
    del ev
    gc.collect()
    assert ref() is None
    new = base.event(mid="a2", text="我换一个问题")
    env.plugin._coalesce(new, env.plugin.rooms.get(new.unified_msg_origin), env.cid)
    assert not new.get_extra("_jev_merged_message_ids")
    assert new.get_message_str() == "我换一个问题"


@pytest.mark.asyncio
async def test_rejection_while_delivery_checks_context_cannot_send_old_reply(env):
    ev = base.event()
    send = AsyncMock()
    ev.send = send
    await env.admit(ev)
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="旧回复")
    )
    ev.set_result(ev.plain_result("旧回复"))
    await env.plugin.before_output(ev)
    entered, release = asyncio.Event(), asyncio.Event()

    async def current(umo):
        entered.set()
        await release.wait()
        return env.cid

    env.ctx.conversation_manager.get_curr_conversation_id = current
    task = asyncio.create_task(ev.send(ev.plain_result("旧回复")))
    await entered.wait()
    env.plugin._reject(ev, None, "request_lease_expired")
    release.set()
    with pytest.raises(DeliveryDeclined):
        await task
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_fragment_arriving_during_conversation_lookup_is_not_lost(env):
    env.cfg["burst_merge_enabled"] = True
    entered, release = asyncio.Event(), asyncio.Event()
    first_call = True

    async def current(umo):
        nonlocal first_call
        if first_call:
            first_call = False
            entered.set()
            await release.wait()
        return env.cid

    env.ctx.conversation_manager.get_curr_conversation_id = current
    one = base.event(mid="a1", text="前半段关键条件")
    two = base.event(mid="a2", text="后半段问题")
    t1 = asyncio.create_task(env.plugin.on_message(one))
    await entered.wait()
    t2 = asyncio.create_task(env.plugin.on_message(two))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(t1, t2)
    assert two.get_extra("_jev_merged_message_ids") == ["a1", "a2"]
    assert one.get_extra(MARK) is None
    assert two.get_extra(MARK) is not None
    assert (
        "前半段关键条件"
        in env.plugin.client.evaluate.await_args.args[0]["target_message"]["content"]
    )


@pytest.mark.asyncio
async def test_new_fragment_during_network_send_does_not_replace_or_replay(env):
    env.cfg["burst_merge_enabled"] = True
    entered, finish = asyncio.Event(), asyncio.Event()

    async def slow_send(*args):
        entered.set()
        await finish.wait()

    first = base.event(mid="a1", text="第一条")
    first.send = AsyncMock(side_effect=slow_send)
    await env.admit(first)
    await env.plugin.on_response(
        first, LLMResponse(role="assistant", completion_text="第一条的回复")
    )
    first.set_result(first.plain_result("第一条的回复"))
    await env.plugin.before_output(first)
    sending = asyncio.create_task(first.send(first.get_result()))
    await entered.wait()
    second = base.event(mid="a2", text="之后补充")
    room = env.plugin.rooms.get(first.unified_msg_origin)
    env.plugin._coalesce(second, room, env.cid)
    assert not second.get_extra("_jev_merged_message_ids")
    assert not first.get_extra(MARK)["rejected"]
    finish.set()
    await sending
    assert first.get_extra(burst.FRAME).committed


@pytest.mark.asyncio
async def test_committed_split_finishes_after_unrelated_message_but_respects_stop(env):
    ev = base.event()
    send = AsyncMock()
    ev.send = send
    await env.admit(ev)
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="完整两段回复")
    )
    ev.set_result(ev.plain_result("完整两段回复"))
    await env.plugin.before_output(ev)
    await ev.send(ev.plain_result("第一段"))
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    room.observe({"content": "另一个人的闲聊"}, "other-mid")
    await ev.send(ev.plain_result("第二段"))
    assert send.await_count == 2
    room.quiet_until = time.monotonic() + 60
    with pytest.raises(DeliveryDeclined):
        await ev.send(ev.plain_result("不该发送的第三段"))
    assert send.await_count == 2


@pytest.mark.asyncio
async def test_long_configured_lifetime_is_not_silently_released_at_120_seconds(env):
    env.cfg["request_lifetime_seconds"] = 180
    ev = await env.admit()
    room = env.plugin.rooms.get(ev.unified_msg_origin)
    room.pending_since -= 130
    assert room.busy(time.monotonic())
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="draft")
    )
    ev.get_extra(MARK)["created"] -= 130
    ev.set_result(ev.plain_result("draft"))
    await env.plugin.before_output(ev)
    assert not ev.get_extra(MARK)["rejected"]


@pytest.mark.asyncio
async def test_owned_local_image_survives_old_event_cleanup_after_merge(env, tmp_path):
    env.cfg["burst_merge_enabled"] = True
    path = tmp_path / "手机上传.png"
    path.write_bytes(b"test attachment")
    one = base.event(mid="a1", parts=[Image.fromFileSystem(path)])
    one.track_temporary_local_file(str(path))
    await env.admit(one)
    two = base.event(mid="a2", text="图里右下角")
    env.plugin._coalesce(two, env.plugin.rooms.get(one.unified_msg_origin), env.cid)
    one.cleanup_temporary_local_files()
    assert path.is_file()
    assert str(path) in two._temporary_local_files
    two.cleanup_temporary_local_files()
    assert not path.exists()


@pytest.mark.asyncio
async def test_native_request_collects_merged_text_and_two_images(
    env, tmp_path, monkeypatch
):
    from astrbot.core import astr_main_agent

    env.cfg["burst_merge_enabled"] = True
    paths = [tmp_path / "one.png", tmp_path / "two.png"]
    for p in paths:
        p.write_bytes(b"attachment")
    one = base.event(
        mid="a1", text="第一张", parts=[Plain("第一张"), Image.fromFileSystem(paths[0])]
    )
    two = base.event(
        mid="a2",
        text="第二张一起比较",
        parts=[Plain("第二张一起比较"), Image.fromFileSystem(paths[1])],
    )
    room = env.plugin.rooms.get(one.unified_msg_origin)
    env.plugin._coalesce(one, room, env.cid)
    env.plugin._coalesce(two, room, env.cid)
    from types import SimpleNamespace as NS

    monkeypatch.setattr(
        astr_main_agent,
        "_get_session_conv",
        AsyncMock(return_value=NS(cid=env.cid, history="[]")),
    )
    req, _ = await base.collect_request(
        two,
        env.ctx,
        astr_main_agent.MainAgentBuildConfig(
            tool_call_timeout=30, provider_settings={}
        ),
    )
    assert "第一张" in req.prompt and "第二张一起比较" in req.prompt
    assert set(req.image_urls) == {str(p) for p in paths}


@pytest.mark.asyncio
async def test_many_speakers_never_cross_merge_or_grow_unbounded(env):
    env.cfg["burst_merge_enabled"] = True
    room = env.plugin.rooms.get(base.event().unified_msg_origin)
    for user in range(80):
        ev = base.event(mid=f"{user}-1", sender=str(user), text="a" * 1000)
        env.plugin._coalesce(ev, room, env.cid)
    assert len(room.bursts.entries) <= 32
    assert sum(f.size for f in room.bursts.entries.values()) <= 262144
    latest = base.event(mid="79-2", sender="79", text="补充")
    env.plugin._coalesce(latest, room, env.cid)
    assert latest.get_extra("_jev_merged_message_ids") == ["79-1", "79-2"]


@pytest.mark.asyncio
async def test_child_task_cannot_inherit_a_delivery_bypass(env):
    ev = base.event()
    release = asyncio.Event()
    child = None
    sent = []

    async def operation(value):
        nonlocal child
        sent.append(value)
        if value == "parent":

            async def later():
                await release.wait()
                await ev.send("child")

            child = asyncio.create_task(later())

    ev.send = operation
    await env.admit(ev)
    await env.plugin.on_response(
        ev, LLMResponse(role="assistant", completion_text="draft")
    )
    ev.set_result(ev.plain_result("draft"))
    await env.plugin.before_output(ev)
    await ev.send("parent")
    env.plugin._release(ev)
    release.set()
    with pytest.raises(DeliveryDeclined):
        await child
    assert sent == ["parent"]


@pytest.mark.asyncio
async def test_existing_native_reply_finishes_before_new_autonomous_admission(
    env, monkeypatch
):
    from astrbot.core.pipeline.process_stage import follow_up
    from types import SimpleNamespace as NS

    direct = base.event(mid="native")
    done = [False]
    runner = NS(done=lambda: done[0], run_context=NS(context=NS(event=direct)))
    monkeypatch.setitem(
        follow_up._ACTIVE_AGENT_RUNNERS, direct.unified_msg_origin, runner
    )
    new = base.event(mid="later")
    task = asyncio.create_task(env.plugin.on_message(new))
    await asyncio.sleep(0.03)
    env.plugin.client.evaluate.assert_not_awaited()
    done[0] = True
    env.plugin.rooms.get(new.unified_msg_origin).notify()
    await task
    assert new.get_extra(MARK)
    assert not direct.is_stopped()


@pytest.mark.asyncio
async def test_native_reply_starting_during_judgment_prevents_double_ownership(
    env, monkeypatch
):
    from astrbot.core.pipeline.process_stage import follow_up
    from types import SimpleNamespace as NS

    direct = base.event(mid="native")
    runner = NS(done=lambda: False, run_context=NS(context=NS(event=direct)))

    async def judge(*args):
        monkeypatch.setitem(
            follow_up._ACTIVE_AGENT_RUNNERS, direct.unified_msg_origin, runner
        )
        return env.response

    env.plugin.client.evaluate = AsyncMock(side_effect=judge)
    ev = base.event(mid="candidate")
    await env.plugin.on_message(ev)
    assert not ev.get_extra(MARK)
    assert not env.plugin.rooms.get(ev.unified_msg_origin).pending


@pytest.mark.asyncio
async def test_large_context_silent_turn_does_not_spend_a_summary_call(env):
    from types import SimpleNamespace as NS
    from astrbot.core.provider.entities import ProviderRequest

    env.cfg["second_review"] = "always"
    env.ctx.llm_generate = AsyncMock()
    ev = await env.admit()
    req = ProviderRequest(
        prompt="原消息",
        system_prompt="large memory " * 30000,
        conversation=NS(cid=env.cid),
    )
    await env.plugin.on_request(ev, req)
    assert ev.get_extra(MARK)["evidence_overflow"]
    assert not ev.is_stopped()
    await env.plugin.keep_silent(ev)
    env.ctx.llm_generate.assert_not_awaited()
    assert not env.plugin.pending_events


@pytest.mark.asyncio
async def test_seeded_burst_lifecycle_stress_never_replays_committed_fragments(
    env, monkeypatch
):
    import random
    from types import SimpleNamespace as NS
    from .. import main

    rng = random.Random(81237)
    now = [1000.0]
    monkeypatch.setattr(main, "time", NS(monotonic=lambda: now[0], time=time.time))
    env.cfg.update(burst_merge_enabled=True, burst_window_seconds=10)
    room = env.plugin.rooms.get(base.event().unified_msg_origin)
    committed = set()
    for n in range(500):
        now[0] += rng.uniform(0.05, 3)
        uid = f"person-{rng.randrange(8)}"
        mid = f"{uid}/{n}"
        ev = base.event(mid=mid, sender=uid, text=f"segment-{n}")
        env.plugin._coalesce(ev, room, env.cid)
        ids = ev.get_extra("_jev_merged_message_ids") or [mid]
        assert len(ids) == len(set(ids))
        assert not committed.intersection(ids)
        assert all(i.startswith(uid + "/") for i in ids)
        frame = ev.get_extra(burst.FRAME)
        if rng.random() < 0.25:
            frame.committed = frame.closed = True
            committed.update(ids)
        del ev
        if n % 30 == 0:
            gc.collect()
        assert len(room.bursts.entries) <= 32
        assert sum(f.size for f in room.bursts.entries.values()) <= 262144
