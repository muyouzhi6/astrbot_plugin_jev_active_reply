"""Synthetic third-review and production-lifecycle regressions."""

import asyncio
import json

import pytest

from .. import bridge
from ..main import MARK
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import adapter_event, delivery_hooks, env, pipeline

__all__ = ["env", "delivery_hooks"]


def payload(bot):
    return json.dumps(
        [c.kwargs for c in bot.transport.await_args_list], ensure_ascii=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "texts,expected",
    [
        (["Answer", "[PASS]", "。"], ["Answer"]),
        (["Answer", "[PA", "SS]", "。"], ["Answer"]),
        (["Use the character ["], ["Use the character ["]),
        (["Use [PA"], ["Use [PA"]),
        (["["], ["["]),
        (["Use [", "x]"], ["Use [x]"]),
        (["Answer", "[PASS]", "Next", "。"], ["Answer", "Next。"]),
        (["Answer", "[PASS]", "func(", ")"], ["Answer", "func()"]),
    ],
)
async def test_output_fragments(env, native, texts, expected):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")

    async def emit(ev):
        for text in texts:
            if native:
                await ev.bot.send_group_msg(group_id=987654321, message=text)
            else:
                await ev.send(ev.plain_result(text))

    await pipeline(
        env, event, adapter=True, native_send=emit, decoration_priority=-200000
    )
    calls = bot.transport.await_args_list
    actual = [
        c.kwargs["message"]
        if isinstance(c.kwargs["message"], str)
        else "".join(p["data"]["text"] for p in c.kwargs["message"])
        for c in calls
    ]
    assert actual == expected
    assert event.get_extra(MARK)["finished"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["stop", "lease", "early_media_stop", "superseded"])
async def test_production_failure_lifecycle(env, monkeypatch, mode):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    entered, release = asyncio.Event(), asyncio.Event()
    if mode == "superseded":
        original = env.plugin.client.evaluate

        async def judge(*args):
            result = await original(*args)
            env.plugin.rooms.get(event.unified_msg_origin).generation += 1
            monkeypatch.setattr(bridge, "fingerprint", lambda conv: "new-history")
            return result

        env.plugin.client.evaluate = judge
    task = asyncio.create_task(
        pipeline(env, event, adapter=True, generation_gate=(entered, release))
    )
    await asyncio.wait_for(entered.wait(), 3)
    if mode == "early_media_stop":
        await event.bot.send_group_msg(
            group_id=987654321,
            message=[{"type": "image", "data": {"file": "base64://c3ludGhldGlj"}}],
        )
        event.stop_event()
    elif mode == "stop":
        event.stop_event()
    elif mode == "lease":
        env.plugin._expire(event)
        await asyncio.sleep(0.05)
    release.set()
    await task
    assert bot.transport.await_count == 1, payload(bot)
    assert event.get_extra(MARK)["finished"]
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending
