"""Round-15 OneBot string and explicit-wake delivery regressions."""

import json

import pytest

from .. import silence
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import adapter_event, delivery_hooks, env, pipeline

__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
@pytest.mark.parametrize("route", ["send_msg", "call_action", "send"])
@pytest.mark.parametrize("auto_escape", [None, False, True])
async def test_native_cq_string_keeps_escape_contract(env, kind, route, auto_escape):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    target = {"user_id": 123456789} if kind == "private" else {"group_id": 987654321}
    options = {} if auto_escape is None else {"auto_escape": auto_escape}
    body = "A &#91;literal&#93;&amp; [CQ:image,file=base64://c3ludGhldGlj] B"

    async def emit(ev):
        if route == "call_action":
            await ev.bot.call_action("send_msg", **target, message=body, **options)
        elif route == "send":
            await ev.bot.send(ev.message_obj.raw_message, body, **options)
        else:
            await ev.bot.send_msg(**target, message=body, **options)

    await pipeline(env, event, "[PASS]。", adapter=True, native_send=emit)
    bot.transport.assert_awaited_once()
    params = bot.transport.await_args.kwargs
    parts = params.get("messages", params.get("message"))
    assert parts
    assert "[PASS]" not in json.dumps(parts)
    if auto_escape:
        assert parts == [{"type": "text", "data": {"text": body}}]
    else:
        assert parts == [
            {"type": "text", "data": {"text": "A [literal]& "}},
            {"type": "image", "data": {"file": "base64://c3ludGhldGlj"}},
            {"type": "text", "data": {"text": " B"}},
        ]
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
async def test_third_party_cq_media_stop_owns_required_reply(env, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event(kind)
    target = {"user_id": 123456789} if kind == "private" else {"group_id": 987654321}

    async def emit(ev):
        await ev.bot.send_msg(
            **target, message="[CQ:record,file=base64://c3ludGhldGlj]"
        )
        ev.stop_event()

    await pipeline(env, event, "[不回复]", adapter=True, native_send=emit)
    bot.transport.assert_awaited_once()
    params = bot.transport.await_args.kwargs
    parts = params.get("messages", params.get("message"))
    assert parts == [{"type": "record", "data": {"file": "base64://c3ludGhldGlj"}}]
    assert silence.FALLBACK_REPLY not in json.dumps(parts, ensure_ascii=False)
    assert not env.plugin.rooms.get(event.unified_msg_origin).pending


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_escape", [False, True])
async def test_json_text_is_not_reparsed_as_cq(env, auto_escape):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    literal = "&#91;literal&#93; [CQ:image,file=literal]"
    parts = [{"type": "text", "data": {"text": literal}}]

    async def emit(ev):
        await ev.bot.send_msg(
            group_id=987654321, message=parts, auto_escape=auto_escape
        )

    await pipeline(env, event, adapter=True, native_send=emit)
    bot.transport.assert_awaited_once()
    assert bot.transport.await_args.kwargs["message"] == parts
    assert parts == [{"type": "text", "data": {"text": literal}}]


@pytest.mark.asyncio
async def test_forward_node_cq_and_json_text_keep_distinct_semantics(env):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    literal = "[CQ:image,file=literal]"
    parts = [
        {
            "type": "node",
            "data": {
                "user_id": "123",
                "nickname": "Synthetic",
                "content": "&#91;PASS&#93;[CQ:image,file=synthetic&#44;name&amp;value]",
            },
        },
        {
            "type": "node",
            "data": {
                "user_id": "123",
                "nickname": "Synthetic",
                "content": [{"type": "text", "data": {"text": literal}}],
            },
        },
    ]

    async def emit(ev):
        await ev.bot.send_group_forward_msg(group_id=987654321, messages=parts)

    await pipeline(env, event, adapter=True, native_send=emit)
    bot.transport.assert_awaited_once()
    sent = bot.transport.await_args.kwargs["messages"]
    image = {"type": "image", "data": {"file": "synthetic,name&value"}}
    assert image in sent[0]["data"]["content"]
    assert "[PASS]" not in json.dumps(sent)
    assert "&#91;PASS&#93;" not in json.dumps(sent)
    assert sent[1]["data"]["content"] == parts[1]["data"]["content"]
    assert isinstance(parts[0]["data"]["content"], str)
