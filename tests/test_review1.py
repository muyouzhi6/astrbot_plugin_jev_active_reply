"""Regression cases from release review, using real ingress and adapter conversion."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock
import uuid

import pytest
from astrbot.api.message_components import At, Image, Reply
from astrbot.core.provider.entities import LLMResponse

from ..admission import local_prefilter
from ..main import MARK
from . import test_integration as base
from . import test_product
from . import test_v020 as previous

env = previous.env


@pytest.mark.parametrize(
    "field", ["token", "password", "Cookie", "api_key", "Authorization"]
)
@pytest.mark.parametrize(
    "shape",
    ["json", "nested", "quoted_spaces", "single_quotes", "multiline", "incomplete"],
)
@pytest.mark.asyncio
async def test_json_credentials_not_persisted(env, field, shape):
    env.cfg.update(
        decision_mode="natural", natural_reply_threshold=0.5, style="balanced"
    )
    env.plugin.client.evaluate = AsyncMock(return_value={"values": {"reply_now": 0.5}})
    marker = uuid.uuid4().hex[:20]
    texts = {
        "json": json.dumps({field: marker}),
        "nested": json.dumps({field: {"value": marker}}, indent=2),
        "quoted_spaces": json.dumps({field: "first part " + marker}),
        "single_quotes": "{'" + field + "': 'first part " + marker + "'}",
        "multiline": '{"' + field + '"\n:\n"first\n' + marker + '"}',
        "incomplete": '{"' + field + '": "' + marker,
    }
    await env.plugin.on_message(base.event(text=texts[shape]))
    await env.plugin.audit_task
    rows = env.plugin.ledger.db.execute("SELECT details FROM audit").fetchall()
    previews = [
        json.loads(row[0])["trigger_preview"]
        for row in rows
        if "trigger_preview" in json.loads(row[0])
    ]
    assert previews and all("[redacted]" in preview for preview in previews)
    protected = all(marker[:8] not in row[0] for row in rows)
    assert protected, "JSON credential value persisted in audit details"


@pytest.mark.parametrize("merge", [True, False])
@pytest.mark.parametrize(
    "text", ["改成3000元", "改为3000元", "换成3000元", "说错了，3000元"]
)
@pytest.mark.asyncio
async def test_numeric_correction_revalidates_approved_draft(env, merge, text):
    env.cfg.update(
        second_review="adaptive", burst_merge_enabled=merge, candidate_wait_seconds=1
    )
    first = base.event(text="装机预算5000元")
    send = AsyncMock()
    first.send = send
    await asyncio.create_task(env.admit(first))
    await env.plugin.on_response(
        first, LLMResponse(role="assistant", completion_text="按5000元给你推荐配置")
    )
    calls = env.plugin.client.evaluate.await_count
    room = env.plugin.rooms.get(first.unified_msg_origin)
    second = base.event(mid="correction", text=text)
    task = asyncio.create_task(env.plugin.on_message(second))
    try:
        for _ in range(50):
            if room.generation > first.get_extra(MARK)["generation"]:
                break
            await asyncio.sleep(0.001)
        first.set_result(first.plain_result("按5000元给你推荐配置"))
        await env.plugin.before_output(first)
        from ..delivery import DeliveryDeclined

        try:
            await first.send(first.get_result())
        except DeliveryDeclined:
            pass
        guarded = (
            send.await_count == 0 or env.plugin.client.evaluate.await_count > calls
        )
        assert guarded, (
            "Old-budget draft actually reached mock adapter after same-user correction without revalidation"
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_same_user_requested_image_attaches_to_text(env, tmp_path, monkeypatch):
    env.cfg["burst_merge_enabled"] = True
    first = await env.admit(base.event(text="帮忙看看下面这张截图里的报错"))
    path = tmp_path / "error.png"
    path.write_bytes(b"attachment")
    second = base.event(mid="image", text="", parts=[Image.fromFileSystem(path)])
    room = env.plugin.rooms.get(first.unified_msg_origin)
    env.plugin._coalesce(second, room, env.cid)
    merged = bool(second.get_extra("_jev_merged_message_ids"))
    assert merged, "Requested screenshot sent separately does not join pending text"
    assert first.get_extra(MARK)["rejected"]
    from astrbot.core import astr_main_agent
    from types import SimpleNamespace

    monkeypatch.setattr(
        astr_main_agent,
        "_get_session_conv",
        AsyncMock(return_value=SimpleNamespace(cid=env.cid, history="[]")),
    )
    request, _ = await base.collect_request(
        second,
        env.ctx,
        astr_main_agent.MainAgentBuildConfig(
            tool_call_timeout=30, provider_settings={}
        ),
    )
    assert "截图里的报错" in request.prompt
    assert request.image_urls == [str(path)]


@pytest.mark.parametrize(
    "subtypes,caption,wake,expected",
    [
        ([1], "", False, "local_prefilter_sticker"),
        (["1"], "", False, "local_prefilter_sticker"),
        ([0], "", False, None),
        ([1, 0], "", False, None),
        ([0, 1], "", False, None),
        ([1, 1], "", False, "local_prefilter_sticker"),
        ([1], "这个贴纸出自哪部动画", False, None),
        ([1], "", True, None),
    ],
)
@pytest.mark.asyncio
async def test_real_onebot_sticker_filtered(subtypes, caption, wake, expected):
    from aiocqhttp import Event
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_platform_adapter import (
        AiocqhttpAdapter,
    )

    adapter = object.__new__(AiocqhttpAdapter)
    incoming = Event(
        {
            "time": 1,
            "self_id": 12345,
            "post_type": "message",
            "message_type": "group",
            "sub_type": "normal",
            "group_id": 22222,
            "user_id": 33333,
            "message_id": 1,
            "sender": {"user_id": 33333, "nickname": "tester"},
            "message": [
                {
                    "type": "image",
                    "data": {
                        "file": "https://example.invalid/sticker.png",
                        "url": "https://example.invalid/sticker.png",
                        "sub_type": subtype,
                    },
                }
                for subtype in subtypes
            ]
            + ([{"type": "text", "data": {"text": caption}}] if caption else []),
            "raw_message": "",
            "font": 0,
        }
    )
    converted = await adapter._convert_handle_message_event(incoming)
    ev = base.event(text=converted.message_str, parts=converted.message)
    ev.message_obj.raw_message = converted.raw_message
    if wake:
        ev.message_obj.message.insert(0, At(qq="bot"))
    assert local_prefilter(ev, {}) == expected
    assert local_prefilter(ev, {"local_prefilter_enabled": False}) is None


def test_product_patch_test_reads_only_checkout(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    original = Path.read_text

    def checkout_only(path, *args, **kwargs):
        assert path.resolve().is_relative_to(root), (
            "Product test requires an external task fixture"
        )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", checkout_only)
    test_product.test_patch_removes_only_legacy_duplicates_and_preserves_effective_values()


@pytest.mark.parametrize(
    "text", ["改成3000元", "改为3000元", "换成3000元", "说错了，3000元"]
)
def test_same_user_correction_is_related(text):
    from ..state import Room

    room = Room()
    room.observe({"sender_id": "human", "content": text, "reply_to": ""}, "2")
    meta = {
        "generation": 0,
        "state": {
            "target_message": {"sender_id": "human", "content": "装机预算5000元"}
        },
    }
    assert base.JevActiveReply._related_generation(room, meta)
    meta["state"]["target_message"]["sender_id"] = "someone_else"
    assert not base.JevActiveReply._related_generation(room, meta)


@pytest.mark.asyncio
async def test_blank_image_reply_to_pending_message_merges(env):
    env.cfg["burst_merge_enabled"] = True
    first = await env.admit(base.event(text="帮我看看这个错误"))
    image = base.event(
        mid="reply-image",
        text="",
        parts=[
            Reply(id="1", sender_id="human"),
            Image.fromURL("https://example.invalid/a.png"),
        ],
    )
    env.plugin._coalesce(image, env.plugin.rooms.get(first.unified_msg_origin), env.cid)
    assert image.get_extra("_jev_merged_message_ids") == ["1", "reply-image"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["other_sender", "expired", "closed"])
async def test_requested_image_respects_burst_ownership_and_lifetime(env, case):
    from .. import burst

    env.cfg["burst_merge_enabled"] = True
    first = await env.admit(base.event(text="帮忙看看下面这张截图里的报错"))
    frame = first.get_extra(burst.FRAME)
    if case == "expired":
        frame.last -= 60
    if case == "closed":
        frame.closed = True
    second = base.event(
        mid="image",
        sender="other" if case == "other_sender" else "human",
        text="",
        parts=[Image.fromURL("https://example.invalid/a.png")],
    )
    env.plugin._coalesce(
        second, env.plugin.rooms.get(first.unified_msg_origin), env.cid
    )
    assert not second.get_extra("_jev_merged_message_ids")
