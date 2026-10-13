"""Real AstrBot entities; isolated providers, no network and no QQ sends."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import At, Image, Plain
from astrbot.core.agent.tool import ToolSet
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from ..main import MARK, JevActiveReply


def event(mid="1", text="这个我也遇到了", sender="human", group="group", parts=None):
    message = AstrBotMessage()
    message.type = MessageType.GROUP_MESSAGE
    message.self_id = "bot"
    message.sender = MessageMember(sender, sender)
    message.message_id = mid
    message.message_str = text
    message.group_id = group
    message.message = parts if parts is not None else [Plain(text)]
    message.raw_message = {}
    return AstrMessageEvent(
        text, message, PlatformMetadata("aiocqhttp", "test", "test"), group
    )


async def collect_request(event, context, config):
    """Exercise the native collector on both supported Core versions."""
    from astrbot.core import astr_main_agent

    if hasattr(astr_main_agent, "collect_initial_request"):
        return await astr_main_agent.collect_initial_request(event, context, config)

    class Collected(Exception):
        pass

    async def capture(event, request, *args, **kwargs):
        raise Collected(request)

    with patch.object(astr_main_agent, "_decorate_llm_request", capture):
        try:
            await astr_main_agent.build_main_agent(
                event=event, plugin_context=context, config=config, provider=object()
            )
        except Collected as result:
            return result.args[0], None
    raise AssertionError("Native collector did not produce a request")


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cid = "conversation-a"
        self.history = "[]"
        self.stars = []
        self.settings = {
            "provider_settings": {
                "enable": True,
                "streaming_response": False,
                "show_tool_use_status": False,
            },
            "agent_runner": {"runner_type": "local"},
        }

        async def current(umo):
            return self.cid

        async def get(umo, cid):
            return NS(cid=cid, history=self.history, persona_id="persona-a")

        self.ctx = NS(
            conversation_manager=NS(
                get_curr_conversation_id=current, get_conversation=get
            ),
            persona_manager=NS(
                resolve_selected_persona=AsyncMock(
                    return_value=(
                        "persona-a",
                        {"prompt": "安静但喜欢一起聊游戏的朋友"},
                        None,
                        False,
                    )
                )
            ),
            get_config=lambda **kw: self.settings,
            get_all_stars=lambda: self.stars,
        )
        self.cfg = {
            "enabled_sessions": ["*"],
            "dry_run": False,
            "debounce_seconds": 0,
            "evaluation_interval": 4,
            "strict_conflicts": True,
            "second_review": "off",
            "image_decision_mode": "native_vision",
            "burst_merge_enabled": False,
            "nanbei": {"keys": ["test-secret"]},
        }
        with patch(
            "astrbot_plugin_jev_active_reply.main.StarTools.get_data_dir",
            return_value=Path(self.tmp.name),
        ):
            self.plugin = JevActiveReply(self.ctx, self.cfg)
        self.cfg = self.plugin.config
        self.response = {
            "values": dict(
                addressed=0.85, continuation=0.7, worthwhile=0.9, intrusive=0.1
            ),
            "channel": "nanbei",
            "model": "jev-1.13.0",
            "key_id": "masked",
        }
        self.plugin.client.evaluate = AsyncMock(return_value=self.response)

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.tmp.cleanup()

    async def admit(self, ev=None):
        ev = ev or event()
        await self.plugin.on_message(ev)
        self.assertIsNotNone(ev.get_extra(MARK))
        return ev

    async def test_native_trigger_preserves_entire_event(self):
        image = Image.fromURL("https://example.invalid/image.png")
        ev = event(parts=[Plain("看看这个"), image])
        before = list(ev.get_messages())
        await self.admit(ev)
        self.assertTrue(ev.is_at_or_wake_command)
        self.assertEqual(ev.get_messages(), before)
        self.assertIsNone(ev.get_extra("provider_request"))
        self.assertTrue(ev.get_extra("_active_trigger"))
        self.assertFalse(ev.is_stopped())

    async def test_explicit_request_not_gated(self):
        ev = event()
        ev.is_at_or_wake_command = True
        await self.plugin.on_message(ev)
        self.plugin.client.evaluate.assert_not_called()
        self.assertFalse(ev.is_stopped())

    async def test_other_person_at_skipped(self):
        await self.plugin.on_message(event(parts=[At(qq="other"), Plain("问题")]))
        self.plugin.client.evaluate.assert_not_called()

    async def test_self_and_unapproved_skipped(self):
        await self.plugin.on_message(event(sender="bot"))
        self.cfg["scope_groups"] = []
        await self.plugin.on_message(event(mid="2"))
        self.plugin.client.evaluate.assert_not_called()

    async def test_stopped_event_never_revived(self):
        ev = event()
        ev.stop_event()
        await self.plugin.on_message(ev)
        self.plugin.client.evaluate.assert_not_called()
        self.assertTrue(ev.is_stopped())

    async def test_dry_run_does_not_trigger_or_stop(self):
        self.cfg["dry_run"] = True
        ev = event()
        await self.plugin.on_message(ev)
        self.plugin.client.evaluate.assert_awaited_once()
        self.assertFalse(ev.is_at_or_wake_command)
        self.assertFalse(ev.is_stopped())

    async def test_old_waker_conflict_blocks_only_own_trigger(self):
        self.stars.append(
            NS(
                name="astrbot_plugin_wakepro",
                activated=True,
                star_cls=NS(cfg=NS(pipeline=NS(steps=["wake(智能唤醒)"]))),
            )
        )
        ev = event()
        await self.plugin.on_message(ev)
        self.plugin.client.evaluate.assert_not_called()
        self.assertFalse(ev.is_stopped())

    async def test_contextaware_public_api_and_life_cache(self):
        recent = NS(
            get_recent_messages=lambda umo, count: [
                dict(content="我们上一轮聊的截图", sender_name="甲", is_bot=False)
            ]
        )
        life = NS(get_life_context=AsyncMock(return_value={"schedule": "休息"}))
        self.stars.extend(
            [
                NS(name="astrbot_plugin_context_aware", star_cls=recent),
                NS(name="astrbot_plugin_life_scheduler", star_cls=life),
            ]
        )
        await self.admit()
        state = self.plugin.client.evaluate.await_args.args[0]
        self.assertIn("上一轮", str(state))
        life.get_life_context.assert_awaited_once_with(allow_generate=False)
        self.ctx.persona_manager.resolve_selected_persona.assert_awaited_once()

    async def test_tools_copied_not_globally_mutated(self):
        ev = await self.admit()
        tools = ToolSet()
        tools.tools = [
            NS(name="keep_silent"),
            NS(name="context_aware_view_images"),
            NS(name="ban_group_user"),
        ]
        req = ProviderRequest(
            prompt="原文",
            conversation=NS(cid=self.cid),
            func_tool=tools,
            contexts=[dict(role="user", content="历史")],
            system_prompt="人格+世界书",
        )
        await self.plugin.on_request(ev, req)
        await self.plugin.on_request(ev, req)
        self.assertEqual(len(req.extra_user_content_parts), 2)
        self.assertTrue(req.extra_user_content_parts[0]._no_save)
        self.assertEqual(len(tools.tools), 3)
        self.assertEqual(len(req.func_tool.tools), 2)
        self.assertEqual(req.system_prompt, "人格+世界书")
        self.assertEqual(req.contexts[0]["content"], "历史")

    async def test_pass_removed_before_history_hooks(self):
        ev = await self.admit()
        response = LLMResponse(role="assistant", completion_text="[PASS]")
        await self.plugin.on_response(ev, response)
        self.assertEqual(response.completion_text, "")
        self.assertTrue(ev.is_stopped())
        self.assertEqual(self.plugin.ledger.count(ev.unified_msg_origin, "sent"), 0)
        self.assertFalse(self.plugin.rooms.get(ev.unified_msg_origin).pending)

    async def test_keep_silent_and_recall_win(self):
        for index, marker in enumerate(
            ("_suanle_silent_requested", "agent_stop_requested")
        ):
            ev = await self.admit(event(mid=str(index), group=str(index)))
            ev.set_extra(marker, True)
            response = LLMResponse(role="assistant", completion_text="late draft")
            await self.plugin.on_response(ev, response)
            self.assertEqual(response.completion_text, "")
            self.assertTrue(ev.get_extra(MARK)["rejected"])

    async def test_response_checks_before_output_patch(self):
        self.cfg["second_review"] = "always"
        ev = await self.admit()
        self.plugin.client.evaluate.return_value = {
            "values": {"appropriate": 0.1, "redundant": 0.1}
        }
        response = LLMResponse(role="assistant", completion_text="不合适的草稿")
        await self.plugin.on_response(ev, response)
        ev.set_result(ev.plain_result("should never send"))
        await self.plugin.before_output(ev)
        self.assertFalse(ev.get_result().chain)
        self.assertEqual(response.completion_text, "")

    async def test_timeout_review_fails_closed(self):
        self.cfg["second_review"] = "always"
        ev = await self.admit()
        self.plugin.client.evaluate.side_effect = TimeoutError
        response = LLMResponse(role="assistant", completion_text="草稿")
        await self.plugin.on_response(ev, response)
        self.assertTrue(ev.is_stopped())

    async def test_new_conversation_cancels_old_request(self):
        ev = await self.admit()
        self.cid = "conversation-b"
        response = LLMResponse(role="assistant", completion_text="旧会话的话")
        await self.plugin.on_response(ev, response)
        self.assertEqual(response.completion_text, "")

    async def test_success_commits_once_and_unknown_does_not(self):
        ev = await self.admit()
        response = LLMResponse(role="assistant", completion_text="确实挺有意思")
        await self.plugin.on_response(ev, response)
        ev.set_result(ev.plain_result(response.completion_text))
        await self.plugin.before_output(ev)
        await self.plugin.after_sent(ev)
        self.assertEqual(self.plugin.ledger.count(ev.unified_msg_origin, "sent"), 0)
        ev._has_send_oper = True
        await self.plugin.after_sent(ev)
        await self.plugin.after_sent(ev)
        self.assertEqual(self.plugin.ledger.count(ev.unified_msg_origin, "sent"), 1)

    async def test_direct_reply_seeds_continuation(self):
        ev = event()
        ev.is_at_or_wake_command = True
        await self.plugin.on_message(ev)
        await self.plugin.on_response(
            ev, LLMResponse(role="assistant", completion_text="刚才的回复")
        )
        ev._has_send_oper = True
        await self.plugin.after_sent(ev)
        room = self.plugin.rooms.get(ev.unified_msg_origin)
        self.assertEqual(room.cid, self.cid)
        self.assertEqual(room.last_text, "刚才的回复")
        self.assertGreater(room.last_sent, 0)

    async def test_new_message_during_decision_discards_result(self):
        ev = event()

        async def evaluate(*args):
            room = self.plugin.rooms.get(ev.unified_msg_origin)
            room.observe({"content": "别人已回答"}, "newer")
            return self.response

        self.plugin.client.evaluate.side_effect = evaluate
        await self.plugin.on_message(ev)
        self.assertFalse(ev.is_at_or_wake_command)
        self.assertIsNone(ev.get_extra(MARK))

    async def test_unapproved_plugin_is_not_read(self):
        ev = event()
        ev.plugins_name = ["astrbot_plugin_jev_active_reply"]
        plugin = NS(get_recent_messages=lambda *a, **kw: self.fail("unauthorized"))
        self.stars.append(NS(name="astrbot_plugin_context_aware", star_cls=plugin))
        await self.admit(ev)

    async def test_native_process_stage_runs_once(self):
        from astrbot.core.pipeline.process_stage.stage import ProcessStage

        ev = event()
        ev.set_extra("activated_handlers", [object()])

        async def handlers(e):
            await self.plugin.on_message(e)
            yield None

        calls = []

        async def agent(e):
            calls.append(e)
            yield None

        stage = ProcessStage()
        stage.ctx = NS(astrbot_config=self.settings)
        stage.star_request_sub_stage = NS(process=handlers)
        stage.agent_sub_stage = NS(process=agent)
        async for _ in stage.process(ev):
            pass
        self.assertEqual(calls, [ev])

    async def test_config_schema_parses_with_current_core(self):
        from astrbot.core.config.astrbot_config import AstrBotConfig

        schema = json.loads(
            (Path(__file__).parents[1] / "_conf_schema.json").read_text(
                encoding="utf-8"
            )
        )
        cfg = AstrBotConfig(str(Path(self.tmp.name) / "config.json"), schema=schema)
        self.assertEqual(cfg["nanbei"]["keys"], [])
        self.assertEqual(cfg["typesafe"]["keys"], [])

    async def test_voice_cache_only_not_reawakened(self):
        ev = event()
        ev.set_extra("_gemini_stt_transcript", "语音内容")
        ev.set_extra("_gemini_stt_cache_only", True)
        await self.plugin.on_message(ev)
        self.plugin.client.evaluate.assert_not_called()

    async def test_image_recall_signal_not_sent_as_image_to_jev(self):
        self.cfg["second_review"] = "always"
        ev = await self.admit()
        await self.plugin.on_tool_result(
            ev,
            NS(name="context_aware_view_images"),
            {},
            NS(content=[NS(type="image", data="SECRET_BASE64")]),
        )
        self.plugin.client.evaluate.return_value = {
            "values": {"appropriate": 0.9, "redundant": 0.1}
        }
        await self.plugin.on_response(
            ev, LLMResponse(role="assistant", completion_text="图中有个按钮")
        )
        state = self.plugin.client.evaluate.await_args.args[0]
        self.assertTrue(state["perception"]["main_model_received_images"])
        self.assertNotIn("SECRET_BASE64", str(state))

    async def test_repeated_message_only_evaluated_once(self):
        ev = event()
        self.cfg["dry_run"] = True
        await self.plugin.on_message(ev)
        await self.plugin.on_message(event())
        self.plugin.client.evaluate.assert_awaited_once()

    async def test_native_collector_keeps_image_and_conversation(self):
        from astrbot.core.astr_main_agent import (
            MainAgentBuildConfig,
        )

        image = Image.fromURL("https://example.invalid/image.png")
        ev = await self.admit(event(parts=[Plain("看这个"), image]))
        with patch.object(
            Image, "convert_to_file_path", AsyncMock(return_value="/test/unchanged.png")
        ):
            request, _ = await collect_request(
                ev,
                self.ctx,
                MainAgentBuildConfig(tool_call_timeout=30, provider_settings={}),
            )
        self.assertEqual(request.image_urls, ["/test/unchanged.png"])
        self.assertEqual(request.conversation.cid, self.cid)
        self.assertEqual(request.contexts, [])
