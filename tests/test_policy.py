import json
import tempfile
import unittest
from pathlib import Path

from ..policy import decide, pack_state
from ..state import Ledger, Room, RoomBook
from ..admission import local_prefilter


class PolicyTests(unittest.TestCase):
    def test_same_person_is_not_automatic_reply(self):
        self.assertFalse(
            decide(
                dict(addressed=0.2, continuation=0.2, worthwhile=0.3, intrusive=0.1),
                True,
                0,
            ).speak
        )

    def test_followup_need_not_be_question(self):
        verdict = decide(
            dict(addressed=0.5, continuation=0.9, worthwhile=0.7, intrusive=0.1),
            True,
            0,
        )
        self.assertEqual(verdict.mode, "continue")

    def test_intrusion_always_blocks(self):
        self.assertFalse(
            decide(
                dict(addressed=0.99, continuation=0.99, worthwhile=0.99, intrusive=0.8),
                True,
                0,
            ).speak
        )

    def test_more_initiations_raise_threshold(self):
        values = dict(addressed=0.1, continuation=0.1, worthwhile=0.82, intrusive=0.1)
        self.assertTrue(decide(values, False, 0).speak)
        self.assertFalse(decide(values, False, 2).speak)

    def test_context_budget_in_utf8_bytes(self):
        state = pack_state(
            "设" * 300,
            [dict(sender_name="甲", content="话" * 5000) for _ in range(96)]
            + [dict(sender_name="甲", content="话" * 100) for _ in range(4)],
            dict(content="说" * 600),
            {},
            max_bytes=8000,
        )
        self.assertLessEqual(len(json.dumps(state, ensure_ascii=False).encode()), 8000)
        self.assertTrue(state["truncated"])
        self.assertEqual(state["target_message"]["content"], "说" * 600)
        self.assertEqual(state["bot_persona"], "设" * 300)

    def test_protected_persona_is_not_silently_cut(self):
        with self.assertRaisesRegex(ValueError, "protected_context"):
            pack_state("设" * 9000, [], {"content": "原消息"}, {}, max_bytes=8000)

    def test_no_media_url_in_state(self):
        state = pack_state(
            "",
            [
                dict(
                    content="[图片]",
                    image_urls=["secret-url"],
                    image_local_paths=["private-path"],
                )
            ],
            {},
            {},
        )
        self.assertNotIn("secret-url", str(state))
        self.assertNotIn("private-path", str(state))

    def test_message_dedup_and_ticket_release(self):
        room = Room()
        self.assertTrue(room.observe({}, "1"))
        self.assertFalse(room.observe({}, "1"))
        room.pending = "new"
        room.release("old")
        self.assertEqual(room.pending, "new")

    def test_rooms_bounded(self):
        rooms = RoomBook(2)
        rooms.get("a")
        rooms.get("b")
        rooms.get("c")
        self.assertEqual(len(rooms.rooms), 2)

    def test_counters_survive_restart_without_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.sqlite3"
            ledger = Ledger(path)
            ledger.bump("room", "sent")
            ledger.record(
                "room", "ok", {"content": "private chat", "api_key": "SECRET"}
            )
            ledger.close()
            ledger = Ledger(path)
            self.assertEqual(ledger.count("room", "sent"), 1)
            self.assertNotIn("private chat", path.read_bytes().decode(errors="ignore"))
            self.assertNotIn("SECRET", path.read_bytes().decode(errors="ignore"))
            ledger.close()

    def test_daily_summary_migrates_and_retains_long_term_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.sqlite3"
            ledger = Ledger(path)
            ledger.record("room", "natural_silence", {"elapsed_ms": 100})
            ledger.record("room", "adapter_send_completed", {"elapsed_ms": 200})
            table = ledger.db.execute(
                "SELECT reason,n FROM daily_summary ORDER BY reason"
            ).fetchall()
            self.assertEqual(
                table, [("adapter_send_completed", 1), ("natural_silence", 1)]
            )
            ledger.close()

    def test_local_prefilter_keeps_explicit_wakes(self):
        class Part:
            def __init__(self, name, **attrs):
                self.__class__ = type(name, (), {})
                self.__dict__.update(attrs)

        class Event:
            def __init__(self, text, parts):
                self.text, self.parts = text, parts

            def get_messages(self):
                return self.parts

            def get_message_str(self):
                return self.text

            def get_self_id(self):
                return "bot"

            def get_extra(self, key):
                return None

        cfg = {"local_prefilter_enabled": True, "wake_names": ["吱吱"]}
        self.assertEqual(
            local_prefilter(Event("哈哈", [Part("Plain")]), cfg),
            "local_prefilter_low_information",
        )
        self.assertIsNone(local_prefilter(Event("吱吱 哈哈", [Part("Plain")]), cfg))
        self.assertEqual(
            local_prefilter(Event("🙂🙂", [Part("Plain")]), cfg),
            "local_prefilter_emoji",
        )
        self.assertEqual(
            local_prefilter(Event("话题", [Part("Plain")]), cfg, "话题", 90, 100),
            "local_prefilter_recent_topic",
        )
