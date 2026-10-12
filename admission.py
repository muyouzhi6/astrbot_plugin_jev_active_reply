"""Standalone input rules; no imports, state or callbacks from WakePro."""

import re


def normalize(text):
    return re.sub(r"[^\w\u4e00-\u9fff]", "", str(text).lower())


def topic_terms(text):
    text = str(text or "")[:4096].lower()
    terms = set(re.findall(r"[a-z][a-z0-9_]+", text))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(run[i : i + 2] for i in range(len(run) - 1))
    return terms - {
        "这个",
        "那个",
        "我们",
        "你们",
        "今天",
        "现在",
        "可以",
        "一下",
        "怎么",
        "什么",
        "问题",
        "时候",
        "还是",
        "就是",
        "the",
        "and",
        "this",
        "that",
        "with",
        "have",
        "what",
        "when",
        "you",
        "are",
        "for",
        "can",
        "how",
        "today",
    }


def related_topic(left, right):
    a, b = topic_terms(left), topic_terms(right)
    return bool(a & b)


def correction(text):
    return str(text).lstrip().startswith(("改成", "改为", "换成", "说错了"))


def awaiting_image(text):
    return bool(
        re.search(
            r"(?:看|检查|分析|识别|读|下面|这张|附|发).{0,24}(?:截图|图片|照片|图)"
            r"|(?:截图|图片|照片).{0,24}(?:如下|在下面|发给你)",
            str(text),
        )
    )


def _sticker_image_ids(event, parts):
    images = [p for p in parts if type(p).__name__ == "Image"]
    marked = {id(p) for p in images if str(getattr(p, "sub_type", "0")) == "1"}
    if not images:
        return marked
    # AstrBot's OneBot Image model drops sub_type; the original event retains it.
    raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
    segments = raw.get("message") if isinstance(raw, dict) else None
    if not isinstance(segments, list):
        return marked
    originals = [
        p.get("data", {})
        for p in segments
        if isinstance(p, dict) and p.get("type") == "image"
    ]
    # A different image count means the original segments no longer align.
    if len(originals) == len(images):
        marked.update(
            id(image)
            for image, data in zip(images, originals)
            if isinstance(data, dict) and str(data.get("sub_type", "0")) == "1"
        )
    return marked


def explicit_wake(event, config):
    if getattr(event, "is_at_or_wake_command", False) or event.get_extra("_jev_takeover"):
        return True
    parts = event.get_messages()
    bot = str(event.get_self_id())
    if any(
        type(p).__name__ == "At" and str(getattr(p, "qq", "")) == bot for p in parts
    ):
        return True
    if any(
        type(p).__name__ == "Reply" and str(getattr(p, "sender_id", "")) == bot
        for p in parts
    ):
        return True
    plain = str(event.get_message_str() or "")
    names = [
        str(n).strip()
        for n in [*config.get("wake_names", []), *config.get("admin_wake_names", [])]
        if str(n).strip()
    ]
    return any(name in plain for name in names)


def local_prefilter(event, config, recent_text="", recent_at=0.0, now=0.0):
    """Filter content-free interjections and repeated declined topics, not photos."""
    if not config.get("local_prefilter_enabled", True) or explicit_wake(event, config):
        return None
    parts = event.get_messages()
    # Platform image sub_type=1 is a sticker; ordinary images still use text_gate.
    stickers = {"Face", "Emoji", "Sticker", "MFace"}
    sticker_images = _sticker_image_ids(event, parts)
    substantive = any(
        type(p).__name__ in ("Record", "Video", "File")
        or (type(p).__name__ == "Image" and id(p) not in sticker_images)
        for p in parts
    )
    if substantive:
        return None
    plain = str(
        event.get_extra("_gemini_stt_transcript") or event.get_message_str() or ""
    ).strip()
    text = re.sub(r"[\s，。！!？?,.]+", "", plain.lower())
    if (
        plain
        and len(plain) <= 16
        and (not text or re.fullmatch(r"(?:哈|呵|嘿|嗯|哦|额|啊|6)+|ok|好的", text))
    ):
        return "local_prefilter_low_information"
    if any(
        type(p).__name__ in stickers or id(p) in sticker_images for p in parts
    ) and not any(
        type(p).__name__ == "Plain" and str(getattr(p, "text", "")).strip()
        for p in parts
    ):
        return "local_prefilter_sticker"
    if plain and all(
        c.isspace()
        or c in "\u200d\ufe0f\u20e3©®™"
        or ("\u20e3" in plain and c in "0123456789#*")
        or 0x1F000 <= ord(c) <= 0x1FAFF
        or 0x2600 <= ord(c) <= 0x27BF
        for c in plain
    ):
        return "local_prefilter_emoji"
    cooldown = max(0.0, min(300.0, float(config.get("prefilter_cooldown_seconds", 20))))
    if len(plain) > 4096:
        return None
    a, b = topic_terms(plain), topic_terms(recent_text)
    similar = bool(a and b and len(a & b) / max(len(a), len(b)) >= 0.8)
    if (
        recent_text
        and 0 <= now - recent_at < cooldown
        and (normalize(plain) == normalize(recent_text) or similar)
    ):
        return "local_prefilter_recent_topic"
    return None


def guard_exempt(event, config):
    targets = {
        str(event.get_sender_id()),
        str(event.get_group_id()),
        event.unified_msg_origin,
    }
    return bool(
        targets.intersection(str(x) for x in config.get("guard_exempt_targets", []))
    )


def input_decision(event, config, recent_bot_text="", prefixes=()):
    sender, bot = str(event.get_sender_id()), str(event.get_self_id())
    targets = {sender, str(event.get_group_id()), event.unified_msg_origin}
    if sender == bot:
        return "skip", "self_message"
    if targets.intersection(str(x) for x in config.get("blocked_targets", [])):
        return "block", "blocked_target"
    exempt = guard_exempt(event, config)
    parts = event.get_messages()
    plain = str(
        event.get_extra("_gemini_stt_transcript")
        or "".join(
            str(getattr(p, "text", "")) for p in parts if type(p).__name__ == "Plain"
        )
        or event.get_message_str()
        or ""
    ).strip()
    first = plain.split(maxsplit=1)[0] if plain else ""
    command = first.lstrip("/!！")
    if command in {
        "jev状态",
        "jev诊断",
        "jev恢复",
        "jev静音",
        "jev开口",
        "jev拉黑",
        "jev取消拉黑",
        "jev黑名单",
        "拉黑",
        "取消拉黑",
        "黑名单",
    }:
        return "pass", "management_command"
    if not exempt:
        if sender in {str(x) for x in config.get("other_bot_ids", [])}:
            return "block", "known_bot"
        if config.get("block_official_qq_bots", True) and sender.isdigit():
            uid = int(sender)
            if any(
                low <= uid <= high
                for low, high in (
                    (3328144510, 3328144510),
                    (2854196301, 2854216399),
                    (66600000, 66600000),
                    (3889000000, 3889999999),
                    (4010000000, 4019999999),
                )
            ):
                return "block", "official_bot"
        if any(
            word and str(word) in plain for word in config.get("blocked_keywords", [])
        ):
            return "block", "blocked_keyword"
        if (
            config.get("block_repeated_input", True)
            and normalize(plain)
            and recent_bot_text
            and normalize(plain) == normalize(recent_bot_text)
        ):
            return "block", "repeated_bot_message"
        if config.get("block_builtin_commands", False) and command in config.get(
            "blocked_commands", []
        ):
            return "block", "blocked_command"
        if any(prefix and plain.startswith(prefix) for prefix in prefixes):
            registered = bool(event.get_extra("handlers_parsed_params"))
            if config.get(
                "block_prefixed_commands" if registered else "block_prefixed_llm", False
            ):
                return "block", "blocked_prefix"
    at_self = any(type(p).__name__ == "At" and str(p.qq) == bot for p in parts)
    reply_self = any(
        type(p).__name__ == "Reply" and str(getattr(p, "sender_id", "")) == bot
        for p in parts
    )
    directed_other = any(
        type(p).__name__ == "AtAll" or (type(p).__name__ == "At" and str(p.qq) != bot)
        for p in parts
    )
    reply_other = any(
        type(p).__name__ == "Reply" and str(getattr(p, "sender_id", "")) != bot
        for p in parts
    )
    if not (at_self or reply_self) and directed_other:
        return "skip", "addressed_elsewhere"
    if (
        not (at_self or reply_self)
        and reply_other
        and config.get("skip_reply_to_others", False)
    ):
        return "skip", "quoted_elsewhere"
    aliases = [str(n).strip() for n in config.get("wake_names", []) if str(n).strip()]
    if event.is_admin():
        aliases += [
            str(n).strip() for n in config.get("admin_wake_names", []) if str(n).strip()
        ]
    named = False
    remainder = plain
    for alias in sorted(set(aliases), key=len, reverse=True):
        match = re.match(
            r"^"
            + re.escape(alias)
            + r"(?=$|[\s,，:：!?！？。~～]|在|帮|你|能|可以|我|给|怎么|别|不要|闭嘴|停)",
            plain,
            re.IGNORECASE,
        )
        if match:
            named = True
            remainder = plain[match.end() :].strip(" ,，:：!！?？。~～")
            break
    explicit = at_self or reply_self or named
    if explicit and config.get("respect_quiet_requests", True):
        stop = r"(?:闭嘴|别说话|不要说话|别回复|不要回复|别吵|别插嘴|别打扰我|先安静|安静一会|停一下)(?:了|吧|好吗|谢谢)?[。.!！?？\s]*"
        resume = r"(?:可以说话了|继续聊|出来吧|回来吧|解除沉默)[。.!！?？\s]*"
        if re.fullmatch(stop, remainder, re.IGNORECASE):
            return "quiet", "explicit_quiet_request"
        if re.fullmatch(resume, remainder, re.IGNORECASE):
            return "resume", "explicit_resume_request"
    if at_self:
        return "direct", "explicit_at"
    if reply_self and config.get("reply_to_bot_wakes", True):
        return "direct", "reply_to_bot"
    if named:
        return "direct", "named_wake"
    if reply_self and not config.get("reply_to_bot_wakes", True):
        return "skip", "reply_wake_disabled"
    return "pass", "not_explicitly_addressed"
