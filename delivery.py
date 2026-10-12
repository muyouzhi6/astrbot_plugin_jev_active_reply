"""Event-scoped delivery control with a transparent Core completion adapter."""

import copy
from functools import wraps
from types import SimpleNamespace

from astrbot.api.message_components import Node, Nodes

from . import silence


async def plan_onebot(event, fragments):
    """Serialize once, retaining opaque native segments and their order."""
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
        AiocqhttpMessageEvent,
    )

    parts = []
    for native, value in fragments:
        if native:
            if isinstance(value, str):
                value = [{"type": "text", "data": {"text": value}}]
            if isinstance(value, dict):
                value = value.get("messages", [value])
            parts.extend(value)
        else:
            for component in value.chain:
                if isinstance(component, Nodes):
                    parts.extend((await component.to_dict())["messages"])
                elif isinstance(component, Node):
                    parts.append(await component.to_dict())
                else:
                    parts.extend(
                        await AiocqhttpMessageEvent._parse_onebot_json(
                            event.chain_result([component])
                        )
                    )
    root = {"message": parts}
    fields = list(silence._native_text_fields(root, "message"))
    silence.clean_delivery_fields(fields)
    parts = silence.repair_native_reply(parts, fallback=False)
    if parts is None:
        return None
    # Records cannot be mixed with text or other records in a normal QQ message.
    forward = any(p.get("type") == "node" for p in parts) or (
        len(parts) > 1 and any(p.get("type") in {"record", "file"} for p in parts)
    )
    if forward:
        parts = [
            p
            if p.get("type") == "node"
            else {
                "type": "node",
                "data": {
                    "user_id": str(event.get_self_id()),
                    "nickname": "AstrBot",
                    "content": [p],
                },
            }
            for p in parts
        ]
    group = bool(event.get_group_id())
    target = {
        "group_id" if group else "user_id": int(
            event.get_group_id() if group else event.get_sender_id()
        )
    }
    raw = getattr(event.message_obj, "raw_message", None)
    if isinstance(raw, dict) and raw.get("self_id"):
        target["self_id"] = raw["self_id"]
    action = (
        ("send_group_forward_msg" if group else "send_private_forward_msg")
        if forward
        else "send_msg"
    )
    return action, {**target, "messages" if forward else "message": parts}


def install_completion_adapter():
    """Core has no completion hook for empty/stopped results; scope by event."""
    from astrbot.core.pipeline.process_stage.stage import ProcessStage

    original = ProcessStage.process
    if getattr(original, "_jev_completion", False):
        return

    @wraps(original)
    async def process(stage, event):
        try:
            async for item in original(stage, event):
                yield item
        finally:
            finish = event.get_extra("_jev_complete_output")
            if finish is not None:
                await finish()

    process._jev_completion = True
    ProcessStage.process = process


class DeliveryDeclined(Exception):
    pass


class EventBotProxy:
    def __init__(self, bot, sender):
        self._bot, self._sender = bot, sender

    def __getattr__(self, name):
        original = getattr(self._bot, name)
        if name in {
            "send",
            "send_msg",
            "send_group_msg",
            "send_private_msg",
            "send_group_forward_msg",
            "send_private_forward_msg",
        }:

            async def guarded(*args, **kwargs):
                return await self._sender(original, *args, **kwargs)

            return guarded
        if name == "call_action":

            async def action(action, *args, **kwargs):
                if str(action).startswith("send_"):
                    return await self._sender(original, action, *args, **kwargs)
                return await original(action, *args, **kwargs)

            return action
        return original


class ScopedContext:
    def __init__(self, original, event):
        self._original, self._event = original, event

    def __getattr__(self, name):
        return getattr(self._original, name)

    async def send_message(self, umo, chain, **kwargs):
        if str(umo) != str(self._event.unified_msg_origin):
            raise DeliveryDeclined("autonomous_cross_session_send")
        return await self._event.send(chain)


class SplitAdapters:
    """Copy only a marked split invocation; other plugin invocations are unchanged."""

    def __init__(self, mark):
        self.mark = mark
        self.installed = []

    def attach(self, plugin):
        if plugin is None:
            return
        for step in getattr(getattr(plugin, "pipeline", None), "_steps", []):
            if type(step).__name__ != "SplitStep" or any(
                item[0] is step for item in self.installed
            ):
                continue
            original = step.handle

            async def scoped(ctx, _step=step, _original=original):
                if not ctx.event.get_extra(self.mark):
                    return await _original(ctx)
                clone = copy.copy(_step)
                clone.plugin_config = copy.copy(_step.plugin_config)
                clone.plugin_config.context = ScopedContext(
                    _step.plugin_config.context, ctx.event
                )
                clone.context = clone.plugin_config.context
                try:
                    return await type(_step).handle(clone, ctx)
                except DeliveryDeclined:
                    ctx.chain.clear()
                    return SimpleNamespace(
                        ok=True, abort=True, msg="Autonomous delivery cancelled"
                    )

            step.handle = scoped
            self.installed.append((step, original, scoped))

    def restore(self):
        for step, original, replacement in self.installed:
            if step.handle is replacement:
                step.handle = original
        self.installed.clear()
