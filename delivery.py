"""Event-scoped delivery control with a transparent Core completion adapter."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from types import SimpleNamespace

from aiocqhttp.message import Message
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
    from astrbot.core.pipeline.respond.stage import RespondStage
    from astrbot.core.pipeline.scheduler import PipelineScheduler
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
        InternalAgentSubStage,
    )

    original = ProcessStage.process
    if getattr(original, "_jev_completion", False):
        return

    @wraps(original)
    async def process(stage, event):
        try:
            async for item in original(stage, event):
                yield item
        except BaseException:
            failed = event.get_extra("_jev_generation_failed")
            if failed is not None:
                failed()
            raise
        finally:
            finish = event.get_extra("_jev_complete_output")
            if finish is not None:
                await finish()

    process._jev_completion = True
    ProcessStage.process = process

    original_error = InternalAgentSubStage._send_llm_error_message

    @wraps(original_error)
    async def error(stage, event, message):
        failed = event.get_extra("_jev_generation_failed")
        if failed is not None:
            failed()
            return
        return await original_error(stage, event, message)

    InternalAgentSubStage._send_llm_error_message = error

    original_stages = PipelineScheduler._process_stages

    @wraps(original_stages)
    async def stages(scheduler, event, from_stage=0):
        try:
            return await original_stages(scheduler, event, from_stage)
        finally:
            if from_stage == 0:
                finish = event.get_extra("_jev_complete_output")
                if finish is not None:
                    await finish()

    PipelineScheduler._process_stages = stages
    original_respond = RespondStage.process

    @wraps(original_respond)
    async def respond(stage, event):
        if event.get_extra("_jev_complete_output") is not None:
            # A required turn submits once; Core's per-part delays cannot apply.
            stage = copy.copy(stage)
            stage.enable_seg = False
        return await original_respond(stage, event)

    RespondStage.process = respond


def install_native_reply_guard(event):
    """Clean a native wake's final sends without taking over its generation."""
    if event.get_extra("_jev_native_reply_guard") is not None:
        return
    original_send = event.send
    sent = False

    async def send(chain, *args, **kwargs):
        nonlocal sent
        texts = [part.text for part in silence._plain_components(chain.chain)]
        if silence.MARKER.search("".join(texts)):
            chain = copy.deepcopy(chain)
            silence.repair_required_reply(result=chain, fallback=not sent)
            if not silence.has_reply_content(chain.chain):
                return None
        result = await original_send(chain, *args, **kwargs)
        sent = True
        return result

    event.send = send
    event.set_extra("_jev_native_reply_guard", lambda: sent)


class DeliveryDeclined(Exception):
    pass


def is_message_action(action):
    # OneBot async/rate-limited variants still send messages, unlike send_like/poke.
    name = str(action)
    for suffix in ("_async", "_rate_limited"):
        name = name.removesuffix(suffix)
    return name in {
        "send",
        "send_group_ai_record",
        "forward_group_single_msg",
        "forward_friend_single_msg",
    } or (name.startswith("send_") and name.endswith("_msg"))


def _native_segments(value, *, auto_escape=False):
    """Convert only native CQ strings; JSON text segments remain opaque."""
    if isinstance(value, str):
        if auto_escape:
            return [{"type": "text", "data": {"text": value}}]
        return [dict(segment) for segment in Message(value)]
    if isinstance(value, dict):
        if "messages" in value:
            return _native_segments(value["messages"])
        value = [value]
    parts = []
    for part in value or []:
        if isinstance(part, str):
            parts.extend(_native_segments(part))
            continue
        part = copy.deepcopy(part)
        if part.get("type") == "node" and isinstance(part.get("data"), dict):
            data = part["data"]
            if "content" in data:
                data["content"] = _native_segments(data["content"])
        parts.append(part)
    return parts


@dataclass
class QueuedDelivery(Mapping):
    """Deferred receipt mapping; platform fields appear after turn completion."""

    action: str
    args: tuple
    params: dict
    status: str = "queued"
    receipt: object = None

    def message_segments(self):
        value = self.params.get("message", self.params.get("messages"))
        if value is None and len(self.args) > 1:
            value = self.args[1]
        return _native_segments(
            value, auto_escape=self.params.get("auto_escape", False)
        )

    def __getitem__(self, key):
        if not isinstance(self.receipt, Mapping):
            raise KeyError(key)
        return self.receipt[key]

    def __iter__(self):
        if isinstance(self.receipt, Mapping):
            return iter(self.receipt)
        return iter(())

    def __len__(self):
        return len(self.receipt) if isinstance(self.receipt, Mapping) else 0


def install_send_tool_adapter(mark):
    """Route the Core built-in current-session tool through Jev's event queue."""
    from astrbot.core.tools.message_tools import SendMessageToUserTool

    original = SendMessageToUserTool.call
    if getattr(original, "_jev_send_tool", False):
        return

    @wraps(original)
    async def call(tool, run_context, **kwargs):
        event = getattr(run_context.context, "event", None)
        if (
            event is None
            or not event.get_extra(mark)
            or not event.get_extra("_jev_complete_output")
        ):
            return await original(tool, run_context, **kwargs)
        agent_context = copy.copy(run_context.context)
        agent_context.context = ScopedContext(agent_context.context, event)
        scoped_context = copy.copy(run_context)
        scoped_context.context = agent_context
        result = await original(tool, scoped_context, **kwargs)
        return result

    call._jev_send_tool = True
    SendMessageToUserTool.call = call


def native_delivery(event, action, args, params):
    """Reject foreign or non-mergeable native calls before collecting their body."""
    args, params = copy.deepcopy(args), copy.deepcopy(params)
    request = QueuedDelivery(action, args, params)
    if action == "call_action":
        action, *rest = args
        args = tuple(rest)
    target = params
    if action == "send":
        target = args[0] if args else params.get("event", {})
        if len(args) > 2 or set(params) - {"event", "message", "auto_escape"}:
            raise DeliveryDeclined("native_send_options_not_mergeable")
    elif args or set(params) - {
        "group_id",
        "user_id",
        "message",
        "messages",
        "message_type",
        "self_id",
        "auto_escape",
    }:
        raise DeliveryDeclined("native_send_options_not_mergeable")
    if "forward_msg" in action and "auto_escape" in params:
        raise DeliveryDeclined("native_send_options_not_mergeable")
    group = bool(event.get_group_id())
    key = "group_id" if group else "user_id"
    expected = event.get_group_id() if group else event.get_sender_id()
    kind = "group" if group else "private"
    if (
        action
        not in {
            "send",
            "send_msg",
            "send_forward_msg",
            f"send_{kind}_msg",
            f"send_{kind}_forward_msg",
        }
        or str(target.get(key, "")) != str(expected)
        or (action != "send" and target.get("user_id" if group else "group_id"))
        or target.get("message_type", kind) != kind
        or (
            target.get("self_id") and str(target["self_id"]) != str(event.get_self_id())
        )
    ):
        raise DeliveryDeclined("autonomous_cross_session_send")
    return request


class EventBotProxy:
    def __init__(self, bot, sender):
        self._bot, self._sender = bot, sender

    def __getattr__(self, name):
        original = getattr(self._bot, name)
        if is_message_action(name):

            async def guarded(*args, **kwargs):
                return await self._sender(original, *args, _jev_action=name, **kwargs)

            return guarded
        if name == "call_action":

            async def action(action, *args, **kwargs):
                if is_message_action(action):
                    return await self._sender(
                        original, action, *args, _jev_action="call_action", **kwargs
                    )
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
        await self._event.send(chain)
        return True


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
