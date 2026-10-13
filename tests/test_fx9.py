"""Ninth-review regressions with synthetic ingress and real Core stages."""

from types import SimpleNamespace as NS

import pytest
from astrbot.core.star.star import star_map
from astrbot.core.star.star_handler import (
    EventType,
    StarHandlerMetadata,
    star_handlers_registry,
)

from ..main import MARK
from . import test_v0101_explicit_wake as wake
from .test_v0102_wake_warnings import adapter_event, delivery_hooks, env, pipeline
from .test_fx4 import payload

__all__ = ["env", "delivery_hooks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", [200000, 0, -200000])
@pytest.mark.parametrize("empty", ["none", "chain", "exception"])
async def test_consumed_result_finishes_with_delivery(
    env, monkeypatch, priority, empty
):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")

    async def consume(ev):
        ev.set_result(ev.plain_result("") if empty == "chain" else None)
        if empty == "exception":
            raise RuntimeError("synthetic decorator failure")

    name = "fx9_consumer"
    monkeypatch.setitem(star_map, name, NS(name=name, activated=True, reserved=False))
    star_handlers_registry.append(
        StarHandlerMetadata(
            EventType.OnDecoratingResultEvent,
            name,
            name,
            name,
            consume,
            [],
            extras_configs={"priority": priority},
        )
    )
    await pipeline(env, event, adapter=True)
    assert bot.transport.await_count == 1, payload(bot)
    assert event.get_extra(MARK)["finished"]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["request", "permit", "decorate", "scan"])
async def test_deadline_before_timer_completes(env, entry):
    wake.configure(env)
    wake.judge_result(env, "reply")
    event, bot = adapter_event("at")
    await env.plugin.on_input(event)
    await env.plugin.on_message(event)
    meta = event.get_extra(MARK)
    meta["created"] -= env.plugin._lifetime() + 1
    if entry == "request":
        await env.plugin.on_request(event, NS())
    elif entry == "permit":
        assert not await env.plugin._permit(event)
    elif entry == "decorate":
        await env.plugin.before_output(event)
    else:
        other, _ = adapter_event("at")
        await env.plugin.on_input(other)
        await env.plugin.on_message(other)
    await env.plugin._complete_output(event, expired=True)
    assert bot.transport.await_count == 0, payload(bot)
    assert meta["finished"]
    assert not meta.get("lease")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["at", "name", "private"])
async def test_full_capacity_closes_wake_without_allocating_room(env, kind):
    wake.configure(env)
    wake.judge_result(env, "reply")
    for n in range(env.plugin.rooms.capacity):
        env.plugin.rooms.get(f"synthetic:{n}").claim(f"pending:{n}", 0)
    event, bot = adapter_event(kind)
    await pipeline(env, event, adapter=True)
    assert bot.transport.await_count == 0, payload(bot)
    assert not event.get_extra(MARK)
    assert len(env.plugin.rooms.rooms) == env.plugin.rooms.capacity
