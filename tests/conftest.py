"""Tests may use loopback HTTP, never an external supplier endpoint."""

from urllib.parse import urlsplit

import aiohttp
import pytest
import sys
import importlib

# Repository checkout names may differ from the published plugin package.
package = __package__.rsplit(".", 1)[0]
sys.modules.setdefault(
    "astrbot_plugin_jev_active_reply", importlib.import_module(package)
)


@pytest.fixture(autouse=True)
def prohibit_external_http(monkeypatch):
    original = aiohttp.ClientSession._request

    async def local_only(self, method, url, *args, **kwargs):
        if urlsplit(str(url)).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise AssertionError("External network calls are forbidden in tests")
        return await original(self, method, url, *args, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "_request", local_only)
