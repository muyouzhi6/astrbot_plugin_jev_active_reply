"""Site migration must not leak historical provider credentials."""

import math

import pytest

from ..client import Adapter, KeyPool
from ..settings import ConfigView


@pytest.mark.parametrize("primary", ["mindshub", "mindshub_air", "typesafe", "nanbei"])
def test_remove_old_credentials_for_every_primary(primary):
    raw = {
        "primary_channel": primary,
        "mindshub": {"keys": ["mdb_old"]},
        "mindshub_air": {"keys": ["mdb_old2"]},
        "typesafe": {"keys": ["apikey_example"]},
    }
    config = ConfigView(raw)
    assert "mindshub" not in raw and "mindshub_air" not in raw
    assert "nanbei" not in raw
    assert "primary_channel" not in raw
    assert config["relay_mode"] == ("nanbei" if primary == "nanbei" else "typesafe")


def test_legacy_without_official_key_stays_unconfigured():
    raw = {"primary_channel": "mindshub_air", "mindshub_air": {"keys": ["mdb_old"]}}
    ConfigView(raw)
    assert raw["relay_mode"] == "nanbei" and "primary_channel" not in raw
    assert "nanbei" not in raw and "mindshub_air" not in raw
    assert raw["dry_run"] is False


def test_native_site_contract_and_recharge_recovery():
    adapter = Adapter.create("nanbei", {})
    assert adapter.endpoint == "https://hajimi.165201.xyz/v1/systemone"
    assert adapter.model == "jev-latest"
    reason, wait, shared = adapter.classify(403, {"error": {"message": "balance"}}, {})
    assert reason == "site_access_or_balance" and math.isfinite(wait) and shared
    assert adapter.classify(401, {"error": {}}, {})[0] == "auth_disabled"


@pytest.mark.parametrize(
    "kind,key",
    [
        ("nanbei", "apikey_example"),
        ("nanbei", "mdb_example"),
        ("typesafe", "sk-example"),
        ("typesafe", "mdb_example"),
    ],
)
def test_wrong_credential_provider_is_rejected_locally(kind, key):
    with pytest.raises(ValueError, match="requires"):
        KeyPool(kind, {"keys": [key]})
