"""Tests for the SearXNG provider-side quota gate + ddgs fallback (LLM-043).

Complements ``test_web_providers_searxng.py`` (which covers the legacy path
and must keep passing unchanged). These tests monkeypatch
``ddgs_fallback.check_quota`` / ``ddgs_search`` and ``httpx.get`` so no
network, subprocess, or real config is touched.

Cases (per the execution plan):
1. quota ``blocked`` -> SearXNG httpx NOT called, ``escalated:
   "quota-blocked"`` + ddgs results.
2. quota ``denied`` (warn tier) -> allowed: SearXNG IS called (documents the
   two-tier intent).
3. quota ``fail_open`` (gate down) -> SearXNG called normally.
4. quota disabled (default cfg {}) -> SearXNG called, no quota side-effects.
5. 2 results + ``unresponsive_engines: ["braveapi"]`` -> ddgs retry,
   ``escalated: "braveapi-unresponsive"``.
6. 2 results + unresponsive ``["yep"]`` (not braveapi) -> no escalation,
   original results returned.
7. full success (10 results) -> untouched, no fallback.
Plus: hard SearXNG failure -> ddgs fallback when enabled, legacy error when
disabled; and unit tests of ``check_quota`` verdict mapping itself.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from unittest.mock import MagicMock, call

import pytest

from tests.tools.conftest import register_all_web_providers  # noqa: F401 (parity)

# Capture the REAL recorder + identity + hash functions at import time
# (before any fixture patches them). The conftest autouse fixture no-ops
# ``sp._record_usage`` per test; tests that exercise the *real* implementation
# restore these references via monkeypatch.setattr.
import plugins.web.searxng.provider as _sp_module
_REAL_RECORD_USAGE = _sp_module._record_usage
_REAL_CURRENT_PROFILE = _sp_module._current_profile
_REAL_QUERY_HASH = _sp_module._query_hash


CFG_BLOCKED = {
    "searxng_quota": {"enabled": True},
    "searxng_fallback": {"enabled": True},
}
CFG_ON = {
    "searxng_quota": {"enabled": True},
    "searxng_fallback": {"enabled": True},
}
CFG_OFF = {}  # gates off (code defaults)


def _sample(n: int, unresponsive=None) -> dict:
    return {
        "results": [
            {
                "title": f"R{i}",
                "url": f"https://r{i}.example.com",
                "content": f"d{i}",
                "score": 1.0 - i * 0.01,
            }
            for i in range(n)
        ],
        "unresponsive_engines": unresponsive or [],
    }


def _mock_resp(json_data):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = json_data
    m.raise_for_status = MagicMock()
    return m


DDGS_OK = {
    "success": True,
    "data": {
        "web": [
            {
                "title": "D1",
                "url": "https://d1.example.com",
                "description": "dd",
                "position": 1,
            }
        ]
    },
    "backend_used": "searxng",
    "backend": "ddgs-fallback",
    "note": "searxng/braveapi unavailable -> ddgs (keyless) fallback",
}


@pytest.fixture(autouse=True)
def _fresh_provider(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
    # Hermetic: the provider loads the web config via tools.web_tools.
    monkeypatch.setattr(
        "tools.web_tools._load_web_config", lambda: CFG_OFF, raising=False
    )
    import plugins.web.searxng.ddgs_fallback as fb

    monkeypatch.setattr(fb, "clear_web_cfg_cache", lambda: None)
    yield
    import plugins.web.searxng.provider as _p  # noqa: F401


@pytest.fixture
def provider():
    from plugins.web.searxng.provider import SearXNGWebSearchProvider

    return SearXNGWebSearchProvider()


class TestQuotaGate:
    def test_blocked_skips_searxng_and_escates_to_ddgs(self, monkeypatch, provider):
        monkeypatch.setattr(
            "tools.web_tools._load_web_config", lambda: CFG_BLOCKED, raising=False
        )
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(fb, "check_quota", lambda cfg=None: ("blocked", "budget exhausted"))
        monkeypatch.setattr(fb, "ddgs_search", lambda q, l, cfg=None: dict(DDGS_OK))

        with patch_httpx() as gets:
            result = provider.search("q", limit=5)

        assert gets.call_count == 0, "SearXNG httpx must NOT be called when blocked"
        assert result["escalated"] == "quota-blocked"
        assert result["quota"] == "budget exhausted"
        assert result["data"]["web"][0]["title"] == "D1"
        assert result["backend_used"] == "searxng"

    def test_denied_warn_tier_still_allowed(self, monkeypatch, provider):
        """Warn tier (>=80% used) is NOT a stop — documents the two-tier intent."""
        monkeypatch.setattr(
            "tools.web_tools._load_web_config", lambda: CFG_ON, raising=False
        )
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(fb, "check_quota", lambda cfg=None: ("denied", "85% of budget used"))
        monkeypatch.setattr(fb, "ddgs_search", lambda q, l, cfg=None: (_ for _ in ()).throw(AssertionError("ddgs must not run in warn tier")))

        with patch_httpx(_sample(10)) as gets:
            result = provider.search("q", limit=5)

        assert gets.call_count == 1
        assert result["success"] is True
        assert "escalated" not in result
        assert len(result["data"]["web"]) == 5

    def test_fail_open_proceeds_to_searxng(self, monkeypatch, provider):
        monkeypatch.setattr(
            "tools.web_tools._load_web_config", lambda: CFG_ON, raising=False
        )
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(fb, "check_quota", lambda cfg=None: ("fail_open", "quota server unreachable"))
        monkeypatch.setattr(fb, "ddgs_search", lambda q, l, cfg=None: (_ for _ in ()).throw(AssertionError("ddgs must not run on fail_open")))

        with patch_httpx(_sample(10)) as gets:
            result = provider.search("q", limit=5)

        assert gets.call_count == 1
        assert result["success"] is True
        assert "escalated" not in result

    def test_quota_disabled_no_side_effects(self, monkeypatch, provider):
        """Default cfg: no quota call, no fallback, exact legacy path."""
        import plugins.web.searxng.ddgs_fallback as fb

        def boom(q, l, cfg=None):
            raise AssertionError("ddgs must not run when gates are off")

        monkeypatch.setattr(fb, "check_quota", lambda cfg=None: ("ok", "quota gate disabled"))
        monkeypatch.setattr(fb, "ddgs_search", boom)

        with patch_httpx(_sample(10)) as gets:
            result = provider.search("q", limit=5)

        assert gets.call_count == 1
        assert result["success"] is True
        assert "escalated" not in result
        assert result["data"]["web"][0]["title"] == "R0"


class TestThinResultEscalation:
    def test_few_results_plus_braveapi_unresponsive(self, monkeypatch, provider):
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(fb, "check_quota", lambda cfg=None: ("ok", "x"))
        monkeypatch.setattr(fb, "ddgs_search", lambda q, l, cfg=None: dict(DDGS_OK))

        with patch_httpx(_sample(2, unresponsive=["braveapi"])) as gets:
            result = provider.search("q", limit=5)

        assert gets.call_count == 1  # SearXNG consulted once, then fallback
        assert result["escalated"] == "braveapi-unresponsive"
        assert result["data"]["web"][0]["title"] == "D1"

    def test_few_results_other_engine_unresponsive_no_escalation(
        self, monkeypatch, provider
    ):
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(fb, "check_quota", lambda cfg=None: ("ok", "x"))
        monkeypatch.setattr(fb, "ddgs_search", lambda q, l, cfg=None: (_ for _ in ()).throw(AssertionError("no escalation for non-braveapi unresponsive")))

        with patch_httpx(_sample(2, unresponsive=["yep"])) as gets:
            result = provider.search("q", limit=5)

        assert gets.call_count == 1
        assert "escalated" not in result
        assert len(result["data"]["web"]) == 2
        assert result["data"]["web"][0]["title"] == "R0"

    def test_full_success_untouched(self, monkeypatch, provider):
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(fb, "check_quota", lambda cfg=None: ("ok", "x"))
        monkeypatch.setattr(fb, "ddgs_search", lambda q, l, cfg=None: (_ for _ in ()).throw(AssertionError("no fallback on full success")))

        with patch_httpx(_sample(10)) as gets:
            result = provider.search("q", limit=5)

        assert gets.call_count == 1
        assert "escalated" not in result
        assert len(result["data"]["web"]) == 5
        assert result["data"]["web"][0]["title"] == "R0"


class TestHardFailureFallback:
    def test_request_error_uses_ddgs_when_enabled(self, monkeypatch, provider):
        import httpx as httpx_mod
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(
            "tools.web_tools._load_web_config", lambda: CFG_ON, raising=False
        )
        monkeypatch.setattr(fb, "check_quota", lambda cfg=None: ("ok", "x"))
        monkeypatch.setattr(fb, "ddgs_search", lambda q, l, cfg=None: dict(DDGS_OK))

        with patch_httpx_raises(httpx_mod.RequestError("boom")) as gets:
            result = provider.search("q", limit=5)

        assert gets.call_count == 1
        assert result["escalated"] == "searxng-failed"
        assert result["data"]["web"][0]["title"] == "D1"
        assert "boom" in result["note"]

    def test_request_error_legacy_when_fallback_disabled(self, monkeypatch, provider):
        import httpx as httpx_mod

        monkeypatch.setattr(
            "tools.web_tools._load_web_config", lambda: CFG_OFF, raising=False
        )
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(fb, "check_quota", lambda cfg=None: ("ok", "x"))
        monkeypatch.setattr(fb, "ddgs_search", lambda q, l, cfg=None: (_ for _ in ()).throw(AssertionError("fallback disabled")))

        with patch_httpx_raises(httpx_mod.RequestError("boom")) as gets:
            result = provider.search("q", limit=5)

        assert gets.call_count == 1
        assert result["success"] is False
        assert "Could not reach SearXNG" in result["error"]


# ---------------------------------------------------------------------------
# check_quota verdict mapping (subprocess-level, no network)
# ---------------------------------------------------------------------------


def _fake_proc(returncode=0, stdout="{}", stderr=""):
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


class TestCheckQuotaMapping:
    def _cfg(self, client=None):
        import plugins.web.searxng.ddgs_fallback as fb

        # Default to the real (existing) client path so the os.path.exists
        # guard passes and the subprocess branch is exercised.
        return {
            "searxng_quota": {
                "enabled": True,
                "client": client or fb.AGENT_QUOTA_CLIENT,
                "backend": "brave",
            }
        }

    @pytest.mark.parametrize(
        "stdout,rc,expected",
        [
            (json.dumps({"verdict": "ALLOW", "reason": "10% of budget used", "remaining": 4.5}), 0, "ok"),
            (json.dumps({"verdict": "WARN", "reason": "85% of budget used"}), 0, "denied"),
            (json.dumps({"verdict": "DENY", "reason": "budget exhausted"}), 1, "blocked"),
            (json.dumps({"verdict": "UNKNOWN", "reason": "no such backend"}), 2, "blocked"),
            ("not json", 0, "blocked"),
        ],
    )
    def test_verdicts(self, monkeypatch, stdout, rc, expected):
        import subprocess
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_proc(rc, stdout))
        verdict, reason = fb.check_quota(self._cfg())
        assert verdict == expected, (verdict, reason)

    def test_client_missing_fails_open(self, monkeypatch):
        import os
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(os.path, "exists", lambda p: False)
        verdict, reason = fb.check_quota(self._cfg(client="/no/such/script.py"))
        assert verdict == "fail_open"
        assert "not found" in reason

    def test_client_crash_fails_open(self, monkeypatch):
        import subprocess
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: (_ for _ in ()).throw(OSError("spawn failed")),
        )
        verdict, reason = fb.check_quota(self._cfg())
        assert verdict == "fail_open"
        assert "spawn failed" in reason

    def test_disabled_returns_ok_without_client(self, monkeypatch):
        import subprocess
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("client must not run when gate is disabled")
            ),
        )
        verdict, reason = fb.check_quota({"searxng_quota": {"enabled": False}})
        assert verdict == "ok"


# ---------------------------------------------------------------------------
# per-profile usage log (task 8)
# ---------------------------------------------------------------------------


class TestUsageLog:
    """One NDJSON line per search(), tagged with the caller profile.

    These tests drive the REAL recorder (not a duplicate): each sets
    ``HERMES_HOME`` to a per-test temp dir and restores the captured
    ``_REAL_RECORD_USAGE`` reference, so the production code path runs and
    writes into the temp home. They verify the *wrapper* contract — one line
    per attempt, across every return path, with the right profile/backend/
    escalation tags. Field-level behavior of the recorder itself is covered
    by TestRealUsageRecorder.
    """

    def _log(self, tmp_path):
        return tmp_path / "web_search_usage.ndjson"

    def test_success_search_appends_usage_line(self, monkeypatch, provider, tmp_path):
        import plugins.web.searxng.provider as sp

        monkeypatch.setenv("HERMES_PROFILE", "tenant-a")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(sp, "_record_usage", _REAL_RECORD_USAGE)
        log = self._log(tmp_path)

        with patch_httpx(_sample(5)):
            result = provider.search("usage test query", limit=5)

        assert result["success"] is True
        assert log.exists()
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1  # exactly one line per search attempt
        rec = json.loads(lines[0])
        assert rec["profile"] == "tenant-a"
        assert rec["profile_source"] == "env"
        assert rec["success"] is True
        assert rec["n_results"] == 5
        assert rec["escalated"] is None
        assert rec["latency_ms"] >= 0
        assert "ts" in rec
        assert "query_hash" in rec

    def test_ddgs_fallback_records_backend_and_escalation(self, monkeypatch, provider, tmp_path):
        import plugins.web.searxng.provider as sp

        monkeypatch.setenv("HERMES_PROFILE", "tenant-b")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(sp, "_record_usage", _REAL_RECORD_USAGE)
        log = self._log(tmp_path)
        monkeypatch.setattr(
            "tools.web_tools._load_web_config", lambda: CFG_BLOCKED, raising=False
        )
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(fb, "check_quota", lambda cfg=None: ("blocked", "budget exhausted"))
        monkeypatch.setattr(fb, "ddgs_search", lambda q, l, cfg=None: dict(DDGS_OK))

        with patch_httpx() as gets:
            provider.search("q", limit=5)

        assert gets.call_count == 0  # SearXNG never called
        rec = json.loads(log.read_text().strip())
        assert rec["profile"] == "tenant-b"
        assert rec["success"] is True
        assert rec["backend"] == "ddgs-fallback"
        assert rec["escalated"] == "quota-blocked"

    def test_failed_search_logged_with_success_false(self, monkeypatch, provider, tmp_path):
        import httpx as httpx_mod
        import plugins.web.searxng.provider as sp

        monkeypatch.delenv("HERMES_PROFILE", raising=False)  # default profile
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(sp, "_record_usage", _REAL_RECORD_USAGE)
        log = self._log(tmp_path)
        # cfg {} -> fallback disabled -> legacy error path
        monkeypatch.setattr(
            "tools.web_tools._load_web_config", lambda: CFG_OFF, raising=False
        )
        import plugins.web.searxng.ddgs_fallback as fb

        monkeypatch.setattr(fb, "check_quota", lambda cfg=None: ("ok", "x"))

        with patch_httpx_raises(httpx_mod.RequestError("boom")):
            result = provider.search("q", limit=5)

        assert result["success"] is False
        rec = json.loads(log.read_text().strip())
        assert rec["success"] is False
        assert rec["n_results"] == 0

    def test_usage_log_failure_never_breaks_search(self, monkeypatch, provider, tmp_path):
        import plugins.web.searxng.provider as sp

        def explode(outcome, query, started):
            raise OSError("disk full")

        monkeypatch.setattr(sp, "_record_usage", explode)
        with patch_httpx(_sample(5)):
            result = provider.search("q", limit=5)
        assert result["success"] is True


class TestRealUsageRecorder:
    """Exercise the REAL ``_record_usage`` / ``_current_profile`` (no
    monkeypatch of them) — closes the gap where TestUsageLog only verified
    a copy of the recorder around the wrapper.

    The conftest autouse fixture no-ops ``sp._record_usage``; each test here
    restores the real captured reference via ``monkeypatch.setattr`` so the
    production code path (not a test duplicate) is what runs.
    """

    def test_real_recorder_writes_line_to_hermes_home(self, monkeypatch, provider, tmp_path):
        import json

        import plugins.web.searxng.provider as sp

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        # Restore the real recorder (conftest no-ops it by default).
        monkeypatch.setattr(sp, "_record_usage", _REAL_RECORD_USAGE)
        # Guard against the dependency-patching hole (Claude review 2026-09-01):
        # the real _record_usage resolves module-level names at call time, so
        # fail loudly if any fixture left the deps patched to fakes.
        assert sp._record_usage is _REAL_RECORD_USAGE
        assert sp._current_profile is _REAL_CURRENT_PROFILE

        with patch_httpx(_sample(5)):
            result = provider.search("real recorder query", limit=5)
        assert result["success"] is True

        log = tmp_path / "web_search_usage.ndjson"
        assert log.exists()
        rec = json.loads(log.read_text().strip())
        # Privacy: raw query NOT stored; hash is the salted HMAC of the query.
        assert "query" not in rec
        assert rec["query_hash"] == _REAL_QUERY_HASH("real recorder query")
        assert rec["profile"]  # non-empty identity
        assert rec["profile_source"] in ("env", "home", "fallback")
        assert rec["n_results"] == 5

    def test_salt_hash_is_stable_and_salt_dependent(self, monkeypatch, tmp_path):
        import os

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        salt_path = tmp_path / ".usage_salt"
        # Same query + same salt -> same hash (distinct-count works).
        first = _REAL_QUERY_HASH("repeat me")
        assert _REAL_QUERY_HASH("repeat me") == first
        # A different salt must change the hash — the salt is what makes it
        # private. _load_or_create_salt re-reads the file each call, so
        # overwriting it changes the hash of the same query.
        salt_path.write_bytes(os.urandom(32))
        assert _REAL_QUERY_HASH("repeat me") != first

    def test_real_recorder_unwritable_home_never_raises(self, monkeypatch, tmp_path):
        import plugins.web.searxng.provider as sp

        monkeypatch.setattr(sp, "_record_usage", _REAL_RECORD_USAGE)
        # Point HERMES_HOME at a *file*, not a directory: the write fails.
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        monkeypatch.setenv("HERMES_HOME", str(blocker))
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        # Must return silently — the whole contract is "never break search".
        sp._record_usage({"success": True, "data": {}}, "q", time.monotonic())

    def test_current_profile_env_wins(self, monkeypatch):
        monkeypatch.setenv("HERMES_PROFILE", "tenant-x")
        assert _REAL_CURRENT_PROFILE() == ("tenant-x", "env")

    def test_current_profile_whitespace_env_falls_back(self, monkeypatch):
        # A whitespace-only HERMES_PROFILE must NOT yield an empty profile —
        # it should fall through to the home-based identity.
        monkeypatch.setenv("HERMES_PROFILE", "   ")
        profile, source = _REAL_CURRENT_PROFILE()
        assert profile != ""
        assert source != "env"

    def test_current_profile_falls_back_to_hermes_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        # Simulate a profile home: <tmp>/profiles/<name>.
        fake_home = tmp_path / "profiles" / "tenant-y"
        fake_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(fake_home))
        # get_active_profile_name() compares against the *real* profiles
        # root, so a tmp fake home resolves to "custom" — still a stable,
        # non-empty identity, which is the contract here.
        profile, source = _REAL_CURRENT_PROFILE()
        assert profile in ("custom", "tenant-y")
        assert source == "home"

    def test_current_profile_never_raises(self, monkeypatch):
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        monkeypatch.setenv("HERMES_HOME", "/nonexistent/hermes/home")
        profile, source = _REAL_CURRENT_PROFILE()
        assert profile in ("default", "custom")
        assert source in ("home", "fallback")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class patch_httpx:
    """Patch httpx.get; return a MagicMock whose .call_count is the SearXNG hit count."""

    def __init__(self, json_data=None, exc=None):
        self.json_data = json_data
        self.exc = exc

    def __enter__(self):
        import httpx as httpx_mod
        from unittest.mock import patch

        if self.exc is not None:
            self.mock = MagicMock(side_effect=self.exc)
        else:
            self.mock = MagicMock(return_value=_mock_resp(self.json_data))
        p = patch.object(httpx_mod, "get", self.mock)
        p.start()
        self._patch = p
        return self.mock

    def __exit__(self, *exc_info):
        self._patch.stop()
        return False


def patch_httpx_raises(exc):
    return patch_httpx(exc=exc)
