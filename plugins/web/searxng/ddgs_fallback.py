"""Quota gate + keyless DuckDuckGo fallback for the SearXNG provider.

Two responsibilities, both driven by the ``web:`` section of
``~/.hermes/config.yaml`` (both OFF by default in code):

1. ``check_quota`` — preflight against the shared ``agent-quota`` ledger
   before a SearXNG request is sent. The ledger's ``brave`` budget is the
   only metered one, and the handover mandate is "never send a request
   once the shared monthly budget is exhausted". The gate runs the client
   script (``agent-quota check brave --json``), which consults the ledger
   server for the cross-host merged view and falls back to the local
   event log (degraded, but still a reading) when the server is down.
   Verdict mapping (two-tier, mirroring the client's design):

   - ``ALLOW``  -> ``ok``          (proceed to SearXNG)
   - ``WARN``   -> ``denied``      (>=80% used; still allowed — the warn
                                    tier keeps working so the user sees it)
   - ``DENY``   -> ``blocked``     (budget exhausted; do NOT send)
   - ``UNKNOWN``-> ``blocked``     (backend not registered; fail safe —
                                    diverting to keyless ddgs is free)
   - script missing / crash / timeout -> ``fail_open`` (handover mandate:
     a downed gate means the user loses the warning, not correctness; the
     sidecar reconciles the ledger from SearXNG metrics when it returns)

2. ``ddgs_search`` — keyless DuckDuckGo search via the ``ddgs`` package
   (installed in the Hermes venv). Mirrors the GIL-safety design of
   ``plugins/web/ddgs/provider.py`` (#68096): the blocking ddgs call runs
   in a disposable child process that the parent terminates on timeout —
   a ``ThreadPoolExecutor`` + ``future.result(timeout=…)`` cap cannot fire
   while native code holds the GIL, so a thread-based timeout would freeze
   the whole agent process. The child reuses the ddgs provider's worker
   machinery (``_search_worker.py`` + ``_run_ddgs_search``) via the
   ``HERMES_DDGS_SEARXNG_FB=1`` env flag, which adds the configured
   ``region`` (default ``wt-wt``, US-centric) to the query.

``ddgs`` is never imported at module load, so this module (and the
provider) import cleanly even if the package is ever uninstalled, and the
existing searxng tests — which exercise the old code path unchanged —
still pass.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

logger = None  # set lazily by _get_logger (keep module import side-effect-free)

# Absolute path by design: ``~/.local/bin`` is not guaranteed to be on the
# PATH a provider subprocess inherits.
AGENT_QUOTA_CLIENT = os.path.expanduser(
    "~/PERSO/Knowledge/AI_knowledge/bin/agent_quota.py"
)

# Overall wall-clock cap for the ddgs fallback child.
_FB_TIMEOUT_SECS = 30
_POLL_INTERVAL_SECS = 0.1
_TERMINATE_GRACE_SECS = 1.0


def _get_logger():
    global logger
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)
    return logger


# ---------------------------------------------------------------------------
# web: config access (config-aware, like the provider's _searxng_url)
# ---------------------------------------------------------------------------

_web_cfg_cache: Optional[dict] = None
_web_cfg_ts: float = 0.0
_WEB_CFG_TTL = 30.0  # re-read config if older than this (config edits apply)


def web_cfg() -> dict:
    """Return the ``web:`` section of the Hermes config (cached, TTL 30 s).

    The provider's ``search(query, limit)`` receives no config object, so
    the fallback module loads it itself — the same config-aware path as
    ``tools.web_tools._load_web_config`` (``hermes_cli.config`` first, raw
    process env / file as fallback). Failures degrade to ``{}``, which
    means every gate stays off and the old behavior is preserved.
    """
    global _web_cfg_cache, _web_cfg_ts
    now = time.monotonic()
    if _web_cfg_cache is not None and now - _web_cfg_ts < _WEB_CFG_TTL:
        return _web_cfg_cache
    cfg: dict = {}
    try:
        from hermes_cli.config import load_config

        cfg = load_config().get("web") or {}
    except Exception:  # noqa: BLE001 — degrade to defaults (gates off)
        cfg = {}
    _web_cfg_cache = cfg
    _web_cfg_ts = now
    return cfg


def clear_web_cfg_cache() -> None:
    """Test hook: force a config re-read on the next :func:`web_cfg` call."""
    global _web_cfg_cache, _web_cfg_ts
    _web_cfg_cache = None
    _web_cfg_ts = 0.0


# ---------------------------------------------------------------------------
# 1. Quota preflight
# ---------------------------------------------------------------------------


def check_quota(cfg: Optional[dict] = None) -> tuple[str, str]:
    """Preflight the shared budget ledger before sending a SearXNG request.

    ``cfg`` is the ``web:`` config section; when omitted it is loaded via
    :func:`web_cfg` (cached). Returns ``(verdict, reason)`` with verdict in
    ``{"ok", "denied", "blocked", "fail_open"}``. Only ``ok`` (plus the
    warn-tier ``denied``, which the provider still lets through) may
    proceed; ``blocked`` diverts to the ddgs fallback so no Brave request
    is sent.
    """
    quota = (cfg if cfg is not None else web_cfg()).get("searxng_quota") or {}
    if not quota.get("enabled", False):
        return "ok", "quota gate disabled"

    client = (quota.get("client") or AGENT_QUOTA_CLIENT).strip()
    backend = str(quota.get("backend", "brave"))
    timeout = float(quota.get("timeout_s", 5))

    if not os.path.exists(client):
        # Handover-mandated: fail OPEN when the gate cannot be consulted.
        return "fail_open", f"agent-quota client not found ({client})"

    try:
        proc = subprocess.run(
            [client, "check", backend, "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "fail_open", f"agent-quota client error: {exc}"

    if proc.returncode not in (0, 1):
        # 2 = UNKNOWN backend (or unexpected failure). Failing safe here
        # (divert to keyless ddgs) is free: it costs no budget.
        detail = (proc.stdout or proc.stderr or "").strip()[:200]
        return "blocked", f"agent-quota exit {proc.returncode}: {detail}"

    try:
        data = json.loads(proc.stdout or "{}")
    except (ValueError, TypeError):
        data = {}
    verdict = str(data.get("verdict") or "").upper()
    reason = str(data.get("reason") or "").strip()
    remaining = data.get("remaining")

    if verdict == "ALLOW":
        return "ok", reason or (f"{remaining} left" if remaining is not None else "")
    if verdict == "WARN":
        # Warn tier: budget nearly spent but not exhausted — the call is
        # allowed (the user sees the warning), matching the client's
        # two-tier design.
        return "denied", reason or "warn tier"
    if verdict == "DENY":
        return "blocked", reason or "budget exhausted"
    # UNKNOWN / unparsable: treat as blocked (fail safe, free fallback).
    return "blocked", f"unknown verdict ({verdict or 'none'}): {reason[:200]}"


# ---------------------------------------------------------------------------
# 2. ddgs keyless fallback (subprocess-isolated, #68096)
# ---------------------------------------------------------------------------

_DDG_PROVIDER_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ddgs")
)
_SEARXNG_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


def _plugins_path_entry() -> str:
    """``sys.path`` entry that makes ``import plugins`` work in the child."""
    try:
        import plugins as plugins_pkg

        pkg_file = getattr(plugins_pkg, "__file__", None)
        if pkg_file:
            return os.path.dirname(
                os.path.dirname(os.path.abspath(pkg_file))
            )
    except Exception:  # noqa: BLE001
        pass
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )


def ddgs_search(query: str, limit: int = 5, cfg: Optional[dict] = None) -> Dict[str, Any]:
    """Keyless DuckDuckGo fallback search.

    ``cfg`` is the ``web:`` config section; when omitted it is loaded via
    :func:`web_cfg` (cached). Returns a provider-shaped response
    (``success`` / ``data.web``) with ``backend_used: "searxng"`` kept
    stable for callers/caches and a ``backend: "ddgs-fallback"`` marker
    (the caller adds ``escalated``). Never raises — any failure
    (including timeout) returns ``{"success": False, "error": ...}``.
    """
    fb = (cfg if cfg is not None else web_cfg()).get("searxng_fallback") or {}
    region = fb.get("region", "wt-wt")
    timeout = float(fb.get("timeout_s", 20))
    safe_limit = max(1, min(int(limit or 5), 20))

    # The blocking ddgs call must run in a disposable child process
    # (#68096): ddgs/primp can hold the GIL inside native code, where a
    # thread-based timeout cannot fire and the whole agent would freeze.
    # Reuse the ddgs provider's proven worker machinery.
    worker_path = os.path.join(_DDG_PROVIDER_DIR, "_search_worker.py")
    if not os.path.exists(worker_path):
        return {
            "success": False,
            "error": "ddgs fallback unavailable (worker missing)",
            "backend_used": "searxng",
            "backend": "ddgs-fallback",
        }

    request: dict[str, Any] = {
        "query": query,
        "safe_limit": safe_limit,
        "region": str(region or "wt-wt"),
    }

    env = dict(os.environ)
    # Make `import plugins.web.ddgs.provider` resolvable in the child: the
    # venv's site-packages (ddgs) is already on the child's default path;
    # add the tree root for the plugins package. (The worker's own
    # directory is sys.path[0] via the script path, as with the ddgs
    # provider's own worker.)
    child_pythonpath = env.get("PYTHONPATH", "")
    path_entry = _plugins_path_entry()
    if path_entry and path_entry not in child_pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = (
            path_entry + os.pathsep + child_pythonpath if child_pythonpath else path_entry
        )

    try:
        # input= writes the request to stdin and closes it atomically in
        # Popen itself (the ddgs provider's own worker is driven the same
        # way) — manually write+close+communicate() hits "I/O operation on
        # closed file" because communicate() re-flushes stdin.
        proc = subprocess.Popen(
            [sys.executable, worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        return {
            "success": False,
            "error": f"ddgs fallback spawn failed: {exc}",
            "backend_used": "searxng",
            "backend": "ddgs-fallback",
        }

    raw = ""
    timed_out = False
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ddgs-fb")
    fut = pool.submit(proc.communicate, json.dumps(request))
    try:
        deadline = time.monotonic() + timeout
        while True:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                timed_out = True
                break
            try:
                # communicate(input=...) writes the request to stdin and
                # closes it atomically (the ddgs provider drives its
                # worker the same way; a manual write+close+communicate()
                # re-flushes the closed pipe and dies with "I/O operation
                # on closed file").
                out, _err = fut.result(timeout=min(_POLL_INTERVAL_SECS, remaining_time))
                raw = out or ""
                break
            except cf.TimeoutError:
                continue
    finally:
        # Terminate + kill the child; never join a possibly-GIL-holding
        # process on the main path.
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=_TERMINATE_GRACE_SECS)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except OSError:
            pass
        # After kill, communicate should return promptly; don't block
        # forever.
        if not fut.done():
            try:
                out, _err = fut.result(timeout=_TERMINATE_GRACE_SECS)
                if not raw:
                    raw = out or ""
            except Exception:  # noqa: BLE001
                pass
        pool.shutdown(wait=False, cancel_futures=True)

    if timed_out and proc.poll() is None:
        return {
            "success": False,
            "error": f"ddgs fallback timed out after {timeout:.0f}s",
            "backend_used": "searxng",
            "backend": "ddgs-fallback",
        }

    raw = raw.strip()
    if not raw:
        return {
            "success": False,
            "error": f"ddgs fallback worker exited without a result (code={proc.poll()})",
            "backend_used": "searxng",
            "backend": "ddgs-fallback",
        }
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": f"ddgs fallback worker returned invalid JSON: {raw[:200]!r}",
            "backend_used": "searxng",
            "backend": "ddgs-fallback",
        }
    if not isinstance(envelope, dict) or not envelope.get("ok"):
        return {
            "success": False,
            "error": str(envelope.get("error") if isinstance(envelope, dict) else envelope)
            or "ddgs fallback failed",
            "backend_used": "searxng",
            "backend": "ddgs-fallback",
        }

    results = envelope.get("results") or []
    web_results = [
        {
            "title": str(r.get("title", "")),
            "url": str(r.get("url", "")),
            "description": str(r.get("description", ""))[:400],
            "position": i + 1,
        }
        for i, r in enumerate(results[:safe_limit])
        if isinstance(r, dict)
    ]

    _get_logger().info(
        "ddgs fallback for '%s': %d results (limit %d)",
        query, len(web_results), safe_limit,
    )
    return {
        "success": True,
        "data": {"web": web_results},
        "backend_used": "searxng",
        "backend": "ddgs-fallback",
        "note": "searxng/braveapi unavailable -> ddgs (keyless) fallback",
    }
