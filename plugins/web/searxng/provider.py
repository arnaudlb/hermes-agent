"""SearXNG search — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Same JSON
API call (``/search?format=json``), same result normalization. The legacy
in-tree module ``tools.web_providers.searxng`` was removed in the same
commit that moved this code under ``plugins/``; this file is now the
canonical implementation.

Search-only — SearXNG aggregates results from upstream engines but does not
fetch/extract arbitrary URLs. ``supports_extract()`` returns False.

Config keys this provider responds to::

    web:
      search_backend: "searxng"     # explicit per-capability
      backend: "searxng"            # shared fallback

Env var::

    SEARXNG_URL=http://localhost:8080
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _usage_log_path() -> str:
    """Per-profile search usage log (LLM-043 task 8): one NDJSON line per
    ``search()`` call so per-user (per-profile) counters can be aggregated
    for multi-tenant visibility and future per-profile quotas."""
    hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return os.path.join(hermes_home, "web_search_usage.ndjson")


def _query_salt_path() -> str:
    hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return os.path.join(hermes_home, ".usage_salt")


def _load_or_create_salt() -> bytes:
    """Per-install HMAC salt for query hashing (0600, never in log/KB).

    A plain SHA-256 of a query is NOT private: queries are short, low-entropy,
    and drawn from a guessable space, so anyone with the log can hash a
    wordlist and match. Salting with a per-install secret defeats the
    dictionary attack while keeping distinct-query counting exact (same query
    + same salt -> same hash). Losing the salt loses cross-history
    comparability and nothing else — the right failure mode. Consequence:
    salted hashes are not comparable *across machines*; a shared salt is a
    shared secret.
    """
    import secrets

    path = _query_salt_path()
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        if data:
            return data
    except (FileNotFoundError, OSError):
        pass
    salt = secrets.token_bytes(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(salt)
    return salt


def _query_hash(query: str) -> str:
    """Salted HMAC of the query, truncated to 16 hex chars (64 bits).

    64 bits is more than enough for a distinct-count on an unbounded set of
    free-text queries (birthday bound: a 50% collision needs ~2^32 distinct
    inputs). The salt is what makes it private, not the width.
    """
    import hashlib
    import hmac

    return hmac.new(_load_or_create_salt(), query.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def _current_profile() -> tuple[str, str]:
    """Best-effort caller identity for the usage log.

    Returns ``(profile, source)`` where source is one of:
    - ``"env"`` — ``HERMES_PROFILE`` was explicitly set (kanban workers set
      it for their workers; ``hermes -p <profile>`` does NOT, per the
      comment at hermes_cli/kanban_db.py:10838).
    - ``"home"`` — inferred from ``HERMES_HOME`` via
      ``get_active_profile_name()``: a profile's home is
      ``~/.hermes/profiles/<name>``, so multi-profile deployments are
      attributed without env plumbing; the default home reports
      ``("default", "home")``.
    - ``"fallback"`` — nothing observable; catch-all bucket. The source
      field is what makes the attribution gap measurable in the data
      (``web_usage_report.py`` shows how many searches are unattributed)
      instead of quietly claiming they belong to ``default``.
    """
    env_profile = (os.getenv("HERMES_PROFILE") or "").strip()
    if env_profile:
        return env_profile, "env"
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name(), "home"
    except Exception:  # noqa: BLE001 — identity is best-effort
        return "default", "fallback"


def _record_usage(outcome: Dict[str, Any], query: str, started: float) -> None:
    """Append one usage line. Must never break search — all failures silent.

    Privacy: the raw query is NOT stored — the whole point of the
    self-hosted SearXNG setup is that query text is sensitive (see the
    yandex exclusion on jurisdiction grounds). A salted HMAC keeps
    distinct-query counts without reconstructing the query.
    """
    try:
        import datetime

        profile, profile_source = _current_profile()
        rec = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "profile": profile,
            "profile_source": profile_source,
            "query_hash": _query_hash(query),
            "success": bool(outcome.get("success")),
            "backend": outcome.get("backend") or None,
            "escalated": outcome.get("escalated"),
            "n_results": len((outcome.get("data") or {}).get("web") or []),
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
        with open(_usage_log_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — usage logging is best-effort
        pass


def _searxng_url() -> str:
    """Return SEARXNG_URL from Hermes config-aware env, falling back to process env."""
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value("SEARXNG_URL")
    except Exception:
        val = None
    if val is None:
        val = os.getenv("SEARXNG_URL", "")
    return (val or "").strip()


class SearXNGWebSearchProvider(WebSearchProvider):
    """Search via a user-hosted SearXNG instance."""

    @property
    def name(self) -> str:
        return "searxng"

    @property
    def display_name(self) -> str:
        return "SearXNG"

    def is_available(self) -> bool:
        """Return True when ``SEARXNG_URL`` is set."""
        return bool(_searxng_url())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the configured SearXNG instance.

        Provider-side escalation (LLM-043): a preflight quota gate keeps
        the request off SearXNG/Brave once the shared budget is exhausted,
        and a thin/unresponsive result — or a hard SearXNG failure —
        diverts to the keyless ddgs fallback. Both are opt-in via
        ``web.searxng_quota`` / ``web.searxng_fallback`` (off by default),
        so the historical behavior is unchanged without config.

        Every call also appends one line to the per-profile usage log
        (``$HERMES_HOME/web_search_usage.ndjson``) for multi-tenant
        per-user counters and future per-profile quotas.
        """
        started = time.monotonic()
        result = self._search_impl(query, limit)
        try:
            _record_usage(result, query, started)
        except Exception:  # noqa: BLE001 — usage logging must never break search
            pass
        return result

    def _search_impl(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx

        base_url = _searxng_url().rstrip("/")
        if not base_url:
            return {"success": False, "error": "SEARXNG_URL is not set"}

        # Load the web config once; the quota gate and fallback read their
        # knobs from it. Failure to load degrades to {} = both gates off.
        try:
            from tools.web_tools import _load_web_config

            cfg = _load_web_config()
        except Exception:  # noqa: BLE001
            cfg = {}

        from .ddgs_fallback import check_quota, ddgs_search

        # --- Quota preflight (BEFORE the SearXNG call) -----------------
        # The gate's only hard stop is `blocked` (budget exhausted); the
        # warn tier (`denied`) stays allowed so the user keeps seeing it.
        try:
            verdict, qreason = check_quota(cfg)
        except Exception as exc:  # noqa: BLE001 — gate must never break search
            logger.warning("quota gate error (%s); proceeding", exc)
            verdict, qreason = "fail_open", str(exc)
        if verdict == "blocked":
            logger.warning(
                "Brave budget exhausted (%s) — skipping SearXNG, using ddgs "
                "fallback for '%s'", qreason, query,
            )
            r = ddgs_search(query, limit, cfg)
            r["escalated"] = "quota-blocked"
            r["quota"] = qreason
            return r
        if verdict == "fail_open":
            logger.warning(
                "quota gate unavailable (%s) — proceeding to SearXNG "
                "(fail-open, per design)", qreason,
            )

        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }

        def _searxng_request(p: Dict[str, Any]):
            """One SearXNG HTTP call. Raises httpx errors like before."""
            resp = httpx.get(
                f"{base_url}/search",
                params=p,
                timeout=15,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            return resp

        try:
            resp = _searxng_request(params)
        except httpx.HTTPStatusError as exc:
            logger.warning("SearXNG HTTP error: %s", exc)
            return self._fallback_or_error(
                cfg, ddgs_search, query, limit,
                f"SearXNG returned HTTP {exc.response.status_code}",
            )
        except httpx.RequestError as exc:
            logger.warning("SearXNG request error: %s", exc)
            return self._fallback_or_error(
                cfg, ddgs_search, query, limit,
                f"Could not reach SearXNG at {base_url}: {exc}",
            )

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("SearXNG response parse error: %s", exc)
            return self._fallback_or_error(
                cfg, ddgs_search, query, limit,
                "Could not parse SearXNG response as JSON",
            )

        raw_results = data.get("results", [])
        unresponsive = data.get("unresponsive_engines") or []

        # --- Post-response escalation (thin + braveapi unresponsive) ---
        # A few results while the paid engine is down means the free tier
        # alone could not answer — ddgs adds a keyless second opinion.
        if len(raw_results) < 3 and any(
            "braveapi" in str(e) for e in unresponsive
        ):
            logger.warning(
                "SearXNG thin (%d results) with braveapi unresponsive for "
                "'%s' — using ddgs fallback", len(raw_results), query,
            )
            r = ddgs_search(query, limit, cfg)
            r["escalated"] = "braveapi-unresponsive"
            return r

        # SearXNG may return a score field; sort descending and cap to limit.
        sorted_results = sorted(
            raw_results,
            key=lambda r: float(r.get("score", 0)),
            reverse=True,
        )[:limit]

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("content", "")),
                "position": i + 1,
            }
            for i, r in enumerate(sorted_results)
        ]

        logger.info(
            "SearXNG search '%s': %d results (from %d raw, limit %d)",
            query,
            len(web_results),
            len(raw_results),
            limit,
        )

        return {"success": True, "data": {"web": web_results}}

    def _fallback_or_error(
        self,
        cfg: Dict[str, Any],
        ddgs_search,
        query: str,
        limit: int,
        error: str,
    ) -> Dict[str, Any]:
        """SearXNG failed hard — use the ddgs fallback if enabled, else error.

        A local SearXNG failure (HTTP error, unreachable, bad JSON, or zero
        results) is one Brave cannot fix, so the keyless fallback is the
        right answer; it costs no budget. Without
        ``web.searxng_fallback.enabled`` the historical error response is
        returned unchanged.
        """
        fb = (cfg or {}).get("searxng_fallback") or {}
        if fb.get("enabled", False):
            logger.warning(
                "SearXNG failed (%s) — using ddgs fallback for '%s'",
                error, query,
            )
            r = ddgs_search(query, limit, cfg)
            r["escalated"] = "searxng-failed"
            if r.get("success"):
                r["note"] = (
                    f"searxng unavailable ({error}) -> ddgs (keyless) fallback"
                )
            return r
        return {"success": False, "error": error}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "SearXNG",
            "badge": "free · self-hosted",
            "tag": "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.",
            "env_vars": [
                {
                    "key": "SEARXNG_URL",
                    "prompt": "SearXNG instance URL (e.g. http://localhost:8080)",
                    "url": "https://searx.space/",
                },
            ],
        }
