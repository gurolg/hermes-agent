"""Issue #9: read-only, authenticated, live work-type diagnostics.

Native behavior tests against the pinned Hermes source revision. No API keys,
model calls, chat contents or real gateways are used.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.run_shutdown import GatewayShutdownMixin


def _runner(*, chat=0, managed=0, cron=0, api=0, deferred=0):
    def api_count(*, diagnostic_strict=False):
        assert diagnostic_strict is True
        return api

    return SimpleNamespace(
        _running_agent_count=lambda: chat,
        _running_cron_job_count=lambda: cron,
        _active_deferred_agent_worker_count=lambda: deferred,
        adapters={
            Platform.API_SERVER: SimpleNamespace(active_agent_work_count=api_count)
        },
        mock_managed=managed,
        # These must never enter a report:
        private_session="SECRET_SESSION",
        private_model="SECRET_MODEL",
        private_auth="SECRET_AUTH",
    )


def test_live_work_counts_match_five_sources_without_disclosing_private_state():
    runner = _runner(chat=1, managed=2, cron=3, api=4, deferred=5)
    with patch("gateway.run_shutdown.managed_turn_count",
               side_effect=lambda obj: obj.mock_managed):
        report = GatewayShutdownMixin._live_work_kind_counts(runner)

    assert report["status"] == "live"
    assert report["kind_counts"] == {
        "chat": 1, "managed": 2, "cron": 3, "api": 4, "deferred": 5
    }
    assert report["total"] == 15
    assert report["diagnostic_only"] is True
    assert isinstance(report["sampled_at_unix_ms"], int)
    assert isinstance(report["pid"], int)
    assert set(report) == {
        "status", "sampled_at_unix_ms", "pid", "kind_counts", "total",
        "diagnostic_only"
    }
    rendered = json.dumps(report)
    assert "SECRET_" not in rendered
    assert "session" not in rendered
    assert "model" not in rendered
    assert "prompt" not in rendered
    assert "job_id" not in rendered


@pytest.mark.parametrize("invalid", [True, -1, 10**9])
def test_invalid_live_work_count_fails_closed(invalid):
    runner = _runner(chat=invalid)
    with patch("gateway.run_shutdown.managed_turn_count", return_value=0):
        report = GatewayShutdownMixin._live_work_kind_counts(runner)

    assert report["status"] == "unknown"
    assert report["total"] is None
    assert report["kind_counts"] is None
    assert report["diagnostic_only"] is True


def test_missing_and_broken_work_sources_are_unknown_not_zero():
    runner = _runner()
    runner.adapters = {}
    result = GatewayShutdownMixin._live_work_kind_counts(runner)
    assert result["status"] == "unknown"
    assert result["kind_counts"] is None

    def broken():
        raise RuntimeError("SECRET_EXCEPTION_MESSAGE")

    runner = _runner()
    runner._running_cron_job_count = broken
    with patch("gateway.run_shutdown.managed_turn_count", return_value=0):
        report = GatewayShutdownMixin._live_work_kind_counts(runner)
    assert report["status"] == "unknown"
    assert "SECRET_EXCEPTION" not in json.dumps(report)


def test_api_count_preserves_existing_shutdown_semantics():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._pending_agent_requests = 1
    adapter._inflight_agent_runs = 2
    adapter._active_run_tasks = {
        "pending": SimpleNamespace(done=lambda: False),
        "finished": SimpleNamespace(done=lambda: True),
    }
    assert adapter.active_agent_work_count() == 4
    assert adapter.active_agent_work_count(diagnostic_strict=True) == 4

    adapter._active_run_tasks = None
    assert adapter.active_agent_work_count() == 0
    with pytest.raises((TypeError, AttributeError)):
        adapter.active_agent_work_count(diagnostic_strict=True)


@pytest.mark.asyncio
async def test_local_endpoint_denies_unauthenticated_and_returns_safe_live_counts():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    report = {
        "status": "live",
        "sampled_at_unix_ms": 1790000000000,
        "pid": 321,
        "kind_counts": {"chat": 1, "managed": 0, "cron": 0,
                        "api": 0, "deferred": 0},
        "total": 1,
        "diagnostic_only": True,
    }
    adapter.gateway_runner = SimpleNamespace(
        _live_work_kind_counts=lambda: report,
        secret="SECRET_SHOULD_NEVER_LEAK",
    )
    app = web.Application()
    app.router.add_get("/health/work-types", adapter._handle_health_work_types)

    async with TestClient(TestServer(app, host="127.0.0.1")) as client:
        with patch.object(adapter, "_check_auth",
                          return_value=web.Response(status=401)):
            resp = await client.get("/health/work-types")
            assert resp.status == 401

        with patch.object(adapter, "_check_auth", return_value=None):
            resp = await client.get("/health/work-types")
            assert resp.status == 200
            received = await resp.json()
            assert received == report
            assert "SECRET_" not in json.dumps(received)


@pytest.mark.asyncio
async def test_endpoint_rejects_other_profile_and_external_socket():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = SimpleNamespace(
        _live_work_kind_counts=lambda: {
            "status": "live", "total": 0, "diagnostic_only": True
        }
    )
    request = SimpleNamespace(
        transport=SimpleNamespace(
            get_extra_info=lambda key: ("10.1.1.176", 63422)),
        app={},
    )
    # Exercise handler body without auth wrapper only to verify the
    # *additional* socket restriction. Other test verifies auth wrapper.
    method = adapter._handle_health_work_types.__wrapped__
    result = await method(adapter, request)
    assert result.status == 403

    request.transport.get_extra_info = lambda key: ("127.0.0.1", 1234)
    token = _api_request_profile.set("unrelated-profile")
    try:
        result = await method(adapter, request)
    finally:
        _api_request_profile.reset(token)
    assert result.status == 403


@pytest.mark.asyncio
async def test_endpoint_missing_runner_fails_without_raw_error_text():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = None
    req = SimpleNamespace(
        transport=SimpleNamespace(get_extra_info=lambda key: ("127.0.0.1", 1234)),
        app={},
    )
    resp = await adapter._handle_health_work_types.__wrapped__(adapter, req)
    assert resp.status == 503
    assert json.loads(resp.text) == {"status": "unknown"}


def test_route_registered_exactly_once():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    names = [(method, path) for method, path, handler
             in adapter._http_route_table()
             if path == "/health/work-types"]
    assert names == [("GET", "/health/work-types")]
