"""Cost regressions use local SQLite and injected transports only."""
import copy
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import api_attempt_audit as audit, cost_settings, full_scan, peer_cache
from backend.app import Settings, create_app
from backend.auto_ria import RiaError
from backend.models import Filters, FullScan, SourceBudget
from backend.ria_budget import BudgetLimits
from backend.ria_search import RiaSearch, parse_car
from backend.tests.test_backend import TOKEN, SECRET, headers
from backend.tests.test_ria_search import engine, raw
from backend.tests.test_full_scan import start, finish, runner, provider, status
from backend.tests.test_valuation import sample
from backend.valuation import estimate


def test_shared_completed_scan_keeps_private_ids_and_observation_times(engine):
    calls = []
    first = start(engine)
    done = finish(engine, first, runner(engine, provider(calls, 3)))
    before = len(calls)
    second = start(engine, uid=222)
    assert second != first
    with Session(engine) as db:
        result = full_scan.view(db, 222, second)
        assert full_scan.view(db, 222, first) is None
    assert result["complete"] and result["inspected"] == 3
    assert result["cars"] == done["cars"]
    assert result["requests_used"] == 0
    runner(engine, provider(calls, 3)).tick()
    assert len(calls) == before


def test_concurrent_users_enumerate_once(engine):
    with ThreadPoolExecutor(2) as pool:
        ids = list(pool.map(lambda uid: start(engine, uid=uid), (111, 222)))
    calls = []
    worker = runner(engine, provider(calls, 3))
    for _ in range(4):
        with Session(engine) as db:
            for row in db.scalars(select(FullScan)):
                row.next_run = 0
            db.commit()
        worker.tick()
    with Session(engine) as db:
        assert all(db.get(FullScan, sid).status == "completed" for sid in ids)
    assert sum(path == "search" for path, _ in calls) == 1
    assert sum(path == "info" for path, _ in calls) == 3


def test_fresh_scan_discovers_new_ids_but_reuses_valid_details(engine):
    calls = []
    sid = start(engine)
    finish(engine, sid, runner(engine, provider(calls, 2)))
    start(engine, restart=True)
    result = finish(engine, sid, runner(engine, provider(calls, 3)))
    assert result["discovered"] == 3
    assert sum(path == "search" for path, _ in calls) == 2
    assert sum(path == "info" for path, _ in calls) == 3


def test_scan_and_detail_expiration_are_separate(engine, monkeypatch):
    calls = []
    sid = start(engine)
    done = finish(engine, sid, runner(engine, provider(calls, 2)))
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now + 1000)
    assert start(engine) == sid
    reused = status(engine, sid)
    assert reused["generation"] == done["generation"] and not reused["stale"]
    assert all(car["stale"] for car in reused["cars"])
    monkeypatch.setattr(time, "time", lambda: now + 3601)
    assert status(engine, sid)["stale"]
    start(engine)
    assert status(engine, sid)["generation"] != done["generation"]


def test_fresh_scan_cannot_reuse_old_data_copied_into_a_newer_private_scan(engine):
    calls = []
    first = start(engine)
    worker = runner(engine, provider(calls, 2))
    finish(engine, first, worker)
    fresh = start(engine, uid=222, restart=True)
    copied = start(engine, uid=333)
    with Session(engine) as db:
        assert db.get(FullScan, copied).status == "completed"
    worker.tick()
    with Session(engine) as db:
        assert db.get(FullScan, fresh).status == "completed"
    assert sum(path == "search" for path, _ in calls) == 2


def test_scan_failures_back_off_without_spending_during_wait(engine, monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    calls = []
    def fail(*args):
        calls.append(args)
        raise RiaError("upstream_error")
    sid = start(engine)
    worker = runner(engine, fail)
    for index, wait in enumerate((60, 120, 240), 1):
        worker.tick()
        with Session(engine) as db:
            row = db.get(FullScan, sid)
            assert row.next_run == clock[0] + wait
        worker.tick()
        assert len(calls) == index
        clock[0] += wait


def test_scan_fingerprint_normalizes_text_order_and_duplicates():
    a = Filters(brand=" Volkswagen ", model="GOLF", fuel=["Дизель", "Бензин", "Дизель"])
    b = Filters(brand="volkswagen", model="golf", fuel=["Бензин", "Дизель"], onlyDeals=False)
    assert full_scan.fingerprint(a) == full_scan.fingerprint(b)


def test_running_scan_and_provider_leases_fence_another_process(engine):
    sid = start(engine)
    first = runner(engine, provider([]))
    assert first.claim() == sid
    source = RiaSearch(engine, "fake")
    source.acquire()
    from backend.monitor import Monitor, initialize
    initialize(engine)
    monitor = Monitor(engine, Settings(str(engine.url), TOKEN, SECRET))
    assert monitor.claim()
    script = '''
import sys
from sqlalchemy import create_engine
from backend.full_scan import Scanner
from backend.ria_search import RiaSearch
from backend.auto_ria import RiaError
from backend.monitor import Monitor
from backend.app import Settings
e = create_engine(sys.argv[1])
assert Scanner(e, "fake").claim() is None
assert not Monitor(e, Settings(sys.argv[1], "fake", "x" * 32)).claim()
try:
    RiaSearch(e, "fake").acquire()
except RiaError as exc:
    assert str(exc) == "busy"
else:
    raise AssertionError("duplicate provider lease")
'''
    try:
        result = subprocess.run([sys.executable, "-c", script, str(engine.url)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
    finally:
        source.release()
        monitor.release("idle")


def test_paused_scan_progress_can_be_resumed_by_another_user(engine):
    calls = []
    first = start(engine)
    worker = runner(engine, provider(calls, 60))
    worker.tick()
    with Session(engine) as db:
        assert db.get(FullScan, first).discovered == 60
        assert full_scan.change(db, 111, first, False)
        db.commit()
    second = start(engine, uid=222)
    for _ in range(5):
        worker.tick()
        with Session(engine) as db:
            row = db.get(FullScan, second)
            if row.status == "completed":
                break
            row.next_run = 0
            db.commit()
    with Session(engine) as db:
        assert db.get(FullScan, second).status == "completed"
        assert db.get(FullScan, first).status == "paused"
        full_scan.change(db, 111, first, True)
        db.commit()
    worker.tick()
    assert status(engine, first)["complete"]
    assert len(status(engine, first)["cars"]) == 60
    assert sum(path == "search" for path, _ in calls) == 2
    assert sum(path == "info" for path, _ in calls) == 60


def test_aged_peer_observations_reused_without_refreshing_candidate_policy(engine, monkeypatch):
    candidate, peers = sample()
    now = time.time()
    for peer in peers:
        peer["observed_at"] = now - 1800
        peer_cache.observe(engine, peer, create=True)
    cached = peer_cache.candidates(engine, candidate)
    rating = estimate(candidate, cached, now=now)
    assert rating["comparables"] == 5 and rating["market"] is not None
    assert rating["valuation_evidence"]["confidence"] == "aged_observations"
    assert estimate({**candidate, "observed_at": now - 901}, cached, now=now)["market"] is None
    monkeypatch.setattr(time, "time", lambda: now + 1801)
    assert not peer_cache.candidates(engine, {**candidate, "observed_at": now + 1801})


def test_peer_lookup_never_refreshes_observation_or_reuses_tombstone(engine):
    car = parse_car(raw(), "123")
    peer_cache.observe(engine, car, create=True)
    source = RiaSearch(engine, "fake", lambda *args: pytest.fail("unexpected HTTP"))
    assert source.peer_car("123")["observed_at"] == car["observed_at"]
    peer_cache.invalidate(engine, "123")
    with pytest.raises(RiaError, match="listing_unavailable"):
        source.peer_car("123")


def test_newer_manual_details_invalidate_evidence_without_entering_peer_pool(engine, monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    source = RiaSearch(engine, "fake", lambda *args: raw(USD=20000))
    original = parse_car(raw(USD=10000), "123")
    clock[0] += 1
    source.acquire()
    try:
        source.car("123")
    finally:
        source.release()
    with Session(engine) as db:
        assert not peer_cache.evidence_current(db, {"candidate": original, "peers": []})
    assert peer_cache.observation(engine, "123") is None


def test_normalized_concurrent_requests_spend_one_reservation(engine):
    calls = []
    source = RiaSearch(engine, "fake", lambda *args: calls.append(args) or {"ok": True})
    source.acquire()
    try:
        children = [source.fork(), source.fork()]
        with ThreadPoolExecutor(2) as pool:
            result = list(pool.map(lambda pair: pair[0].request("search", {"page": pair[1]}, lambda data: data),
                                   zip(children, (1, "1"))))
        assert result == [{"ok": True}, {"ok": True}] and len(calls) == 1
        with Session(engine) as db:
            assert db.get(SourceBudget, "auto_ria").total == 3
    finally:
        source.release()


def test_background_cannot_spend_interactive_reserve(engine):
    now = time.time()
    with Session(engine) as db:
        db.get(SourceBudget, "auto_ria").calls = [now] * 9
        db.commit()
    source = RiaSearch(engine, "fake", lambda *args: {"ok": True}, BudgetLimits(10, 20, 100))
    source.acquire()
    try:
        source.request_policy = lambda *args: audit.authorization("publication_search", "group")
        with pytest.raises(RiaError, match="quota_exceeded"):
            source.request("search", {}, lambda data: data)
        source.request_policy = lambda *args: audit.authorization("authenticated_search", "group")
        assert source.request("search", {}, lambda data: data) == {"ok": True}
        # Cache hits still succeed at the hard cap without another reservation.
        assert source.request("search", {}, lambda data: data) == {"ok": True}
        with pytest.raises(RiaError, match="quota_exceeded"):
            source.request("search", {"page": 2}, lambda data: data)
    finally:
        source.release()


def test_concurrent_comparisons_share_search_and_each_peer_detail(engine):
    calls = []
    def fetch(key, path, params):
        calls.append(path)
        if path == "search":
            return {"result": {"search_result": {"ids": list(range(90000, 90005)), "count": 5}}}
        return raw(params["auto_id"])
    source = RiaSearch(engine, "fake", fetch)
    source.acquire()
    try:
        with ThreadPoolExecutor(2) as pool:
            batches = list(pool.map(lambda child: child.comparisons(parse_car(raw(), "123")),
                                    [source.fork(), source.fork()]))
        assert all(len(batch) == 5 for batch in batches)
        assert calls.count("search") == 1 and calls.count("info") == 5
    finally:
        source.release()


def test_request_specific_ttl_does_not_mistake_old_response_for_fresh(engine, monkeypatch):
    calls, now = [], time.time()
    source = RiaSearch(engine, "fake", lambda *args: calls.append(args) or {"ok": True})
    source.acquire()
    source.request("search", {}, lambda raw: raw, ttl=3600)
    source.release()
    monkeypatch.setattr(time, "time", lambda: now + 1000)
    source.acquire()
    try:
        source.request("search", {}, lambda raw: raw, ttl=900)
        assert len(calls) == 2
    finally:
        source.release()


def test_incomplete_search_response_is_not_cached(engine):
    from backend.ria_search import parse_ids
    calls = []
    def fetch(*args):
        calls.append(args)
        return {"result": {"search_result": {"ids": ["123"], "count": 3}}}
    source = RiaSearch(engine, "fake", fetch)
    source.acquire()
    try:
        for _ in range(2):
            source.request("search", {"countpage": 3, "page": 0}, parse_ids)
        assert len(calls) == 2
    finally:
        source.release()


def test_missing_provider_valuation_is_cached_without_crash_or_repeat_cost(engine):
    calls = []
    source = RiaSearch(engine, "fake", lambda *args: calls.append(args) or {})
    source.acquire()
    try:
        for _ in range(2):
            assert source.request("ai-avarage-price", {"auto_id": "123"}, lambda data: None) is None
        assert len(calls) == 1
    finally:
        source.release()


def test_usage_report_counts_reservations_transport_and_repeats_separately(engine):
    def fetch(*args):
        audit.current_observer()({"event": "transport_started"})
        return {"ok": True}
    source = RiaSearch(engine, "fake", fetch)
    source.request_policy = lambda *args: audit.authorization("authenticated_search", "private-user-context")
    source.acquire()
    try:
        source.request("search", {}, lambda data: data)
        source.request("search", {}, lambda data: data)  # no reservation
        source.request("search", {}, lambda data: data, force=True)
    finally:
        source.release()
    with Session(engine) as db:
        report = audit.usage_report(db, time.time() - 60, time.time() + 1)
    assert report["reservations"] == report["transport_observed_attempts"] == 2
    assert report["initiators"] == {"user": 2, "background": 0, "unknown": 0}
    assert report["forced_reservations"] == 1
    assert report["call_relations"]["repeat_after_success"] == 1
    assert sum(bucket["count"] for bucket in report["per_hour"]["http_attempts"]) == 2
    assert "private-user-context" not in str(report)


def test_usage_report_requires_verified_admin_and_never_claims_provider_billing(engine, monkeypatch):
    monkeypatch.setenv("SUBSCRIPTION_EXPECTED_ADMIN_ID", "111")
    settings = Settings(str(engine.url), TOKEN, SECRET, admin_telegram_id=111)
    client = TestClient(create_app(settings, engine))
    url = "/api/admin/api-usage?after=0&before=3600"
    assert client.get(url).status_code == 401
    assert client.get(url, headers=headers(222)).status_code == 403
    response = client.get(url, headers=headers())
    assert response.status_code == 200
    report = response.json()
    assert report["provider_charged_units"] is None
    assert report["cache_misses"] is None and report["cache_hit_calls"] is None
    assert client.get("/api/admin/api-usage?after=0&before=9999999", headers=headers()).status_code == 422
    monkeypatch.setenv("SUBSCRIPTION_EXPECTED_ADMIN_ID", "222")
    assert client.get(url, headers=headers()).status_code == 403


@pytest.mark.parametrize("name,value", [("COMPLETED_SCAN_REUSE_SECONDS", "3599"),
    ("RIA_PEER_CACHE_TTL_SECONDS", "3601"), ("RIA_ACTIVE_WINDOW_INTERVAL_SECONDS", "0"),
    ("RIA_ACTIVE_WINDOW_ENABLED", "maybe")])
def test_invalid_cost_configuration_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        cost_settings.validate()
