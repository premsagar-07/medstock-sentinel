from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from io import StringIO
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import database.supabase_client as db
from database.supabase_client import InventoryStore
from modules.ai_recommendations import (
    build_inventory_context,
    create_gemini_client,
    get_cached_recommendations,
    get_recommendations,
    rule_based_recommendations,
)
from modules.inventory import InventoryValidationError, add_inventory_intelligence, load_inventory_csv
from modules.medicine_updates import changed_medicine_fields, discard_pending_update


def medicine(**updates):
    item = {
        "medicine_id": "T001", "medicine_name": "Synthetic Paracetamol", "category": "Analgesic",
        "current_stock": 50, "average_daily_consumption": 10, "supplier_lead_time": 3,
        "safety_stock": 10, "reorder_quantity": 100, "manufacturing_date": date(2025, 1, 1),
        "expiry_date": date(2027, 1, 1), "supplier_name": "Demo Supplier",
    }
    item.update(updates)
    return item


class FakeQuery:
    def __init__(self, client, table):
        self.client, self.table = client, table
        self.operation, self.payload, self.filters = "select", None, []
        self.limit_value = None

    def select(self, *_args):
        if self.operation == "select": self.operation = "select"
        return self
    def order(self, *_args, **_kwargs): return self
    def limit(self, value): self.limit_value = value; return self
    def upsert(self, rows, **_kwargs): self.operation, self.payload = "upsert", rows; return self
    def insert(self, row): self.operation, self.payload = "insert", row; return self
    def update(self, values): self.operation, self.payload = "update", values; self.client.update_payloads.append(dict(values)); return self
    def delete(self): self.operation = "delete"; return self
    def eq(self, key, value): self.filters.append((key, value)); return self

    def execute(self):
        if self.client.fail_select and self.operation == "select":
            raise ConnectionError("private test transport failure")
        rows = self.client.tables.setdefault(self.table, [])
        if self.operation == "select":
            result = [row for row in rows if all(row.get(k) == v for k, v in self.filters)]
            result = result[:self.limit_value] if self.limit_value else result
            return SimpleNamespace(data=result)
        if self.operation == "upsert":
            values = self.payload if isinstance(self.payload, list) else [self.payload]
            for item in values:
                rows[:] = [old for old in rows if old.get("medicine_id") != item.get("medicine_id")]
                rows.append(dict(item))
            return SimpleNamespace(data=values)
        if self.operation == "delete":
            rows[:] = [row for row in rows if not all(row.get(k) == v for k, v in self.filters)]
            return SimpleNamespace(data=[])
        if self.operation == "insert":
            rows.append(dict(self.payload))
            return SimpleNamespace(data=[self.payload])
        if self.operation == "update":
            matched = []
            for row in rows:
                if all(row.get(k) == v for k, v in self.filters):
                    row.update(self.payload)
                    matched.append(dict(row))
            return SimpleNamespace(data=matched)
        return SimpleNamespace(data=[])


class FakeSupabase:
    def __init__(self):
        self.tables = {}
        self.fail_select = False
        self.update_payloads = []

    def table(self, name): return FakeQuery(self, name)
    def rpc(self, name, params):
        assert name == "record_medicine_consumption"
        row = next(row for row in self.tables["medicines"] if row["medicine_id"] == params["p_medicine_id"])
        row["current_stock"] -= params["p_quantity"]
        self.tables.setdefault("consumption_history", []).append(params)
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=[]))


def isolated_store(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setattr(db, "LOCAL_PATH", tmp_path / "inventory.csv")
    monkeypatch.setattr(db, "LOCAL_ANALYSIS_PATH", tmp_path / "analysis.json")
    monkeypatch.setattr(db, "LOCAL_CONSUMPTION_PATH", tmp_path / "consumption.csv")
    return InventoryStore()


def test_csv_import_and_validation():
    source = StringIO("medicine_id,medicine_name,category,current_stock,average_daily_consumption,supplier_lead_time,safety_stock,reorder_quantity,manufacturing_date,expiry_date,supplier_name\nT1,Example,Analgesic,5,1,2,1,10,2025-01-01,2027-01-01,Supplier\n")
    assert load_inventory_csv(source).iloc[0].medicine_id == "T1"
    with pytest.raises(InventoryValidationError):
        load_inventory_csv(StringIO("not,an,inventory\n1,2,3\n"))


def test_local_fallback_inventory_consumption_and_analysis_history(monkeypatch, tmp_path):
    store = isolated_store(monkeypatch, tmp_path)
    data = store.load()
    assert len(data) == 24
    assert not store.cloud
    data.loc[data["medicine_id"] == "MED001", "current_stock"] = 419
    store.save(data)
    assert store.load().set_index("medicine_id").loc["MED001", "current_stock"] == 419
    store.record_consumption("MED001", 2, "2026-01-01", "demo use")
    assert store.load().set_index("medicine_id").loc["MED001", "current_stock"] == 417
    assert store.load_consumption_history(1)[0]["medicine_id"] == "MED001"
    calculated = add_inventory_intelligence(store.load(), today=date(2026, 1, 1))
    store.record_analysis(calculated, "Synthetic test summary", "rules")
    history = store.load_analysis_history()
    assert history[0]["provider"] == "rules"
    assert json.loads((tmp_path / "analysis.json").read_text())[0]["recommendation"] == "Synthetic test summary"


def test_cloud_crud_and_history_use_supabase(monkeypatch, tmp_path):
    store = isolated_store(monkeypatch, tmp_path)
    fake = FakeSupabase()
    store.client = fake
    assert store.check_connection()
    original = pd.DataFrame([medicine(), medicine(medicine_id="T002", medicine_name="Synthetic Ibuprofen")])
    store.save(original)
    assert {r["medicine_id"] for r in fake.tables["medicines"]} == {"T001", "T002"}
    store.save(pd.DataFrame([medicine(current_stock=35)]))
    assert len(fake.tables["medicines"]) == 1
    assert fake.tables["medicines"][0]["current_stock"] == 35
    store.record_consumption("T001", 3, "2026-02-01")
    assert fake.tables["medicines"][0]["current_stock"] == 32
    assert len(fake.tables["consumption_history"]) == 1
    store.record_analysis(add_inventory_intelligence(pd.DataFrame([medicine()])), "Grounded synthetic analysis", "gemini")
    assert store.load_analysis_history()[0]["provider"] == "gemini"
    store.delete(["T001"])
    assert fake.tables["medicines"] == []
    assert not (tmp_path / "inventory.csv").exists()


def test_id_lookup_existing_and_missing_medicine(monkeypatch, tmp_path):
    store = isolated_store(monkeypatch, tmp_path)
    fake = FakeSupabase()
    store.client = fake
    store.upsert(pd.DataFrame([medicine()]))
    found = store.get_medicine("T001")
    assert found["medicine_name"] == "Synthetic Paracetamol"
    assert store.get_medicine("MISSING-ID") is None


def test_update_stock_only_sends_changed_stock_field(monkeypatch, tmp_path):
    store = isolated_store(monkeypatch, tmp_path)
    fake = FakeSupabase()
    store.client = fake
    store.upsert(pd.DataFrame([medicine()]))
    updated = store.update_medicine("T001", {"current_stock": 12})
    assert updated["current_stock"] == 12
    assert fake.update_payloads[-1] == {"current_stock": 12}
    saved = store.get_medicine("T001")
    assert saved["medicine_name"] == "Synthetic Paracetamol"
    assert saved["category"] == "Analgesic"


def test_update_multiple_fields_and_readback_after_store_restart(monkeypatch, tmp_path):
    store = isolated_store(monkeypatch, tmp_path)
    fake = FakeSupabase()
    store.client = fake
    store.upsert(pd.DataFrame([medicine()]))
    updated = store.update_medicine("T001", {
        "medicine_name": "Synthetic Paracetamol Updated",
        "category": "Pain relief",
        "current_stock": 22,
        "supplier_name": "Updated Synthetic Supplier",
    })
    assert updated["medicine_name"] == "Synthetic Paracetamol Updated"
    assert set(fake.update_payloads[-1]) == {"medicine_name", "category", "current_stock", "supplier_name"}
    restarted_store = isolated_store(monkeypatch, tmp_path)
    restarted_store.client = fake
    persisted = restarted_store.get_medicine("T001")
    assert persisted["current_stock"] == 22
    assert persisted["supplier_name"] == "Updated Synthetic Supplier"


def test_cancel_discards_loaded_update_without_database_changes():
    state = {"update_record": medicine(), "update_generation": 4}
    discard_pending_update(state)
    assert state["update_record"] is None
    assert state["update_generation"] == 5


def test_update_validation_and_no_change_detection():
    current = {key: value.isoformat() if hasattr(value, "isoformat") else value for key, value in medicine().items()}
    assert changed_medicine_fields(current, {"current_stock": 50}) == {}
    assert changed_medicine_fields(current, {"expiry_date": date(2027, 1, 1)}) == {}
    with pytest.raises(InventoryValidationError):
        changed_medicine_fields(current, {"current_stock": -2})
    with pytest.raises(ValueError):
        changed_medicine_fields(current, {"medicine_id": "OTHER"})


def test_cloud_read_failure_is_not_silently_saved_locally(monkeypatch, tmp_path):
    store = isolated_store(monkeypatch, tmp_path)
    store.client = FakeSupabase()
    store.client.fail_select = True
    with pytest.raises(RuntimeError, match="cloud data was not saved locally"):
        store.load()
    assert store.reachable is False
    assert not (tmp_path / "inventory.csv").exists()


def test_partial_cloud_configuration_is_explicit_local_mode(monkeypatch, tmp_path):
    store = isolated_store(monkeypatch, tmp_path)
    monkeypatch.setenv("SUPABASE_URL", "https://example.invalid")
    store = InventoryStore()
    assert not store.cloud
    assert "incomplete" in store.error
    assert "LOCAL DEMONSTRATION MODE" in store.connection_status


def test_supabase_rest_suffix_is_normalized_before_client_creation(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://unit-test.supabase.co/rest/v1/")
    monkeypatch.setenv("SUPABASE_KEY", "test-only-secret-key")
    store = InventoryStore()
    assert store.url == "https://unit-test.supabase.co"
    assert store.url_normalized
    assert store.client is not None
    assert store.config_warning
    assert store.client.postgrest.session.event_hooks["response"]


def test_cloud_diagnostics_log_status_and_error_without_secrets(monkeypatch, caplog):
    import logging

    raw_url = "https://unit-test.supabase.co/rest/v1"
    api_key = "sb_secret_test-only-never-log"
    monkeypatch.setenv("SUPABASE_URL", raw_url)
    monkeypatch.setenv("SUPABASE_KEY", api_key)
    store = InventoryStore()

    error = RuntimeError("Invalid path at " + raw_url + " using " + api_key)
    error.code = "PGRST125"
    error.message = "Invalid path specified in request URL: " + raw_url
    error.details = "test detail"
    error.hint = None

    class Response:
        status_code = 404
        is_error = True
        request = SimpleNamespace(method="GET")

    with caplog.at_level(logging.ERROR, logger="database.supabase_client"):
        store._log_http_response(Response())
        store._cloud_error("read", error)
    assert "status=404" in caplog.text
    assert "PGRST125" in caplog.text
    assert "Invalid path specified" in caplog.text
    assert raw_url not in caplog.text
    assert api_key not in caplog.text


def test_gemini_grounded_success_empty_and_failure_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-only-key")
    monkeypatch.setenv("GEMINI_MODEL", "mock-model")
    calculated = add_inventory_intelligence(pd.DataFrame([medicine(current_stock=10)]), today=date(2026, 1, 1))
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text="T001 is critical; review replenishment.")
    client_factory = Mock(return_value=client)
    monkeypatch.setattr("modules.ai_recommendations.create_gemini_client", client_factory)
    answer, used, notice, duration = get_recommendations(calculated)
    assert used and notice is None and "T001" in answer
    assert duration >= 0
    client_factory.assert_called_once_with("mock-only-key")
    assert client.models.generate_content.call_args.kwargs["model"] == "mock-model"
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert '"stockout_risk": "Critical"' in prompt
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens == 450
    assert "two or three short sentences" in prompt and "four to six short bullets" in prompt

    client.models.generate_content.return_value = SimpleNamespace(text="  ")
    answer, used, notice, _duration = get_recommendations(calculated)
    assert not used and "Rule-based inventory summary" in answer and "validated" in notice

    client.models.generate_content.return_value = SimpleNamespace(text="T001 should be replenished in 2 days.")
    answer, used, notice, _duration = get_recommendations(calculated)
    assert not used and "Rule-based inventory summary" in answer and "validated" in notice

    monkeypatch.setattr("modules.ai_recommendations.create_gemini_client", Mock(side_effect=TimeoutError()))
    answer, used, notice, duration = get_recommendations(calculated)
    assert not used and "Rule-based inventory summary" in answer and "Could not reach Gemini" in notice
    assert duration >= 0


def test_gemini_error_diagnostic_includes_status_and_redacts_secret(monkeypatch, caplog):
    api_key = "mock-secret-value"
    monkeypatch.setattr(
        "modules.ai_recommendations.setting",
        lambda name, default="": api_key if name == "GEMINI_API_KEY" else default,
    )

    class ForbiddenError(Exception):
        response = SimpleNamespace(status_code=403)

        def __str__(self):
            return f"Access denied for api_key={api_key}"

    monkeypatch.setattr("modules.ai_recommendations.create_gemini_client", Mock(side_effect=ForbiddenError()))
    with caplog.at_level("ERROR", logger="modules.ai_recommendations"):
        result, used, notice, _duration = get_recommendations(add_inventory_intelligence(pd.DataFrame([medicine()])))
    assert not used and "Rule-based inventory summary" in result
    assert "403" in notice and "GEMINI_API_KEY" in notice
    assert "ForbiddenError" in caplog.text and "403" in caplog.text
    assert "Access denied" in caplog.text
    assert api_key not in notice and api_key not in caplog.text


def test_gemini_uses_default_model_when_unset(monkeypatch):
    monkeypatch.setattr(
        "modules.ai_recommendations.setting",
        lambda name, default="": "mock-api-key" if name == "GEMINI_API_KEY" else default,
    )
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text="Review inventory trends and supplier coordination.")
    monkeypatch.setattr("modules.ai_recommendations.create_gemini_client", lambda _key: client)
    answer, used, notice, _duration = get_recommendations(add_inventory_intelligence(pd.DataFrame([medicine()])))
    assert used and notice is None and answer
    assert client.models.generate_content.call_args.kwargs["model"] == "gemini-2.5-flash"


def test_missing_gemini_key_uses_rule_based_fallback(monkeypatch):
    monkeypatch.setattr("modules.ai_recommendations.setting", lambda name, default="": default)
    answer, used, notice, duration = get_recommendations(add_inventory_intelligence(pd.DataFrame([medicine()])))
    assert not used
    assert "Rule-based inventory summary" in answer
    assert "not configured" in notice
    assert duration == 0


def test_gemini_sdk_client_uses_bounded_http_timeout(monkeypatch):
    factory = Mock(return_value=object())
    monkeypatch.setattr("google.genai.Client", factory)
    create_gemini_client("mock-only-key")
    options = factory.call_args.kwargs["http_options"]
    assert options.timeout == 30_000


def test_gemini_rate_limit_uses_clear_rule_fallback(monkeypatch):
    monkeypatch.setattr("modules.ai_recommendations.setting", lambda name, default="": "mock-api-key" if name == "GEMINI_API_KEY" else default)

    class ResourceExhausted(Exception):
        code = 429

    monkeypatch.setattr("modules.ai_recommendations.create_gemini_client", Mock(side_effect=ResourceExhausted()))
    result, used_gemini, notice, _duration = get_recommendations(add_inventory_intelligence(pd.DataFrame([medicine()])))
    assert not used_gemini
    assert "Rule-based inventory summary" in result
    assert "rate limit" in notice


def test_gemini_503_keeps_python_recommendations_available(monkeypatch):
    calculated = add_inventory_intelligence(
        pd.DataFrame([medicine(current_stock=5)]), today=date(2026, 1, 1)
    )
    python_recommendations = rule_based_recommendations(calculated)
    request_started = Event()
    release_request = Event()

    class ServiceUnavailable(Exception):
        code = 503

    def delayed_503(**_kwargs):
        request_started.set()
        assert release_request.wait(timeout=3)
        raise ServiceUnavailable("provider temporarily overloaded")

    client = Mock()
    client.models.generate_content.side_effect = delayed_503
    monkeypatch.setattr("modules.ai_recommendations.create_gemini_client", lambda _key: client)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            get_cached_recommendations,
            calculated,
            None,
            api_key="mock-only-key",
            model="mock-model",
        )
        assert request_started.wait(timeout=2)
        # Python's result is already available while the simulated API call is still pending.
        assert python_recommendations.startswith("Rule-based inventory summary")
        assert not future.done()
        release_request.set()
        result, used, notice, _cache_hit, cache_entry, _duration, _age = future.result(timeout=3)

    assert result == python_recommendations
    assert not used and cache_entry is None
    assert "503" in notice and "overloaded" in notice


def test_gemini_context_contains_only_summary_and_at_risk_medicines():
    data = add_inventory_intelligence(pd.DataFrame([
        medicine(medicine_id="RISK001", current_stock=5),
        medicine(medicine_id="HEALTHY002", current_stock=500),
    ]), today=date(2026, 1, 1))
    context = build_inventory_context(data)
    included_ids = {item["medicine_id"] for item in context["at_risk_medicines"]}
    assert "RISK001" in included_ids
    assert "HEALTHY002" not in included_ids
    assert context["python_calculated_summary"]["total_medicines"] == 2


def test_gemini_context_keeps_expiry_alerts_even_with_many_stock_alerts():
    records = [
        medicine(medicine_id=f"RISK{i:03}", current_stock=1, expiry_date=date(2027, 1, 1))
        for i in range(30)
    ]
    records.append(medicine(medicine_id="EXPIRY001", current_stock=500, expiry_date=date(2026, 1, 10)))
    calculated = add_inventory_intelligence(pd.DataFrame(records), today=date(2026, 1, 1))
    context = build_inventory_context(calculated)
    included_ids = {item["medicine_id"] for item in context["at_risk_medicines"]}
    assert "RISK000" in included_ids
    assert "EXPIRY001" in included_ids
    assert len(included_ids) <= 24


def test_successful_gemini_response_is_cached_per_inventory_session(monkeypatch):
    monkeypatch.setattr("modules.ai_recommendations.setting", lambda name, default="": "mock-api-key" if name == "GEMINI_API_KEY" else default)
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text="T001 requires a procurement review.")
    factory = Mock(return_value=client)
    monkeypatch.setattr("modules.ai_recommendations.create_gemini_client", factory)
    data = add_inventory_intelligence(pd.DataFrame([
        medicine(current_stock=10),
        medicine(medicine_id="HEALTHY002", current_stock=500),
    ]), today=date(2026, 1, 1))
    first = get_cached_recommendations(data, now=1000)
    second = get_cached_recommendations(data, first[4], now=1100)
    assert first[1] and not first[3]
    assert second[1] and second[3]
    assert first[5] >= 0 and second[5] == 0 and second[6] == 100
    assert first[4]["request_duration_seconds"] == first[5]
    assert factory.call_count == 1
    assert "mock-api-key" not in str(first[4])

    # A healthy row is omitted from Gemini's prompt but remains part of the cache fingerprint.
    changed = add_inventory_intelligence(pd.DataFrame([
        medicine(current_stock=10),
        medicine(medicine_id="HEALTHY002", current_stock=501),
    ]), today=date(2026, 1, 1))
    third = get_cached_recommendations(changed, second[4], now=1150)
    assert not third[3]
    assert factory.call_count == 2

    expired = get_cached_recommendations(changed, third[4], now=1150 + 12 * 60 + 1)
    assert not expired[3]
    assert factory.call_count == 3
