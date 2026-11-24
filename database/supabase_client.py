"""Supabase persistence with an explicit local demonstration fallback."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import pandas as pd

from modules.config import normalize_supabase_url, setting
from modules.inventory import REQUIRED_COLUMNS, validate_inventory
from modules.medicine_updates import changed_medicine_fields

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATH = ROOT / "data" / "local_inventory.csv"
LOCAL_ANALYSIS_PATH = ROOT / "data" / "local_analysis_history.json"
LOCAL_CONSUMPTION_PATH = ROOT / "data" / "local_consumption_history.csv"
logger = logging.getLogger(__name__)


class InventoryStore:
    """Persistence service. Configured cloud failures are surfaced, never masked."""

    def __init__(self) -> None:
        raw_url = setting("SUPABASE_URL")
        self.url, self.url_normalized = normalize_supabase_url(raw_url)
        self.key = setting("SUPABASE_KEY")
        self.table = setting("SUPABASE_TABLE", "medicines") or "medicines"
        self.client = None
        self.error: str | None = None
        self.config_warning: str | None = None
        self.reachable: bool | None = None
        if self.url_normalized:
            self.config_warning = "SUPABASE_URL included /rest/v1; removed that suffix because the SDK adds it automatically."
            logger.warning(self.config_warning)
        if bool(self.url) != bool(self.key):
            self.error = "Supabase configuration is incomplete; local demonstration mode is active."
        elif self.url and self.key:
            try:
                from supabase import create_client
                self.client = create_client(self.url, self.key)
                self.client.postgrest.session.event_hooks.setdefault("response", []).append(self._log_http_response)
            except Exception as exc:
                self.error = f"Supabase client initialization failed ({type(exc).__name__})."

    @property
    def cloud(self) -> bool:
        """True only when cloud credentials constructed a client."""
        return self.client is not None

    @property
    def connection_status(self) -> str:
        if not self.cloud:
            return "CLOUD CONFIGURED · connection not verified" if self.url and self.key else "LOCAL DEMONSTRATION MODE"
        if self.reachable is True:
            return "CLOUD · Supabase connected"
        if self.reachable is False:
            return "CLOUD CONFIGURED · Supabase unavailable"
        return "CLOUD CONFIGURED · connection not verified"

    def _safe_log_value(self, value: Any) -> str:
        text = str(value)
        secrets = [self.url, self.key]
        secrets.extend(setting(name) for name in ("SUPABASE_URL", "SUPABASE_KEY", "GEMINI_API_KEY"))
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"(?i)https?://[^\s\"']+", "[URL REDACTED]", text)
        text = re.sub(r"(?i)sb_secret_[A-Za-z0-9_-]+|eyJ[A-Za-z0-9._-]+", "[KEY REDACTED]", text)
        text = re.sub(r"(?i)(apikey|authorization|password|token)\s*[:=]\s*[^,\s]+", r"\1=[REDACTED]", text)
        return text.replace("\r", " ").replace("\n", " ")[:1000]

    def _log_http_response(self, response: Any) -> None:
        if response.is_error:
            logger.error(
                "Supabase HTTP error status=%s method=%s",
                response.status_code,
                response.request.method,
            )

    def _cloud_error(self, action: str, exc: Exception) -> RuntimeError:
        self.reachable = False
        response = getattr(exc, "response", None)
        status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
        code = getattr(exc, "code", None)
        message = getattr(exc, "message", None) or str(exc)
        details = getattr(exc, "details", None)
        hint = getattr(exc, "hint", None)
        body = None
        if response is not None:
            try:
                body = response.text
            except Exception:
                body = None
        logger.error(
            "Supabase operation=%s exception_type=%s http_status=%s code=%s message=%s details=%s hint=%s response=%s",
            action,
            type(exc).__name__,
            status if status is not None else "not exposed by SDK; see HTTP response log",
            self._safe_log_value(code) if code is not None else "",
            self._safe_log_value(message),
            self._safe_log_value(details) if details is not None else "",
            self._safe_log_value(hint) if hint is not None else "",
            self._safe_log_value(body) if body else "",
        )
        self.error = f"Supabase {action} failed ({type(exc).__name__}); cloud data was not saved locally."
        return RuntimeError(self.error)

    @staticmethod
    def _records(data: pd.DataFrame) -> list[dict[str, Any]]:
        clean = validate_inventory(data)
        records = json.loads(clean.to_json(orient="records", date_format="iso"))
        for record in records:
            for field in ("current_stock", "average_daily_consumption", "supplier_lead_time", "safety_stock", "reorder_quantity"):
                record[field] = float(record[field])
        return records

    def check_connection(self) -> bool:
        if not self.client:
            self.reachable = False
            return False
        try:
            self.client.table(self.table).select("medicine_id").limit(1).execute()
            self.reachable = True
            self.error = None
            return True
        except Exception as exc:
            raise self._cloud_error("connection check", exc)

    def load(self) -> pd.DataFrame:
        if self.client:
            try:
                result = self.client.table(self.table).select(",".join(REQUIRED_COLUMNS)).order("medicine_id").execute()
                self.reachable = True
                self.error = None
                rows = result.data or []
                return validate_inventory(pd.DataFrame(rows, columns=REQUIRED_COLUMNS))
            except Exception as exc:
                raise self._cloud_error("read", exc)
        if LOCAL_PATH.exists():
            return validate_inventory(pd.read_csv(LOCAL_PATH))
        sample = ROOT / "data" / "sample_inventory.csv"
        return validate_inventory(pd.read_csv(sample))

    def get_medicine(self, medicine_id: str) -> dict[str, Any] | None:
        """Fetch exactly one medicine by ID from the active store."""
        medicine_id = str(medicine_id).strip()
        if not medicine_id:
            raise ValueError("Enter a medicine ID.")
        if self.client:
            try:
                result = self.client.table(self.table).select(",".join(REQUIRED_COLUMNS)).eq("medicine_id", medicine_id).limit(1).execute()
                self.reachable = True
                self.error = None
                if not result.data:
                    return None
                return self._records(pd.DataFrame(result.data, columns=REQUIRED_COLUMNS))[0]
            except Exception as exc:
                raise self._cloud_error("medicine lookup", exc)
        data = self.load()
        matches = data[data["medicine_id"] == medicine_id]
        if matches.empty:
            return None
        return self._records(matches)[0]

    def update_medicine(self, medicine_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Validate and update only changed fields for one selected medicine."""
        if not isinstance(updates, dict):
            raise ValueError("Update contains an unsupported medicine field.")
        current = self.get_medicine(medicine_id)
        if current is None:
            return None
        changed = changed_medicine_fields(current, updates)
        if not changed:
            return current
        if self.client:
            try:
                result = self.client.table(self.table).update(changed).eq("medicine_id", medicine_id).select(",".join(REQUIRED_COLUMNS)).execute()
                self.reachable = True
                self.error = None
                if not result.data:
                    return None
                return self._records(pd.DataFrame(result.data, columns=REQUIRED_COLUMNS))[0]
            except Exception as exc:
                raise self._cloud_error("medicine update", exc)
        data = self.load()
        mask = data["medicine_id"] == medicine_id
        for field, value in changed.items():
            data.loc[mask, field] = value
        LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        validate_inventory(data).to_csv(LOCAL_PATH, index=False)
        return {**current, **changed}

    def upsert(self, data: pd.DataFrame) -> None:
        """Insert or update one or more medicine records."""
        records = self._records(data)
        if self.client:
            try:
                if records:
                    self.client.table(self.table).upsert(records, on_conflict="medicine_id").execute()
                self.reachable = True
                self.error = None
                return
            except Exception as exc:
                raise self._cloud_error("write", exc)
        current = self.load()
        merged = pd.concat([current[~current["medicine_id"].isin([r["medicine_id"] for r in records])], pd.DataFrame(records)], ignore_index=True)
        LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        validate_inventory(merged).to_csv(LOCAL_PATH, index=False)

    def delete(self, medicine_ids: list[str]) -> None:
        """Delete selected medicine IDs from the configured persistence layer."""
        if not medicine_ids:
            return
        if self.client:
            try:
                for medicine_id in medicine_ids:
                    self.client.table(self.table).delete().eq("medicine_id", medicine_id).execute()
                self.reachable = True
                self.error = None
                return
            except Exception as exc:
                raise self._cloud_error("delete", exc)
        current = self.load()
        remaining = current[~current["medicine_id"].isin(medicine_ids)]
        LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        validate_inventory(remaining).to_csv(LOCAL_PATH, index=False)

    def save(self, data: pd.DataFrame) -> None:
        """Synchronize the inventory snapshot, including cloud deletions."""
        clean = validate_inventory(data)
        current = self.load()
        incoming_ids = set(clean["medicine_id"])
        stale_ids = [value for value in current["medicine_id"] if value not in incoming_ids]
        self.upsert(clean)
        if stale_ids:
            self.delete(stale_ids)
        if not self.client and clean.empty:
            LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            clean.to_csv(LOCAL_PATH, index=False)

    def record_consumption(self, medicine_id: str, quantity: float, consumed_on: str, note: str = "", medicine_name: str = "") -> None:
        if quantity <= 0:
            raise ValueError("Consumption quantity must be greater than zero.")
        entry = {"medicine_id": medicine_id, "medicine_name": medicine_name, "quantity": float(quantity), "consumed_on": consumed_on, "note": note.strip()}
        if self.client:
            try:
                self.client.rpc("record_medicine_consumption", {
                    "p_medicine_id": medicine_id,
                    "p_medicine_name": medicine_name,
                    "p_quantity": float(quantity),
                    "p_consumed_on": consumed_on,
                    "p_note": note.strip(),
                }).execute()
                self.reachable = True
                return
            except Exception as exc:
                raise self._cloud_error("consumption history write", exc)
        current = self.load()
        selected = current.index[current["medicine_id"] == medicine_id]
        if len(selected) != 1:
            raise ValueError("Medicine ID was not found.")
        idx = selected[0]
        if quantity > current.at[idx, "current_stock"]:
            raise ValueError("Consumption quantity exceeds current stock.")
        entry["medicine_name"] = medicine_name or str(current.at[idx, "medicine_name"])
        current.at[idx, "current_stock"] -= quantity
        LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        validate_inventory(current).to_csv(LOCAL_PATH, index=False)
        LOCAL_CONSUMPTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        prior = pd.read_csv(LOCAL_CONSUMPTION_PATH) if LOCAL_CONSUMPTION_PATH.exists() else pd.DataFrame(columns=list(entry))
        pd.concat([prior, pd.DataFrame([entry])], ignore_index=True).to_csv(LOCAL_CONSUMPTION_PATH, index=False)

    def load_consumption_history(self, limit: int = 20) -> list[dict[str, Any]]:
        if self.client:
            try:
                result = self.client.table("consumption_history").select("id,medicine_id,medicine_name,quantity,consumed_on,note,created_at").order("consumed_on", desc=True).limit(limit).execute()
                self.reachable = True
                return result.data or []
            except Exception as exc:
                raise self._cloud_error("consumption history read", exc)
        if not LOCAL_CONSUMPTION_PATH.exists():
            return []
        rows = pd.read_csv(LOCAL_CONSUMPTION_PATH).tail(limit).iloc[::-1]
        return rows.where(pd.notna(rows), None).to_dict(orient="records")

    def record_analysis(self, data: pd.DataFrame, recommendation: str, provider: str) -> None:
        snapshot = json.loads(data.to_json(orient="records", date_format="iso"))
        entry = {"provider": provider, "recommendation": recommendation, "inventory_snapshot": snapshot}
        if self.client:
            try:
                self.client.table("recommendation_history").insert(entry).execute()
                self.reachable = True
                return
            except Exception as exc:
                raise self._cloud_error("recommendation history write", exc)
        LOCAL_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
        history = json.loads(LOCAL_ANALYSIS_PATH.read_text(encoding="utf-8")) if LOCAL_ANALYSIS_PATH.exists() else []
        entry["created_at"] = datetime.now(timezone.utc).isoformat()
        history.insert(0, entry)
        LOCAL_ANALYSIS_PATH.write_text(json.dumps(history[:50], indent=2), encoding="utf-8")

    def load_analysis_history(self, limit: int = 10) -> list[dict[str, Any]]:
        if self.client:
            try:
                result = self.client.table("recommendation_history").select("id,provider,recommendation,created_at").order("created_at", desc=True).limit(limit).execute()
                self.reachable = True
                return result.data or []
            except Exception as exc:
                raise self._cloud_error("recommendation history read", exc)
        if not LOCAL_ANALYSIS_PATH.exists():
            return []
        entries = json.loads(LOCAL_ANALYSIS_PATH.read_text(encoding="utf-8"))[:limit]
        return [{key: entry.get(key) for key in ("provider", "recommendation", "created_at")} for entry in entries]
