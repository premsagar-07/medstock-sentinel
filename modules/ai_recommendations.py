"""Grounded Gemini inventory explanations with rule-based fallback."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

import pandas as pd

from modules.config import setting

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 12 * 60
API_TIMEOUT_MS = 30_000
MAX_OUTPUT_TOKENS = 450
AI_DETAIL_COLUMNS = [
    "medicine_id", "medicine_name", "current_stock", "average_daily_consumption",
    "supplier_lead_time", "safety_stock", "reorder_point", "days_until_stockout",
    "stockout_risk", "expiry_date", "days_to_expiry", "expiry_risk",
    "suggested_reorder_quantity",
]


def rule_based_recommendations(data: pd.DataFrame) -> str:
    """Deterministic inventory summary that needs no Gemini credentials."""
    if data.empty:
        return "No medicine records are available. Add inventory to receive recommendations."
    high = data[data["stockout_risk"].isin(["Critical", "High", "Moderate"])].sort_values(
        ["stockout_risk", "days_until_stockout"], na_position="last"
    )
    expiring = data[data["expiry_risk"].isin(["Expired", "Critical", "Soon"])].sort_values("days_to_expiry")
    lines = ["Rule-based inventory summary (calculated from the current inventory):"]
    if high.empty:
        lines.append("• No medicines currently meet the replenishment risk thresholds.")
    else:
        lines.append("• Prioritise replenishment:")
        for row in high.head(8).itertuples():
            cover = "unknown cover" if pd.isna(row.days_until_stockout) else f"about {int(row.days_until_stockout)} days of cover"
            lines.append(f"  - {row.medicine_name} ({row.medicine_id}): {row.stockout_risk} risk, {cover}; suggested order {int(row.suggested_reorder_quantity)} units. Lead time {int(row.supplier_lead_time)} days.")
    if not expiring.empty:
        lines.append("• Review near-expiry stock and follow local rotation/disposal policy:")
        for row in expiring.head(5).itertuples():
            lines.append(f"  - {row.medicine_name} ({row.medicine_id}): {row.expiry_risk}, {int(row.days_to_expiry)} days to expiry.")
    return "\n".join(lines)


def create_gemini_client(api_key: str):
    """Construct the official Google GenAI SDK client (separated for tests)."""
    from google import genai
    from google.genai import types
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=API_TIMEOUT_MS))


def contains_unverified_numeric_claim(answer: str, medicine_ids: list[str]) -> bool:
    """Reject generated quantities/dates so Python remains the numeric authority."""
    prose = answer
    for medicine_id in sorted(medicine_ids, key=len, reverse=True):
        prose = re.sub(re.escape(medicine_id), "MEDICINE_ID", prose, flags=re.IGNORECASE)
    number_words = (
        r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"dozen|couple|few|several|half|quarter|double|twice|hundred|thousand"
    )
    return bool(re.search(r"\d|\b(?:" + number_words + r")\b", prose, flags=re.IGNORECASE))


def build_inventory_context(data: pd.DataFrame) -> dict[str, Any]:
    """Build compact aggregates and the highest priority stock/expiry alerts."""
    missing = [column for column in AI_DETAIL_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Inventory intelligence fields are missing: " + ", ".join(missing))
    stock_alert = data["stockout_risk"].isin(["Critical", "High"])
    expiry_alert = data["expiry_risk"].isin(["Expired", "Critical", "Soon"])
    summary = {
        "total_medicines": int(len(data)),
        "total_inventory_units": float(data["current_stock"].sum()),
        "low_stock_medicines": int(stock_alert.sum()),
        "critical_stock_alerts": int(data["stockout_risk"].eq("Critical").sum()),
        "expiring_medicines": int(expiry_alert.sum()),
        "expired_medicines": int(data["expiry_risk"].eq("Expired").sum()),
    }
    detail_columns = [
        "medicine_id", "medicine_name", "current_stock", "average_daily_consumption",
        "supplier_lead_time", "reorder_point", "days_until_stockout", "stockout_risk",
        "expiry_date", "days_to_expiry", "expiry_risk",
    ]
    priority = {"Critical": 0, "High": 1}
    stock_rows = data.loc[stock_alert, detail_columns].copy()
    stock_rows["_stock_priority"] = stock_rows["stockout_risk"].map(priority).fillna(2)
    stock_rows = stock_rows.sort_values(
        ["_stock_priority", "days_until_stockout"], na_position="last"
    ).head(12)
    expiry_rows = data.loc[expiry_alert, detail_columns].copy()
    expiry_rows["_stock_priority"] = expiry_rows["stockout_risk"].map(priority).fillna(2)
    expiry_rows = expiry_rows.sort_values("days_to_expiry", na_position="last").head(12)
    attention = pd.concat([stock_rows, expiry_rows]).drop_duplicates("medicine_id", keep="first")
    attention = attention.sort_values(["_stock_priority", "days_to_expiry"], na_position="last")
    details = json.loads(attention.drop(columns="_stock_priority").to_json(orient="records", date_format="iso"))
    return {"python_calculated_summary": summary, "at_risk_medicines": details}


def _exception_status(exc: Exception) -> str | None:
    """Extract a safe HTTP status or provider code from SDK error shapes."""
    response = getattr(exc, "response", None)
    status = (
        getattr(exc, "status_code", None)
        or getattr(response, "status_code", None)
        or getattr(exc, "code", None)
    )
    return str(status)[:40] if status is not None else None


def _safe_error_message(exc: Exception, api_key: str) -> str:
    """Sanitize SDK error text before writing it to logs or showing it in the UI."""
    message = str(exc).strip() or "No error message was provided by the SDK."
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(r"(?i)(api[_-]?key|access[_-]?token|password)(\s*[=:]\s*)[^&\s,;]+", r"\1\2[REDACTED]", message)
    message = re.sub(r"https?://\S+", lambda match: re.sub(r"(?i)(key|token)=([^&\s]+)", r"\1=[REDACTED]", match.group(0)), message)
    return message[:600]


def _friendly_fallback_notice(exc: Exception, *, status: str | None, safe_message: str) -> str:
    name = type(exc).__name__
    try:
        numeric_status = int(status) if status and status.isdigit() else None
    except ValueError:
        numeric_status = None
    if numeric_status == 429 or name.lower() in {"resourceexhausted", "toomanyrequests", "ratelimiterror"}:
        guidance = "Gemini quota or rate limit reached; check project quota and billing."
    elif numeric_status in (401, 403):
        guidance = "Gemini rejected the credential; verify GEMINI_API_KEY and that the key can use the Generative Language API."
    elif numeric_status == 404:
        guidance = "Gemini could not find the selected model; verify GEMINI_MODEL is available to this API key."
    elif numeric_status == 400:
        guidance = "Gemini rejected the request; verify the selected model and request configuration."
    elif numeric_status in (500, 502, 503, 504):
        guidance = "Gemini service is temporarily overloaded or unavailable; retry after a short wait."
    elif name in {"ConnectError", "ConnectTimeout", "ReadTimeout", "TimeoutError"}:
        guidance = "Could not reach Gemini; check outbound network access, DNS, proxy, and timeout settings."
    elif isinstance(exc, ValueError) or name == "InvalidGeminiResponse":
        guidance = "Gemini returned a response that could not be validated."
    else:
        guidance = "Gemini request failed. Use the diagnostic details to investigate the provider or SDK error."
    status_detail = f" HTTP status/provider code: {status}." if status else ""
    return f"{guidance} ({name}).{status_detail} Details: {safe_message} Showing rule-based recommendations."


def get_recommendations(
    data: pd.DataFrame,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[str, bool, str | None, float]:
    """Return recommendation text, whether Gemini was used, and a safe notice."""
    fallback = rule_based_recommendations(data)
    key = setting("GEMINI_API_KEY") if api_key is None else api_key.strip()
    selected_model = setting("GEMINI_MODEL", "gemini-2.5-flash") if model is None else model.strip()
    selected_model = selected_model or "gemini-2.5-flash"
    if not key:
        return fallback, False, "Gemini is not configured. Showing rule-based recommendations.", 0.0
    try:
        context = build_inventory_context(data)
        known_ids = [str(item["medicine_id"]) for item in context["at_risk_medicines"]]
        prompt = (
            "Write a concise inventory operations briefing using only this Python-calculated data. "
            "Use these sections: Overall health (two or three short sentences); Critical stock risks; "
            "Important expiry risks; Actionable recommendations (four to six short bullets). "
            "Prioritize Critical and High stock risks and Expired, Critical, or Soon expiry risks. "
            "Explain why listed medicines need attention, but never invent a medicine, stock level, risk, date, "
            "quantity, calculation, or duration. Do not include digits, number words, dates, or numbered lists; "
            "exact values are displayed separately from Python. Do not refer to medicines absent from the supplied alerts. "
            "Keep the whole briefing under two hundred words. This is inventory support, not medical advice.\n"
            "CALCULATED INVENTORY CONTEXT:\n" + json.dumps(context, ensure_ascii=False)
        )
        request_started = time.perf_counter()
        client = create_gemini_client(key)
        from google.genai import types
        response = client.models.generate_content(
            model=selected_model,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS, temperature=0.2),
        )
        request_duration = time.perf_counter() - request_started
        answer = getattr(response, "text", None)
        answer = answer.strip() if isinstance(answer, str) else ""
        if not answer:
            raise ValueError("Empty Gemini response")
        if contains_unverified_numeric_claim(answer, known_ids):
            raise ValueError("Gemini response contained numeric claims")
        logger.info("Gemini request completed model=%s duration_seconds=%.3f", selected_model, request_duration)
        return answer, True, None, request_duration
    except Exception as exc:
        request_duration = time.perf_counter() - request_started if "request_started" in locals() else 0.0
        status = _exception_status(exc)
        safe_message = _safe_error_message(exc, key)
        logger.error(
            "Gemini request failed model=%s type=%s status=%s duration_seconds=%.3f message=%s",
            selected_model, type(exc).__name__, status or "unavailable", request_duration, safe_message,
        )
        return fallback, False, _friendly_fallback_notice(exc, status=status, safe_message=safe_message), request_duration


def _cache_fingerprint(data: pd.DataFrame, api_key: str, model: str) -> str:
    # Hash all calculated medicine fields, including healthy rows omitted from the prompt,
    # so any relevant inventory edit invalidates the successful response.
    inventory_snapshot = data[AI_DETAIL_COLUMNS].sort_values("medicine_id").to_json(
        orient="records", date_format="iso", double_precision=15
    )
    key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return hashlib.sha256(f"{model}\0{key_fingerprint}\0{inventory_snapshot}".encode("utf-8")).hexdigest()


def inventory_data_fingerprint(data: pd.DataFrame) -> str:
    """Return a secret-free fingerprint for checking whether an async result is stale."""
    snapshot = data[AI_DETAIL_COLUMNS].sort_values("medicine_id").to_json(
        orient="records", date_format="iso", double_precision=15
    )
    return hashlib.sha256(snapshot.encode("utf-8")).hexdigest()


def get_cached_recommendations(
    data: pd.DataFrame,
    cache_entry: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    now: float | None = None,
) -> tuple[str, bool, str | None, bool, dict[str, Any] | None, float, float | None]:
    """Reuse a successful per-session result for the same inventory and model."""
    key = setting("GEMINI_API_KEY") if api_key is None else api_key.strip()
    model = (setting("GEMINI_MODEL", "gemini-2.5-flash") if model is None else model.strip()) or "gemini-2.5-flash"
    timestamp = time.monotonic() if now is None else now
    if key:
        fingerprint = _cache_fingerprint(data, key, model)
        if (
            cache_entry
            and cache_entry.get("fingerprint") == fingerprint
            and timestamp - float(cache_entry.get("created_at", 0)) < CACHE_TTL_SECONDS
        ):
            age = max(0.0, timestamp - float(cache_entry.get("created_at", timestamp)))
            return str(cache_entry["text"]), True, None, True, cache_entry, 0.0, age
    else:
        fingerprint = ""
    text, used_gemini, notice, request_duration = get_recommendations(data, api_key=key, model=model)
    updated_cache = (
        {
            "fingerprint": fingerprint,
            "created_at": timestamp,
            "text": text,
            "request_duration_seconds": request_duration,
        }
        if key and used_gemini
        else None
    )
    return text, used_gemini, notice, False, updated_cache, request_duration, None
