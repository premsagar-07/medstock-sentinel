"""Validation helpers for the staged, ID-based medicine update workflow."""
from __future__ import annotations

from collections.abc import MutableMapping
from datetime import date
from typing import Any

import pandas as pd

from modules.inventory import REQUIRED_COLUMNS, validate_inventory

EDITABLE_FIELDS = tuple(field for field in REQUIRED_COLUMNS if field != "medicine_id")


def changed_medicine_fields(current: dict[str, Any], proposed: dict[str, Any]) -> dict[str, Any]:
    """Validate a complete proposed record and return only modified fields."""
    unsupported = set(proposed) - set(EDITABLE_FIELDS)
    if unsupported:
        raise ValueError("Update contains an unsupported medicine field.")
    candidate = {**current, **proposed, "medicine_id": current["medicine_id"]}
    clean = validate_inventory(pd.DataFrame([candidate])).iloc[0]
    changes: dict[str, Any] = {}
    for field in proposed:
        value = clean[field]
        if field in ("manufacturing_date", "expiry_date") and isinstance(value, date):
            value = value.isoformat()
        elif hasattr(value, "item"):
            value = value.item()
        old_value = current[field]
        if field in ("manufacturing_date", "expiry_date"):
            old_value = pd.Timestamp(old_value).date().isoformat()
        if value != old_value:
            changes[field] = value
    return changes


def discard_pending_update(state: MutableMapping[str, Any]) -> None:
    """Forget a loaded record and rotate widget keys so canceled edits cannot reappear."""
    state["update_record"] = None
    state["update_generation"] = int(state.get("update_generation", 0)) + 1
