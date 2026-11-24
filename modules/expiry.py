"""Expiry risk helpers."""
from datetime import date


def classify_expiry(expiry_date: date, *, today: date | None = None) -> tuple[int, str]:
    days = (expiry_date - (today or date.today())).days
    return days, "Expired" if days < 0 else "Critical" if days <= 30 else "Soon" if days <= 90 else "Low"
