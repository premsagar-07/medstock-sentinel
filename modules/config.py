"""Read server-side settings from environment or Streamlit secrets."""
from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

load_dotenv()


def setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        import streamlit as st
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or default).strip()


def normalize_supabase_url(url: str) -> tuple[str, bool]:
    """Return the project base URL; supabase-py appends ``/rest/v1`` itself."""
    raw = (url or "").strip()
    if not raw:
        return raw, False
    parts = urlsplit(raw)
    path = parts.path.rstrip("/")
    normalized = False
    if path.lower().endswith("/rest/v1"):
        path = path[:-len("/rest/v1")]
        normalized = True
    if path == "/":
        path = ""
    result = urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
    return result, normalized
