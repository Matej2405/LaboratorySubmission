"""Lightweight text/JSON helpers shared across notebooks and agents.

Kept dependency-light (stdlib only) so it imports cleanly inside Spark UDFs.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

_PIN_RE = re.compile(r"\b(\d{6})\b")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "punjab", "rajasthan", "sikkim",
    "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "west bengal", "andaman and nicobar islands", "chandigarh",
    "dadra and nagar haveli and daman and diu", "delhi", "jammu and kashmir",
    "ladakh", "lakshadweep", "puducherry",
}

STATE_ALIASES = {
    "orissa": "odisha",
    "uttaranchal": "uttarakhand",
    "pondicherry": "puducherry",
    "ncr": "delhi",
    "new delhi": "delhi",
    "j&k": "jammu and kashmir",
    "uttar pradesh ": "uttar pradesh",
    "tn": "tamil nadu",
    "ap": "andhra pradesh",
    "mp": "madhya pradesh",
    "up": "uttar pradesh",
    "wb": "west bengal",
    "tg": "telangana",
}


def normalize_state(raw: str | None) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    s = STATE_ALIASES.get(s, s)
    if s in INDIAN_STATES:
        return s.title().replace("And", "and").replace("Of", "of")
    for canon in INDIAN_STATES:
        if canon in s or s in canon:
            return canon.title().replace("And", "and").replace("Of", "of")
    return raw.strip().title() or None


def extract_pin(*candidates: Any) -> str | None:
    for c in candidates:
        if c is None:
            continue
        s = str(c)
        m = _PIN_RE.search(s)
        if m:
            return m.group(1)
    return None


def parse_json_list(value: Any) -> list[str]:
    """Parse a JSON-string-encoded list of strings; tolerate raw strings & None."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    if not isinstance(value, str):
        return []
    s = value.strip()
    if not s or s in {"[]", "null", "None"}:
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if x is not None]
        if isinstance(parsed, str):
            return [parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    return [p.strip() for p in s.strip("[]").split(",") if p.strip()]


def split_sentences(text: str | None) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    out = []
    for p in parts:
        p = p.strip()
        if 4 < len(p) < 600:
            out.append(p)
    return out


def _coerce_str(v: Any) -> str | None:
    """Pandas / Excel cells sometimes hand us floats (NaN), bools, etc."""
    if v is None:
        return None
    if isinstance(v, float):
        import math
        return None if math.isnan(v) else str(v)
    s = str(v).strip()
    return s or None


def _coerce_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if x is not None and str(x).strip()]
    s = _coerce_str(v)
    return [s] if s else []


def to_unstructured_blob(
    description: Any,
    specialties: Iterable[str],
    procedure: Iterable[str],
    capability: Iterable[str],
    equipment: Iterable[str],
) -> str:
    """Compose the prompt-ready blob fed to the extractor LLM.

    Tolerant of NaN / float / bool values that arise when reading the raw
    Excel; everything is coerced to a clean string before composing.
    """
    parts: list[str] = []
    desc = _coerce_str(description)
    if desc:
        parts.append(f"DESCRIPTION:\n{desc}")
    sp = _coerce_list(specialties)
    if sp:
        parts.append("SPECIALTIES:\n- " + "\n- ".join(sp))
    pr = _coerce_list(procedure)
    if pr:
        parts.append("PROCEDURES:\n- " + "\n- ".join(pr))
    cap = _coerce_list(capability)
    if cap:
        parts.append("CAPABILITY CLAIMS:\n- " + "\n- ".join(cap))
    eq = _coerce_list(equipment)
    if eq:
        parts.append("EQUIPMENT:\n- " + "\n- ".join(eq))
    else:
        parts.append("EQUIPMENT:\n(none listed)")
    return "\n\n".join(parts)
