"""Extractor Agent - structured capability extraction from messy facility notes.

Backed by Databricks Agent Bricks (Foundation Model API) when running on
Databricks; falls back to OpenAI-compatible APIs when running locally.

Uses pydantic structured output via the OpenAI SDK pattern that the Databricks
Foundation Models endpoint speaks (`response_format={"type": "json_object"}` +
schema injected in the prompt). MLflow 3 tracing wraps every call.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Iterable, Optional

import mlflow
from pydantic import ValidationError

from schemas.virtue_foundation import (
    CapabilityClaim,
    FacilityExtraction,
    OperatingHours,
    StaffingProfile,
    empty_extraction,
)

LOG = logging.getLogger("vf_health.extractor")

SYSTEM_PROMPT = """You are a meticulous Indian healthcare facility auditor.

Your job is to read a single facility's free-form notes and extract a STRICT,
verifiable JSON record. You MUST:

1. Only mark a capability as `claimed=True` when an explicit sentence in the
   notes says the facility offers it. Marketing terms like "best in city" do
   NOT count as evidence.
2. For every claimed capability, copy 1-3 verbatim sentences from the notes
   into `evidence_sentences`. Do not paraphrase.
3. Distinguish "claimed" from "functional". A facility that says "ICU" but
   lists no ventilator and no anesthesiologist gets a low
   `functional_confidence`.
4. Fill `supporting_equipment` and `supporting_staff` ONLY with items
   actually mentioned in the EQUIPMENT or notes. Empty lists are fine.
5. Output JSON conforming exactly to the provided schema. No extra keys, no
   commentary, no markdown fences.
"""

USER_TEMPLATE = """FACILITY ID: {facility_id}

STRUCTURED METADATA:
- Type: {facility_type}
- Operator: {operator_type}
- City / State: {city} / {state}
- Doctors listed (structured): {n_doctors}
- Bed capacity (structured): {capacity}

NOTES (free-form, may be noisy):
---
{blob}
---

JSON SCHEMA YOU MUST RETURN (top-level FacilityExtraction):
{schema}

Return ONLY the JSON object. No markdown."""


def detect_endpoint_family(endpoint: str) -> str:
    """Map an endpoint name to a backend family.

    Returns one of: 'databricks', 'openai', 'anthropic'.
    """
    name = (endpoint or "").lower()
    if name.startswith("databricks-") or "databricks" in name:
        return "databricks"
    if name.startswith("claude") or "anthropic" in name:
        return "anthropic"
    return "openai"


def _openai_client(endpoint: str):
    """Return an OpenAI-compatible client appropriate for the endpoint."""
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("openai SDK is required") from e

    family = detect_endpoint_family(endpoint)
    if family == "databricks":
        if not (os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN")):
            raise RuntimeError(
                "Endpoint looks like a Databricks model but DATABRICKS_HOST / "
                "DATABRICKS_TOKEN are not set."
            )
        return OpenAI(
            api_key=os.environ["DATABRICKS_TOKEN"],
            base_url=f"{os.environ['DATABRICKS_HOST'].rstrip('/')}/serving-endpoints",
        )
    if family == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAI()
    raise RuntimeError(f"Unsupported family for OpenAI client: {family}")


def _anthropic_client():
    """Return a native Anthropic client (used for `claude-*` endpoint names)."""
    try:
        import anthropic  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "anthropic SDK is required for Claude endpoints. `pip install anthropic`."
        ) from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    return anthropic.Anthropic()


def _call_anthropic(
    endpoint: str,
    system: str,
    user: str,
    *,
    temperature: float,
    timeout_s: int,
) -> str:
    """Single-call wrapper around Anthropic's Messages API. Returns raw text."""
    client = _anthropic_client()
    msg = client.messages.create(
        model=endpoint,
        system=system,
        max_tokens=4000,
        temperature=temperature,
        messages=[{"role": "user", "content": user}],
        timeout=timeout_s,
    )
    parts = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts) or "{}"


def _schema_hint() -> str:
    """A compact JSON-schema hint for the prompt (small enough to fit context)."""
    return json.dumps(FacilityExtraction.model_json_schema(), indent=0)


def _coerce(raw: dict[str, Any], facility_id: str) -> FacilityExtraction:
    """Best-effort coerce LLM JSON into FacilityExtraction."""
    raw.setdefault("facility_id", facility_id)
    caps_in = raw.get("capabilities") or []
    fixed_caps: list[CapabilityClaim] = []
    for c in caps_in:
        try:
            fixed_caps.append(CapabilityClaim.model_validate(c))
        except ValidationError as e:  # drop malformed claim, keep going
            LOG.warning("dropping malformed capability for %s: %s", facility_id, e)
    raw["capabilities"] = [c.model_dump() for c in fixed_caps]
    raw.setdefault("staffing", {})
    raw.setdefault("hours", {})
    return FacilityExtraction.model_validate(raw)


@mlflow.trace(span_type="LLM", name="extractor.extract_one")
def extract_one(
    facility_id: str,
    blob: str,
    *,
    facility_type: Optional[str] = None,
    operator_type: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    n_doctors: Optional[int] = None,
    capacity: Optional[int] = None,
    endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
    temperature: float = 0.0,
    max_retries: int = 2,
    timeout_s: int = 60,
) -> FacilityExtraction:
    """Run the LLM once for a single facility row, returning a structured record."""
    if not blob or not blob.strip():
        return empty_extraction(facility_id)

    user = USER_TEMPLATE.format(
        facility_id=facility_id,
        facility_type=facility_type or "?",
        operator_type=operator_type or "?",
        city=city or "?",
        state=state or "?",
        n_doctors=n_doctors if n_doctors is not None else "?",
        capacity=capacity if capacity is not None else "?",
        blob=blob[:8000],
        schema=_schema_hint()[:6000],
    )

    family = detect_endpoint_family(endpoint)
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            if family == "anthropic":
                text = _call_anthropic(
                    endpoint,
                    SYSTEM_PROMPT,
                    user + "\n\nReturn ONLY the JSON object. No markdown.",
                    temperature=temperature,
                    timeout_s=timeout_s,
                )
            else:
                client = _openai_client(endpoint)
                resp = client.chat.completions.create(
                    model=endpoint,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    timeout=timeout_s,
                )
                text = resp.choices[0].message.content or "{}"
            elapsed_ms = (time.time() - t0) * 1000
            try:
                mlflow.log_metric("extractor_latency_ms", elapsed_ms)
            except Exception:
                pass
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                start = text.find("{")
                end = text.rfind("}")
                payload = json.loads(text[start : end + 1])
            return _coerce(payload, facility_id)
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            LOG.warning("parse retry %s for %s: %s", attempt, facility_id, e)
        except Exception as e:  # network / quota
            last_err = e
            LOG.warning("api retry %s for %s: %s", attempt, facility_id, e)
            time.sleep(0.5 * (attempt + 1))
    LOG.error("extractor gave up on %s after %s tries: %s", facility_id, max_retries + 1, last_err)
    return empty_extraction(facility_id)


def extract_batch(
    rows: Iterable[dict[str, Any]],
    *,
    endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
    temperature: float = 0.0,
) -> list[FacilityExtraction]:
    """Sequential batch extractor.

    Designed to be called from Spark `applyInPandas` so each executor handles
    its own slice in parallel; we keep the Python side sequential to respect
    per-endpoint rate limits.
    """
    out: list[FacilityExtraction] = []
    for r in rows:
        out.append(
            extract_one(
                facility_id=str(r["facility_id"]),
                blob=str(r.get("unstructured_blob") or ""),
                facility_type=r.get("facilityTypeId"),
                operator_type=r.get("operatorTypeId"),
                city=r.get("city"),
                state=r.get("state"),
                n_doctors=r.get("numberDoctors"),
                capacity=r.get("capacity"),
                endpoint=endpoint,
                temperature=temperature,
            )
        )
    return out
