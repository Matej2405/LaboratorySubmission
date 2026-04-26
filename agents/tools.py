"""Tools used by the Reasoning Agent.

Each tool is a thin Python function decorated with `@mlflow.trace` so its inputs
and outputs land as a span in the MLflow trace tree. Tools are deliberately
small and pure-ish so the agent (and unit tests) can compose them.

Where Spark/Vector Search are unavailable (local dev), the tools fall back to
in-memory pandas DataFrames provided by the caller via `set_local_state(...)`.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import mlflow

# ---------- Local-mode state ----------------------------------------------

@dataclass
class _LocalState:
    silver_pdf: Any | None = None
    gold_pdf: Any | None = None
    trust_pdf: Any | None = None
    summaries_pdf: Any | None = None
    chunks_pdf: Any | None = None


_LOCAL = _LocalState()


def set_local_state(
    *,
    silver_pdf=None,
    gold_pdf=None,
    trust_pdf=None,
    summaries_pdf=None,
    chunks_pdf=None,
):
    """Inject pandas DataFrames so tools work without Spark / VS (tests + UI)."""
    _LOCAL.silver_pdf = silver_pdf if silver_pdf is not None else _LOCAL.silver_pdf
    _LOCAL.gold_pdf = gold_pdf if gold_pdf is not None else _LOCAL.gold_pdf
    _LOCAL.trust_pdf = trust_pdf if trust_pdf is not None else _LOCAL.trust_pdf
    _LOCAL.summaries_pdf = summaries_pdf if summaries_pdf is not None else _LOCAL.summaries_pdf
    _LOCAL.chunks_pdf = chunks_pdf if chunks_pdf is not None else _LOCAL.chunks_pdf


def _spark():
    """Try to grab the active Spark session (Databricks runtime)."""
    try:
        from pyspark.sql import SparkSession  # type: ignore
        return SparkSession.getActiveSession()
    except Exception:
        return None


def _contains(xs, cap) -> bool:
    """List-safe membership check that also handles None and numpy arrays."""
    if xs is None:
        return False
    try:
        if hasattr(xs, "tolist"):
            xs = xs.tolist()
        return cap in (xs or [])
    except Exception:
        return False


# ---------- Tool: distance ------------------------------------------------

@mlflow.trace(span_type="TOOL", name="tools.distance_km")
def distance_km(
    lat_a: float, lon_a: float, lat_b: float, lon_b: float
) -> float:
    """Great-circle distance in km using haversine."""
    R = 6371.0
    p1, p2 = math.radians(lat_a), math.radians(lat_b)
    dp = math.radians(lat_b - lat_a)
    dl = math.radians(lon_b - lon_a)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * R * math.asin(math.sqrt(a)), 2)


# ---------- Tool: find_facilities -----------------------------------------

@mlflow.trace(span_type="TOOL", name="tools.find_facilities")
def find_facilities(
    state: Optional[str] = None,
    city: Optional[str] = None,
    capabilities: Optional[list[str]] = None,
    min_trust: float = 0.0,
    facility_type: Optional[str] = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Structured filter over the gold facility_summaries view."""
    capabilities = capabilities or []

    spark = _spark()
    if spark is not None:
        from pyspark.sql import functions as F
        df = spark.table("vf_health.gold.facility_summaries")
        if state:
            df = df.filter(F.lower(F.col("state")) == state.lower())
        if city:
            df = df.filter(F.lower(F.col("city")) == city.lower())
        if facility_type:
            df = df.filter(F.col("facilityTypeId") == facility_type)
        if min_trust:
            df = df.filter(F.coalesce(F.col("trust_score"), F.lit(0.0)) >= min_trust)
        for cap in capabilities:
            df = df.filter(F.array_contains(F.col("claimed_capabilities"), cap))
        return [r.asDict(recursive=True) for r in df.limit(limit).collect()]

    df = _LOCAL.summaries_pdf
    if df is None:
        return []
    out = df
    if state:
        out = out[out["state"].str.lower() == state.lower()]
    if city:
        out = out[out["city"].str.lower() == city.lower()]
    if facility_type:
        out = out[out["facilityTypeId"] == facility_type]
    if min_trust:
        out = out[out["trust_score"].fillna(0) >= min_trust]
    for cap in capabilities:
        out = out[out["claimed_capabilities"].apply(lambda xs: _contains(xs, cap))]
    return out.head(limit).to_dict(orient="records")


# ---------- Tool: semantic_search -----------------------------------------

@mlflow.trace(span_type="TOOL", name="tools.semantic_search")
def semantic_search(
    query: str,
    *,
    state: Optional[str] = None,
    capabilities: Optional[list[str]] = None,
    min_trust: float = 0.0,
    k: int = 8,
    target: str = "summaries",
) -> list[dict[str, Any]]:
    """Mosaic AI Vector Search hybrid call.

    `target='summaries'` -> the per-facility paragraph index
    `target='notes'`     -> sentence-level chunks (use for evidence retrieval)
    """
    try:
        from databricks.vector_search.client import VectorSearchClient
    except ImportError:
        VectorSearchClient = None  # type: ignore

    if VectorSearchClient is None or os.environ.get("VF_HEALTH_LOCAL"):
        return _local_semantic_search(query, state=state, capabilities=capabilities, min_trust=min_trust, k=k, target=target)

    from agents.config import CFG
    vsc = VectorSearchClient(disable_notice=True)
    index_name = CFG.summaries_index if target == "summaries" else CFG.notes_index
    fq_index = CFG.fq(CFG.idx_schema, index_name)
    idx = vsc.get_index(endpoint_name=CFG.vs_endpoint, index_name=fq_index)
    filters = []
    if state:
        filters.append({"state": state})
    if min_trust and target == "summaries":
        filters.append({"trust_score >=": float(min_trust)})
    if capabilities and target == "summaries":
        for c in capabilities:
            filters.append({"claimed_capabilities": c})
    res = idx.similarity_search(
        query_text=query,
        columns=None,
        num_results=k,
        filters={"AND": filters} if filters else None,
    )
    rows = res.get("result", {}).get("data_array", [])
    cols = [c["name"] for c in res.get("manifest", {}).get("columns", [])]
    return [dict(zip(cols, r)) for r in rows]


def _local_semantic_search(query, *, state, capabilities, min_trust, k, target):
    """Naive token-overlap fallback used in tests and local Streamlit dev."""
    df = _LOCAL.summaries_pdf if target == "summaries" else _LOCAL.chunks_pdf
    if df is None:
        return []
    pdf = df.copy()
    text_col = "summary_text" if target == "summaries" else "chunk_text"
    if state and "state" in pdf.columns:
        pdf = pdf[pdf["state"].str.lower() == state.lower()]
    if min_trust and "trust_score" in pdf.columns:
        pdf = pdf[pdf["trust_score"].fillna(0) >= min_trust]
    if capabilities and "claimed_capabilities" in pdf.columns:
        for cap in capabilities:
            pdf = pdf[pdf["claimed_capabilities"].apply(lambda xs: _contains(xs, cap))]
    if pdf.empty:
        return []
    if text_col not in pdf.columns:
        # Synthesize a searchable text column from whichever fields exist so we
        # never crash on a missing summary_text/chunk_text column.
        parts = []
        for col in ("name", "city", "state", "description", "unstructured_blob"):
            if col in pdf.columns:
                parts.append(pdf[col].fillna("").astype(str))
        if not parts:
            return []
        synthesized = parts[0]
        for extra in parts[1:]:
            synthesized = synthesized.str.cat(extra, sep=" | ")
        pdf = pdf.assign(_text=synthesized)
        text_col = "_text"
    q_tokens = set(query.lower().split())
    pdf = pdf.assign(
        _score=pdf[text_col].fillna("").astype(str).str.lower().apply(
            lambda t: len(q_tokens & set(t.split())) / max(1, len(q_tokens))
        )
    )
    drop_cols = [c for c in ("_score", "_text") if c in pdf.columns]
    return pdf.sort_values("_score", ascending=False).head(k).drop(columns=drop_cols).to_dict(orient="records")


# ---------- Tool: get_evidence --------------------------------------------

@mlflow.trace(span_type="TOOL", name="tools.get_evidence")
def get_evidence(facility_id: str, capability: Optional[str] = None) -> list[dict[str, Any]]:
    """Return the verbatim sentences justifying a facility's claim(s).

    When `capability` is supplied, only its evidence is returned. Otherwise we
    return up to 5 sentences from the most recent extraction.
    """
    spark = _spark()
    if spark is not None:
        from pyspark.sql import functions as F
        df = spark.table("vf_health.gold.capability_claims").filter(F.col("facility_id") == facility_id)
        rows = df.limit(1).collect()
        if not rows:
            return []
        from schemas.virtue_foundation import FacilityExtraction
        ex = FacilityExtraction.model_validate_json(rows[0]["extraction_json"])
    else:
        if _LOCAL.gold_pdf is None:
            return []
        sub = _LOCAL.gold_pdf[_LOCAL.gold_pdf["facility_id"] == facility_id]
        if sub.empty:
            return []
        from schemas.virtue_foundation import FacilityExtraction
        ex = FacilityExtraction.model_validate_json(sub.iloc[0]["extraction_json"])

    out: list[dict[str, Any]] = []
    for c in ex.capabilities:
        if not c.claimed:
            continue
        if capability and c.name != capability:
            continue
        for sent in c.evidence_sentences:
            out.append({"capability": c.name, "sentence": sent, "functional_confidence": c.functional_confidence})
        if capability:
            break
    return out[:5]
