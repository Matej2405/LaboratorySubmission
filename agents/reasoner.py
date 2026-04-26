"""Reasoning Agent - LangGraph-style multi-step plan: parse query -> retrieve
-> filter -> cite -> answer. Wrapped in MLflow 3 spans end-to-end.

We deliberately keep the orchestration as plain Python (no heavy framework
dependency) so the agent runs identically on Databricks and locally. The
LLM that does query understanding can be the same Foundation Model endpoint
as the extractor.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import mlflow

from agents import tools
from agents.config import CFG
from agents.text_utils import normalize_state, INDIAN_STATES
from schemas.virtue_foundation import HIGH_ACUITY_CAPABILITIES

LOG = logging.getLogger("vf_health.reasoner")


PLANNER_SYSTEM = """You translate a natural-language question about Indian
healthcare facilities into a structured query plan.

Reply ONLY with JSON of the shape:
{
  "intent": "find_facility" | "list_capability_in_region" | "trust_audit" | "general",
  "state": "<one Indian state name, lowercased, or null>",
  "city": "<lowercase city name or null>",
  "capabilities": ["icu", "general_surgery", ...],   // canonical tags
  "min_trust": 0..100,
  "facility_type": "hospital" | "clinic" | null,
  "semantic_query": "<short query for vector search, may include rural/24x7/part-time>",
  "k": 1..25
}

Canonical capability tags you may use:
icu, nicu, dialysis, oncology, trauma_emergency, general_surgery,
emergency_appendectomy, cardiac_care, obgyn_delivery, neonatal_care,
radiology_xray, radiology_ct, radiology_mri, ultrasound, blood_bank,
ambulance, oxygen_supply, emergency_24x7, pharmacy, lab_diagnostics,
telemedicine, mental_health, dental, ophthalmology, physiotherapy.

If the question mentions emergency, set capability emergency_24x7. If it
mentions specific procedures (appendectomy, dialysis, chemo, etc.), pick the
matching high-acuity tag.
"""


@dataclass
class AgentAnswer:
    answer: str
    facilities: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None


def _client():
    from openai import OpenAI

    if os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"):
        return OpenAI(
            api_key=os.environ["DATABRICKS_TOKEN"],
            base_url=f"{os.environ['DATABRICKS_HOST'].rstrip('/')}/serving-endpoints",
        )
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI()
    raise RuntimeError("No LLM credentials configured.")


def _has_llm_creds() -> bool:
    return bool(
        (os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"))
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


_CAP_KEYWORDS = {
    "icu": ["icu", "intensive care"],
    "nicu": ["nicu", "neonatal"],
    "dialysis": ["dialysis", "kidney"],
    "oncology": ["oncology", "cancer", "chemo"],
    "trauma_emergency": ["trauma"],
    "general_surgery": ["surgery", "surgical", "operation theatre", "operating theatre"],
    "emergency_appendectomy": ["appendectomy", "appendicitis"],
    "cardiac_care": ["cardiac", "cardio", "heart"],
    "obgyn_delivery": ["obstetric", "delivery", "labour", "labor", "maternity"],
    "neonatal_care": ["newborn", "neonatal"],
    "emergency_24x7": ["emergency", "24x7", "24/7", "round the clock"],
    "pharmacy": ["pharmacy", "chemist"],
    "lab_diagnostics": ["lab", "diagnostic", "pathology"],
}


def _heuristic_plan(question: str) -> dict[str, Any]:
    """Regex-based fallback planner used when no LLM credentials are configured.

    Mirrors the JSON shape the LLM planner produces so downstream code is
    unchanged. Wired into the demo so the Streamlit "Ask the agent" tab is
    never blocked by missing credentials.
    """
    q = (question or "").lower()
    state: Optional[str] = None
    for s in INDIAN_STATES:
        if s in q:
            state = s
            break
    if state is None:
        norm = normalize_state(question)
        if norm and norm.lower() in INDIAN_STATES:
            state = norm.lower()

    capabilities: list[str] = []
    for cap, kws in _CAP_KEYWORDS.items():
        if any(kw in q for kw in kws):
            capabilities.append(cap)
    capabilities = list(dict.fromkeys(capabilities))

    facility_type: Optional[str] = None
    if "hospital" in q:
        facility_type = "hospital"
    elif "clinic" in q:
        facility_type = "clinic"

    intent = "find_facility"
    if any(w in q for w in ("desert", "gap", "missing", "no facility")):
        intent = "list_capability_in_region"
    elif any(w in q for w in ("trust", "fake", "claim", "contradiction")):
        intent = "trust_audit"

    semantic_bits: list[str] = []
    for tag in ("rural", "urban", "24x7", "24/7", "part-time", "part time", "private", "public", "ngo"):
        if tag in q:
            semantic_bits.append(tag)
    if "emergency" in q and "emergency_24x7" not in capabilities:
        semantic_bits.append("emergency")

    plan = {
        "intent": intent,
        "state": state,
        "city": None,
        "capabilities": capabilities,
        "min_trust": 50 if "trust" in q or "trustworthy" in q else 0,
        "facility_type": facility_type,
        "semantic_query": " ".join(semantic_bits) or question[:120],
        "k": 10,
        "_planner": "heuristic",
    }
    return plan


def _to_list(v) -> list:
    """Coerce numpy arrays / None / scalars into a plain list for safe truthiness."""
    if v is None:
        return []
    if hasattr(v, "tolist"):
        try:
            return list(v.tolist())
        except Exception:
            return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _heuristic_compose(question: str, plan: dict, facilities: list[dict], citations: list[dict]) -> str:
    """Deterministic answer composition used when no LLM is available."""
    if not facilities:
        gap = plan.get("capabilities") or ["the requested capability"]
        loc = (plan.get("state") or "the requested area").title()
        return (
            f"No facilities in the cache match {', '.join(gap)} in {loc} with the "
            f"current trust threshold ({plan.get('min_trust')}). The closest signal "
            "is summarized in the Trust audit tab; consider relaxing the trust "
            "threshold or expanding to a neighbouring state."
        )
    top = facilities[: min(3, len(facilities))]
    loc = (plan.get("state") or "").title()
    caps = ", ".join(plan.get("capabilities") or []) or "the requested capability"
    bullets = []
    for f in top:
        flags = _to_list(f.get("trust_flags"))
        flag_note = (
            f" (flags: {', '.join(str(x) for x in flags[:2])})" if flags else ""
        )
        try:
            ts = float(f.get("trust_score") or 0)
        except (TypeError, ValueError):
            ts = 0.0
        bullets.append(
            f"- **{f.get('name','?')}** - {f.get('city','?')}, {f.get('state','?')} - "
            f"trust {ts:.0f}/100{flag_note}"
        )
    cite_lines = []
    for c in citations[:3]:
        s = (c.get("sentence") or "").strip().replace("\n", " ")
        if s:
            cite_lines.append(f'  - [{c.get("facility_id")}] _{s[:200]}_')
    cited_block = "\n\n**Cited evidence:**\n" + "\n".join(cite_lines) if cite_lines else ""
    return (
        f"Found {len(facilities)} facilities matching {caps}"
        + (f" in {loc}" if loc else "")
        + ". Top by trust score:\n\n"
        + "\n".join(bullets)
        + cited_block
        + "\n\n_Heuristic-only reasoning: planner and answer composition fell "
        "back to local rules because no LLM credentials are configured. "
        "All retrievals and citations are real cache lookups._"
    )


@mlflow.trace(span_type="LLM", name="reasoner.plan")
def _plan(question: str, endpoint: str) -> dict[str, Any]:
    if not _has_llm_creds():
        return _heuristic_plan(question)
    try:
        client = _client()
        resp = client.chat.completions.create(
            model=endpoint,
            temperature=0.0,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            timeout=30,
        )
        text = resp.choices[0].message.content or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            s, e = text.find("{"), text.rfind("}")
            return json.loads(text[s : e + 1]) if s != -1 and e != -1 else _heuristic_plan(question)
    except Exception as exc:
        LOG.warning("LLM planner failed (%s); falling back to heuristic.", exc)
        return _heuristic_plan(question)


@mlflow.trace(span_type="CHAIN", name="reasoner.retrieve")
def _retrieve(plan: dict[str, Any]) -> list[dict[str, Any]]:
    structured = tools.find_facilities(
        state=plan.get("state"),
        city=plan.get("city"),
        capabilities=plan.get("capabilities") or [],
        min_trust=float(plan.get("min_trust") or 0.0),
        facility_type=plan.get("facility_type"),
        limit=int(plan.get("k") or 10),
    )
    if structured:
        return structured
    return tools.semantic_search(
        query=plan.get("semantic_query") or "",
        state=plan.get("state"),
        capabilities=plan.get("capabilities") or [],
        min_trust=float(plan.get("min_trust") or 0.0),
        k=int(plan.get("k") or 10),
        target="summaries",
    )


@mlflow.trace(span_type="CHAIN", name="reasoner.cite")
def _cite(facilities: list[dict[str, Any]], capabilities: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in facilities[:5]:
        fid = f.get("facility_id")
        if not fid:
            continue
        for cap in capabilities[:2] or [None]:
            for ev in tools.get_evidence(fid, capability=cap):
                out.append({"facility_id": fid, **ev})
    return out


ANSWER_SYSTEM = """You are a careful healthcare-access assistant for an Indian
NGO planner. Always:

* Cite facility names AND state/city.
* Distinguish "claimed" from "verified" capabilities, and call out trust flags.
* Prefer facilities with higher trust scores; explicitly mention trust.
* If results are sparse, name the closest near-misses and explain the gap.
* Use the provided FACTS only. Do not invent facilities.
"""


@mlflow.trace(span_type="LLM", name="reasoner.compose")
def _compose(question: str, plan: dict, facilities: list[dict], citations: list[dict], endpoint: str) -> str:
    if not _has_llm_creds():
        return _heuristic_compose(question, plan, facilities, citations)
    facts = {
        "plan": plan,
        "facilities": [
            {
                "facility_id": f.get("facility_id"),
                "name": f.get("name"),
                "city": f.get("city"),
                "state": f.get("state"),
                "trust_score": f.get("trust_score"),
                "claimed_capabilities": f.get("claimed_capabilities"),
                "trust_flags": f.get("trust_flags"),
            }
            for f in facilities[:8]
        ],
        "citations": citations[:10],
    }
    try:
        client = _client()
        resp = client.chat.completions.create(
            model=endpoint,
            temperature=0.1,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": f"QUESTION:\n{question}\n\nFACTS:\n{json.dumps(facts, default=str)[:6000]}"},
            ],
            timeout=45,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        LOG.warning("LLM composer failed (%s); falling back to heuristic.", exc)
        return _heuristic_compose(question, plan, facilities, citations)


@mlflow.trace(span_type="AGENT", name="reasoner.answer")
def answer(
    question: str,
    *,
    endpoint: str = CFG.extractor_endpoint,
) -> AgentAnswer:
    plan = _plan(question, endpoint)
    facilities = _retrieve(plan)
    citations = _cite(facilities, plan.get("capabilities") or [])
    text = _compose(question, plan, facilities, citations, endpoint)
    trace = mlflow.get_current_active_trace_id() if hasattr(mlflow, "get_current_active_trace_id") else None
    return AgentAnswer(
        answer=text,
        facilities=facilities,
        citations=citations,
        plan=plan,
        trace_id=trace,
    )
