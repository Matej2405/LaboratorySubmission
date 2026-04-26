"""Validator Agent - cross-references the Extractor's claims against the
medical-standards KB and re-prompts an LLM only when the rule-based check
flags a disagreement.

Outputs `ValidatorVerdict` per (facility, capability), all wrapped in MLflow
spans so each step is visible in the trace UI.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import mlflow

from agents.config import CFG
from agents.extractor import detect_endpoint_family, _call_anthropic, _openai_client
from agents.medical_kb import KB, required_for
from schemas.virtue_foundation import (
    CapabilityClaim,
    CapabilityName,
    FacilityExtraction,
    HIGH_ACUITY_CAPABILITIES,
    ValidatorVerdict,
)

LOG = logging.getLogger("vf_health.validator")


VALIDATOR_SYSTEM = """You are a medical-records validator. Your job is to
decide whether a facility's claim of a clinical capability is actually
supported by the provided evidence sentences and equipment list.

Reply ONLY with JSON of the form:
{
  "claim_supported": boolean,
  "rationale": "<= 2 short sentences",
  "flagged_evidence": ["<verbatim sentence that looks weak or contradictory>"]
}
"""


@mlflow.trace(span_type="LLM", name="validator.llm_judge")
def _llm_judge(
    capability: CapabilityName,
    claim: CapabilityClaim,
    blob: str,
    endpoint: str,
) -> dict[str, Any]:
    payload = {
        "capability": capability,
        "claim": {
            "evidence_sentences": claim.evidence_sentences,
            "supporting_equipment": claim.supporting_equipment,
            "supporting_staff": claim.supporting_staff,
        },
        "facility_notes_excerpt": blob[:3000],
        "kb_requirement_notes": (required_for(capability).notes if required_for(capability) else ""),
    }
    user = json.dumps(payload, indent=0)
    family = detect_endpoint_family(endpoint)
    if family == "anthropic":
        text = _call_anthropic(
            endpoint,
            VALIDATOR_SYSTEM,
            user + "\n\nReturn ONLY the JSON object. No markdown.",
            temperature=0.0,
            timeout_s=45,
        )
    else:
        client = _openai_client(endpoint)
        resp = client.chat.completions.create(
            model=endpoint,
            temperature=0.0,
            messages=[
                {"role": "system", "content": VALIDATOR_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            timeout=45,
        )
        text = resp.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s : e + 1]) if s != -1 and e != -1 else {}


@mlflow.trace(span_type="CHAIN", name="validator.kb_check")
def _kb_check(claim: CapabilityClaim) -> list[str]:
    """Pure-Python rule check: does the claim violate the medical KB?"""
    req = required_for(claim.name)
    if req is None:
        return []
    return req.missing_for(claim.supporting_equipment, claim.supporting_staff)


@mlflow.trace(span_type="AGENT", name="validator.validate_facility")
def validate_facility(
    extraction: FacilityExtraction,
    blob: str,
    *,
    endpoint: str = CFG.validator_endpoint,
    only_high_acuity: bool = True,
) -> list[ValidatorVerdict]:
    verdicts: list[ValidatorVerdict] = []
    for claim in extraction.capabilities:
        if not claim.claimed:
            continue
        if only_high_acuity and claim.name not in HIGH_ACUITY_CAPABILITIES:
            continue

        missing = _kb_check(claim)
        if not missing:
            verdicts.append(
                ValidatorVerdict(
                    facility_id=extraction.facility_id,
                    capability=claim.name,
                    original_claim=True,
                    validator_claim=True,
                    agreement=True,
                    rationale="KB check passed.",
                )
            )
            continue

        try:
            judge = _llm_judge(claim.name, claim, blob, endpoint)
        except Exception as e:
            LOG.warning("validator LLM failed for %s/%s: %s", extraction.facility_id, claim.name, e)
            judge = {"claim_supported": False, "rationale": f"LLM judge errored: {e}", "flagged_evidence": []}

        validator_claim = bool(judge.get("claim_supported", False))
        verdicts.append(
            ValidatorVerdict(
                facility_id=extraction.facility_id,
                capability=claim.name,
                original_claim=True,
                validator_claim=validator_claim,
                agreement=validator_claim,
                rationale=str(judge.get("rationale", ""))[:500],
                missing_required=missing,
                flagged_evidence=[str(s) for s in (judge.get("flagged_evidence") or [])][:3],
            )
        )

    return verdicts
