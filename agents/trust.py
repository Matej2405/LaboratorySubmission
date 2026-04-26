"""Trust Scorer - 0..100 score per facility built from three sub-scores.

* Completeness  (40%): how filled-in is the structured extraction?
* Consistency   (40%): do contradiction rules pass? Each violation -> flag.
* Source agreement (20%): how often did the Validator Agent agree with the
  Extractor for this facility's high-acuity claims?

Each flag carries a verbatim cited sentence so the UI can show "we flagged
this because the note says X but no Y".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from agents.medical_kb import KB, required_for
from schemas.virtue_foundation import (
    CapabilityClaim,
    FacilityExtraction,
    HIGH_ACUITY_CAPABILITIES,
    TrustScore,
    ValidatorVerdict,
)


@dataclass
class StructuredSignals:
    """Signals harvested from the Bronze/Silver structured columns."""

    number_doctors: Optional[float] = None
    capacity: Optional[float] = None
    has_equipment_evidence: bool = False
    n_capability_claims: int = 0
    has_followers: bool = False
    has_official_website: bool = False


# ----- Contradiction rule library -----------------------------------------

def _flag(name: str, claim: CapabilityClaim, missing: list[str]) -> tuple[str, str]:
    cite = claim.evidence_sentences[0] if claim.evidence_sentences else ""
    detail = f"{name}: {', '.join(missing[:2])}"
    return detail, cite


def contradiction_flags(
    extraction: FacilityExtraction,
    structured: StructuredSignals,
) -> list[tuple[str, str]]:
    """Return list of (flag_label, cited_sentence) tuples."""
    out: list[tuple[str, str]] = []

    by_name = {c.name: c for c in extraction.capabilities if c.claimed}

    for cap_name, claim in by_name.items():
        req = required_for(cap_name)
        if req is None:
            continue
        missing = req.missing_for(claim.supporting_equipment, claim.supporting_staff)
        if missing and cap_name in HIGH_ACUITY_CAPABILITIES:
            out.append(_flag(f"{cap_name}_missing", claim, missing))

    if "general_surgery" in by_name and not extraction.staffing.has_anesthesiologist:
        out.append(_flag("surgery_no_anesthesiologist", by_name["general_surgery"], ["anesthesiologist"]))

    if "icu" in by_name and not any(
        "ventilator" in (e.lower()) for e in by_name["icu"].supporting_equipment
    ):
        out.append(_flag("icu_no_ventilator", by_name["icu"], ["ventilator"]))

    if "nicu" in by_name and not (
        extraction.staffing.has_pediatrician or "neonatologist" in " ".join(by_name["nicu"].supporting_staff).lower()
    ):
        out.append(_flag("nicu_no_pediatrician", by_name["nicu"], ["pediatrician/neonatologist"]))

    if extraction.hours.is_24x7 and (structured.number_doctors or 0) <= 1:
        cite = extraction.hours.evidence_sentences[0] if extraction.hours.evidence_sentences else ""
        out.append(("emergency_24x7_understaffed", cite))

    if "cardiac_care" in by_name and not extraction.staffing.has_cardiologist:
        out.append(_flag("cardiac_no_cardiologist", by_name["cardiac_care"], ["cardiologist"]))

    if "oncology" in by_name and not extraction.staffing.has_oncologist:
        out.append(_flag("oncology_no_oncologist", by_name["oncology"], ["oncologist"]))

    if structured.n_capability_claims >= 3 and not structured.has_equipment_evidence:
        out.append(("claims_with_zero_equipment", "structured 'equipment' column is empty []"))

    return out


# ----- Sub-scores ---------------------------------------------------------

def completeness(extraction: FacilityExtraction, structured: StructuredSignals) -> float:
    fields_filled = 0
    fields_total = 8
    if extraction.capabilities:
        fields_filled += 1
    if extraction.staffing.total_doctors_estimate is not None or structured.number_doctors:
        fields_filled += 1
    if extraction.inpatient_beds_estimate is not None or structured.capacity:
        fields_filled += 1
    if extraction.hours.is_24x7 or extraction.hours.has_emergency_dept:
        fields_filled += 1
    if structured.has_equipment_evidence:
        fields_filled += 1
    if any(c.evidence_sentences for c in extraction.capabilities):
        fields_filled += 1
    if structured.has_official_website:
        fields_filled += 1
    if structured.has_followers:
        fields_filled += 1
    return fields_filled / fields_total


def consistency(extraction: FacilityExtraction, structured: StructuredSignals) -> tuple[float, list[tuple[str, str]]]:
    flags = contradiction_flags(extraction, structured)
    n_high_acuity_claims = sum(
        1 for c in extraction.capabilities if c.claimed and c.name in HIGH_ACUITY_CAPABILITIES
    ) or 1
    raw = 1.0 - min(1.0, len(flags) / (n_high_acuity_claims + 1))
    return max(0.0, raw), flags


def source_agreement(verdicts: Iterable[ValidatorVerdict]) -> float:
    verdicts = list(verdicts)
    if not verdicts:
        return 1.0
    return sum(1 for v in verdicts if v.agreement) / len(verdicts)


# ----- Public API ---------------------------------------------------------

def score_facility(
    extraction: FacilityExtraction,
    structured: StructuredSignals,
    verdicts: Iterable[ValidatorVerdict] = (),
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> TrustScore:
    comp = completeness(extraction, structured)
    cons, flags = consistency(extraction, structured)
    agree = source_agreement(verdicts)
    w_comp, w_cons, w_agree = weights
    raw = w_comp * comp + w_cons * cons + w_agree * agree
    return TrustScore(
        facility_id=extraction.facility_id,
        score=round(raw * 100.0, 2),
        completeness=round(comp, 4),
        consistency=round(cons, 4),
        source_agreement=round(agree, 4),
        flags=[f for f, _ in flags],
        flag_evidence=[c for _, c in flags],
    )
