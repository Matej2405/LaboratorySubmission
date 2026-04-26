"""Virtue Foundation extension schema for Indian healthcare facility extraction.

Single source of truth - imported by the extractor, validator, trust scorer,
the eval harness, and the Streamlit app.

Design notes
------------
* Every claim field carries `evidence_sentences` so we can do row-level citation
  per the "Agentic Traceability" stretch goal.
* Capability names are a closed `Literal` set so structured output from
  Agent Bricks (`response_format=pydantic`) gives us SQL-friendly tags.
* The high-acuity capabilities (Oncology, Dialysis, Trauma, NICU, ICU,
  emergency surgery) are split out so Trust Scorer rules can target them.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


CapabilityName = Literal[
    "icu",
    "nicu",
    "dialysis",
    "oncology",
    "trauma_emergency",
    "general_surgery",
    "emergency_appendectomy",
    "cardiac_care",
    "obgyn_delivery",
    "neonatal_care",
    "radiology_xray",
    "radiology_ct",
    "radiology_mri",
    "ultrasound",
    "blood_bank",
    "ambulance",
    "oxygen_supply",
    "emergency_24x7",
    "pharmacy",
    "lab_diagnostics",
    "telemedicine",
    "mental_health",
    "dental",
    "ophthalmology",
    "physiotherapy",
]

HIGH_ACUITY_CAPABILITIES: tuple[CapabilityName, ...] = (
    "icu",
    "nicu",
    "dialysis",
    "oncology",
    "trauma_emergency",
    "general_surgery",
    "emergency_appendectomy",
    "cardiac_care",
    "neonatal_care",
)


class CapabilityClaim(BaseModel):
    name: CapabilityName = Field(
        ...,
        description="Canonical capability tag the facility appears to offer.",
    )
    claimed: bool = Field(
        ...,
        description="True if the facility's notes assert this capability.",
    )
    functional_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How confident are we that this capability is functional, not "
            "just claimed? 0=mere mention, 1=multiple corroborating signals."
        ),
    )
    evidence_sentences: List[str] = Field(
        default_factory=list,
        description="Verbatim sentences from the source notes that support the claim.",
    )
    supporting_equipment: List[str] = Field(
        default_factory=list,
        description="Equipment items mentioned in the notes that back the claim.",
    )
    supporting_staff: List[str] = Field(
        default_factory=list,
        description="Specialists / staff types mentioned that back the claim.",
    )

    @field_validator("evidence_sentences", "supporting_equipment", "supporting_staff")
    @classmethod
    def _strip_and_dedupe(cls, v: List[str]) -> List[str]:
        seen: list[str] = []
        for item in v or []:
            s = (item or "").strip()
            if s and s not in seen:
                seen.append(s)
        return seen


class StaffingProfile(BaseModel):
    total_doctors_estimate: Optional[int] = Field(
        None, description="Best estimate of doctor headcount from notes."
    )
    has_anesthesiologist: bool = False
    has_surgeon: bool = False
    has_pediatrician: bool = False
    has_cardiologist: bool = False
    has_oncologist: bool = False
    has_radiologist: bool = False
    uses_part_time_doctors: bool = Field(
        False, description="True if the facility relies on visiting/part-time doctors."
    )
    uses_visiting_specialists: bool = False
    evidence_sentences: List[str] = Field(default_factory=list)


class OperatingHours(BaseModel):
    is_24x7: bool = False
    has_emergency_dept: bool = False
    evidence_sentences: List[str] = Field(default_factory=list)


class FacilityExtraction(BaseModel):
    """Top-level structured extraction for a single facility row."""

    facility_id: str
    capabilities: List[CapabilityClaim] = Field(default_factory=list)
    staffing: StaffingProfile = Field(default_factory=StaffingProfile)
    hours: OperatingHours = Field(default_factory=OperatingHours)
    inpatient_beds_estimate: Optional[int] = None
    icu_beds_estimate: Optional[int] = None
    extraction_notes: Optional[str] = Field(
        None,
        description=(
            "Free-form summary the LLM emits; useful for debugging and for the "
            "Vector Search summary index."
        ),
    )

    def claimed_capabilities(self) -> List[str]:
        return [c.name for c in self.capabilities if c.claimed]

    def evidence_for(self, capability: str) -> List[str]:
        for c in self.capabilities:
            if c.name == capability and c.claimed:
                return c.evidence_sentences
        return []


class ValidatorVerdict(BaseModel):
    """Output of the self-correction Validator Agent."""

    facility_id: str
    capability: CapabilityName
    original_claim: bool
    validator_claim: bool
    agreement: bool
    rationale: str
    missing_required: List[str] = Field(default_factory=list)
    flagged_evidence: List[str] = Field(default_factory=list)


class TrustScore(BaseModel):
    """Per-facility 0-100 trust score plus contributing components."""

    facility_id: str
    score: float = Field(..., ge=0.0, le=100.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    consistency: float = Field(..., ge=0.0, le=1.0)
    source_agreement: float = Field(..., ge=0.0, le=1.0)
    flags: List[str] = Field(default_factory=list)
    flag_evidence: List[str] = Field(default_factory=list)


def empty_extraction(facility_id: str) -> FacilityExtraction:
    return FacilityExtraction(facility_id=facility_id)
