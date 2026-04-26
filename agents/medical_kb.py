"""Knowledge base of medical-standards expectations.

Used by both the Validator Agent (cross-check claims against required
equipment / staff) and the Trust Scorer (penalize claims that violate these
rules).

Sources are intentionally conservative - we encode what is uncontroversial in
WHO / Indian Public Health Standards (IPHS) facility guidelines rather than
trying to model every nuance.

The matching is keyword-based with a small synonym list so we tolerate the
free-form spelling variations in Indian facility notes (`x-ray`, `xray`,
`X Ray`, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from schemas.virtue_foundation import CapabilityName


@dataclass(frozen=True)
class Requirement:
    capability: CapabilityName
    required_equipment_any: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    required_staff_any: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    notes: str = ""

    def missing_for(self, equipment: Iterable[str], staff: Iterable[str]) -> list[str]:
        eq_blob = " | ".join(e.lower() for e in equipment)
        st_blob = " | ".join(s.lower() for s in staff)
        missing: list[str] = []
        for group in self.required_equipment_any:
            if not any(syn in eq_blob for syn in group):
                missing.append(f"equipment: any of {list(group)}")
        for group in self.required_staff_any:
            if not any(syn in st_blob for syn in group):
                missing.append(f"staff: any of {list(group)}")
        return missing


KB: dict[CapabilityName, Requirement] = {
    "icu": Requirement(
        capability="icu",
        required_equipment_any=(
            ("ventilator", "bipap", "cpap"),
            ("monitor", "cardiac monitor", "vital sign"),
        ),
        required_staff_any=(
            ("intensivist", "critical care", "icu nurse", "anesthesiologist"),
        ),
        notes="A functional ICU expects ventilators + continuous monitoring + trained staff.",
    ),
    "nicu": Requirement(
        capability="nicu",
        required_equipment_any=(
            ("incubator", "warmer", "neonatal ventilator", "phototherapy"),
        ),
        required_staff_any=(
            ("neonatologist", "pediatrician", "neonatal nurse"),
        ),
        notes="Neonatal ICU expects incubator/warmer + neonatologist or pediatrician.",
    ),
    "neonatal_care": Requirement(
        capability="neonatal_care",
        required_staff_any=(
            ("pediatrician", "neonatologist"),
        ),
    ),
    "dialysis": Requirement(
        capability="dialysis",
        required_equipment_any=(
            ("dialysis machine", "hemodialysis", "ro plant", "dialyzer"),
        ),
        required_staff_any=(
            ("nephrologist", "dialysis technician"),
        ),
    ),
    "oncology": Requirement(
        capability="oncology",
        required_staff_any=(
            ("oncologist", "radiation oncologist", "medical oncologist"),
        ),
        notes="At minimum needs an oncologist; chemo/radiation imply more.",
    ),
    "trauma_emergency": Requirement(
        capability="trauma_emergency",
        required_equipment_any=(
            ("trauma kit", "defibrillator", "emergency", "resuscitation"),
        ),
    ),
    "general_surgery": Requirement(
        capability="general_surgery",
        required_equipment_any=(
            ("operation theatre", "ot table", "operating", "anesthesia machine", "surgical"),
        ),
        required_staff_any=(
            ("surgeon", "general surgeon"),
            ("anesthesiologist", "anesthetist"),
        ),
        notes="Surgery without an anesthesiologist is the canonical Trust Scorer flag.",
    ),
    "emergency_appendectomy": Requirement(
        capability="emergency_appendectomy",
        required_equipment_any=(
            ("operation theatre", "ot ", "operating room", "anesthesia machine"),
        ),
        required_staff_any=(
            ("surgeon",),
            ("anesthesiologist", "anesthetist"),
        ),
        notes="Emergency appendectomy = OT + surgeon + anesthesiologist + 24x7.",
    ),
    "cardiac_care": Requirement(
        capability="cardiac_care",
        required_equipment_any=(
            ("ecg", "ekg", "echocardiogram", "cath lab", "cardiac monitor"),
        ),
        required_staff_any=(
            ("cardiologist",),
        ),
    ),
    "obgyn_delivery": Requirement(
        capability="obgyn_delivery",
        required_staff_any=(
            ("obstetrician", "gynecologist", "midwife"),
        ),
    ),
    "radiology_xray": Requirement(
        capability="radiology_xray",
        required_equipment_any=(
            ("x-ray", "x ray", "xray", "radiograph"),
        ),
    ),
    "radiology_ct": Requirement(
        capability="radiology_ct",
        required_equipment_any=(
            ("ct scan", "ct ", "computed tomography"),
        ),
        required_staff_any=(
            ("radiologist",),
        ),
    ),
    "radiology_mri": Requirement(
        capability="radiology_mri",
        required_equipment_any=(
            ("mri", "magnetic resonance"),
        ),
        required_staff_any=(
            ("radiologist",),
        ),
    ),
    "ultrasound": Requirement(
        capability="ultrasound",
        required_equipment_any=(
            ("ultrasound", "sonography", "usg"),
        ),
    ),
    "blood_bank": Requirement(
        capability="blood_bank",
        required_equipment_any=(
            ("blood bank", "blood storage", "refrigerator"),
        ),
    ),
    "ambulance": Requirement(
        capability="ambulance",
        required_equipment_any=(("ambulance",),),
    ),
    "oxygen_supply": Requirement(
        capability="oxygen_supply",
        required_equipment_any=(
            ("oxygen", "o2 cylinder", "psa plant", "concentrator"),
        ),
    ),
    "emergency_24x7": Requirement(
        capability="emergency_24x7",
        notes="No equipment requirement, but >=2 doctors on rotation expected.",
    ),
    "lab_diagnostics": Requirement(
        capability="lab_diagnostics",
        required_equipment_any=(
            ("lab", "pathology", "analyzer", "microscope"),
        ),
    ),
}


def required_for(capability: CapabilityName) -> Requirement | None:
    return KB.get(capability)
