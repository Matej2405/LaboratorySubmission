"""Build a local parquet cache from the raw Excel + a synthetic extraction.

When no Databricks workspace is available, this script lets the Streamlit app
run end-to-end against a deterministic local snapshot of the pipeline:

    python scripts/build_local_cache.py
    streamlit run app/streamlit_app.py

It does NOT call any LLM. The "extraction" is rule-based fan-out from the
existing structured columns (specialties / procedure / capability / equipment),
intended for the offline demo and the eval harness baseline.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.text_utils import (
    normalize_state, extract_pin, parse_json_list, to_unstructured_blob,
    split_sentences,
)
from agents.medical_kb import KB
from agents.trust import score_facility, StructuredSignals
from schemas.virtue_foundation import (
    CapabilityClaim, FacilityExtraction, OperatingHours, StaffingProfile,
    HIGH_ACUITY_CAPABILITIES,
)


CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
RAW = ROOT / "data" / "raw" / "VF_Hackathon_Dataset_India_Large.xlsx"


# ---------- Heuristic capability detector ---------------------------------

CAP_PATTERNS: dict[str, list[str]] = {
    "icu": [r"\bicu\b", r"intensive care"],
    "nicu": [r"\bnicu\b", r"neonatal intensive"],
    "dialysis": [r"dialysis", r"hemodialys"],
    "oncology": [r"oncolog", r"cancer", r"chemother", r"radiation therapy"],
    "trauma_emergency": [r"trauma", r"emergency dep", r"casualty"],
    "general_surgery": [r"surgery", r"surgical", r"operation theatre", r"\bot\b"],
    "emergency_appendectomy": [r"appendectom", r"emergency appendec"],
    "cardiac_care": [r"cardiac", r"cardiolog", r"heart"],
    "obgyn_delivery": [r"obgyn", r"obstetri", r"gyn", r"deliver", r"maternity"],
    "neonatal_care": [r"neonat", r"newborn"],
    "radiology_xray": [r"x[- ]?ray", r"radiograph"],
    "radiology_ct": [r"\bct scan\b", r"computed tomography"],
    "radiology_mri": [r"\bmri\b", r"magnetic resonance"],
    "ultrasound": [r"ultrasound", r"sonograph", r"\busg\b"],
    "blood_bank": [r"blood bank", r"transfusion"],
    "ambulance": [r"ambulance"],
    "oxygen_supply": [r"oxygen", r"o2 cylinder"],
    "emergency_24x7": [r"24[ /]?7", r"24 hours", r"round[- ]the[- ]clock"],
    "pharmacy": [r"pharmacy", r"medical store"],
    "lab_diagnostics": [r"\blab\b", r"pathology", r"diagnostic"],
    "telemedicine": [r"tele[- ]?medicine", r"online consult"],
    "mental_health": [r"mental health", r"psychiatr", r"counsel"],
    "dental": [r"dental", r"dentist"],
    "ophthalmology": [r"ophthalm", r"\beye\b"],
    "physiotherapy": [r"physiothera", r"\bpt\b", r"rehabilitat"],
}

STAFF_PATTERNS = {
    "anesthesiologist": [r"anesthesiolog", r"anesthet"],
    "surgeon": [r"surgeon", r"surgical"],
    "pediatrician": [r"pediatric", r"paediatric"],
    "cardiologist": [r"cardiolog"],
    "oncologist": [r"oncolog"],
    "radiologist": [r"radiolog"],
    "obstetrician": [r"obstetri", r"gyn"],
}


def _matches(text: str, patterns: list[str]) -> list[str]:
    out = []
    for sent in split_sentences(text):
        for p in patterns:
            if re.search(p, sent, flags=re.IGNORECASE):
                out.append(sent)
                break
    return out


def _heuristic_extract(row: pd.Series) -> FacilityExtraction:
    blob = to_unstructured_blob(
        row.get("description"),
        parse_json_list(row.get("specialties")),
        parse_json_list(row.get("procedure")),
        parse_json_list(row.get("capability")),
        parse_json_list(row.get("equipment")),
    )
    eq_list = parse_json_list(row.get("equipment"))
    full_text = blob

    capabilities: list[CapabilityClaim] = []
    claimed_text = full_text.lower()
    for cap, patterns in CAP_PATTERNS.items():
        evidence = _matches(full_text, patterns)
        if not evidence:
            continue
        eq_hits = [e for e in eq_list if any(re.search(p, e, re.IGNORECASE) for p in patterns)]
        staff_hits: list[str] = []
        for s_name, s_patterns in STAFF_PATTERNS.items():
            if any(re.search(p, claimed_text, re.IGNORECASE) for p in s_patterns):
                staff_hits.append(s_name)
        functional = 0.4
        if eq_hits:
            functional += 0.3
        if staff_hits:
            functional += 0.2
        functional = min(0.95, functional)
        capabilities.append(CapabilityClaim(
            name=cap,  # type: ignore[arg-type]
            claimed=True,
            functional_confidence=functional,
            evidence_sentences=evidence[:3],
            supporting_equipment=eq_hits[:5],
            supporting_staff=staff_hits[:5],
        ))

    staffing = StaffingProfile(
        total_doctors_estimate=int(row["numberDoctors"]) if pd.notna(row.get("numberDoctors")) else None,
        has_anesthesiologist=any(re.search(p, full_text, re.IGNORECASE) for p in STAFF_PATTERNS["anesthesiologist"]),
        has_surgeon=any(re.search(p, full_text, re.IGNORECASE) for p in STAFF_PATTERNS["surgeon"]),
        has_pediatrician=any(re.search(p, full_text, re.IGNORECASE) for p in STAFF_PATTERNS["pediatrician"]),
        has_cardiologist=any(re.search(p, full_text, re.IGNORECASE) for p in STAFF_PATTERNS["cardiologist"]),
        has_oncologist=any(re.search(p, full_text, re.IGNORECASE) for p in STAFF_PATTERNS["oncologist"]),
        has_radiologist=any(re.search(p, full_text, re.IGNORECASE) for p in STAFF_PATTERNS["radiologist"]),
        uses_part_time_doctors=bool(re.search(r"part[- ]?time|visiting", full_text, re.IGNORECASE)),
        uses_visiting_specialists=bool(re.search(r"visiting", full_text, re.IGNORECASE)),
    )
    hours = OperatingHours(
        is_24x7=bool(re.search(r"24[ /]?7|24 hours|round[- ]the[- ]clock", full_text, re.IGNORECASE)),
        has_emergency_dept=bool(re.search(r"emergency dep|casualty", full_text, re.IGNORECASE)),
        evidence_sentences=_matches(full_text, [r"24[ /]?7", r"emergency"])[:2],
    )
    return FacilityExtraction(
        facility_id=str(row["facility_id"]),
        capabilities=capabilities,
        staffing=staffing,
        hours=hours,
        inpatient_beds_estimate=int(row["capacity"]) if pd.notna(row.get("capacity")) else None,
        extraction_notes=full_text[:240],
    )


def main() -> None:
    print(f"Loading {RAW} ...")
    pdf = pd.read_excel(RAW, engine="openpyxl")
    pdf.columns = [c.strip().replace(" ", "_").replace("/", "_").replace("-", "_") for c in pdf.columns]
    if "facility_id" not in pdf.columns:
        pdf.insert(0, "facility_id", [f"vf_{i:06d}" for i in range(len(pdf))])

    print("Building silver-equivalent ...")
    pdf["state"] = pdf["address_stateOrRegion"].apply(normalize_state)
    pdf["pincode"] = pdf.apply(
        lambda r: extract_pin(r.get("address_zipOrPostcode"), r.get("address_line1"), r.get("address_line2")),
        axis=1,
    )
    pdf["city"] = pdf["address_city"]
    for col in ("specialties", "procedure", "capability", "equipment"):
        pdf[f"{col}_list"] = pdf[col].apply(parse_json_list)
    pdf["unstructured_blob"] = pdf.apply(
        lambda r: to_unstructured_blob(
            r.get("description"), r["specialties_list"], r["procedure_list"],
            r["capability_list"], r["equipment_list"],
        ),
        axis=1,
    )

    print("Running heuristic extraction (no LLM) ...")
    extractions = pdf.apply(_heuristic_extract, axis=1).tolist()
    extraction_rows = [{
        "facility_id": ex.facility_id,
        "extraction_json": ex.model_dump_json(),
        "claimed_capabilities": [c.name for c in ex.capabilities if c.claimed],
        "n_capabilities_claimed": sum(1 for c in ex.capabilities if c.claimed),
        "n_evidence_sentences": sum(len(c.evidence_sentences) for c in ex.capabilities),
        "extractor_endpoint": "local-heuristic",
    } for ex in extractions]
    gold = pd.DataFrame(extraction_rows)

    print("Scoring trust ...")
    trust_rows = []
    for ex, (_, row) in zip(extractions, pdf.iterrows()):
        signals = StructuredSignals(
            number_doctors=row.get("numberDoctors"),
            capacity=row.get("capacity"),
            has_equipment_evidence=bool(row["equipment_list"]),
            n_capability_claims=len(row["capability_list"]) + len(row["procedure_list"]),
            has_followers=bool(row.get("engagement_metrics_n_followers")),
            has_official_website=bool(row.get("officialWebsite")),
        )
        trust_rows.append(score_facility(ex, signals).model_dump())
    trust = pd.DataFrame(trust_rows)

    print("Composing summaries ...")
    summaries = pd.DataFrame({
        "facility_id": pdf["facility_id"],
        "name": pdf["name"],
        "city": pdf["city"],
        "state": pdf["state"],
        "pincode": pdf["pincode"],
        "facilityTypeId": pdf["facilityTypeId"],
        "operatorTypeId": pdf["operatorTypeId"],
        "specialties_list": pdf["specialties_list"],
        "claimed_capabilities": gold["claimed_capabilities"],
        "trust_score": trust["score"],
        "trust_flags": trust["flags"],
        "latitude": pdf["latitude"],
        "longitude": pdf["longitude"],
        "summary_text": pdf.apply(lambda r: " | ".join([
            str(r.get("name", "")), f"{r.get('city','')}, {r.get('state','')}",
            f"type={r.get('facilityTypeId','')}",
            f"specialties={','.join(r['specialties_list'])}",
            str(r.get("description") or ""),
        ]), axis=1),
        "description": pdf["description"],
    })

    print("Assigning districts (point-in-polygon) ...")
    try:
        from scripts.assign_districts import assign_districts
        summaries = assign_districts(summaries)
        district_coverage = summaries["district"].notna().mean() * 100
        print(f"  district coverage: {district_coverage:.1f}%")
    except FileNotFoundError as e:
        print(f"  skipping (run scripts/fetch_reference_data.py first): {e}")
        summaries["district"] = None
        summaries["district_state"] = None

    print("Joining Census 2011 population (fuzzy) ...")
    try:
        from scripts.join_population import join_population
        summaries, unmapped = join_population(summaries)
        pop_coverage = summaries["district_population"].notna().mean() * 100
        print(f"  population coverage: {pop_coverage:.1f}%")
        if unmapped:
            uniq = sorted(set(unmapped))
            (CACHE / "unmapped_districts.json").write_text(
                json.dumps([{"state": s, "district": d} for s, d in uniq], indent=2)
            )
            print(f"  {len(uniq)} unmapped (state, district) pairs - logged to unmapped_districts.json")
    except FileNotFoundError as e:
        print(f"  skipping: {e}")
        summaries["district_population"] = None

    print("Building chunks ...")
    fid_to_district = dict(zip(summaries["facility_id"], summaries["district"]))
    chunks_rows = []
    for _, r in pdf.iterrows():
        for i, sent in enumerate(split_sentences(r["unstructured_blob"])):
            chunks_rows.append({
                "chunk_id": f"{r['facility_id']}::{i}",
                "facility_id": r["facility_id"],
                "state": r["state"],
                "pincode": r["pincode"],
                "city": r["city"],
                "district": fid_to_district.get(r["facility_id"]),
                "chunk_idx": i,
                "chunk_text": sent,
            })
    chunks = pd.DataFrame(chunks_rows)

    print("Computing prevalence + population-aware deserts ...")
    from agents.confidence import trust_weighted_proportion, desert_index
    cap_universe = list(CAP_PATTERNS.keys())

    # State-level prevalence (unchanged) ----------------------------------
    prev_rows = []
    for state, sub in summaries.groupby("state"):
        weights = (sub["trust_score"].fillna(50.0) / 100.0).tolist()
        for cap in cap_universe:
            indicators = sub["claimed_capabilities"].apply(
                lambda xs: 1 if isinstance(xs, list) and cap in xs else 0
            ).tolist()
            iv = trust_weighted_proportion(weights, indicators)
            prev_rows.append({
                "state": state, "capability": cap,
                "p_hat": iv.point, "low": iv.lower, "high": iv.upper,
                "n_eff": iv.n, "n_facilities": len(sub),
            })

    # District-level deserts (population-aware) ---------------------------
    # Prefer the GeoJSON-derived `district_state` over the noisy free-form `state`.
    if "district_state" in summaries.columns:
        summaries["state_canonical"] = summaries["district_state"].fillna(summaries["state"])
    else:
        summaries["state_canonical"] = summaries["state"]

    desert_rows = []
    pop_lookup: dict[tuple[str, str], int] = {}
    if "district" in summaries.columns and "district_population" in summaries.columns:
        for (state, district), sub in summaries.dropna(subset=["district"]).groupby(["state_canonical", "district"]):
            pop_vals = sub["district_population"].dropna()
            if not pop_vals.empty:
                pop_lookup[(state, district)] = int(pop_vals.iloc[0])

    if pop_lookup:
        for (state, district), sub in summaries.dropna(subset=["district"]).groupby(["state_canonical", "district"]):
            n_fac = len(sub)
            population = pop_lookup.get((state, district))
            if population is None:
                continue
            weights = (sub["trust_score"].fillna(50.0) / 100.0).tolist()
            for cap in cap_universe:
                indicators = sub["claimed_capabilities"].apply(
                    lambda xs: 1 if isinstance(xs, list) and cap in xs else 0
                ).tolist()
                iv = trust_weighted_proportion(weights, indicators)
                di = desert_index(population, n_fac, iv.point)
                desert_rows.append({
                    "state": state,
                    "district": district,
                    "capability": cap,
                    "p_hat": iv.point,
                    "low": iv.lower,
                    "high": iv.upper,
                    "n_facilities": n_fac,
                    "population": population,
                    "desert_score": round(di.point, 2),
                    "desert_low": round(di.lower, 2),
                    "desert_high": round(di.upper, 2),
                })
    else:
        # Fallback to state-level when no district population is available.
        for state, sub in summaries.groupby("state"):
            weights = (sub["trust_score"].fillna(50.0) / 100.0).tolist()
            for cap in cap_universe:
                indicators = sub["claimed_capabilities"].apply(
                    lambda xs: 1 if isinstance(xs, list) and cap in xs else 0
                ).tolist()
                iv = trust_weighted_proportion(weights, indicators)
                desert_rows.append({
                    "state": state, "district": None, "capability": cap,
                    "p_hat": iv.point, "low": iv.lower, "high": iv.upper,
                    "n_facilities": len(sub), "population": None,
                    "desert_score": round(1 - iv.upper, 4),
                    "desert_low": None, "desert_high": None,
                })

    print("Writing parquet cache ...")
    summaries.to_parquet(CACHE / "facility_summaries.parquet", index=False)
    trust.to_parquet(CACHE / "trust_scores.parquet", index=False)
    gold.to_parquet(CACHE / "capability_claims.parquet", index=False)
    chunks.to_parquet(CACHE / "notes_chunks.parquet", index=False)
    pd.DataFrame(prev_rows).to_parquet(CACHE / "capability_prevalence.parquet", index=False)
    pd.DataFrame(desert_rows).to_parquet(CACHE / "desert_scores.parquet", index=False)

    silver = pd.DataFrame({
        "facility_id": pdf["facility_id"],
        "name": pdf["name"],
        "city": pdf["city"],
        "state": pdf["state"],
        "facilityTypeId": pdf["facilityTypeId"],
        "operatorTypeId": pdf["operatorTypeId"],
        "numberDoctors": pdf.get("numberDoctors"),
        "capacity": pdf.get("capacity"),
        "unstructured_blob": pdf["unstructured_blob"],
    })
    silver.to_parquet(CACHE / "facility_silver.parquet", index=False)

    print("Done. Files in", CACHE)
    for f in CACHE.glob("*.parquet"):
        print(" -", f.name, f.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
