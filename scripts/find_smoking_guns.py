"""Surface the most demo-worthy "smoking-gun" contradictions in the cache.

A smoking-gun is a facility that:
* claims a high-acuity capability (ICU / surgery / oncology / cardiac / NICU)
* has a contradiction flag from `agents.trust.contradiction_flags`
* has a non-trivial verbatim cited sentence (>= 6 words)
* and lives in a state that is not already represented (geographic spread).

Writes `data/cache/smoking_guns.json` - a small, hand-curatable list the
Streamlit "Featured findings" card and the README quote.

Usage:
    python scripts/find_smoking_guns.py
    python scripts/find_smoking_guns.py --top 5 --pin 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "cache"

PRIORITY_FLAGS = (
    "icu_no_ventilator",
    "surgery_no_anesthesiologist",
    "oncology_no_oncologist",
    "cardiac_no_cardiologist",
    "nicu_no_pediatrician",
    "emergency_24x7_understaffed",
    "icu_missing",
    "general_surgery_missing",
)

HIGH_ACUITY = {
    "icu", "nicu", "dialysis", "oncology", "trauma_emergency",
    "general_surgery", "emergency_appendectomy", "cardiac_care",
}

PRETTY_HEADLINE = {
    "icu_no_ventilator": "ICU claimed but ventilator not documented",
    "surgery_no_anesthesiologist": "Surgery claimed but no anesthesiologist on staff",
    "oncology_no_oncologist": "Oncology claimed but no oncologist documented",
    "cardiac_no_cardiologist": "Cardiac care claimed but no cardiologist",
    "nicu_no_pediatrician": "NICU claimed but no pediatrician/neonatologist",
    "emergency_24x7_understaffed": "24x7 emergency claimed with <=1 doctor",
}


def _to_list(xs):
    if xs is None:
        return []
    try:
        return list(xs)
    except TypeError:
        return []


def _flag_root(flag: str) -> str:
    """Strip 'foo_missing: equipment: any of...' down to 'foo_missing'."""
    return flag.split(":")[0].strip()


def _missing_requirement(flag: str) -> str:
    """Return the human-readable missing chip from the flag string."""
    parts = flag.split(":")
    if len(parts) <= 1:
        return parts[0].strip()
    rest = ":".join(parts[1:]).strip()
    rest = rest.replace("equipment: any of", "equipment").replace("staff: any of", "staff")
    return rest[:120]


CAP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "icu": ("icu", "intensive care"),
    "nicu": ("nicu", "neonatal intensive"),
    "general_surgery": ("surgery", "surgical", "operation"),
    "emergency_appendectomy": ("appendec",),
    "oncology": ("oncolog", "cancer", "chemo"),
    "cardiac_care": ("cardiac", "cardio", "heart"),
    "trauma_emergency": ("trauma", "emergency"),
    "dialysis": ("dialysis",),
    "neonatal_care": ("neonatal", "newborn"),
}


def _clean_sentence(s: str, prefer_keywords: tuple[str, ...] = ()) -> str:
    """Pick the single most-relevant sub-line from a multi-line evidence chunk."""
    text = s.strip().replace("\r", "")
    for hdr in ("DESCRIPTION:", "EQUIPMENT:", "SPECIALTIES:", "PROCEDURES:", "CAPABILITIES:"):
        text = text.replace(hdr, "")
    raw_parts = [p.strip(" -*") for p in text.split("\n") if p.strip(" -*")]
    long_parts = [p for p in raw_parts if len(p.split()) >= 6]

    chosen: str | None = None
    if prefer_keywords:
        # Try keyword hits in long_parts first, then any keyword-matching line >= 3 words.
        kw_hits = [p for p in long_parts if any(k.lower() in p.lower() for k in prefer_keywords)]
        if not kw_hits:
            kw_hits = [
                p for p in raw_parts
                if len(p.split()) >= 3 and any(k.lower() in p.lower() for k in prefer_keywords)
            ]
        if kw_hits:
            chosen = kw_hits[0]
    if chosen is None:
        chosen = long_parts[0] if long_parts else (raw_parts[0] if raw_parts else text.strip())

    if len(chosen) > 240:
        chosen = chosen[:237].rsplit(" ", 1)[0] + "..."
    return chosen


def _claim_evidence(claim_dict: dict, capability: str) -> str | None:
    """Find the verbatim sentence backing a specific capability claim."""
    if not isinstance(claim_dict, dict):
        return None
    keywords = CAP_KEYWORDS.get(capability, (capability.replace("_", " "),))
    for c in claim_dict.get("capabilities", []):
        if c.get("name") == capability and c.get("evidence_sentences"):
            for s in c["evidence_sentences"]:
                if not isinstance(s, str):
                    continue
                cleaned = _clean_sentence(s, prefer_keywords=keywords)
                if len(cleaned.split()) >= 3:
                    return cleaned
    return None


def _capability_for_flag(flag: str, claimed: list[str]) -> str | None:
    root = _flag_root(flag)
    mapping = {
        "icu_no_ventilator": "icu",
        "icu_missing": "icu",
        "surgery_no_anesthesiologist": "general_surgery",
        "general_surgery_missing": "general_surgery",
        "emergency_appendectomy_missing": "emergency_appendectomy",
        "oncology_no_oncologist": "oncology",
        "oncology_missing": "oncology",
        "cardiac_no_cardiologist": "cardiac_care",
        "cardiac_care_missing": "cardiac_care",
        "nicu_no_pediatrician": "nicu",
        "nicu_missing": "nicu",
        "dialysis_missing": "dialysis",
        "neonatal_care_missing": "neonatal_care",
        "trauma_emergency_missing": "trauma_emergency",
        "emergency_24x7_understaffed": None,
    }
    cap = mapping.get(root)
    if cap is None:
        for c in claimed:
            if c in HIGH_ACUITY:
                return c
    return cap


def _priority_idx(flag: str) -> int:
    root = _flag_root(flag)
    for i, p in enumerate(PRIORITY_FLAGS):
        if root.startswith(p):
            return i
    return len(PRIORITY_FLAGS) + 1


def find_smoking_guns(top: int = 5) -> list[dict]:
    summaries = pd.read_parquet(CACHE / "facility_summaries.parquet")
    trust = pd.read_parquet(CACHE / "trust_scores.parquet")
    extractions = pd.read_parquet(CACHE / "capability_claims.parquet")

    df = summaries.merge(
        trust[["facility_id", "score", "flags", "flag_evidence"]],
        on="facility_id",
        how="left",
    ).merge(
        extractions[["facility_id", "extraction_json"]],
        on="facility_id",
        how="left",
    )

    candidates = []
    for _, row in df.iterrows():
        flags = _to_list(row.get("flags"))
        claimed = _to_list(row.get("claimed_capabilities"))
        if not flags or len(claimed) < 2:
            continue
        if not any(c in HIGH_ACUITY for c in claimed):
            continue
        # pick the highest-priority flag for this row
        best = None
        for f in flags:
            if not isinstance(f, str):
                continue
            idx = _priority_idx(f)
            if idx >= len(PRIORITY_FLAGS):
                continue
            if best is None or idx < best[0]:
                best = (idx, f)
        if best is None:
            continue
        flag = best[1]
        cap = _capability_for_flag(flag, claimed)
        try:
            extraction = json.loads(row["extraction_json"]) if row.get("extraction_json") else {}
        except Exception:
            extraction = {}
        cited = _claim_evidence(extraction, cap) if cap else None
        if not cited:
            evidences = _to_list(row.get("flag_evidence"))
            for ev in evidences:
                if not isinstance(ev, str):
                    continue
                cleaned = _clean_sentence(ev)
                if len(cleaned.split()) >= 6:
                    cited = cleaned
                    break
        if not cited:
            continue
        candidates.append({
            "facility_id": row["facility_id"],
            "name": row.get("name"),
            "city": row.get("city"),
            "state": row.get("state"),
            "district": row.get("district"),
            "trust_score": float(row.get("score") or 0.0),
            "claimed_capabilities": claimed,
            "flag": flag,
            "flag_root": _flag_root(flag),
            "headline": PRETTY_HEADLINE.get(_flag_root(flag), flag),
            "missing_requirement": _missing_requirement(flag),
            "cited_sentence": cited,
            "capability": cap,
            "_priority": best[0],
        })

    # sort by (priority, trust_score asc to favour suspect ones)
    candidates.sort(key=lambda c: (c["_priority"], c["trust_score"]))

    # Stratify across states for geographic spread.
    seen_states: set[str] = set()
    seen_caps: set[str] = set()
    chosen: list[dict] = []
    for c in candidates:
        state = c.get("state") or "unknown"
        cap = c.get("capability") or ""
        if state in seen_states and len(chosen) >= 2:
            continue
        if cap and cap in seen_caps and len(chosen) >= 3:
            continue
        chosen.append(c)
        seen_states.add(state)
        if cap:
            seen_caps.add(cap)
        if len(chosen) >= top:
            break

    for c in chosen:
        c.pop("_priority", None)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=5, help="How many findings to keep.")
    parser.add_argument(
        "--out",
        default=str(CACHE / "smoking_guns.json"),
        help="Output JSON path (default: data/cache/smoking_guns.json).",
    )
    args = parser.parse_args()
    findings = find_smoking_guns(top=args.top)
    Path(args.out).write_text(json.dumps(findings, indent=2, ensure_ascii=False))
    print(f"wrote {args.out} with {len(findings)} findings")
    for i, f in enumerate(findings, 1):
        print(
            f"  {i}. [{f['state']}] {f['name']} - {f['headline']} "
            f"(trust {f['trust_score']:.0f})"
        )


if __name__ == "__main__":
    main()
