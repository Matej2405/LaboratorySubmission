"""Build a stratified 50-row golden subset for human labeling.

Stratification:
* facility type x state quartile (urban / semi-urban / rural surrogate based
  on web-presence signal)
* trust-score band (low / mid / high) so we get hard cases
* presence of high-acuity claim (icu / oncology / surgery / dialysis)

Output: `evals/golden_subset.csv` with the columns a human needs to label
plus blank columns for the labels themselves. The eval notebook reads this
back and computes precision / recall.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "cache"
OUT = ROOT / "evals" / "golden_subset.csv"


HIGH_ACUITY = {"icu", "nicu", "dialysis", "oncology", "trauma_emergency",
               "general_surgery", "emergency_appendectomy", "cardiac_care"}


def main(n: int = 50, include_evidence: bool = False) -> None:
    summaries = pd.read_parquet(CACHE / "facility_summaries.parquet")
    trust = pd.read_parquet(CACHE / "trust_scores.parquet")
    df = summaries.merge(trust[["facility_id", "score"]], on="facility_id", how="left")
    if include_evidence:
        silver_path = CACHE / "facility_silver.parquet"
        if silver_path.exists():
            silver = pd.read_parquet(silver_path)[["facility_id", "unstructured_blob"]]
            df = df.merge(silver, on="facility_id", how="left")

    def _to_list(xs):
        if xs is None:
            return []
        try:
            return list(xs)
        except TypeError:
            return []

    df["claim_count"] = df["claimed_capabilities"].apply(lambda xs: len(_to_list(xs)))
    df["has_high_acuity"] = df["claimed_capabilities"].apply(
        lambda xs: any(c in HIGH_ACUITY for c in _to_list(xs))
    )
    df["trust_band"] = pd.qcut(
        df["score"].fillna(0), q=[0, 0.33, 0.66, 1.0], labels=["low", "mid", "high"]
    )

    strata = []
    for (band, hi), sub in df.groupby(["trust_band", "has_high_acuity"], observed=False):
        take = max(2, min(len(sub), n // 6))
        strata.append(sub.sample(min(take, len(sub)), random_state=42))
    sample = pd.concat(strata).head(n).reset_index(drop=True)

    label_cols = [
        "label_icu", "label_nicu", "label_dialysis", "label_oncology",
        "label_general_surgery", "label_trauma_emergency",
        "label_emergency_appendectomy", "label_cardiac_care",
        "label_emergency_24x7", "label_anesthesiologist",
        "notes_for_labeler",
    ]
    for c in label_cols:
        sample[c] = ""

    keep = [
        "facility_id", "name", "city", "state", "facilityTypeId",
        "operatorTypeId", "claimed_capabilities", "score", "summary_text",
        "description",
    ]
    if include_evidence and "unstructured_blob" in sample.columns:
        keep.append("unstructured_blob")
    keep.extend(label_cols)
    sample[keep].to_csv(OUT, index=False)
    print(f"Wrote golden subset n={len(sample)} -> {OUT}")
    if include_evidence:
        print("(includes `unstructured_blob` so the LLM judge has full context.)")
    print("\nNext: open the CSV, fill the label_* columns with 1/0 (or leave blank for unsure),")
    print("then run notebooks/09_eval_harness.py to score the agent against your labels.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="Sample size.")
    parser.add_argument(
        "--include_evidence",
        action="store_true",
        help="Add the unstructured_blob column for the LLM judge.",
    )
    args = parser.parse_args()
    main(n=args.n, include_evidence=args.include_evidence)
