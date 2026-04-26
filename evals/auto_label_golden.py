"""LLM-as-judge auto-labeler for the golden subset.

For each row in `evals/golden_subset.csv`, asks a *different model family* than
the extractor whether the facility offers each high-acuity capability. The
judge returns a structured `JudgeVerdict` (Pydantic) with a 0/1/unsure label
plus a verbatim cited sentence, so a human can spot-check it.

Why a different family?
    Cross-family agreement is a stronger signal than self-agreement: if Claude
    and Llama both say "ICU is functional", that's a meaningful trust signal.
    See README's Methodology section.

Output:
    `evals/golden_subset.labeled.csv` - same columns as the input plus filled
    `label_*` cells (0/1 or blank for unsure).

Usage:
    python evals/auto_label_golden.py --judge claude-3-5-sonnet-20241022
    python evals/auto_label_golden.py --judge gpt-4o-mini --concurrency 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.extractor import detect_endpoint_family, _call_anthropic, _openai_client

LOG = logging.getLogger("vf_health.auto_label")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GOLDEN_PATH = ROOT / "evals" / "golden_subset.csv"
LABELED_PATH = ROOT / "evals" / "golden_subset.labeled.csv"
SPOT_CHECK_PATH = ROOT / "evals" / "spot_check.md"

CAPABILITY_TO_LABEL = {
    "icu": "label_icu",
    "nicu": "label_nicu",
    "dialysis": "label_dialysis",
    "oncology": "label_oncology",
    "general_surgery": "label_general_surgery",
    "trauma_emergency": "label_trauma_emergency",
    "emergency_appendectomy": "label_emergency_appendectomy",
    "cardiac_care": "label_cardiac_care",
    "emergency_24x7": "label_emergency_24x7",
    "anesthesiologist": "label_anesthesiologist",
}


class CapabilityVerdict(BaseModel):
    capability: str = Field(..., description="Capability name being judged.")
    label: int = Field(..., ge=0, le=1, description="1 if functional, 0 if not.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    cited_sentence: Optional[str] = Field(
        None, description="Verbatim sentence from the notes that justifies the label."
    )


class JudgeVerdict(BaseModel):
    facility_id: str
    capabilities: list[CapabilityVerdict] = Field(default_factory=list)


SYSTEM_PROMPT = """You are an experienced Indian healthcare auditor.

Your job: given a single facility's free-form notes, decide for each
capability in the list whether the facility actually OFFERS it (label=1) or
does NOT (label=0). Be conservative:

* Mark 1 only when an explicit sentence in the notes describes the capability
  being delivered, OR when both equipment and trained staff are present.
* Marketing terms like "best in city" or "world-class" do NOT count.
* "Anesthesiologist" is the staff role, not a capability - mark 1 only if a
  named or titled anesthesiologist appears in the notes.
* Always cite a verbatim sentence (or "" if no evidence at all).
* Confidence in [0, 1].

Return JSON conforming exactly to the JudgeVerdict schema. No commentary.
"""

USER_TEMPLATE = """FACILITY ID: {facility_id}
NAME: {name}
LOCATION: {city}, {state}
TYPE: {facility_type}

CAPABILITIES TO JUDGE:
{cap_list}

NOTES (free-form, possibly noisy):
---
{blob}
---

JSON SCHEMA:
{schema}

Return ONLY the JSON object.
"""


def _judge_one(
    row: dict[str, Any],
    endpoint: str,
    capabilities: list[str],
    *,
    timeout_s: int = 60,
) -> JudgeVerdict:
    family = detect_endpoint_family(endpoint)
    user = USER_TEMPLATE.format(
        facility_id=row.get("facility_id", "?"),
        name=row.get("name", "?"),
        city=row.get("city", "?"),
        state=row.get("state", "?"),
        facility_type=row.get("facilityTypeId", "?"),
        cap_list="\n".join(f"- {c}" for c in capabilities),
        blob=str(row.get("unstructured_blob") or row.get("description") or row.get("summary_text") or "")[:8000],
        schema=json.dumps(JudgeVerdict.model_json_schema(), indent=0)[:4000],
    )

    if family == "anthropic":
        text = _call_anthropic(
            endpoint,
            SYSTEM_PROMPT,
            user + "\n\nReturn ONLY the JSON object. No markdown.",
            temperature=0.0,
            timeout_s=timeout_s,
        )
    else:
        client = _openai_client(endpoint)
        resp = client.chat.completions.create(
            model=endpoint,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            timeout=timeout_s,
        )
        text = resp.choices[0].message.content or "{}"

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        payload = json.loads(text[start : end + 1])
    payload.setdefault("facility_id", str(row.get("facility_id")))
    return JudgeVerdict.model_validate(payload)


def _to_list(xs: Any) -> list[str]:
    if xs is None or (isinstance(xs, float)):
        return []
    if isinstance(xs, list):
        return [str(x) for x in xs]
    if isinstance(xs, str):
        s = xs.strip()
        if not s:
            return []
        try:
            return [str(x) for x in json.loads(s.replace("'", '"'))]
        except Exception:
            return [c.strip() for c in s.strip("[]").split(",") if c.strip()]
    return list(xs)


def auto_label(
    *,
    judge: str,
    concurrency: int,
    capabilities: list[str],
    golden_path: Path,
    out_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not golden_path.exists():
        raise FileNotFoundError(
            f"{golden_path} missing - run "
            "`python evals/golden_subset.py --include_evidence` first."
        )
    df = pd.read_csv(golden_path)
    LOG.info("loaded %d golden rows from %s", len(df), golden_path)
    LOG.info("judge endpoint=%s family=%s", judge, detect_endpoint_family(judge))

    results: dict[str, JudgeVerdict] = {}
    rows_iter = df.to_dict(orient="records")

    def _task(row: dict[str, Any]) -> tuple[str, JudgeVerdict | Exception]:
        fid = str(row.get("facility_id"))
        try:
            v = _judge_one(row, judge, capabilities)
            return fid, v
        except Exception as e:
            return fid, e

    if concurrency <= 1:
        for row in rows_iter:
            fid, out = _task(row)
            if isinstance(out, Exception):
                LOG.warning("judge failed for %s: %s", fid, out)
            else:
                results[fid] = out
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_task, r) for r in rows_iter]
            for f in as_completed(futs):
                fid, out = f.result()
                if isinstance(out, Exception):
                    LOG.warning("judge failed for %s: %s", fid, out)
                else:
                    results[fid] = out

    LOG.info("collected verdicts for %d/%d facilities", len(results), len(df))

    out = df.copy()
    raw_records: list[dict[str, Any]] = []
    for cap in capabilities:
        col = CAPABILITY_TO_LABEL.get(cap)
        if not col:
            continue
        if col not in out.columns:
            out[col] = ""
        for i, fid in enumerate(out["facility_id"].astype(str).tolist()):
            v = results.get(fid)
            if not v:
                out.at[i, col] = ""
                continue
            match = next((c for c in v.capabilities if c.capability == cap), None)
            if match is None:
                out.at[i, col] = ""
            else:
                out.at[i, col] = int(match.label)
                raw_records.append({
                    "facility_id": fid,
                    "capability": cap,
                    "label": int(match.label),
                    "confidence": float(match.confidence),
                    "cited_sentence": (match.cited_sentence or "").strip(),
                })

    out.to_csv(out_path, index=False)
    LOG.info("wrote %s", out_path)
    raw_df = pd.DataFrame(raw_records)
    return out, raw_df


def write_spot_check(
    df: pd.DataFrame,
    raw: pd.DataFrame,
    out_path: Path = SPOT_CHECK_PATH,
    top: int = 10,
) -> None:
    """Surface the top-N disagreements between heuristic and judge."""
    if raw.empty:
        out_path.write_text("# Spot check\n\nNo judge verdicts collected.\n")
        return

    long_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        claimed = _to_list(row.get("claimed_capabilities"))
        for cap, col in CAPABILITY_TO_LABEL.items():
            heur = 1 if cap in claimed else 0
            judge_label = row.get(col)
            if judge_label in ("", None) or pd.isna(judge_label):
                continue
            try:
                judge_label = int(judge_label)
            except (TypeError, ValueError):
                continue
            if heur == judge_label:
                continue
            long_rows.append({
                "facility_id": row["facility_id"],
                "name": row.get("name"),
                "state": row.get("state"),
                "city": row.get("city"),
                "capability": cap,
                "heuristic": heur,
                "judge": judge_label,
            })

    if not long_rows:
        out_path.write_text(
            "# Spot check\n\nNo disagreements between heuristic and judge.\n"
        )
        LOG.info("no disagreements - %s says clean run", out_path)
        return

    long_df = pd.DataFrame(long_rows)
    long_df = long_df.merge(
        raw[["facility_id", "capability", "confidence", "cited_sentence"]],
        on=["facility_id", "capability"], how="left",
    ).sort_values("confidence", ascending=False)

    head = long_df.head(top)
    lines: list[str] = [
        "# Spot check - heuristic vs LLM-judge disagreements",
        "",
        f"Total disagreements: **{len(long_df)}**.",
        f"Listing the top {len(head)} by judge confidence so a human can sanity-check the pipeline.",
        "",
    ]
    for i, r in enumerate(head.itertuples(index=False), 1):
        cited = (r.cited_sentence or "(no sentence)").replace("\n", " ").strip()
        lines.append(
            f"### {i}. {r.name} - {r.city}, {r.state}\n\n"
            f"- **Capability:** `{r.capability}`\n"
            f"- **Heuristic says:** {r.heuristic}\n"
            f"- **Judge says:** {r.judge}  (confidence {r.confidence:.2f})\n"
            f"- **Judge cited:** _{cited[:280]}_\n"
        )
    lines.append("\n*Generated by `evals/auto_label_golden.py`.*")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("wrote %s with %d disagreements", out_path, len(head))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", required=True, help="Judge model endpoint (different family from the extractor).")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--in", dest="in_path", default=str(GOLDEN_PATH), help="Input golden_subset.csv.")
    parser.add_argument("--out", default=str(LABELED_PATH), help="Output golden_subset.labeled.csv.")
    parser.add_argument(
        "--capabilities",
        default="icu,nicu,dialysis,oncology,general_surgery,trauma_emergency,emergency_appendectomy,cardiac_care",
        help="Comma-separated capabilities to judge.",
    )
    parser.add_argument("--top_disagreements", type=int, default=10)
    args = parser.parse_args()
    capabilities = [c.strip() for c in args.capabilities.split(",") if c.strip()]
    try:
        labeled, raw = auto_label(
            judge=args.judge,
            concurrency=args.concurrency,
            capabilities=capabilities,
            golden_path=Path(args.in_path),
            out_path=Path(args.out),
        )
    except RuntimeError as e:
        LOG.error("Cannot run judge: %s", e)
        return 2
    write_spot_check(labeled, raw, top=args.top_disagreements)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
