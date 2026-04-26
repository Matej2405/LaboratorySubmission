"""CLI runner that calls the real LLM extractor over the silver dataset.

Usage:
    python scripts/run_llm_extraction.py \\
        --endpoint databricks-meta-llama-3-3-70b-instruct \\
        --limit 200 \\
        --concurrency 4 \\
        --out data/cache/capability_claims.parquet

Behavior:
* Auto-detects the endpoint family (Databricks, OpenAI, Anthropic) and uses
  `agents.extractor.extract_one`, which now supports all three.
* Reads silver-equivalent rows from `data/cache/facility_silver.parquet`. If
  that file is missing the runner exits with a one-line setup hint.
* Bounded `ThreadPoolExecutor` concurrency to respect provider rate limits.
* Persists incrementally every `--persist-every` rows so a quota hit only
  loses the in-flight batch (resume-safe via `facility_id` set check).
* Emits a side log `data/cache/llm_extraction_log.jsonl` with timing per row.

Without credentials, the extractor module raises at first call with a clear
error - this script will report that and exit so the heuristic baseline is
still safe to demo.
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
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.extractor import detect_endpoint_family, extract_one

LOG = logging.getLogger("vf_health.llm_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CACHE = ROOT / "data" / "cache"
SILVER_PATH = CACHE / "facility_silver.parquet"
DEFAULT_OUT = CACHE / "capability_claims.parquet"
LOG_PATH = CACHE / "llm_extraction_log.jsonl"


def _existing_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    try:
        df = pd.read_parquet(out_path, columns=["facility_id", "extractor_endpoint"])
    except Exception:
        return set()
    # Only resume rows that were written by an LLM; do not overwrite heuristic rows.
    llm_rows = df[df["extractor_endpoint"] != "local-heuristic"]
    return set(llm_rows["facility_id"].astype(str).tolist())


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {
        "facility_id": str(row["facility_id"]),
        "blob": str(row.get("unstructured_blob") or ""),
        "facility_type": row.get("facilityTypeId"),
        "operator_type": row.get("operatorTypeId"),
        "city": row.get("city"),
        "state": row.get("state"),
        "n_doctors": row.get("numberDoctors"),
        "capacity": row.get("capacity"),
    }


def _extract_with_log(args: dict[str, Any], endpoint: str, log_fp) -> dict[str, Any]:
    fid = args["facility_id"]
    t0 = time.time()
    try:
        ex = extract_one(
            args["facility_id"],
            args["blob"],
            facility_type=args["facility_type"],
            operator_type=args["operator_type"],
            city=args["city"],
            state=args["state"],
            n_doctors=args["n_doctors"],
            capacity=args["capacity"],
            endpoint=endpoint,
        )
        elapsed = time.time() - t0
        record = {
            "facility_id": ex.facility_id,
            "extraction_json": ex.model_dump_json(),
            "claimed_capabilities": [c.name for c in ex.capabilities if c.claimed],
            "n_capabilities_claimed": sum(1 for c in ex.capabilities if c.claimed),
            "n_evidence_sentences": sum(len(c.evidence_sentences) for c in ex.capabilities),
            "extractor_endpoint": endpoint,
        }
        log_fp.write(json.dumps({
            "facility_id": fid,
            "endpoint": endpoint,
            "elapsed_s": round(elapsed, 3),
            "ok": True,
            "n_capabilities": record["n_capabilities_claimed"],
        }) + "\n")
        log_fp.flush()
        return record
    except Exception as e:
        elapsed = time.time() - t0
        log_fp.write(json.dumps({
            "facility_id": fid,
            "endpoint": endpoint,
            "elapsed_s": round(elapsed, 3),
            "ok": False,
            "error": str(e)[:200],
        }) + "\n")
        log_fp.flush()
        raise


def _persist(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        # drop heuristic / earlier rows for the same facility_id when LLM rows arrive
        existing = existing[~existing["facility_id"].isin(new_df["facility_id"])]
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_parquet(out_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Model endpoint (e.g. databricks-meta-llama-3-3-70b-instruct, gpt-4o-mini, claude-3-5-sonnet-20241022).")
    parser.add_argument("--limit", type=int, default=200, help="Max number of facilities to process this run.")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel inflight requests.")
    parser.add_argument("--persist-every", type=int, default=50, help="Persist after every N rows.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output parquet (default: data/cache/capability_claims.parquet).")
    parser.add_argument("--silver", default=str(SILVER_PATH), help="Silver parquet input.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle silver rows before slicing limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true", help="Re-run rows that already have LLM output.")
    args = parser.parse_args()

    silver_path = Path(args.silver)
    if not silver_path.exists():
        LOG.error(
            "Silver parquet missing at %s. Run `python scripts/build_local_cache.py` first.",
            silver_path,
        )
        return 2

    family = detect_endpoint_family(args.endpoint)
    LOG.info("endpoint=%s family=%s", args.endpoint, family)

    silver = pd.read_parquet(silver_path)
    if args.shuffle:
        silver = silver.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already_done: set[str] = set() if args.no_resume else _existing_ids(out_path)
    if already_done:
        LOG.info("resume: %d facilities already have LLM output, skipping.", len(already_done))

    pending = silver[~silver["facility_id"].astype(str).isin(already_done)]
    if args.limit and args.limit > 0:
        pending = pending.head(args.limit)
    LOG.info("processing %d facilities (concurrency=%d, persist_every=%d)",
             len(pending), args.concurrency, args.persist_every)

    tasks = [_row_to_dict(r) for _, r in pending.iterrows()]
    if not tasks:
        LOG.info("nothing to do.")
        return 0

    buf: list[dict[str, Any]] = []
    written_total = 0
    n_errors = 0

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as log_fp:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(_extract_with_log, t, args.endpoint, log_fp): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                t = futures[fut]
                try:
                    rec = fut.result()
                    buf.append(rec)
                except Exception as e:
                    n_errors += 1
                    LOG.warning("extraction failed for %s: %s", t["facility_id"], e)
                if len(buf) >= args.persist_every or i == len(futures):
                    _persist(buf, out_path)
                    written_total += len(buf)
                    LOG.info("persisted %d rows (total=%d, errors=%d, processed=%d/%d)",
                             len(buf), written_total, n_errors, i, len(futures))
                    buf = []

    LOG.info("done. wrote %d rows to %s; errors=%d; log=%s",
             written_total, out_path, n_errors, LOG_PATH)
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
