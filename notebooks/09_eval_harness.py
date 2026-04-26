# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Evaluation Harness
# MAGIC
# MAGIC Three sub-evals:
# MAGIC 1. **Extraction precision/recall** vs the hand-labeled `golden_subset.csv`
# MAGIC    (50 facilities, stratified urban/rural x trust band x high-acuity).
# MAGIC 2. **Validator agreement rate** - how often the Validator confirms the
# MAGIC    Extractor for high-acuity claims.
# MAGIC 3. **Reasoning agent smoke test** - run the canned queries, log latency
# MAGIC    and citation count to MLflow.

# COMMAND ----------

# MAGIC %pip install -q pydantic==2.9.2 mlflow==3.0.0rc0 openai==1.55.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys, os, json
sys.path.append(os.path.abspath(".."))

import pandas as pd
import mlflow
from pyspark.sql import functions as F

from agents.config import CFG
from schemas.virtue_foundation import FacilityExtraction

mlflow.set_experiment(CFG.mlflow_experiment)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Extraction precision / recall
# MAGIC
# MAGIC Reads `evals/golden_subset.csv` (you fill the `label_*` columns by hand)
# MAGIC and compares against `gold.capability_claims`.

# COMMAND ----------

LABELED_PATH = os.path.abspath("../evals/golden_subset.labeled.csv")
GOLDEN_PATH = os.path.abspath("../evals/golden_subset.csv")
LABEL_SOURCE = "auto-judge" if os.path.exists(LABELED_PATH) else (
    "human" if os.path.exists(GOLDEN_PATH) else "missing"
)
print(f"label source: {LABEL_SOURCE}")

if LABEL_SOURCE == "missing":
    print("Run `python evals/golden_subset.py --include_evidence` and "
          "`python evals/auto_label_golden.py --judge <model>` first.")
else:
    golden = pd.read_csv(LABELED_PATH if LABEL_SOURCE == "auto-judge" else GOLDEN_PATH)
    label_cols = [c for c in golden.columns if c.startswith("label_")]
    cap_for_label = {c: c.replace("label_", "") for c in label_cols}

    gold = spark.table(CFG.fq(CFG.gold_schema, CFG.gold_capabilities)).toPandas()
    merged = golden.merge(gold[["facility_id", "claimed_capabilities"]], on="facility_id", how="left")

    rows = []
    for cap_label, cap in cap_for_label.items():
        labels = pd.to_numeric(merged[cap_label], errors="coerce")
        gold_pred = merged["claimed_capabilities"].apply(
            lambda xs: 1 if isinstance(xs, list) and cap in xs else 0
        )
        mask = labels.notna()
        if mask.sum() < 5:
            continue
        tp = int(((labels[mask] == 1) & (gold_pred[mask] == 1)).sum())
        fp = int(((labels[mask] == 0) & (gold_pred[mask] == 1)).sum())
        fn = int(((labels[mask] == 1) & (gold_pred[mask] == 0)).sum())
        tn = int(((labels[mask] == 0) & (gold_pred[mask] == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and (prec + rec) > 0) else float("nan")
        rows.append({"capability": cap, "n": int(mask.sum()), "tp": tp, "fp": fp, "fn": fn,
                     "tn": tn, "precision": prec, "recall": rec, "f1": f1})
    metrics = pd.DataFrame(rows)
    print(metrics.round(3).to_string(index=False))

    macro_f1 = metrics["f1"].dropna().mean() if not metrics.empty else float("nan")
    macro_prec = metrics["precision"].dropna().mean() if not metrics.empty else float("nan")
    macro_rec = metrics["recall"].dropna().mean() if not metrics.empty else float("nan")
    print(f"\nMacro precision={macro_prec:.3f}  recall={macro_rec:.3f}  F1={macro_f1:.3f}  source={LABEL_SOURCE}")

    with mlflow.start_run(run_name="eval_extraction_precision_recall"):
        mlflow.log_param("label_source", LABEL_SOURCE)
        for _, r in metrics.iterrows():
            mlflow.log_metric(f"precision_{r['capability']}", float(r["precision"] or 0))
            mlflow.log_metric(f"recall_{r['capability']}", float(r["recall"] or 0))
            mlflow.log_metric(f"f1_{r['capability']}", float(r["f1"] or 0))
            mlflow.log_metric(f"n_{r['capability']}", int(r["n"]))
        if pd.notna(macro_f1):
            mlflow.log_metric("macro_precision", float(macro_prec))
            mlflow.log_metric("macro_recall", float(macro_rec))
            mlflow.log_metric("macro_f1", float(macro_f1))
        mlflow.log_text(metrics.to_csv(index=False), "extraction_metrics.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validator agreement

# COMMAND ----------

verdicts_table = CFG.fq(CFG.gold_schema, "validator_verdicts")
if spark.catalog.tableExists(verdicts_table):
    v = spark.table(verdicts_table)
    n = v.count()
    n_agree = v.filter(F.col("agreement")).count()
    rate = n_agree / max(1, n)
    print(f"Validator-Extractor agreement: {rate:.3%}  ({n_agree}/{n})")
    with mlflow.start_run(run_name="eval_validator_agreement"):
        mlflow.log_metric("validator_agreement_rate", rate)
        mlflow.log_metric("n_validations", n)

    display(
        v.groupBy("capability").agg(
            F.count("*").alias("n"),
            (F.sum(F.when(F.col("agreement"), 1).otherwise(0)) / F.count("*")).alias("agree_rate"),
        ).orderBy("agree_rate")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reasoning agent smoke test

# COMMAND ----------

QUERIES = [
    "Find the nearest facility in rural Bihar that can perform an emergency appendectomy and typically leverages part-time doctors.",
    "Which Tamil Nadu hospitals claim NICU but show no neonatologist or pediatrician?",
    "Show me the top 5 most trustworthy oncology centres in Maharashtra.",
    "Which districts in Uttar Pradesh appear to be dialysis deserts?",
    "List 24x7 emergency hospitals in West Bengal with a cardiologist on staff.",
]

from agents.reasoner import answer
import time

with mlflow.start_run(run_name="eval_reasoner_smoke"):
    for i, q in enumerate(QUERIES):
        t0 = time.time()
        try:
            res = answer(q)
            elapsed = (time.time() - t0) * 1000
            mlflow.log_metric(f"latency_ms_q{i}", elapsed)
            mlflow.log_metric(f"n_facilities_q{i}", len(res.facilities))
            mlflow.log_metric(f"n_citations_q{i}", len(res.citations))
            mlflow.log_text(json.dumps({
                "question": q, "plan": res.plan, "answer": res.answer,
                "n_facilities": len(res.facilities), "n_citations": len(res.citations),
            }, indent=2, default=str), f"q{i}.json")
            print(f"[Q{i}] {elapsed:.0f}ms, facilities={len(res.facilities)}, citations={len(res.citations)}")
        except Exception as e:
            print(f"[Q{i}] FAILED: {e}")
            mlflow.log_text(str(e), f"q{i}_error.txt")
