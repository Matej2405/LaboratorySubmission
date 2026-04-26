# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Validator Agent
# MAGIC
# MAGIC For every Gold extraction, run the Validator: rule-based KB check first,
# MAGIC then LLM-judge only on disagreements. Persists per-facility verdicts to
# MAGIC `vf_health.gold.validator_verdicts`.

# COMMAND ----------

# MAGIC %pip install -q openai==1.55.0 mlflow==3.0.0rc0 pydantic==2.9.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys, os, json
sys.path.append(os.path.abspath(".."))

import pandas as pd
import mlflow
from pyspark.sql import functions as F, types as T

from agents.config import CFG
from agents.validator import validate_facility
from schemas.virtue_foundation import FacilityExtraction

mlflow.set_experiment(CFG.mlflow_experiment)

# COMMAND ----------

ONLY_HIGH_ACUITY = True
# Use a deliberately *different* foundation-model family than the extractor,
# so the agreement metric reflects cross-family agreement rather than
# self-agreement (see README "Methodology").
ENDPOINT = CFG.validator_endpoint
print(f"validator endpoint = {ENDPOINT} (extractor = {CFG.extractor_endpoint})")

# COMMAND ----------

silver = spark.table(CFG.fq(CFG.silver_schema, CFG.silver_table)).select(
    "facility_id", "unstructured_blob"
)
gold = spark.table(CFG.fq(CFG.gold_schema, CFG.gold_capabilities)).select(
    "facility_id", "extraction_json"
)
joined = gold.join(silver, on="facility_id", how="inner")
print(f"Validating {joined.count():,} facilities")

# COMMAND ----------

OUT_SCHEMA = T.StructType([
    T.StructField("facility_id", T.StringType()),
    T.StructField("capability", T.StringType()),
    T.StructField("original_claim", T.BooleanType()),
    T.StructField("validator_claim", T.BooleanType()),
    T.StructField("agreement", T.BooleanType()),
    T.StructField("rationale", T.StringType()),
    T.StructField("missing_required", T.ArrayType(T.StringType())),
    T.StructField("flagged_evidence", T.ArrayType(T.StringType())),
])


def _validate_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in pdf.iterrows():
        try:
            ex = FacilityExtraction.model_validate_json(r["extraction_json"])
        except Exception:
            continue
        verdicts = validate_facility(
            ex, str(r.get("unstructured_blob") or ""),
            endpoint=ENDPOINT, only_high_acuity=ONLY_HIGH_ACUITY,
        )
        for v in verdicts:
            rows.append(v.model_dump())
    return pd.DataFrame(rows, columns=[f.name for f in OUT_SCHEMA.fields])


verdicts_df = (
    joined.repartition(50)
    .groupBy("facility_id")
    .applyInPandas(_validate_partition, schema=OUT_SCHEMA)
)

# COMMAND ----------

verdicts_table = CFG.fq(CFG.gold_schema, "validator_verdicts")
(
    verdicts_df.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(verdicts_table)
)
print(f"Wrote {spark.table(verdicts_table).count():,} verdicts -> {verdicts_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Disagreement summary

# COMMAND ----------

display(
    spark.table(verdicts_table)
    .groupBy("capability")
    .agg(
        F.count("*").alias("n_claims"),
        F.sum(F.when(F.col("agreement"), 1).otherwise(0)).alias("n_agree"),
        F.sum(F.when(~F.col("agreement"), 1).otherwise(0)).alias("n_flagged"),
    )
    .orderBy(F.desc("n_flagged"))
)
