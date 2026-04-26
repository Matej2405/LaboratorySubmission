# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Gold Extraction (Agentic IDP)
# MAGIC
# MAGIC Runs the Extractor Agent (Agent Bricks Foundation Model) over the 10k Silver
# MAGIC rows and writes structured `capability_claims` to Gold with row-level
# MAGIC citations.
# MAGIC
# MAGIC Knobs:
# MAGIC * `LIMIT_ROWS` - cap for fast iteration (set None for full run)
# MAGIC * `BATCH_SIZE` - rows per Spark partition handed to one executor
# MAGIC * `ENDPOINT` - Foundation Model endpoint name
# MAGIC
# MAGIC Idempotent: writes to a Delta table with MERGE on `facility_id` so reruns
# MAGIC patch only the rows you re-extract.

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
from agents.extractor import extract_batch
from schemas.virtue_foundation import FacilityExtraction

mlflow.set_experiment(CFG.mlflow_experiment)

# COMMAND ----------

LIMIT_ROWS = 200       # set to None for full 10k after the dry-run looks good
BATCH_SIZE = 25
ENDPOINT = CFG.extractor_endpoint

# COMMAND ----------

silver = spark.table(CFG.fq(CFG.silver_schema, CFG.silver_table))
work = silver.select(
    "facility_id",
    "facilityTypeId",
    "operatorTypeId",
    "city",
    "state",
    "numberDoctors",
    "capacity",
    "unstructured_blob",
)
if LIMIT_ROWS:
    work = work.orderBy(F.rand(seed=42)).limit(LIMIT_ROWS)
print(f"Will extract from {work.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## applyInPandas dispatch
# MAGIC
# MAGIC Each Spark partition becomes one synchronous mini-batch. We stream the
# MAGIC extractor's pydantic output back as a flat row.

# COMMAND ----------

OUT_SCHEMA = T.StructType([
    T.StructField("facility_id", T.StringType()),
    T.StructField("extraction_json", T.StringType()),
    T.StructField("claimed_capabilities", T.ArrayType(T.StringType())),
    T.StructField("n_capabilities_claimed", T.IntegerType()),
    T.StructField("n_evidence_sentences", T.IntegerType()),
    T.StructField("extractor_endpoint", T.StringType()),
])


def _extract_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    rows = pdf.to_dict(orient="records")
    extractions = extract_batch(rows, endpoint=ENDPOINT)
    out = []
    for ex in extractions:
        claimed = ex.claimed_capabilities()
        n_evidence = sum(len(c.evidence_sentences) for c in ex.capabilities)
        out.append({
            "facility_id": ex.facility_id,
            "extraction_json": ex.model_dump_json(),
            "claimed_capabilities": claimed,
            "n_capabilities_claimed": len(claimed),
            "n_evidence_sentences": n_evidence,
            "extractor_endpoint": ENDPOINT,
        })
    return pd.DataFrame(out)


repartitioned = work.repartition(max(1, work.count() // BATCH_SIZE))
extracted = repartitioned.groupBy("facility_id").applyInPandas(_extract_partition, schema=OUT_SCHEMA)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MERGE into Gold

# COMMAND ----------

gold_table = CFG.fq(CFG.gold_schema, CFG.gold_capabilities)

extracted.createOrReplaceTempView("_extract_stage")
spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {gold_table} (
      facility_id STRING,
      extraction_json STRING,
      claimed_capabilities ARRAY<STRING>,
      n_capabilities_claimed INT,
      n_evidence_sentences INT,
      extractor_endpoint STRING,
      extracted_at TIMESTAMP
    ) USING DELTA
    """
)

spark.sql(
    f"""
    MERGE INTO {gold_table} t
    USING (SELECT *, current_timestamp() AS extracted_at FROM _extract_stage) s
    ON t.facility_id = s.facility_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """
)

print(f"Gold table {gold_table} now has {spark.table(gold_table).count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick dashboard

# COMMAND ----------

display(
    spark.table(gold_table)
    .selectExpr("explode(claimed_capabilities) as capability")
    .groupBy("capability")
    .count()
    .orderBy(F.desc("count"))
)
