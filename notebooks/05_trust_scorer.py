# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Trust Scorer
# MAGIC
# MAGIC Combines extraction (Gold) + validator verdicts + Silver structured
# MAGIC signals (web presence, equipment count, staff count) into a 0..100 trust
# MAGIC score per facility, with row-level cited flags.

# COMMAND ----------

# MAGIC %pip install -q pydantic==2.9.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys, os, json
sys.path.append(os.path.abspath(".."))

import pandas as pd
from pyspark.sql import functions as F, types as T

from agents.config import CFG
from agents.trust import score_facility, StructuredSignals
from schemas.virtue_foundation import FacilityExtraction, ValidatorVerdict

# COMMAND ----------

silver = spark.table(CFG.fq(CFG.silver_schema, CFG.silver_table)).select(
    "facility_id",
    "numberDoctors",
    "capacity",
    "has_equipment_evidence",
    "claim_count",
    F.col("engagement_metrics_n_followers").alias("followers"),
    F.col("officialWebsite").alias("official_site"),
)
gold = spark.table(CFG.fq(CFG.gold_schema, CFG.gold_capabilities)).select(
    "facility_id", "extraction_json"
)
verdicts_tbl = CFG.fq(CFG.gold_schema, "validator_verdicts")
verdicts = spark.table(verdicts_tbl).groupBy("facility_id").agg(
    F.collect_list(F.struct(
        "facility_id", "capability", "original_claim", "validator_claim",
        "agreement", "rationale", "missing_required", "flagged_evidence",
    )).alias("verdicts")
)

joined = gold.join(silver, on="facility_id", how="left").join(verdicts, on="facility_id", how="left")
print(f"Scoring {joined.count():,} facilities")

# COMMAND ----------

OUT_SCHEMA = T.StructType([
    T.StructField("facility_id", T.StringType()),
    T.StructField("score", T.DoubleType()),
    T.StructField("completeness", T.DoubleType()),
    T.StructField("consistency", T.DoubleType()),
    T.StructField("source_agreement", T.DoubleType()),
    T.StructField("flags", T.ArrayType(T.StringType())),
    T.StructField("flag_evidence", T.ArrayType(T.StringType())),
])


def _score_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for _, r in pdf.iterrows():
        try:
            extraction = FacilityExtraction.model_validate_json(r["extraction_json"])
        except Exception:
            continue
        structured = StructuredSignals(
            number_doctors=r.get("numberDoctors"),
            capacity=r.get("capacity"),
            has_equipment_evidence=bool(r.get("has_equipment_evidence")),
            n_capability_claims=int(r.get("claim_count") or 0),
            has_followers=bool(r.get("followers")),
            has_official_website=bool(r.get("official_site")),
        )
        verdicts_in = []
        raw_v = r.get("verdicts")
        if raw_v is not None:
            for v in raw_v:
                v_dict = v.asDict() if hasattr(v, "asDict") else dict(v)
                try:
                    verdicts_in.append(ValidatorVerdict.model_validate(v_dict))
                except Exception:
                    pass
        ts = score_facility(extraction, structured, verdicts_in)
        out_rows.append(ts.model_dump())
    return pd.DataFrame(out_rows, columns=[f.name for f in OUT_SCHEMA.fields])


scored = (
    joined.repartition(50)
    .groupBy("facility_id")
    .applyInPandas(_score_partition, schema=OUT_SCHEMA)
)

# COMMAND ----------

trust_table = CFG.fq(CFG.gold_schema, CFG.gold_trust)
(
    scored.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(trust_table)
)
print(f"Wrote {spark.table(trust_table).count():,} trust scores -> {trust_table}")

# COMMAND ----------

display(
    spark.table(trust_table)
    .selectExpr("explode(flags) as flag")
    .groupBy("flag")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

display(
    spark.table(trust_table)
    .selectExpr(
        "round(avg(score),1) avg_score",
        "round(percentile_approx(score,0.5),1) median_score",
        "round(percentile_approx(score,0.1),1) p10_score",
        "round(percentile_approx(score,0.9),1) p90_score",
    )
)
