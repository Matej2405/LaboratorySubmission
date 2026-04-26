# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - Confidence Intervals
# MAGIC
# MAGIC Computes Wilson and trust-weighted intervals for capability prevalence by
# MAGIC state and facility type, plus per-facility Beta posteriors over validator
# MAGIC agreement. Output table powers the "value [lo-hi] (n=...)" chips in the
# MAGIC Crisis Map dashboard.

# COMMAND ----------

# MAGIC %pip install -q scipy==1.13.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys, os
sys.path.append(os.path.abspath(".."))

import pandas as pd
from pyspark.sql import functions as F, types as T

from agents.config import CFG
from agents.confidence import wilson_interval, beta_posterior, trust_weighted_proportion

# COMMAND ----------

silver = spark.table(CFG.fq(CFG.silver_schema, CFG.silver_table)).select(
    "facility_id", "state", "pincode", "city", "facilityTypeId",
    "latitude", "longitude",
)
gold = spark.table(CFG.fq(CFG.gold_schema, CFG.gold_capabilities)).select(
    "facility_id", "claimed_capabilities"
)
trust = spark.table(CFG.fq(CFG.gold_schema, CFG.gold_trust)).select(
    "facility_id", "score"
)
joined = silver.join(gold, "facility_id", "left").join(trust, "facility_id", "left")

# COMMAND ----------

# MAGIC %md
# MAGIC ## State x capability prevalence with intervals

# COMMAND ----------

@F.pandas_udf(T.StructType([
    T.StructField("p_hat", T.DoubleType()),
    T.StructField("low", T.DoubleType()),
    T.StructField("high", T.DoubleType()),
    T.StructField("n_eff", T.IntegerType()),
]), F.PandasUDFType.GROUPED_MAP)
def _interval_for_group(pdf):
    weights = (pdf["score"].fillna(50.0) / 100.0).tolist()
    indicators = pdf["has_cap"].astype(int).tolist()
    iv = trust_weighted_proportion(weights, indicators)
    return pd.DataFrame([{
        "p_hat": iv.point, "low": iv.lower, "high": iv.upper, "n_eff": iv.n,
    }])

# applyInPandas requires the full per-group columns to be returned, so we
# instead do the calculation as a deterministic Python aggregation on the
# driver - dataset is small (~50 states x ~25 capabilities = 1250 cells).

CAPS = [
    "icu", "nicu", "dialysis", "oncology", "trauma_emergency",
    "general_surgery", "emergency_appendectomy", "cardiac_care",
    "obgyn_delivery", "radiology_ct", "radiology_mri", "ultrasound",
    "blood_bank", "ambulance", "oxygen_supply", "emergency_24x7",
    "lab_diagnostics",
]

facilities_pdf = joined.select(
    "facility_id", "state", "claimed_capabilities", "score"
).toPandas()

rows = []
for state, sub in facilities_pdf.groupby("state"):
    weights = (sub["score"].fillna(50.0) / 100.0).tolist()
    for cap in CAPS:
        indicators = sub["claimed_capabilities"].apply(
            lambda xs: 1 if isinstance(xs, list) and cap in xs else 0
        ).tolist()
        iv = trust_weighted_proportion(weights, indicators)
        rows.append({
            "state": state,
            "capability": cap,
            "p_hat": iv.point,
            "low": iv.lower,
            "high": iv.upper,
            "n_eff": iv.n,
            "n_facilities": len(sub),
        })

prev_pdf = pd.DataFrame(rows)
prev_table = CFG.fq(CFG.gold_schema, "capability_prevalence")
spark.createDataFrame(prev_pdf).write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(prev_table)
print(f"Wrote {len(prev_pdf):,} prevalence rows -> {prev_table}")
display(spark.table(prev_table).orderBy("state", "capability").limit(40))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Medical-desert score per (state, capability)
# MAGIC
# MAGIC Desert score = `(1 - upper bound of trust-weighted prevalence)`. We use
# MAGIC the upper bound to be conservative: a region is only declared a desert
# MAGIC when even the optimistic estimate is low.

# COMMAND ----------

desert_pdf = prev_pdf.assign(desert_score=lambda d: (1 - d["high"]).round(4))
desert_table = CFG.fq(CFG.gold_schema, "desert_scores")
spark.createDataFrame(desert_pdf).write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(desert_table)
display(
    spark.table(desert_table)
    .filter(F.col("capability").isin(["icu", "dialysis", "oncology", "trauma_emergency", "emergency_appendectomy"]))
    .orderBy(F.desc("desert_score"))
    .limit(25)
)
