# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Bronze Ingestion
# MAGIC
# MAGIC Reads `VF_Hackathon_Dataset_India_Large.xlsx` from the Unity Catalog volume
# MAGIC and writes a faithful Bronze Delta table (10k rows x 41 columns) preserving
# MAGIC every original column. Columns are normalized to snake_case for SQL safety.
# MAGIC
# MAGIC Profiles unstructured columns so we can size LLM prompts.

# COMMAND ----------

# MAGIC %pip install -q openpyxl==3.1.5 pandas==2.2.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys, os
sys.path.append(os.path.abspath(".."))

import pandas as pd
from pyspark.sql import functions as F

from agents.config import CFG

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Excel from Volume

# COMMAND ----------

xls = pd.ExcelFile(CFG.raw_path, engine="openpyxl")
print(f"Sheets: {xls.sheet_names}")
pdf = xls.parse(xls.sheet_names[0])
print(f"Loaded {len(pdf):,} rows, {len(pdf.columns)} columns")

pdf.columns = [
    c.strip()
    .replace(" ", "_")
    .replace("/", "_")
    .replace("-", "_")
    for c in pdf.columns
]

if "facility_id" not in pdf.columns:
    pdf.insert(0, "facility_id", [f"vf_{i:06d}" for i in range(len(pdf))])

datetime_cols = [c for c in pdf.columns if pd.api.types.is_datetime64_any_dtype(pdf[c])]
for c in datetime_cols:
    pdf[c] = pdf[c].dt.tz_localize(None)

display(pdf.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Bronze (overwriteSchema, preserve every column verbatim)

# COMMAND ----------

sdf = spark.createDataFrame(pdf)
bronze_table = CFG.fq(CFG.bronze_schema, CFG.bronze_table)
(
    sdf.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(bronze_table)
)
print(f"Wrote {sdf.count():,} rows -> {bronze_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profile unstructured columns
# MAGIC
# MAGIC The four payload columns we care about for IDP:
# MAGIC * `description` - free marketing prose
# MAGIC * `specialties` - JSON list of canonical specialty IDs
# MAGIC * `procedure`   - JSON list of free-form procedure claims
# MAGIC * `capability`  - JSON list of free-form capability claims (the messy one)
# MAGIC * `equipment`   - JSON list (often `[]` -> contradiction signal)

# COMMAND ----------

string_cols = [c for c, t in sdf.dtypes if t == "string"]

profile = []
for col in string_cols:
    stats = (
        sdf.select(F.length(F.col(col)).alias("len"))
        .summary("count", "min", "50%", "mean", "max")
        .collect()
    )
    row = {r["summary"]: r["len"] for r in stats}
    row["column"] = col
    profile.append(row)

prof_df = pd.DataFrame(profile).set_index("column")
prof_df["median_len"] = pd.to_numeric(prof_df["50%"], errors="coerce")
prof_df["unstructured"] = prof_df["median_len"] > 80
print("Unstructured columns:")
print(prof_df[prof_df["unstructured"]].sort_values("median_len", ascending=False))

profile_table = CFG.fq(CFG.bronze_schema, "column_profile")
spark.createDataFrame(
    prof_df.reset_index()[["column", "median_len", "unstructured"]]
).write.mode("overwrite").saveAsTable(profile_table)
print(f"Profile written -> {profile_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity counts

# COMMAND ----------

display(
    spark.table(bronze_table)
    .groupBy("facilityTypeId")
    .count()
    .orderBy(F.desc("count"))
)

display(
    spark.table(bronze_table)
    .groupBy("address_stateOrRegion")
    .count()
    .orderBy(F.desc("count"))
    .limit(20)
)
