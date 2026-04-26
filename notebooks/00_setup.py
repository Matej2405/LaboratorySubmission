# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Setup
# MAGIC
# MAGIC One-time bootstrap of Unity Catalog assets used by the Agentic Healthcare
# MAGIC Intelligence pipeline.
# MAGIC
# MAGIC * Catalog: `vf_health`
# MAGIC * Schemas: `bronze`, `silver`, `gold`, `idx`, `eval`
# MAGIC * Volume: `vf_health.bronze.raw` (Excel landing zone)
# MAGIC
# MAGIC Run this notebook **once** before anything else.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "vf_health"
SCHEMAS = ["bronze", "silver", "gold", "idx", "eval"]
RAW_VOLUME = "raw"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
for s in SCHEMAS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{s}")

spark.sql(
    f"CREATE VOLUME IF NOT EXISTS {CATALOG}.bronze.{RAW_VOLUME}"
)

print(f"Catalog {CATALOG} ready with schemas: {SCHEMAS}")
print(f"Volume: /Volumes/{CATALOG}/bronze/{RAW_VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upload the dataset
# MAGIC
# MAGIC Upload `VF_Hackathon_Dataset_India_Large.xlsx` to the volume:
# MAGIC
# MAGIC ```
# MAGIC /Volumes/vf_health/bronze/raw/VF_Hackathon_Dataset_India_Large.xlsx
# MAGIC ```
# MAGIC
# MAGIC Either drag-and-drop in the UI or run:
# MAGIC
# MAGIC ```python
# MAGIC dbutils.fs.cp("file:/local/path/...", "/Volumes/vf_health/bronze/raw/...")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow experiment

# COMMAND ----------

import mlflow

EXPERIMENT_PATH = "/Shared/vf_health_agents"
mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.langchain.autolog()
print(f"MLflow experiment: {EXPERIMENT_PATH}")
