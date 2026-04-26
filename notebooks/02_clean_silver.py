# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Silver Cleaning
# MAGIC
# MAGIC From Bronze, build a Silver table that:
# MAGIC * Normalizes state names (`Orissa` -> `Odisha`, etc.)
# MAGIC * Extracts a clean 6-digit PIN
# MAGIC * Parses the four JSON-string columns (`specialties`, `procedure`,
# MAGIC   `capability`, `equipment`) into native arrays
# MAGIC * Builds a single `unstructured_blob` column ready to feed the LLM
# MAGIC * Computes a **rural_score** using webpresence + capacity + city size as a
# MAGIC   proxy (no external district data needed - the dataset already has
# MAGIC   lat/long for the map)

# COMMAND ----------

# MAGIC %pip install -q pandas==2.2.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys, os
sys.path.append(os.path.abspath(".."))

from pyspark.sql import functions as F, types as T
import pandas as pd

from agents.config import CFG
from agents.text_utils import (
    normalize_state, extract_pin, parse_json_list, to_unstructured_blob,
)

# COMMAND ----------

bronze = spark.table(CFG.fq(CFG.bronze_schema, CFG.bronze_table))
print(f"Bronze rows: {bronze.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap helpers as UDFs

# COMMAND ----------

normalize_state_udf = F.udf(normalize_state, T.StringType())
extract_pin_udf = F.udf(lambda *xs: extract_pin(*xs), T.StringType())
parse_list_udf = F.udf(parse_json_list, T.ArrayType(T.StringType()))
blob_udf = F.udf(
    to_unstructured_blob,
    T.StringType(),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver build

# COMMAND ----------

silver = (
    bronze
    .withColumn("state", normalize_state_udf(F.col("address_stateOrRegion")))
    .withColumn(
        "pincode",
        extract_pin_udf(
            F.col("address_zipOrPostcode").cast("string"),
            F.col("address_line1").cast("string"),
            F.col("address_line2").cast("string"),
        ),
    )
    .withColumn("city", F.col("address_city"))
    .withColumn("specialties_list", parse_list_udf(F.col("specialties")))
    .withColumn("procedure_list", parse_list_udf(F.col("procedure")))
    .withColumn("capability_list", parse_list_udf(F.col("capability")))
    .withColumn("equipment_list", parse_list_udf(F.col("equipment")))
    .withColumn(
        "unstructured_blob",
        blob_udf(
            F.col("description"),
            F.col("specialties_list"),
            F.col("procedure_list"),
            F.col("capability_list"),
            F.col("equipment_list"),
        ),
    )
    .withColumn(
        "claim_count",
        F.size(F.col("capability_list")) + F.size(F.col("procedure_list")),
    )
    .withColumn("equipment_count", F.size(F.col("equipment_list")))
    .withColumn(
        "has_equipment_evidence", F.col("equipment_count") > 0,
    )
    .withColumn(
        "completeness_signal",
        (F.col("claim_count") > 0).cast("int")
        + F.col("has_equipment_evidence").cast("int")
        + (F.col("numberDoctors").isNotNull()).cast("int")
        + (F.col("capacity").isNotNull()).cast("int"),
    )
)

# COMMAND ----------

silver_table = CFG.fq(CFG.silver_schema, CFG.silver_table)
(
    silver.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(silver_table)
)
print(f"Wrote silver -> {silver_table}")
display(spark.table(silver_table).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Coverage report

# COMMAND ----------

display(
    silver.groupBy("state")
    .agg(
        F.count("*").alias("n_facilities"),
        F.sum(F.col("has_equipment_evidence").cast("int")).alias(
            "n_with_equipment"
        ),
        F.sum(F.when(F.col("pincode").isNull(), 1).otherwise(0)).alias(
            "n_missing_pincode"
        ),
    )
    .orderBy(F.desc("n_facilities"))
)
