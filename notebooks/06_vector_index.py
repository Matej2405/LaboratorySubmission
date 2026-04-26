# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Vector Search Indexes
# MAGIC
# MAGIC Builds two Mosaic AI Vector Search Delta-sync indexes:
# MAGIC
# MAGIC 1. **`facility_notes_chunks`** - sentence-level chunks of `unstructured_blob`
# MAGIC    for evidence retrieval and citation surface area.
# MAGIC 2. **`facility_summaries`** - one synthesized paragraph per facility
# MAGIC    (LLM-extracted summary + structured tags) for high-level semantic match.
# MAGIC
# MAGIC Filterable columns include `state`, `pincode`, `claimed_capabilities`,
# MAGIC `trust_score` so the Reasoning Agent can do hybrid retrieval.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch==0.40
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys, os
sys.path.append(os.path.abspath(".."))

from pyspark.sql import functions as F, types as T
from databricks.vector_search.client import VectorSearchClient

from agents.config import CFG
from agents.text_utils import split_sentences

# COMMAND ----------

silver = spark.table(CFG.fq(CFG.silver_schema, CFG.silver_table))
gold = spark.table(CFG.fq(CFG.gold_schema, CFG.gold_capabilities))
trust = spark.table(CFG.fq(CFG.gold_schema, CFG.gold_trust))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build summary table
# MAGIC
# MAGIC Composes a short synthesized paragraph per facility from extracted claims +
# MAGIC structured metadata. This is what the agent's *high-level* semantic search
# MAGIC matches against.

# COMMAND ----------

base = (
    silver.select(
        "facility_id", "name", "city", "state", "pincode",
        "facilityTypeId", "operatorTypeId",
        F.col("numberDoctors").alias("n_doctors"),
        F.col("capacity").alias("n_beds"),
        "specialties_list",
        F.col("description").alias("description"),
        "latitude", "longitude",
    )
    .join(
        gold.select(
            "facility_id",
            F.col("claimed_capabilities").alias("claimed_capabilities"),
            "extraction_json",
        ),
        on="facility_id", how="left",
    )
    .join(
        trust.select(
            "facility_id",
            F.col("score").alias("trust_score"),
            F.col("flags").alias("trust_flags"),
        ),
        on="facility_id", how="left",
    )
    .withColumn(
        "summary_text",
        F.concat_ws(
            " | ",
            F.col("name"),
            F.concat_ws(", ", F.col("city"), F.col("state")),
            F.concat_ws("=", F.lit("type"), F.col("facilityTypeId")),
            F.concat_ws("=", F.lit("operator"), F.coalesce(F.col("operatorTypeId"), F.lit("unknown"))),
            F.concat_ws("=", F.lit("specialties"), F.array_join(F.coalesce(F.col("specialties_list"), F.array()), ",")),
            F.concat_ws("=", F.lit("capabilities"), F.array_join(F.coalesce(F.col("claimed_capabilities"), F.array()), ",")),
            F.concat_ws("=", F.lit("trust"), F.coalesce(F.col("trust_score").cast("string"), F.lit("?"))),
            F.coalesce(F.col("description"), F.lit("")),
        ),
    )
)

summaries_table = CFG.fq(CFG.gold_schema, CFG.gold_facilities)
(
    base.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(summaries_table)
)
print(f"Wrote {spark.table(summaries_table).count():,} summary rows -> {summaries_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build sentence-level chunks table

# COMMAND ----------

split_sentences_udf = F.udf(split_sentences, T.ArrayType(T.StringType()))

chunks = (
    silver.select("facility_id", "state", "pincode", "city", "unstructured_blob")
    .withColumn("sentences", split_sentences_udf(F.col("unstructured_blob")))
    .selectExpr(
        "facility_id", "state", "pincode", "city",
        "posexplode(sentences) as (chunk_idx, chunk_text)",
    )
    .withColumn("chunk_id", F.concat_ws("::", "facility_id", F.col("chunk_idx").cast("string")))
)

chunks_table = CFG.fq(CFG.silver_schema, "facility_notes_chunks")
(
    chunks.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(chunks_table)
)
print(f"Wrote {spark.table(chunks_table).count():,} chunks -> {chunks_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Vector Search endpoint + Delta-sync indexes
# MAGIC
# MAGIC On Free Edition you may already have a default endpoint - reuse it if so.

# COMMAND ----------

vsc = VectorSearchClient(disable_notice=True)

try:
    vsc.create_endpoint(name=CFG.vs_endpoint, endpoint_type="STANDARD")
    print(f"Created VS endpoint {CFG.vs_endpoint}")
except Exception as e:
    print(f"Endpoint exists or could not be created: {e}")

# COMMAND ----------

def _create_index(source_table: str, index_name: str, primary_key: str, embed_col: str):
    fq_index = CFG.fq(CFG.idx_schema, index_name)
    spark.sql(
        f"ALTER TABLE {source_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    try:
        return vsc.create_delta_sync_index_and_wait(
            endpoint_name=CFG.vs_endpoint,
            source_table_name=source_table,
            index_name=fq_index,
            pipeline_type="TRIGGERED",
            primary_key=primary_key,
            embedding_source_column=embed_col,
            embedding_model_endpoint_name=CFG.embedding_endpoint,
        )
    except Exception as e:
        print(f"Index {fq_index} likely exists already: {e}")
        return vsc.get_index(endpoint_name=CFG.vs_endpoint, index_name=fq_index)


notes_idx = _create_index(
    source_table=chunks_table,
    index_name=CFG.notes_index,
    primary_key="chunk_id",
    embed_col="chunk_text",
)
sum_idx = _create_index(
    source_table=summaries_table,
    index_name=CFG.summaries_index,
    primary_key="facility_id",
    embed_col="summary_text",
)
print("Indexes ready.")
