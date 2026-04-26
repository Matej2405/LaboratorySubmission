"""Central configuration for catalog, schema, and model endpoint names.

Importable from notebooks (`from agents.config import CFG`) so we don't sprinkle
magic strings everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    catalog: str = "vf_health"
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"
    gold_schema: str = "gold"
    idx_schema: str = "idx"
    eval_schema: str = "eval"

    raw_volume: str = "raw"
    raw_filename: str = "VF_Hackathon_Dataset_India_Large.xlsx"

    bronze_table: str = "facilities_raw"
    silver_table: str = "facilities_clean"
    gold_capabilities: str = "capability_claims"
    gold_trust: str = "trust_scores"
    gold_facilities: str = "facility_summaries"

    notes_index: str = "facility_notes_chunks"
    summaries_index: str = "facility_summaries"

    extractor_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
    # Dual-model audit: validator deliberately uses a *different* foundation
    # model family than the extractor. Cross-family agreement is a stronger
    # trust signal than self-agreement.
    validator_endpoint: str = "databricks-claude-3-5-sonnet"
    judge_endpoint: str = "databricks-claude-3-5-sonnet"
    embedding_endpoint: str = "databricks-bge-large-en"
    vs_endpoint: str = "vf_health_vs"

    mlflow_experiment: str = "/Shared/vf_health_agents"

    def fq(self, schema: str, table: str) -> str:
        return f"{self.catalog}.{schema}.{table}"

    @property
    def raw_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.bronze_schema}/{self.raw_volume}/{self.raw_filename}"


CFG = Config()
