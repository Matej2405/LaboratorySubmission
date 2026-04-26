"""Data loader for the Streamlit app.

Resolves data sources in this order:
1. **Databricks SQL** (when `DATABRICKS_HOST` is set) - live queries against the
   pipeline tables.
2. **Local cache** under `data/cache/*.parquet` - written by `scripts/build_local_cache.py`
   so the app works without a Databricks cluster (eg. for the local demo).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _has_databricks() -> bool:
    return bool(os.environ.get("DATABRICKS_HOST")) and bool(
        os.environ.get("DATABRICKS_TOKEN") or os.environ.get("DATABRICKS_CLIENT_ID")
    )


def _databricks_sql(table: str) -> pd.DataFrame:
    from databricks import sql  # type: ignore

    http_path = os.environ.get("DATABRICKS_HTTP_PATH")
    if not http_path:
        raise RuntimeError(
            "Set DATABRICKS_HTTP_PATH to your SQL Warehouse HTTP path."
        )
    with sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", "").rstrip("/"),
        http_path=http_path,
        access_token=os.environ.get("DATABRICKS_TOKEN"),
    ) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {table}")
        return cur.fetchall_arrow().to_pandas()


def _cache_or_dataframe(name: str, table: str) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"{name}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    if _has_databricks():
        df = _databricks_sql(table)
        df.to_parquet(cache_file, index=False)
        return df
    return pd.DataFrame()


@lru_cache(maxsize=8)
def load_summaries() -> pd.DataFrame:
    return _cache_or_dataframe("facility_summaries", "vf_health.gold.facility_summaries")


@lru_cache(maxsize=8)
def load_trust() -> pd.DataFrame:
    return _cache_or_dataframe("trust_scores", "vf_health.gold.trust_scores")


@lru_cache(maxsize=8)
def load_extractions() -> pd.DataFrame:
    return _cache_or_dataframe("capability_claims", "vf_health.gold.capability_claims")


@lru_cache(maxsize=8)
def load_chunks() -> pd.DataFrame:
    return _cache_or_dataframe("notes_chunks", "vf_health.silver.facility_notes_chunks")


@lru_cache(maxsize=8)
def load_prevalence() -> pd.DataFrame:
    return _cache_or_dataframe("capability_prevalence", "vf_health.gold.capability_prevalence")


@lru_cache(maxsize=8)
def load_deserts() -> pd.DataFrame:
    return _cache_or_dataframe("desert_scores", "vf_health.gold.desert_scores")


def all_loaded() -> dict[str, pd.DataFrame]:
    return {
        "summaries": load_summaries(),
        "trust": load_trust(),
        "extractions": load_extractions(),
        "chunks": load_chunks(),
        "prevalence": load_prevalence(),
        "deserts": load_deserts(),
    }
