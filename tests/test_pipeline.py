"""Smoke tests that exercise the local pipeline (no LLM, no Spark)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "cache"

if not CACHE.exists() or not list(CACHE.glob("*.parquet")):
    pytest.skip(
        "Local cache not built. Run scripts/build_local_cache.py first.",
        allow_module_level=True,
    )


def _to_list(xs):
    if xs is None:
        return []
    try:
        return list(xs)
    except TypeError:
        return []


@pytest.fixture(scope="module")
def cache():
    summaries = pd.read_parquet(CACHE / "facility_summaries.parquet")
    trust = pd.read_parquet(CACHE / "trust_scores.parquet")
    extractions = pd.read_parquet(CACHE / "capability_claims.parquet")
    chunks = pd.read_parquet(CACHE / "notes_chunks.parquet")
    deserts = pd.read_parquet(CACHE / "desert_scores.parquet")
    for c in ("claimed_capabilities", "trust_flags", "specialties_list"):
        if c in summaries.columns:
            summaries[c] = summaries[c].apply(_to_list)
    if "flags" in trust.columns:
        trust["flags"] = trust["flags"].apply(_to_list)
    return dict(
        summaries=summaries, trust=trust, extractions=extractions,
        chunks=chunks, deserts=deserts,
    )


def test_row_counts(cache):
    assert len(cache["summaries"]) == 10000
    assert len(cache["trust"]) == 10000
    assert len(cache["extractions"]) == 10000
    assert len(cache["chunks"]) > 5000


def test_trust_score_range(cache):
    s = cache["trust"]["score"]
    assert s.min() >= 0
    assert s.max() <= 100


def test_top_flags_present(cache):
    flags = cache["trust"].explode("flags")["flags"].dropna().value_counts()
    assert any("anesthesiologist" in f for f in flags.index[:10])
    assert "claims_with_zero_equipment" in flags.index


def test_local_tools_filter(cache):
    from agents import tools

    tools.set_local_state(
        summaries_pdf=cache["summaries"],
        gold_pdf=cache["extractions"],
        chunks_pdf=cache["chunks"],
        trust_pdf=cache["trust"],
    )

    res = tools.find_facilities(
        state="Tamil Nadu", capabilities=["general_surgery"], min_trust=0, limit=5,
    )
    assert isinstance(res, list)
    if res:
        for r in res:
            assert r.get("state") == "Tamil Nadu"
            assert "general_surgery" in (_to_list(r.get("claimed_capabilities")))


def test_get_evidence_returns_sentences(cache):
    from agents import tools

    tools.set_local_state(
        summaries_pdf=cache["summaries"],
        gold_pdf=cache["extractions"],
        chunks_pdf=cache["chunks"],
    )
    flagged = cache["extractions"][cache["extractions"]["n_evidence_sentences"] > 0]
    assert len(flagged) > 0
    fid = flagged.iloc[0]["facility_id"]
    ev = tools.get_evidence(fid)
    assert isinstance(ev, list)
    assert all("sentence" in e and "capability" in e for e in ev)


def test_desert_scores_make_sense(cache):
    d = cache["deserts"]
    must_have = {"state", "capability", "p_hat", "low", "high", "desert_score"}
    assert must_have.issubset(d.columns)
    assert (d["low"] <= d["high"]).all()
    assert (d["desert_score"] >= 0).all()
    # p_hat is still bounded in [0, 1].
    assert (d["p_hat"] >= 0).all()
    assert (d["p_hat"] <= 1).all()
    # When district population is available the score is a `pop_per_facility`
    # ratio (un-bounded above 1) with Wilson confidence; pure-state fallback
    # rows still use the [0, 1] band. Desert_low <= desert_high always holds
    # when both columns are present.
    if {"desert_low", "desert_high"}.issubset(d.columns):
        both = d.dropna(subset=["desert_low", "desert_high"])
        assert (both["desert_low"] <= both["desert_high"]).all()


def test_confidence_intervals_monotone():
    from agents.confidence import wilson_interval, beta_posterior, trust_weighted_proportion

    iv1 = wilson_interval(0, 10)
    iv2 = wilson_interval(10, 10)
    assert iv1.point == 0 and iv1.upper > 0
    assert iv2.point == 1 and iv2.lower < 1

    iv3 = beta_posterior(5, 10)
    assert 0 < iv3.point < 1

    iv4 = trust_weighted_proportion([0.9] * 10, [1] * 5 + [0] * 5)
    assert 0.3 < iv4.point < 0.7
