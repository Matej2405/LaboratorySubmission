"""Render submission-ready architecture and methodology PNGs with matplotlib.

Usage:
    python scripts/render_submission_assets.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "submission"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DARK_BG = "#0c1420"
PANEL_BG = "#101826"
TEXT = "#f4f6fb"
SUBTLE = "#a3b1c7"
ACCENT = "#ff5a76"
CYAN = "#5eead4"
AMBER = "#fbbf24"
GREEN = "#22c55e"
PURPLE = "#a78bfa"


def _box(ax, x, y, w, h, label, *, color=PANEL_BG, edge=SUBTLE, text_color=TEXT, fontsize=10, weight="normal"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        facecolor=color,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        color=text_color, fontsize=fontsize,
        fontweight=weight, wrap=True,
    )


def _arrow(ax, src, dst, *, color=SUBTLE, width=1.4):
    arr = FancyArrowPatch(
        src, dst,
        arrowstyle="-|>", mutation_scale=14,
        linewidth=width, color=color,
        shrinkA=4, shrinkB=4,
    )
    ax.add_patch(arr)


def render_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        8, 8.4,
        "VF Health  -  System architecture",
        ha="center", va="center",
        color=TEXT, fontsize=22, fontweight="bold",
    )
    ax.text(
        8, 7.95,
        "Bronze - Silver - Gold medallion on Delta Lake; dual-family agents traced in MLflow 3.",
        ha="center", va="center",
        color=SUBTLE, fontsize=11, style="italic",
    )

    # Row 1: Ingest + medallion
    _box(ax, 0.4, 5.5, 2.4, 1.4, "Excel\n10,000 facilities\n41 columns", color="#1c2740", edge=SUBTLE, fontsize=10)
    _box(ax, 3.2, 5.5, 1.8, 1.4, "Bronze\n(raw Delta)", color="#2a1f12", edge=AMBER, fontsize=10)
    _box(ax, 5.4, 5.5, 2.2, 1.4, "Silver\ncleaned + parsed\nUnity Catalog", color="#1f2a16", edge=GREEN, fontsize=10)
    _box(ax, 8.0, 5.5, 2.4, 1.4, "Gold\ncapabilities + citations\n+ trust + districts", color="#2c1f12", edge=AMBER, fontsize=10)

    # Row 1 right: Mosaic Vector Search
    _box(ax, 10.8, 5.5, 2.4, 1.4, "Mosaic AI\nVector Search\nfacility + chunk", color="#1c1530", edge=PURPLE, fontsize=10)

    # Row 1 far right: Confidence layer
    _box(ax, 13.6, 5.5, 2.0, 1.4, "Confidence\nWilson + Beta\nintervals", color="#1a2630", edge=CYAN, fontsize=10)

    # Row 2: Agents
    _box(ax, 3.2, 3.2, 2.4, 1.6, "Extractor agent\nAgent Bricks\nPydantic schema\n(Llama-3-70B)", color="#241125", edge=ACCENT, fontsize=9.5, weight="bold")
    _box(ax, 6.0, 3.2, 2.4, 1.6, "Validator agent\nmedical KB +\nLLM judge\n(Claude-3.5)", color="#241125", edge=ACCENT, fontsize=9.5, weight="bold")
    _box(ax, 8.8, 3.2, 2.4, 1.6, "Trust scorer\ncompleteness +\ncontradiction rules", color="#241125", edge=ACCENT, fontsize=9.5, weight="bold")
    _box(ax, 11.6, 3.2, 2.4, 1.6, "Reasoner\nplan -> retrieve\n-> cite -> compose\n(LangGraph)", color="#241125", edge=ACCENT, fontsize=9.5, weight="bold")

    # Row 3: UI + observability
    _box(ax, 4.6, 1.0, 5.0, 1.4, "Streamlit dashboard\nCrisis Map  /  Trust Audit\nFacility Explorer  /  Ask the Agent", color="#1a1f2c", edge=CYAN, fontsize=10, weight="bold")
    _box(ax, 10.0, 1.0, 3.2, 1.4, "MLflow 3 tracing\nspans, tokens, latency\non every step", color="#1a2630", edge=CYAN, fontsize=10)
    _box(ax, 0.4, 1.0, 3.8, 1.4, "Eval harness\nLLM-as-judge labels +\nmacro-F1 per capability", color="#1a2630", edge=CYAN, fontsize=10)

    # Arrows: medallion flow
    _arrow(ax, (2.8, 6.2), (3.2, 6.2))
    _arrow(ax, (5.0, 6.2), (5.4, 6.2))
    _arrow(ax, (7.6, 6.2), (8.0, 6.2))
    _arrow(ax, (10.4, 6.2), (10.8, 6.2))
    _arrow(ax, (13.2, 6.2), (13.6, 6.2))

    # Silver -> Extractor
    _arrow(ax, (6.5, 5.5), (4.4, 4.8), color=ACCENT)
    # Extractor -> Validator
    _arrow(ax, (5.6, 4.0), (6.0, 4.0), color=ACCENT)
    # Validator -> Trust
    _arrow(ax, (8.4, 4.0), (8.8, 4.0), color=ACCENT)
    # Trust -> Gold
    _arrow(ax, (10.0, 4.8), (9.2, 5.5), color=AMBER)
    # Gold -> Reasoner
    _arrow(ax, (10.4, 5.5), (12.8, 4.8), color=ACCENT)
    # VSearch -> Reasoner
    _arrow(ax, (12.0, 5.5), (12.8, 4.8), color=PURPLE)
    # Reasoner -> Streamlit
    _arrow(ax, (12.0, 3.2), (8.0, 2.4), color=CYAN)
    # Trust -> Streamlit
    _arrow(ax, (10.0, 3.2), (7.0, 2.4), color=CYAN)
    # Confidence -> Streamlit
    _arrow(ax, (14.6, 5.5), (9.0, 2.4), color=CYAN)
    # MLflow ties
    _arrow(ax, (12.8, 3.2), (11.6, 2.4), color=CYAN, width=1.0)
    _arrow(ax, (4.0, 3.2), (3.0, 2.4), color=CYAN, width=1.0)

    # Legend / footnote
    ax.text(
        0.4, 0.35,
        "Arrows: red = LLM agent flow; amber = Delta medallion; cyan = observability + UI; purple = vector retrieval.",
        ha="left", va="center",
        color=SUBTLE, fontsize=8.5, style="italic",
    )

    out = OUT_DIR / "architecture.png"
    fig.savefig(out, facecolor=DARK_BG, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out


def render_methodology() -> Path:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        8, 8.5,
        "Methodology  -  why these numbers are trustworthy",
        ha="center", va="center",
        color=TEXT, fontsize=22, fontweight="bold",
    )
    ax.text(
        8, 8.05,
        "Four engineering decisions that turn an extraction demo into auditable healthcare intelligence.",
        ha="center", va="center",
        color=SUBTLE, fontsize=11, style="italic",
    )

    cards = [
        {
            "title": "1. Cross-family agreement",
            "color": ACCENT,
            "lines": [
                "Extractor and Validator deliberately use",
                "different LLM families (Llama-3-70B + Claude-3.5).",
                "Agreement reflects independent corroboration,",
                "not the same model marking its own homework.",
                "",
                "Provider-agnostic: one env var swap to",
                "Mistral / OpenAI / Databricks / Anthropic.",
            ],
        },
        {
            "title": "2. Wilson + Beta intervals everywhere",
            "color": CYAN,
            "lines": [
                "Every prevalence reported as 95% CI,",
                "never a single point estimate.",
                "",
                "Trust-weighted Wilson for proportions,",
                "Beta posterior for rare events,",
                "n_eff displayed alongside every claim.",
                "Source: agents/confidence.py",
            ],
        },
        {
            "title": "3. Population-aware desert score",
            "color": AMBER,
            "lines": [
                "District score =",
                "  people per 100k served per capable facility.",
                "",
                "Wilson lower/upper band, not raw counts.",
                "Census-2011 joined via rapidfuzz",
                "(98.7% district coverage).",
                "Shapely + cKDTree point-in-polygon mapping.",
            ],
        },
        {
            "title": "4. LLM-as-judge evaluation",
            "color": GREEN,
            "lines": [
                "Stratified 50-row golden subset,",
                "auto-labelled by a different-family judge.",
                "",
                "Per-capability precision / recall / macro-F1.",
                "10 hardest disagreements published in",
                "evals/spot_check.md for human audit.",
                "Source: evals/auto_label_golden.py",
            ],
        },
    ]

    # 2x2 grid of cards
    positions = [(0.4, 4.25), (8.2, 4.25), (0.4, 0.4), (8.2, 0.4)]
    card_w, card_h = 7.4, 3.4

    for (x, y), card in zip(positions, cards):
        _box(ax, x, y, card_w, card_h, "", color=PANEL_BG, edge=card["color"])
        ax.text(
            x + 0.3, y + card_h - 0.45,
            card["title"],
            ha="left", va="center",
            color=card["color"], fontsize=14, fontweight="bold",
        )
        for i, line in enumerate(card["lines"]):
            ax.text(
                x + 0.3, y + card_h - 1.0 - i * 0.32,
                line,
                ha="left", va="center",
                color=TEXT if line.strip() else SUBTLE,
                fontsize=10,
            )

    out = OUT_DIR / "methodology.png"
    fig.savefig(out, facecolor=DARK_BG, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out


if __name__ == "__main__":
    arch = render_architecture()
    meth = render_methodology()
    print(f"Architecture: {arch}")
    print(f"Methodology: {meth}")
