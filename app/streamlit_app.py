"""VF Health - Crisis Map Streamlit App.

Three panels:
1. Chat panel - natural-language Q&A with cited evidence.
2. Map panel  - Pydeck India district choropleth + toggleable facility pins.
3. Trust panel - flag breakdown + suspect facilities.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pydeck as pdk
import streamlit as st

from app import data_loader
from agents import tools


GEOJSON_PATH = ROOT / "data" / "reference" / "india_districts.geojson"
SMOKING_GUNS_PATH = ROOT / "data" / "cache" / "smoking_guns.json"


st.set_page_config(
    page_title="VF Health - Indian Healthcare Intelligence",
    page_icon="+",
    layout="wide",
)


# ---------- Bootstrap data ------------------------------------------------

@st.cache_data(show_spinner=False)
def _bootstrap():
    return data_loader.all_loaded()


data = _bootstrap()
summaries = data["summaries"]
trust = data["trust"]
extractions = data["extractions"]
chunks = data["chunks"]
prevalence = data["prevalence"]
deserts = data["deserts"]

if summaries.empty:
    st.error(
        "No data found. Run the pipeline notebooks 01..08 first, then either "
        "(a) connect Databricks SQL via env vars, or (b) run "
        "`python scripts/build_local_cache.py` to populate `data/cache/*.parquet`."
    )
    st.stop()

def _to_list(xs):
    if xs is None:
        return []
    try:
        return list(xs)
    except TypeError:
        return []


for col in ("claimed_capabilities", "trust_flags", "specialties_list"):
    if col in summaries.columns:
        summaries[col] = summaries[col].apply(_to_list)
if "flags" in trust.columns:
    trust["flags"] = trust["flags"].apply(_to_list)
if "flag_evidence" in trust.columns:
    trust["flag_evidence"] = trust["flag_evidence"].apply(_to_list)

tools.set_local_state(
    summaries_pdf=summaries,
    trust_pdf=trust,
    gold_pdf=extractions,
    chunks_pdf=chunks,
)


@st.cache_data(show_spinner=False)
def _load_geojson(path: str = str(GEOJSON_PATH)) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _load_smoking_guns(path: str = str(SMOKING_GUNS_PATH)) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _desert_color(score: float, score_max: float) -> list[int]:
    """Yellow-orange-red gradient, alpha=180 for visibility."""
    if score_max <= 0:
        return [240, 240, 240, 60]
    t = max(0.0, min(1.0, score / score_max))
    r = int(255 * (0.4 + 0.6 * t))
    g = int(255 * (0.85 - 0.75 * t))
    b = int(50 + 30 * (1 - t))
    return [r, g, b, 180]


# ---------- Sidebar -------------------------------------------------------

st.sidebar.title("VF Health")
st.sidebar.caption("Agentic Healthcare Intelligence for India")
states = sorted(summaries["state"].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("Filter state", ["(all)"] + states, index=0)
selected_capability = st.sidebar.selectbox(
    "High-acuity capability",
    ["icu", "nicu", "dialysis", "oncology", "trauma_emergency",
     "general_surgery", "emergency_appendectomy", "cardiac_care"],
)
min_trust = st.sidebar.slider("Minimum trust score", 0, 100, 0, step=5)

st.sidebar.divider()
st.sidebar.markdown("### Pipeline status")
for k, v in data.items():
    st.sidebar.metric(k, f"{len(v):,}")


# ---------- Header --------------------------------------------------------

st.title("Indian Healthcare Intelligence System")
st.caption(
    "10,000 facilities -> structured, validated, trust-scored, and queryable. "
    "All numbers shown with confidence intervals."
)


# ---------- Tabs ----------------------------------------------------------

tab_chat, tab_map, tab_facilities, tab_trust = st.tabs(
    ["Ask the agent", "Crisis Map", "Facility explorer", "Trust audit"]
)


# ===== TAB: Chat ==========================================================

with tab_chat:
    st.subheader("Natural-language query")
    st.caption("The agent will plan -> retrieve -> cite -> answer. Citations are clickable.")

    query = st.text_input(
        "Ask a question",
        value=(
            "Find the nearest facility in rural Bihar that can perform an "
            "emergency appendectomy and typically leverages part-time doctors."
        ),
    )
    do_call = st.button("Run agent", type="primary")

    if do_call and query:
        try:
            from agents.reasoner import answer
            with st.spinner("Reasoning..."):
                res = answer(query)
            st.markdown("#### Answer")
            st.write(res.answer)

            cols = st.columns(2)
            with cols[0]:
                st.markdown("#### Plan")
                st.json(res.plan)
            with cols[1]:
                st.markdown("#### Trace")
                if res.trace_id:
                    st.code(res.trace_id)
                else:
                    st.caption("(MLflow trace ID surfaces only when running on Databricks.)")

            st.markdown("#### Citations")
            for c in res.citations[:10]:
                st.markdown(
                    f"- **[{c.get('facility_id')}] {c.get('capability')}**: "
                    f"_{c.get('sentence')}_"
                )

            st.markdown("#### Top facilities")
            facs = pd.DataFrame(res.facilities)
            if not facs.empty:
                show_cols = [c for c in [
                    "facility_id", "name", "city", "state", "trust_score",
                    "claimed_capabilities", "trust_flags",
                ] if c in facs.columns]
                st.dataframe(facs[show_cols].head(10), use_container_width=True)
        except Exception as e:
            st.error(
                "Agent call failed - this usually means LLM credentials are missing.\n\n"
                "Set `DATABRICKS_HOST` and `DATABRICKS_TOKEN` (or `OPENAI_API_KEY`) and reload.\n\n"
                f"Error: {e}"
            )


# ===== TAB: Crisis Map ====================================================

with tab_map:
    st.subheader("India Crisis Map")
    st.caption(
        f"District-level desert score for **{selected_capability}** = population per "
        "100 k served per capable facility. Larger = bigger crisis. Wilson-bounded."
    )

    # ---- Featured findings (smoking-gun cards) ---------------------------
    smoking_guns = _load_smoking_guns()
    if smoking_guns:
        st.markdown("##### Featured findings - facilities flagged by the agent")
        gun_cols = st.columns(min(3, len(smoking_guns)))
        for i, finding in enumerate(smoking_guns[:3]):
            with gun_cols[i].container(border=True):
                st.markdown(
                    f"**{finding.get('name', '?')}**  \n"
                    f":red[{finding.get('headline', '?')}]"
                )
                st.caption(
                    f"{finding.get('city','?')}, {finding.get('state','?')} - "
                    f"trust {finding.get('trust_score', 0):.0f}/100"
                )
                cited = finding.get("cited_sentence")
                if cited:
                    st.markdown(f"> _{cited}_")
                missing = finding.get("missing_requirement")
                if missing:
                    st.markdown(f":red-background[missing: **{missing}**]")
    else:
        st.info(
            "No smoking-gun findings cached yet - run "
            "`python scripts/find_smoking_guns.py` to populate this card."
        )

    show_pins = st.checkbox("Overlay individual facilities (pins)", value=False)

    # ---- District-level desert frame -------------------------------------
    desert_for_cap = pd.DataFrame()
    if not deserts.empty and "district" in deserts.columns:
        desert_for_cap = deserts[deserts["capability"] == selected_capability].copy()
        if selected_state != "(all)":
            desert_for_cap = desert_for_cap[desert_for_cap["state"] == selected_state]

    geojson = _load_geojson()
    score_max = float(desert_for_cap["desert_score"].max() or 1.0) if not desert_for_cap.empty else 1.0

    layers: list[pdk.Layer] = []

    if geojson is not None and not desert_for_cap.empty:
        score_lookup: dict[tuple[str, str], dict] = {}
        for _, r in desert_for_cap.iterrows():
            score_lookup[(str(r.get("state")).strip(), str(r.get("district")).strip())] = {
                "desert_score": float(r.get("desert_score") or 0.0),
                "p_hat": float(r.get("p_hat") or 0.0),
                "n_facilities": int(r.get("n_facilities") or 0),
                "population": int(r.get("population") or 0),
                "low": float(r.get("low") or 0.0),
                "high": float(r.get("high") or 0.0),
            }

        decorated_features = []
        for feat in geojson.get("features", []):
            props = dict(feat.get("properties", {}) or {})
            key = (str(props.get("st_nm", "")).strip(), str(props.get("district", "")).strip())
            entry = score_lookup.get(key)
            if entry is None:
                props["desert_score"] = 0.0
                props["fill_color"] = [220, 220, 220, 30]
                props["display_text"] = f"{props.get('district','?')} (no data)"
            else:
                props.update(entry)
                props["fill_color"] = _desert_color(entry["desert_score"], score_max)
                props["display_text"] = (
                    f"{props.get('district','?')}, {props.get('st_nm','?')} - "
                    f"desert {entry['desert_score']:.0f} (pop {entry['population']:,}, "
                    f"{entry['n_facilities']} fac, p={entry['p_hat']*100:.1f}%)"
                )
            decorated_features.append({**feat, "properties": props})
        decorated = {**geojson, "features": decorated_features}

        layers.append(pdk.Layer(
            "GeoJsonLayer",
            decorated,
            stroked=True,
            filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[80, 80, 80, 120],
            line_width_min_pixels=0.5,
            pickable=True,
            auto_highlight=True,
        ))
    elif geojson is None:
        st.warning(
            "District boundary file missing. Run "
            "`python scripts/fetch_reference_data.py` then "
            "`python scripts/build_local_cache.py`. Falling back to facility pins."
        )
        show_pins = True

    if show_pins:
        df = summaries.dropna(subset=["latitude", "longitude"]).copy()
        if selected_state != "(all)":
            df = df[df["state"] == selected_state]
        df = df[df["claimed_capabilities"].apply(lambda xs: selected_capability in (xs or []))]
        df = df[df["trust_score"].fillna(0) >= min_trust]
        df["radius"] = 1500 + (df["trust_score"].fillna(0) * 60)
        df["color_r"] = 220 - (df["trust_score"].fillna(0) * 1.8)
        df["color_g"] = 50 + (df["trust_score"].fillna(0) * 1.5)
        df["color_b"] = 80
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            df,
            get_position="[longitude, latitude]",
            get_radius="radius",
            get_fill_color="[color_r, color_g, color_b, 200]",
            pickable=True,
        ))

    view = pdk.ViewState(latitude=22.5, longitude=80.0, zoom=4.0)
    if not layers:
        st.info("No layers to render. Check that capability/state filters return data.")
    else:
        st.pydeck_chart(pdk.Deck(
            initial_view_state=view,
            layers=layers,
            tooltip={
                "html": "<b>{display_text}</b><br/>{name}",
                "style": {"color": "white", "fontSize": "12px"},
            },
        ))

    st.markdown("#### Top medical deserts (population-aware, Wilson-bounded)")
    if not desert_for_cap.empty:
        top = desert_for_cap.sort_values("desert_score", ascending=False).head(15).copy()
        top["prevalence"] = top.apply(
            lambda r: f"{r['p_hat']*100:.1f}% [{r['low']*100:.1f}-{r['high']*100:.1f}%]", axis=1,
        )
        top["desert"] = top.apply(
            lambda r: f"{r['desert_score']:.0f} [{r.get('desert_low') or 0:.0f}-{r.get('desert_high') or 0:.0f}]",
            axis=1,
        )
        st.dataframe(
            top[["state", "district", "population", "n_facilities", "prevalence", "desert"]],
            use_container_width=True,
            hide_index=True,
        )
    elif not deserts.empty:
        # State-level fallback if district data unavailable
        top = deserts[deserts["capability"] == selected_capability].copy()
        top = top.sort_values("desert_score", ascending=False).head(15)
        top["prevalence"] = top.apply(
            lambda r: f"{r['p_hat']*100:.1f}% [{r['low']*100:.1f}-{r['high']*100:.1f}%]", axis=1,
        )
        st.dataframe(
            top[["state", "capability", "prevalence", "n_facilities", "desert_score"]],
            use_container_width=True,
            hide_index=True,
        )


# ===== TAB: Facility explorer ============================================

with tab_facilities:
    st.subheader("Facility explorer")

    df = summaries.copy()
    if selected_state != "(all)":
        df = df[df["state"] == selected_state]
    df = df[df["trust_score"].fillna(0) >= min_trust]
    df = df[df["claimed_capabilities"].apply(
        lambda xs: selected_capability in (xs or [])
    )]
    df = df.sort_values("trust_score", ascending=False)
    st.write(f"{len(df):,} matching facilities")

    if df.empty:
        st.info("No facilities match the current filters - try lowering trust threshold.")
    else:
        for _, row in df.head(20).iterrows():
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"**{row.get('name', '?')}**")
                    st.caption(
                        f"{row.get('city','?')}, {row.get('state','?')} - "
                        f"type: {row.get('facilityTypeId','?')} - operator: {row.get('operatorTypeId','?')}"
                    )
                    caps = row.get("claimed_capabilities") or []
                    st.write("Claims: " + ", ".join(caps) if caps else "_no claims_")
                    flags = row.get("trust_flags") or []
                    if flags:
                        st.warning("Flags: " + ", ".join(flags))
                with c2:
                    score = row.get("trust_score") or 0
                    st.metric("Trust", f"{score:.0f}/100")
                with c3:
                    if st.button("Show evidence", key=f"ev_{row['facility_id']}"):
                        ev = tools.get_evidence(row["facility_id"], capability=selected_capability)
                        for e in ev:
                            st.caption(f"[{e['capability']}] _{e['sentence']}_")


# ===== TAB: Trust audit ===================================================

with tab_trust:
    st.subheader("Trust audit")

    if trust.empty:
        st.info("No trust scores available.")
    else:
        df = trust.merge(
            summaries[["facility_id", "name", "city", "state", "claimed_capabilities"]],
            on="facility_id", how="left",
        )
        if selected_state != "(all)":
            df = df[df["state"] == selected_state]
        df = df[df["claimed_capabilities"].apply(
            lambda xs: selected_capability in (xs or [])
        )]

        st.markdown("#### Trust score distribution")
        st.bar_chart(df["score"].dropna())

        st.markdown("#### Most common contradiction flags")
        flat = df.explode("flags")["flags"].value_counts().head(15)
        st.dataframe(flat.rename_axis("flag").reset_index(name="count"), use_container_width=True)

        st.markdown("#### Suspect facilities (lowest scores, claiming high-acuity)")
        suspect = df.sort_values("score").head(20)
        for _, row in suspect.iterrows():
            with st.container(border=True):
                st.markdown(
                    f"**{row.get('name','?')}** - {row.get('city','?')}, {row.get('state','?')} "
                    f"-> trust **{row.get('score', 0):.0f}**"
                )
                st.write("Claims: " + ", ".join(row.get("claimed_capabilities") or []))
                if row.get("flags"):
                    for flag, ev in zip(row["flags"], row["flag_evidence"] or []):
                        st.error(f"{flag}: \"{ev}\"")
