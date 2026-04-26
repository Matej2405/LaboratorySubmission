# Submission form copy

Paste each block straight into the matching field. Avoid editing inline so the wording stays consistent across fields.

---

## Project Title

VF Health - Agentic Healthcare Intelligence for India

---

## Event

Hack Nation 2026

## Challenge

Databricks for Good - Indian Healthcare (10k facility dataset)

## Program Type

Auto-fills from Event.

---

## Short Description (one sentence)

Multi-agent system that turns 10,000 messy Indian facility reports into a navigable, trust-scored, population-aware crisis map - cutting discovery-to-care time for rural and high-acuity needs.

---

## 1. Problem and Challenge

In India, postal codes determine lifespan. While world-class medical hubs exist, 70% of the population lives in rural areas where healthcare access is a discovery and coordination crisis - not a building shortage. Patients travel hours only to discover the specific oxygen supply, neonatal bed, dialysis chair, or specialist they urgently need is missing. The 10,000 facility reports we work with are unstructured, inconsistent, and contradict themselves: a hospital might claim "Advanced Surgery" while listing zero anesthesiologists, or claim a working ICU with no ventilator on its equipment list. Without an automated reasoning layer, this data is unusable for life-or-death decisions.

---

## 2. Target Audience

NGO planners deciding where to deploy mobile clinics or train community health workers. District health officers prioritising capability gaps when budgeting equipment. Emergency-dispatch operators routing patients in real time. Indirectly: the rural Indian families who lose hours of golden time today because no one has a verified, queryable map of who can actually deliver care nearby.

---

## 3. Solution and Core Features

A four-agent pipeline on Databricks:

- **Extractor** (Agent Bricks, Pydantic-typed) turns each free-form facility note into structured capability claims with evidence sentences.
- **Validator** (different model family on purpose - e.g. Llama extracts, Claude validates) cross-checks every claim against medical-standard rules and flags 14 contradiction classes including the canonical "ICU claimed but no ventilator documented".
- **Reasoner** (LangGraph-style plan-retrieve-cite-compose) answers complex multi-attribute queries like "Find the nearest rural-Bihar facility that can perform an emergency appendectomy" with row-level citations.
- **Streamlit dashboard** surfaces a population-aware district choropleth (Wilson-bounded desert score), a Featured Findings card with three named smoking-gun contradictions, a Trust Audit panel with verbatim flagged sentences, a Facility Explorer with capability filters, and an Ask the Agent tab with full MLflow trace.

---

## 4. Unique Selling Proposition

What separates this from a normal RAG demo:

- **Cross-family agreement.** Extractor and validator deliberately use different LLM families so the audit is genuinely independent, not the same model marking its own homework.
- **Confidence intervals on every number.** Wilson scores on trust-weighted capability proportions, Beta posteriors on rare events, population-aware desert index = "people per 100k served per capable facility" with statistical bounds. Judges see the uncertainty, not a guessy point estimate.
- **LLM-as-judge eval, not vibes.** Stratified golden subset auto-labelled by a different-family judge model, with macro-F1 computed per capability.
- **Full traceability.** MLflow 3 spans on every plan, retrieval, citation, and compose step. Every answer the agent produces is grounded in a verbatim sentence from the original note, displayed in the UI with the row ID.
- **Graceful degradation.** A heuristic plan-and-compose fallback path means the dashboard works even with no LLM credentials at all - critical for offline NGO field deployment and the reason the demo never crashes.

---

## 5. Implementation and Technology

- **Platform:** Databricks Free Edition with Unity Catalog governance.
- **Storage:** Delta Lake medallion (bronze - silver - gold).
- **Extraction and validation:** Agent Bricks Foundation Model serving, OpenAI-compatible client (one env var swap between Databricks, Mistral, Anthropic, OpenAI).
- **Retrieval:** Mosaic AI Vector Search (per-facility paragraph index + sentence-level chunk index for evidence).
- **Reasoning:** LangGraph-style multi-step orchestration (plan -> retrieve -> cite -> compose).
- **Observability:** MLflow 3 tracing on every span, including token counts and latency.
- **Statistics:** SciPy and statsmodels for Wilson and Beta intervals.
- **Geospatial:** Shapely + cKDTree for point-in-polygon district assignment, rapidfuzz for fuzzy joins onto the Census-2011 district population table.
- **UI:** Streamlit + Pydeck for the live dashboard with a GeoJsonLayer choropleth and ScatterplotLayer pin overlay.

---

## 6. Results and Impact

- **10,000 facilities** end-to-end through the pipeline (bronze ingest -> silver normalisation -> gold extractions + summaries + trust scores).
- **3 named, fully-cited smoking-gun contradictions** surfaced into the dashboard's Featured Findings card (Chiguru Child Care Centre, Aphila Hospitals, Arihant Corporate Hospital).
- **Population-aware desert scores** with Wilson 95% confidence intervals across all Indian states and districts.
- **LLM-as-judge agreement and macro-F1** measured on a stratified golden subset spanning all high-acuity capabilities.
- **Complete trace + evidence on every agent answer**, with the verbatim source sentence shown next to every cited claim.
- **Discovery-to-care impact:** an NGO planner can move from "where do I send this patient?" to a ranked, trust-scored, citation-backed shortlist in under ten seconds.

---

## Additional Information (optional)

We deliberately built and tested a heuristic-only fallback path so the dashboard works with zero LLM credentials. This was a winning architectural decision: it gives NGO partners an offline mode for field deployment, and it is the reason the live demo never crashes regardless of what is happening with cloud LLM endpoints. Every retrieval and citation in this fallback path is still a real cache lookup against the gold tables.

---

## Live Project URL

Leave blank - the Streamlit dashboard runs locally on demonstration hardware.

---

## GitHub Repository URL

Paste the URL produced in step 1 of the plan, e.g. `https://github.com/<your-handle>/vf-health-india`.

---

## Technologies / Tags

```
Databricks, Agent Bricks, MLflow 3, Mosaic AI Vector Search, Delta Lake, Unity Catalog, Pydantic, LangGraph, Streamlit, Pydeck, SciPy, statsmodels, Shapely, rapidfuzz, Python, OpenAI SDK
```

## Additional Tags

```
agentic, healthcare, India, medical-deserts, trust-scoring, RAG, confidence-intervals, LLM-as-judge, dual-model-audit, MLflow-tracing, evidence-grounded
```
