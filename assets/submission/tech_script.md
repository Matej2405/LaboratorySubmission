# Tech Video script (60 seconds)

Goal: Technical explanation. Cover stack, architecture, and implementation. Convince a Databricks judge that this would scale and that the engineering is real.

## Pre-flight (do this BEFORE you hit record)

1. Open the architecture PNG at `assets/submission/architecture.png` full-screen in an image viewer. This is your opening slide.
2. In VS Code, open the following files in tabs in this exact order so cycling left-to-right tells the story:
   - `agents/extractor.py`
   - `agents/validator.py`
   - `agents/confidence.py`
   - `evals/auto_label_golden.py`
   - `notebooks/09_eval_harness.py`
3. Browser at `http://localhost:8888`, **Ask the agent** tab open with the Maharashtra oncology answer already loaded (so the trace span tree is visible without you waiting).
4. Hide the VS Code sidebar and minimap (Cmd/Ctrl+B). Use Zen Mode if you have time.

## Voiceover beats (60 sec total)

| Time | View | What you do | What you say |
|---|---|---|---|
| 0 - 11 s | Architecture PNG | Hold on the diagram | "The stack: bronze-silver-gold medallion on Delta Lake, an Agent Bricks extractor, a different-family validator, a LangGraph reasoner, and Mosaic AI Vector Search - everything traced end-to-end in MLflow 3." |
| 11 - 22 s | `agents/extractor.py` | Scroll once through the file | "Extractor uses a Pydantic schema with retries. The endpoint family is auto-detected, so any OpenAI-compatible provider - Mistral, Anthropic, Databricks, OpenAI - works with one env var swap. Provider-agnostic by design." |
| 22 - 32 s | `agents/validator.py` | Scroll to `_llm_judge` | "Self-correction is real, not theatre. The validator deliberately runs on a different model family than the extractor, so the audit is genuinely independent. Fourteen contradiction classes today, easy to extend." |
| 32 - 42 s | `agents/confidence.py` | Stop on `desert_index` and `wilson_interval` | "Every number ships with statistical bounds. Population-aware desert score, Wilson interval for trust-weighted proportions, Beta posterior for rare events. We do not sell point estimates as truth." |
| 42 - 52 s | `evals/auto_label_golden.py` then `notebooks/09_eval_harness.py` | Quick cut between both | "We measure ourselves: an LLM-as-judge auto-labels a stratified golden subset, and the eval harness reports macro-F1 per capability. The judge runs on a different family from the extractor too." |
| 52 - 60 s | Streamlit Ask the agent (already populated) | Hover the cited sentences | "Result: every answer is grounded in a verbatim sentence from the source notes, with a full MLflow trace from query to citation. No hallucination, no black box." |

## Visual polish

- VS Code theme: dark+ or any high-contrast dark theme. Font size 16+ for legibility on a 1080p export.
- When scrolling code, scroll smoothly with mouse wheel - no cursor-key jumps.
- Pre-collapse function bodies you do not want to dwell on (Ctrl+K Ctrl+0 collapses all in VS Code).

## Save as

`assets/submission/tech.mp4` (mp4, H.264, max 60 sec, target 20-40 MB).
