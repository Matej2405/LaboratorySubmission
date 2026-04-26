# Spot check - heuristic vs LLM-judge disagreements

> **Status:** Not yet generated. Run the LLM judge with valid credentials to populate this file.

## How to generate

```bash
# 1. Build the golden subset with the unstructured blob (LLM judge needs context)
python evals/golden_subset.py --include_evidence

# 2. Run the LLM-as-judge labeler against a *different model family* than the
#    extractor. Examples (set the matching env var first):
ANTHROPIC_API_KEY=...   python evals/auto_label_golden.py --judge claude-3-5-sonnet-20241022
OPENAI_API_KEY=...      python evals/auto_label_golden.py --judge gpt-4o-mini
DATABRICKS_HOST=... DATABRICKS_TOKEN=... \
  python evals/auto_label_golden.py --judge databricks-claude-3-5-sonnet
```

The script writes:

* `evals/golden_subset.labeled.csv` - the original sheet plus filled `label_*` columns (0/1).
* `evals/spot_check.md` (this file) - the top-10 disagreements between the heuristic baseline and the judge, ranked by judge confidence. Use it for a 5-minute human sanity check before publishing eval numbers.

## Why we record it

Hackathon judges should see, line by line, where the agent's heuristic baseline differs from a stronger LLM judge. Cross-family agreement (Llama extractor vs Claude/GPT judge) is the metric quoted in the README, not self-agreement.
