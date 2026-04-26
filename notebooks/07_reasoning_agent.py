# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Reasoning Agent
# MAGIC
# MAGIC End-to-end demo: a few canned NL questions are answered with cited
# MAGIC facilities. Every step is wrapped in MLflow 3 spans so the trace UI
# MAGIC shows planner -> retrieve -> cite -> compose.

# COMMAND ----------

# MAGIC %pip install -q openai==1.55.0 mlflow==3.0.0rc0 databricks-vectorsearch==0.40
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys, os
sys.path.append(os.path.abspath(".."))

import mlflow
from agents.config import CFG
from agents.reasoner import answer

mlflow.set_experiment(CFG.mlflow_experiment)

# COMMAND ----------

QUERIES = [
    "Find the nearest facility in rural Bihar that can perform an emergency appendectomy and typically leverages part-time doctors.",
    "Which Tamil Nadu hospitals claim NICU but show no neonatologist or pediatrician?",
    "Show me the top 5 most trustworthy oncology centres in Maharashtra.",
    "Which districts in Uttar Pradesh appear to be dialysis deserts?",
    "List 24x7 emergency hospitals in West Bengal with cardiologist on staff.",
]

# COMMAND ----------

for q in QUERIES:
    print("\n" + "=" * 90)
    print("Q:", q)
    res = answer(q)
    print("\nPLAN:", res.plan)
    print("\nANSWER:\n", res.answer)
    print("\nCITATIONS:")
    for c in res.citations[:5]:
        print(f"  [{c.get('facility_id')}] {c.get('capability')}: {c.get('sentence')}")
