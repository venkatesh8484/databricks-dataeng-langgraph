# Reference Implementation (Unity Catalog Baseline)

This folder contains a verified, manual (deterministic) implementation of the medallion pipeline designed for **Databricks and Unity Catalog**. 

It runs the exact same transformation, contract checking, and SCD Type 2 logic as the agentic pipeline, but without the LangGraph orchestrator or LLM nodes.

Use it to:
1. Validate that your Databricks cluster has read access to the Raw Volume.
2. Validate that your Unity Catalog permissions allow schema creation and table writes.
3. Contrast agent outputs against a hand-written baseline.

---

## Running the Reference Pipeline

You can run this directly in a Databricks Notebook or as a Databricks Workflow Python Task:

```python
import os
import sys

# Ensure project root is on PYTHONPATH
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from reference_implementation.src.data_platform import run_pipeline
run_pipeline.main()
```
