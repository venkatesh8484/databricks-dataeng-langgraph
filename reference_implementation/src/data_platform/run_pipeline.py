"""
run_pipeline.py
===============
Deterministic entry point that runs the reference Bronze -> Silver -> Gold pipeline in Databricks.
Used for verification of cluster access, schemas, and volume mounts.
"""
from __future__ import annotations

import json
import os
import sys

from . import bronze, silver, gold
from dbricks_lang_agent.data_platform.spark_utils import reset_lake

def main():
    print("Starting Databricks Medallion Reference Pipeline...")
    
    # Optional reset of schemas in Databricks (dangerous in production, used in dev)
    if os.environ.get("RESET_SCHEMAS_BEFORE_RUN", "false").lower() == "true":
        print("Resetting target schemas...")
        reset_lake()

    print("\n=== STEP 1: Bronze Ingestion (Raw Volume -> Bronze Schema) ===")
    bronze_result = bronze.ingest_all()
    print(json.dumps({k: {"row_count": v["row_count"]} for k, v in bronze_result.items()}, indent=2))

    print("\n=== STEP 2: Silver Transformation & Contract Validation ===")
    # Run silver transformations. By default, don't halt on first contract breach
    silver_summary = silver.transform_all(fail_fast_on_hard_breach=False)
    print(json.dumps(silver_summary["tables"], indent=2))
    
    # Save the silver summary JSON locally for tracking
    reports_dir = "/tmp/reports"
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "silver_summary.json"), "w") as f:
        json.dump(silver_summary, f, indent=2, default=str)

    if silver_summary["halted_at"]:
        print(f"!!! PIPELINE HALTED: Hard contract breach on table: '{silver_summary['halted_at']}'")
        sys.exit(1)

    print("\n=== STEP 3: Gold Dimensional Load (Kimball Star Schema) ===")
    gold_result = gold.build_all()
    print(json.dumps(gold_result["row_counts"], indent=2))
    
    # Save the gold summary JSON
    with open(os.path.join(reports_dir, "gold_summary.json"), "w") as f:
        json.dump(gold_result, f, indent=2, default=str)

    print("\nReference Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
