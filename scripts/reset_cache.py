#!/usr/bin/env python3
"""
reset_cache.py — Clear the agent OUTPUT caches so the pipeline regenerates
bronze/silver/gold code (and DQ/contracts/modeling) from scratch.

The dashboard's "Reset pipeline / start fresh" button only deleted the
LangGraph checkpoint (graph state) — it left the codebase cache in Unity
Catalog / local JSON intact, so the Engineer stage kept recalling the same
(possibly broken) cached scripts. Run this to actually flush those caches.

Usage:
    # Full reset — clears ALL cached rows for every dataset fingerprint
    python scripts/reset_cache.py

    # Also wipe the cross-run few-shot learning memory (total cold start)
    python scripts/reset_cache.py --include-fewshot

    # Only clear caches for one dataset fingerprint
    python scripts/reset_cache.py --fingerprint <fingerprint>

Runs against Unity Catalog when a real Spark session is available; otherwise
it clears the local JSON fallbacks under $AGENT_GENERATED_ROOT/config.
"""
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset agent output caches.")
    parser.add_argument("--fingerprint", default=None,
                        help="Only clear caches for this dataset fingerprint (default: clear all).")
    parser.add_argument("--include-fewshot", action="store_true",
                        help="Also wipe cross-run few-shot learning memory.")
    args = parser.parse_args()

    from dbricks_lang_agent.orchestrator import memory

    spark = None
    try:
        from dbricks_lang_agent.data_platform.spark_utils import get_spark
        spark = get_spark()
    except Exception as e:
        print(f"[reset_cache] No Spark session ({e}); clearing local JSON caches only.")

    logs = memory.reset_agent_caches(
        spark=spark,
        fingerprint=args.fingerprint,
        include_fewshot=args.include_fewshot,
    )
    print("\n=== Cache reset ===")
    for line in logs:
        print(f"  • {line}")
    print("Done. Re-run the pipeline to regenerate fresh code.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
