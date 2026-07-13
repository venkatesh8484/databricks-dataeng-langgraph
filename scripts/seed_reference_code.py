"""
seed_reference_code.py
======================
Prepare the pipeline code ONCE for this dataset and store it in Unity Catalog so
the orchestrator reuses it on every run (and can still patch on top of it).

Rationale
---------
The Data-Engineer agent regenerates bronze/silver/gold with an LLM each time,
which is non-deterministic and occasionally hallucinates. Instead, we seed the
codebase cache (`<catalog>.gold.agent_script_codebase_memory`) with the known-good
hand-written reference scripts, keyed by this dataset's schema fingerprint, and
log an "approved" review for the Engineering stage. From then on:

  * engineering_node gets an exact-fingerprint cache hit → reuses the seeded
    code instead of calling the LLM;
  * because the stage is already approved, the run auto-advances past the
    Engineering gate;
  * if a future run's execution fails, the normal self-heal / targeted-patch
    path still kicks in and edits ON TOP of the seeded code — you're not locked
    to it, it's just the trusted starting point;
  * if the source SCHEMA changes (columns added/removed/renamed) the fingerprint
    changes and the pipeline regenerates fresh — re-run this seed afterward if
    you want to pin the new schema too.

Usage (Databricks notebook cell)
--------------------------------
    from scripts import seed_reference_code   # or %run ./scripts/seed_reference_code
    seed_reference_code.seed(spark)

or from the repo root:  python scripts/seed_reference_code.py
"""
from __future__ import annotations

import os


# Driver appended to each reference script so exec()-ing it actually RUNS the
# pipeline stage and writes the summary files execution_node/report expect.
_DRIVERS = {
    "bronze_code": (
        "\n\n# --- pipeline driver (appended by seed_reference_code) ---\n"
        "ingest_all()\n"
    ),
    "silver_code": (
        "\n\n# --- pipeline driver (appended by seed_reference_code) ---\n"
        "import json as _json\n"
        "_summary = transform_all()\n"
        "with open('/tmp/silver_summary.json', 'w') as _f:\n"
        "    _json.dump(_summary, _f, default=str)\n"
        "if _summary.get('halted_at'):\n"
        "    raise RuntimeError(\n"
        "        f\"Promotion blocked at {_summary['halted_at']} — see /tmp/silver_summary.json\"\n"
        "    )\n"
    ),
    "gold_code": (
        "\n\n# --- pipeline driver (appended by seed_reference_code) ---\n"
        "import json as _json\n"
        "_results = build_all()\n"
        "with open('/tmp/gold_summary.json', 'w') as _f:\n"
        "    _json.dump(_results, _f, default=str)\n"
    ),
}

_REF_FILES = {
    "bronze_code": "bronze.py",
    "silver_code": "silver.py",
    "gold_code": "gold.py",
}


def _reference_dir() -> str:
    """Locate reference_implementation/src/data_platform relative to the installed
    package, so this works whether run from the repo or the deployed app."""
    import dbricks_lang_agent
    # dbricks_lang_agent/__init__.py -> src/dbricks_lang_agent -> src -> repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(dbricks_lang_agent.__file__))))
    return os.path.join(repo_root, "reference_implementation", "src", "data_platform")


def build_scripts() -> dict:
    """Read the reference scripts and append the pipeline drivers. Returns
    {'bronze_code':..., 'silver_code':..., 'gold_code':...}."""
    ref_dir = _reference_dir()
    out = {}
    for key, fname in _REF_FILES.items():
        path = os.path.join(ref_dir, fname)
        with open(path, "r") as f:
            out[key] = f.read() + _DRIVERS[key]
    return out


def seed(spark, save_copies_to: str = None) -> None:
    """Assemble the reference code and store it in the UC codebase cache keyed by
    the CURRENT schema fingerprint, then log an approved Engineering review so
    the orchestrator reuses and auto-advances it on subsequent runs.

    save_copies_to: optional directory to also drop the assembled scripts for
    review (e.g. "/tmp/seeded")."""
    from dbricks_lang_agent.orchestrator import memory
    from dbricks_lang_agent.orchestrator.agents import get_schema_fingerprint

    fingerprint = get_schema_fingerprint(spark)
    scripts = build_scripts()

    memory.init_codebase_memory_table(spark)
    for key, code in scripts.items():
        memory.log_script_code(spark, fingerprint, key, code)
        print(f"[seed] wrote {key} to codebase cache (fingerprint {fingerprint[:12]}…)")

    # Log an approved Engineering review so was_previously_approved() lets the
    # cache-hit auto-advance past the review gate on future runs.
    try:
        memory.init_stage_review_table(spark)
        memory.log_stage_review(
            spark,
            pipeline_run_id="seed_reference_code",
            stage_key="engineering",
            agent_name="ManualSeed",
            decision="approved",
            reviewer_comments="Seeded known-good reference bronze/silver/gold code for this dataset.",
            output=scripts,
            dataset_fingerprint=fingerprint,
        )
        print("[seed] logged approved Engineering review — the stage will auto-advance on cache hit.")
    except Exception as e:
        print(f"[seed] WARNING: could not log approval ({e}); the Engineering gate may pause once for approval.")

    if save_copies_to:
        os.makedirs(save_copies_to, exist_ok=True)
        for key, fname in _REF_FILES.items():
            with open(os.path.join(save_copies_to, fname), "w") as f:
                f.write(scripts[key])
        print(f"[seed] wrote assembled copies to {save_copies_to}")

    print(
        "\n[seed] Done. Every subsequent run reuses this code (cache hit) and can still "
        "patch on top of it via self-heal. Re-run this seed if the source SCHEMA changes."
    )


if __name__ == "__main__":
    try:
        from dbricks_lang_agent.data_platform.spark_utils import get_spark
        _spark = get_spark()
    except Exception as e:
        raise SystemExit(f"Could not get a Spark session: {e}")
    seed(_spark, save_copies_to="/tmp/seeded")
