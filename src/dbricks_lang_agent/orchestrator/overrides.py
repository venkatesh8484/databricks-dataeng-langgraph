"""
overrides.py
============
Manually correct any agent's cached output and persist the correction to Unity
Catalog, so subsequent pipeline runs REUSE your edited version instead of
regenerating it — until you change it again (or the source SCHEMA changes).

How the persistence works
-------------------------
Every stage caches its output in a UC Delta table keyed by the *schema
fingerprint* (a structural hash of table + column names only — NOT the content
and NOT the row data). So:

  * Editing the CONTENT of an artifact while the source schema is unchanged
    keeps the same fingerprint → the next run gets a cache hit and reuses your
    edit. This is exactly the "stick until I change it again" behaviour you want.
  * If a column is later added/removed/renamed, the fingerprint changes, the
    cache misses, and the stage regenerates fresh (your edit no longer applies —
    by design, because the schema it was written against no longer exists).

Each setter here also logs an "approved" row for that stage so the review gate
auto-advances on the next run instead of pausing to re-ask you to approve
content you just hand-edited.

Typical Databricks notebook usage
---------------------------------
    from dbricks_lang_agent.orchestrator import overrides

    # See what's cached for the current source schema
    overrides.show(spark)

    # Fetch a contract, edit it, write it back
    y = overrides.get_contract(spark, "accommodations")
    print(y)
    y = y.replace("max_fail_rate: 0.0", "max_fail_rate: 0.05")
    overrides.set_contract(spark, "accommodations", y)

    # Correct a generated script
    code = overrides.get_code(spark, "silver_code")
    code = code.replace("bad_col", "good_col")
    overrides.set_code(spark, "silver_code", code)

    # Correct the Gold DDL and/or data dictionary
    overrides.set_ddl(spark, gold_ddl=my_fixed_ddl)

Then re-run the pipeline: the edited stage(s) come straight from cache, already
approved. No full reset needed.
"""
from __future__ import annotations

from typing import Dict, Optional

from dbricks_lang_agent.orchestrator import memory

# stage_key used by the review gates / was_previously_approved, per artifact.
_STAGE_FOR_ARTIFACT = {
    "dq": "dq",
    "contracts": "contracts",
    "modeling": "modeling",
    "code": "engineering",
}

_CODE_KEYS = ("bronze_code", "silver_code", "gold_code")


def current_fingerprint(spark) -> str:
    """The schema fingerprint every cache is keyed on for the CURRENT source."""
    from dbricks_lang_agent.orchestrator.agents import get_schema_fingerprint
    return get_schema_fingerprint(spark)


def _reaffirm_approval(spark, stage_key: str, output: dict) -> None:
    """Log an 'approved' review row for (stage_key, current fingerprint) so the
    next run auto-advances past this stage's gate instead of re-prompting."""
    try:
        memory.init_stage_review_table(spark)
        memory.log_stage_review(
            spark,
            pipeline_run_id="manual_override",
            stage_key=stage_key,
            agent_name="ManualOverride",
            decision="approved",
            reviewer_comments="Manual override written directly to the Unity Catalog cache.",
            output=output or {},
            dataset_fingerprint=current_fingerprint(spark),
        )
    except Exception as e:
        print(f"[Override] WARNING: wrote the edit but could not re-log approval for '{stage_key}': {e}. "
              f"The stage may pause for approval on the next run.")


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------

def show(spark) -> None:
    """Print what is currently cached for the active source schema."""
    fp = current_fingerprint(spark)
    print(f"Schema fingerprint: {fp}\n")

    contracts = memory.get_contracts_cache(spark, fp) or {}
    print(f"Contracts cached: {sorted(contracts.keys()) or '(none)'}")

    model = memory.get_modeling_cache(spark, fp) or {}
    print(f"Gold DDL cached: {'yes' if model.get('gold_ddl') else 'no'}; "
          f"data dictionary cached: {'yes' if model.get('data_dictionary') else 'no'}")

    dq = memory.get_dq_cache(spark, fp)
    print(f"DQ report cached: {'yes' if dq else 'no'}")

    code = memory.get_stored_codebase(spark, fp) or {}
    print(f"Code cached: {sorted(code.keys()) or '(none)'}")


# ---------------------------------------------------------------------------
# Contracts  (one YAML string per table)
# ---------------------------------------------------------------------------

def get_contract(spark, table: str) -> Optional[str]:
    """Return the cached YAML contract string for `table`, or None."""
    return (memory.get_contracts_cache(spark, current_fingerprint(spark)) or {}).get(table)


def set_contract(spark, table: str, yaml_str: str) -> None:
    """Persist an edited YAML contract for `table`. Other tables' contracts are
    preserved. Validates the YAML before writing."""
    import yaml as _yaml
    parsed = _yaml.safe_load(yaml_str)
    if not isinstance(parsed, dict) or "table" not in parsed:
        raise ValueError("Contract YAML must be a mapping containing a 'table' key.")
    fp = current_fingerprint(spark)
    contracts = dict(memory.get_contracts_cache(spark, fp) or {})
    contracts[table] = yaml_str
    memory.init_contracts_cache_table(spark)
    memory.upsert_contracts_cache(spark, fp, contracts)
    _reaffirm_approval(spark, "contracts", {"contracts": contracts})
    print(f"[Override] Contract for '{table}' saved to Unity Catalog (fingerprint {fp[:12]}…). "
          f"It will be reused on the next run.")


# ---------------------------------------------------------------------------
# Generated code  (bronze_code / silver_code / gold_code)
# ---------------------------------------------------------------------------

def get_code(spark, script_key: str) -> Optional[str]:
    """Return cached code for 'bronze_code' | 'silver_code' | 'gold_code'."""
    if script_key not in _CODE_KEYS:
        raise ValueError(f"script_key must be one of {_CODE_KEYS}")
    return (memory.get_stored_codebase(spark, current_fingerprint(spark)) or {}).get(script_key)


def set_code(spark, script_key: str, code: str) -> None:
    """Persist edited code for one script. NOTE: on the next run the engineering
    stage still passes cached code through the deterministic sanitizer/healer
    (import fixes, etc.), so a correct edit is preserved but may be normalized.
    If it round-trips your edit unexpectedly, tell me and I'll add a verbatim
    bypass flag."""
    if script_key not in _CODE_KEYS:
        raise ValueError(f"script_key must be one of {_CODE_KEYS}")
    fp = current_fingerprint(spark)
    memory.log_script_code(spark, fp, script_key, code)
    # Re-affirm engineering approval only if all three scripts are present, so a
    # partial edit doesn't mark the whole stage approved with a missing script.
    code_now = memory.get_stored_codebase(spark, fp) or {}
    if all(k in code_now for k in _CODE_KEYS):
        _reaffirm_approval(spark, "engineering", code_now)
    print(f"[Override] {script_key} saved to Unity Catalog (fingerprint {fp[:12]}…). "
          f"It will be reused on the next run.")


# ---------------------------------------------------------------------------
# Gold model  (DDL + data dictionary)
# ---------------------------------------------------------------------------

def get_ddl(spark) -> Dict[str, Optional[str]]:
    """Return {'gold_ddl':..., 'data_dictionary':...} from cache."""
    return memory.get_modeling_cache(spark, current_fingerprint(spark)) or {"gold_ddl": None, "data_dictionary": None}


def set_ddl(spark, gold_ddl: Optional[str] = None, data_dictionary: Optional[str] = None) -> None:
    """Persist an edited Gold DDL and/or data dictionary. Pass only what you want
    to change; the other is preserved from cache."""
    fp = current_fingerprint(spark)
    cur = memory.get_modeling_cache(spark, fp) or {}
    new_ddl = gold_ddl if gold_ddl is not None else cur.get("gold_ddl", "")
    new_dd = data_dictionary if data_dictionary is not None else cur.get("data_dictionary", "")
    memory.init_modeling_cache_table(spark)
    memory.upsert_modeling_cache(spark, fp, new_ddl, new_dd)
    _reaffirm_approval(spark, "modeling", {"gold_ddl": new_ddl, "data_dictionary": new_dd})
    print(f"[Override] Gold model saved to Unity Catalog (fingerprint {fp[:12]}…). "
          f"It will be reused on the next run.")


# ---------------------------------------------------------------------------
# Data Quality report
# ---------------------------------------------------------------------------

def get_dq(spark) -> Optional[str]:
    return memory.get_dq_cache(spark, current_fingerprint(spark))


def set_dq(spark, report: str) -> None:
    fp = current_fingerprint(spark)
    memory.init_dq_cache_table(spark)
    memory.upsert_dq_cache(spark, fp, report)
    _reaffirm_approval(spark, "dq", {"dq_report": report})
    print(f"[Override] DQ report saved to Unity Catalog (fingerprint {fp[:12]}…). "
          f"It will be reused on the next run.")
