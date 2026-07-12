"""
memory.py
=========
Handles persisting and retrieving human-approved resolutions to build
a few-shot learning feedback loop for the agents.
Supports both Databricks Delta Tables (Unity Catalog) and local JSON file fallbacks.
"""
from __future__ import annotations

import os
import json
import datetime
from typing import Dict, Any, List, Optional
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, TimestampType

# Local fallback path for development sandbox runs
LOCAL_MEMORY_PATH = "./generated/config/agent_fewshot_memory.json"


def get_memory_table_fqn(spark: SparkSession) -> str:
    """Read the memory table path from Spark configuration or catalog settings."""
    try:
        from dbricks_lang_agent.data_platform.spark_utils import load_config
        cfg = load_config()
        catalog = cfg.get("catalog", "databricks_langgraph")
    except Exception:
        catalog = "databricks_langgraph"
    return f"{catalog}.gold.agent_fewshot_memory"


def init_memory_table(spark: SparkSession) -> bool:
    """Initialize the gold.agent_fewshot_memory Delta Table in Unity Catalog."""
    fqn = get_memory_table_fqn(spark)
    print(f"Initializing few-shot memory table at: {fqn}...")
    
    # Check if we are running locally/mock or on Databricks
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False
        
    if is_local:
        # For local test environments, ensure directory exists
        os.makedirs(os.path.dirname(LOCAL_MEMORY_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_MEMORY_PATH):
            with open(LOCAL_MEMORY_PATH, "w") as f:
                json.dump([], f)
        return True

    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fqn.split('.')[0]}.gold")
        
        # Create Delta table
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                dataset_name STRING,
                issue_type STRING,
                anomalous_fields ARRAY<STRING>,
                resolution_applied STRING,
                human_comments STRING,
                decision STRING,
                timestamp TIMESTAMP
            ) USING DELTA
        """)
        return True
    except Exception as e:
        print(f"[Warning] Failed to initialize UC memory table: {e}. Falling back to local file.")
        os.makedirs(os.path.dirname(LOCAL_MEMORY_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_MEMORY_PATH):
            with open(LOCAL_MEMORY_PATH, "w") as f:
                json.dump([], f)
        return False


def log_approval(
    spark: SparkSession,
    dataset_name: str,
    issue_type: str,
    anomalous_fields: List[str],
    resolution_applied: str,
    human_comments: str
) -> None:
    """Log human approval resolution into memory."""
    timestamp = datetime.datetime.now()
    
    # 1. Try local JSON logger (always update local cache for quick local debugs)
    try:
        os.makedirs(os.path.dirname(LOCAL_MEMORY_PATH), exist_ok=True)
        records = []
        if os.path.exists(LOCAL_MEMORY_PATH):
            with open(LOCAL_MEMORY_PATH, "r") as f:
                records = json.load(f)
        
        records.append({
            "dataset_name": dataset_name,
            "issue_type": issue_type,
            "anomalous_fields": anomalous_fields,
            "resolution_applied": resolution_applied,
            "human_comments": human_comments,
            "record_type": "approval",
            "decision": "approved",
            "timestamp": timestamp.isoformat()
        })

        with open(LOCAL_MEMORY_PATH, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to log approval to local JSON cache: {e}")

    # 2. Try Spark/Delta table log
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if is_local:
        return

    try:
        fqn = get_memory_table_fqn(spark)
        # Create a tiny temp DF and append it
        schema = StructType([
            StructField("dataset_name", StringType(), True),
            StructField("issue_type", StringType(), True),
            StructField("anomalous_fields", ArrayType(StringType()), True),
            StructField("resolution_applied", StringType(), True),
            StructField("human_comments", StringType(), True),
            StructField("decision", StringType(), True),
            StructField("timestamp", TimestampType(), True)
        ])

        row_data = [(dataset_name, issue_type, anomalous_fields, resolution_applied, human_comments, "approved", timestamp)]
        df = spark.createDataFrame(row_data, schema=schema)
        # mergeSchema handles UC tables created before the 'decision' column existed
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(fqn)
        print(f"Successfully logged approval to Delta table '{fqn}'")
    except Exception as e:
        print(f"[Warning] Failed to log approval to Delta table: {e}")


def get_few_shot_context(spark: SparkSession, dataset_name: str, issue_type: str) -> str:
    """Retrieve matching historical resolutions to format as few-shot agent contexts."""
    records: List[Dict[str, Any]] = []
    
    # Try reading from Delta table first
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False
        
    read_success = False
    
    if not is_local:
        try:
            fqn = get_memory_table_fqn(spark)
            # Use PySpark column API — NOT string interpolation — to prevent SQL injection
            df = spark.read.table(fqn).filter(
                (F.col("issue_type") == issue_type) | (F.col("dataset_name") == dataset_name)
            ).orderBy(F.col("timestamp").desc()).limit(5)
            
            rows = df.collect()
            for r in rows:
                r_dict = r.asDict()
                decision = r_dict.get("decision") or "approved"
                records.append({
                    "dataset_name": r.dataset_name,
                    "issue_type": r.issue_type,
                    "anomalous_fields": r.anomalous_fields,
                    "resolution_applied": r.resolution_applied,
                    "human_comments": r.human_comments,
                    "record_type": "approval" if decision == "approved" else "rejection",
                    "decision": decision,
                })
            read_success = True
        except Exception as e:
            print(f"[Warning] Failed to read from UC memory table: {e}. Checking local cache.")
            
    # Fallback to local JSON cache
    if not read_success:
        try:
            if os.path.exists(LOCAL_MEMORY_PATH):
                with open(LOCAL_MEMORY_PATH, "r") as f:
                    all_records = json.load(f)
                
                # Filter matching records
                matched = [
                    r for r in all_records 
                    if r["issue_type"] == issue_type or r["dataset_name"] == dataset_name
                ]
                # Sort descending by timestamp index and take last 5
                matched = matched[-5:]
                records = matched
        except Exception as e:
            print(f"[Warning] Failed to read from local memory cache: {e}")
            
    if not records:
        return "No historical human approvals are registered for this issue category yet."

    # Format as Markdown few-shot context
    markdown = "### Historical Human Approvals for Reference:\n"
    for r in records:
        fields_str = ", ".join(r.get("anomalous_fields") or [])
        record_type = r.get("record_type", "approval")
        type_label = "👍 Approved" if record_type == "approval" else "🚫 Rejected"
        markdown += (
            f"- **Dataset**: {r['dataset_name']} [{type_label}]\n"
            f"  - **Issue Type**: {r['issue_type']}\n"
            f"  - **Affected Fields**: [{fields_str}]\n"
            f"  - **Resolution / Outcome**: {r['resolution_applied']}\n"
            f"  - **Human Feedback**: \"{r['human_comments'] or 'None'}\"\n"
        )
    return markdown


def log_rejection(
    spark: SparkSession,
    dataset_name: str,
    issue_type: str,
    human_comments: str,
    step_key: str,
) -> None:
    """Log human rejection feedback into memory so agents can learn from mistakes.

    Unlike log_approval, rejections store what went wrong so future agent runs
    can avoid repeating the same errors.
    """
    resolution = f"Rejected at step '{step_key}' — agent output was insufficient or incorrect."
    timestamp = datetime.datetime.now()

    record = {
        "dataset_name": dataset_name,
        "issue_type": issue_type,
        "anomalous_fields": [],
        "resolution_applied": resolution,
        "human_comments": human_comments,
        "record_type": "rejection",
        "decision": "rejected",
        "timestamp": timestamp.isoformat()
    }

    # 1. Always update local JSON cache
    try:
        os.makedirs(os.path.dirname(LOCAL_MEMORY_PATH), exist_ok=True)
        records = []
        if os.path.exists(LOCAL_MEMORY_PATH):
            with open(LOCAL_MEMORY_PATH, "r") as f:
                records = json.load(f)
        records.append(record)
        with open(LOCAL_MEMORY_PATH, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to log rejection to local JSON cache: {e}")

    # 2. Try Delta table
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if is_local:
        return

    try:
        fqn = get_memory_table_fqn(spark)
        schema = StructType([
            StructField("dataset_name", StringType(), True),
            StructField("issue_type", StringType(), True),
            StructField("anomalous_fields", ArrayType(StringType()), True),
            StructField("resolution_applied", StringType(), True),
            StructField("human_comments", StringType(), True),
            StructField("decision", StringType(), True),
            StructField("timestamp", TimestampType(), True)
        ])
        row_data = [(dataset_name, issue_type, [], resolution, human_comments, "rejected", timestamp)]
        df = spark.createDataFrame(row_data, schema=schema)
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(fqn)
        print(f"Successfully logged rejection to Delta table '{fqn}'")
    except Exception as e:
        print(f"[Warning] Failed to log rejection to Delta table: {e}")


LOCAL_CODEBASE_MEMORY_PATH = "./generated/config/agent_script_codebase_memory.json"

# NOTE: this cache is keyed per (dataset_fingerprint, script_key) — one row
# per SCRIPT, not one row per fingerprint bundling all three. Earlier this
# was all-or-nothing: if silver failed to compile, bronze and gold (even if
# they compiled perfectly) never got persisted, so every retry regenerated
# all three scripts from scratch. Now each script is cached the moment IT
# individually compiles clean, independent of its siblings' status, and
# get_stored_codebase() below returns whatever subset is actually cached —
# possibly 0, 1, 2, or 3 scripts. This intentionally lives in a differently
# named table (agent_script_codebase_memory) rather than reusing the old
# agent_codebase_memory name, since the old table (if it exists in a
# workspace already) has an incompatible schema (one row per fingerprint
# with bronze_code/silver_code/gold_code columns).

def get_codebase_table_fqn(spark: SparkSession) -> str:
    """Read the codebase memory table path from Spark configuration or catalog settings."""
    try:
        from dbricks_lang_agent.data_platform.spark_utils import load_config
        cfg = load_config()
        catalog = cfg.get("catalog", "databricks_langgraph")
    except Exception:
        catalog = "databricks_langgraph"
    return f"{catalog}.gold.agent_script_codebase_memory"


def init_codebase_memory_table(spark: SparkSession) -> bool:
    """Initialize the gold.agent_script_codebase_memory Delta Table in Unity Catalog."""
    fqn = get_codebase_table_fqn(spark)
    print(f"Initializing script codebase memory table at: {fqn}...")

    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if is_local:
        os.makedirs(os.path.dirname(LOCAL_CODEBASE_MEMORY_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_CODEBASE_MEMORY_PATH):
            with open(LOCAL_CODEBASE_MEMORY_PATH, "w") as f:
                json.dump({}, f)
        return True

    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fqn.split('.')[0]}.gold")

        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                dataset_fingerprint STRING,
                script_key STRING,
                code STRING,
                timestamp TIMESTAMP
            ) USING DELTA
        """)
        return True
    except Exception as e:
        print(f"[Warning] Failed to initialize UC script codebase memory table: {e}. Falling back to local file.")
        os.makedirs(os.path.dirname(LOCAL_CODEBASE_MEMORY_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_CODEBASE_MEMORY_PATH):
            with open(LOCAL_CODEBASE_MEMORY_PATH, "w") as f:
                json.dump({}, f)
        return False


def get_stored_codebase(spark: SparkSession, fingerprint: str) -> Dict[str, str]:
    """Retrieve whichever scripts are cached (individually) for this dataset
    fingerprint. Returns a dict with only the keys that have a known-good
    cached entry — e.g. {'bronze_code': '...'} if only bronze has ever
    compiled clean for this fingerprint. Returns {} if nothing is cached."""
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if not is_local:
        try:
            fqn = get_codebase_table_fqn(spark)
            df = spark.read.table(fqn).filter(F.col("dataset_fingerprint") == fingerprint)
            rows = df.collect()
            if rows:
                # If duplicate rows exist for the same script_key (shouldn't,
                # since we merge/upsert on write), keep the newest.
                out: Dict[str, str] = {}
                latest_ts: Dict[str, Any] = {}
                for r in rows:
                    k = r.script_key
                    if k not in latest_ts or (r.timestamp and r.timestamp > latest_ts[k]):
                        out[k] = r.code
                        latest_ts[k] = r.timestamp
                return out
        except Exception as e:
            print(f"[Warning] Failed to read from UC script codebase memory: {e}. Checking local cache.")

    # Local fallback
    try:
        if os.path.exists(LOCAL_CODEBASE_MEMORY_PATH):
            with open(LOCAL_CODEBASE_MEMORY_PATH, "r") as f:
                data = json.load(f)
            entry = data.get(fingerprint, {})
            return {k: v["code"] for k, v in entry.items() if isinstance(v, dict) and "code" in v}
    except Exception as e:
        print(f"[Warning] Failed to read from local script codebase memory: {e}")

    return {}


def log_script_code(spark: SparkSession, fingerprint: str, script_key: str, code: str) -> None:
    """Persist ONE script's code the moment it individually compiles clean —
    called from inside the compiler loop right after a script passes
    verification, independent of whether its siblings pass. This is what
    lets a proven-good bronze.py stay cached even while silver.py is still
    being iterated on."""
    timestamp = datetime.datetime.now()

    # 1. Update local cache
    try:
        os.makedirs(os.path.dirname(LOCAL_CODEBASE_MEMORY_PATH), exist_ok=True)
        data = {}
        if os.path.exists(LOCAL_CODEBASE_MEMORY_PATH):
            with open(LOCAL_CODEBASE_MEMORY_PATH, "r") as f:
                data = json.load(f)

        data.setdefault(fingerprint, {})
        data[fingerprint][script_key] = {
            "code": code,
            "timestamp": timestamp.isoformat()
        }

        with open(LOCAL_CODEBASE_MEMORY_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to log script code to local JSON cache: {e}")

    # 2. Update Delta table in Unity Catalog
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if is_local:
        return

    try:
        fqn = get_codebase_table_fqn(spark)
        schema = StructType([
            StructField("dataset_fingerprint", StringType(), True),
            StructField("script_key", StringType(), True),
            StructField("code", StringType(), True),
            StructField("timestamp", TimestampType(), True)
        ])

        row_data = [(fingerprint, script_key, code, timestamp)]
        df = spark.createDataFrame(row_data, schema=schema)

        # Merge/Upsert on (fingerprint, script_key) so only this ONE script's
        # row is replaced — siblings cached under the same fingerprint are
        # untouched.
        if spark.catalog.tableExists(fqn):
            from delta.tables import DeltaTable
            target = DeltaTable.forName(spark, fqn)
            target.alias("t").merge(
                df.alias("s"),
                "t.dataset_fingerprint = s.dataset_fingerprint AND t.script_key = s.script_key"
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        else:
            df.write.format("delta").mode("overwrite").saveAsTable(fqn)

        print(f"Successfully logged {script_key} to Delta table '{fqn}'")
    except Exception as e:
        print(f"[Warning] Failed to log script code to Delta table: {e}")


# ---------------------------------------------------------------------------
# Compile Audit Log — append-only record of EVERY compile/self-heal attempt
# made while producing bronze/silver/gold scripts: whether the code fed into
# that attempt came from the Unity Catalog cache untouched, a fresh LLM
# generation, a targeted cache-patch (only the previously-failing script
# regenerated), or an in-loop compiler self-heal retry — plus the resulting
# exit_code/stdout/stderr and a hash of the exact code that ran. This exists
# so "is the agent actually reusing code or generating something new every
# time?" is answerable by reading a table instead of inferring it from stack
# trace line numbers. Always append — never overwrite — one row per attempt.
# ---------------------------------------------------------------------------

LOCAL_COMPILE_AUDIT_PATH = "./generated/config/agent_compile_audit.json"


def get_compile_audit_table_fqn(spark: SparkSession) -> str:
    """Read the compile audit table path from Spark configuration or catalog settings."""
    try:
        from dbricks_lang_agent.data_platform.spark_utils import load_config
        cfg = load_config()
        catalog = cfg.get("catalog", "databricks_langgraph")
    except Exception:
        catalog = "databricks_langgraph"
    return f"{catalog}.gold.agent_compile_audit"


def init_compile_audit_table(spark: SparkSession) -> bool:
    """Initialize the gold.agent_compile_audit Delta Table in Unity Catalog."""
    fqn = get_compile_audit_table_fqn(spark)
    print(f"Initializing compile audit table at: {fqn}...")

    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if is_local:
        os.makedirs(os.path.dirname(LOCAL_COMPILE_AUDIT_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_COMPILE_AUDIT_PATH):
            with open(LOCAL_COMPILE_AUDIT_PATH, "w") as f:
                json.dump([], f)
        return True

    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fqn.split('.')[0]}.gold")

        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                run_id STRING,
                dataset_fingerprint STRING,
                script_name STRING,
                attempt_number INT,
                code_source STRING,
                code_sha256 STRING,
                code_snapshot STRING,
                exit_code INT,
                stdout STRING,
                stderr STRING,
                timestamp TIMESTAMP
            ) USING DELTA
        """)
        return True
    except Exception as e:
        print(f"[Warning] Failed to initialize UC compile audit table: {e}. Falling back to local file.")
        os.makedirs(os.path.dirname(LOCAL_COMPILE_AUDIT_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_COMPILE_AUDIT_PATH):
            with open(LOCAL_COMPILE_AUDIT_PATH, "w") as f:
                json.dump([], f)
        return False


def log_compile_attempt(
    spark: SparkSession,
    run_id: str,
    dataset_fingerprint: str,
    script_name: str,
    attempt_number: int,
    code_source: str,
    code: str,
    exit_code: int,
    stdout: str,
    stderr: str,
) -> None:
    """Append one row per compile/execution attempt.

    code_source should be one of: 'cache_recalled', 'llm_fresh_generated',
    'llm_targeted_patch', 'llm_self_heal_fix', 'execution_node_real_run' —
    callers pick the label that describes how THIS attempt's code came to be.
    Failures here are swallowed (print-only) so audit logging never breaks
    the pipeline it's observing.
    """
    import hashlib
    timestamp = datetime.datetime.now()
    code_sha256 = hashlib.sha256((code or "").encode("utf-8")).hexdigest()
    code_snapshot = (code or "")[:20000]
    stdout_trunc = (stdout or "")[:5000]
    stderr_trunc = (stderr or "")[:5000]

    # 1. Always update local JSON cache
    try:
        os.makedirs(os.path.dirname(LOCAL_COMPILE_AUDIT_PATH), exist_ok=True)
        records = []
        if os.path.exists(LOCAL_COMPILE_AUDIT_PATH):
            with open(LOCAL_COMPILE_AUDIT_PATH, "r") as f:
                records = json.load(f)
        records.append({
            "run_id": run_id,
            "dataset_fingerprint": dataset_fingerprint,
            "script_name": script_name,
            "attempt_number": attempt_number,
            "code_source": code_source,
            "code_sha256": code_sha256,
            "code_snapshot": code_snapshot,
            "exit_code": exit_code,
            "stdout": stdout_trunc,
            "stderr": stderr_trunc,
            "timestamp": timestamp.isoformat(),
        })
        with open(LOCAL_COMPILE_AUDIT_PATH, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to log compile attempt to local JSON cache: {e}")

    # 2. Try Delta table
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if is_local:
        return

    try:
        fqn = get_compile_audit_table_fqn(spark)
        schema = StructType([
            StructField("run_id", StringType(), True),
            StructField("dataset_fingerprint", StringType(), True),
            StructField("script_name", StringType(), True),
            StructField("attempt_number", StringType(), True),
            StructField("code_source", StringType(), True),
            StructField("code_sha256", StringType(), True),
            StructField("code_snapshot", StringType(), True),
            StructField("exit_code", StringType(), True),
            StructField("stdout", StringType(), True),
            StructField("stderr", StringType(), True),
            StructField("timestamp", TimestampType(), True),
        ])
        row_data = [(
            run_id, dataset_fingerprint, script_name, str(attempt_number), code_source,
            code_sha256, code_snapshot, str(exit_code), stdout_trunc, stderr_trunc, timestamp,
        )]
        df = (
            spark.createDataFrame(row_data, schema=schema)
            .withColumn("attempt_number", F.col("attempt_number").cast("int"))
            .withColumn("exit_code", F.col("exit_code").cast("int"))
        )
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(fqn)
        print(f"Successfully logged compile attempt ({script_name}, {code_source}) to Delta table '{fqn}'")
    except Exception as e:
        print(f"[Warning] Failed to log compile attempt to Delta table: {e}")


def get_compile_audit(
    spark: SparkSession,
    fingerprint: str = None,
    run_id: str = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Retrieve compile audit rows, newest first, optionally filtered by
    dataset fingerprint and/or run_id. Used to audit whether a given run
    actually reused cached code or regenerated it, attempt by attempt."""
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if not is_local:
        try:
            fqn = get_compile_audit_table_fqn(spark)
            df = spark.read.table(fqn)
            if fingerprint:
                df = df.filter(F.col("dataset_fingerprint") == fingerprint)
            if run_id:
                df = df.filter(F.col("run_id") == run_id)
            df = df.orderBy(F.col("timestamp").desc()).limit(limit)
            return [r.asDict() for r in df.collect()]
        except Exception as e:
            print(f"[Warning] Failed to read from UC compile audit table: {e}. Checking local cache.")

    try:
        if os.path.exists(LOCAL_COMPILE_AUDIT_PATH):
            with open(LOCAL_COMPILE_AUDIT_PATH, "r") as f:
                records = json.load(f)
            if fingerprint:
                records = [r for r in records if r.get("dataset_fingerprint") == fingerprint]
            if run_id:
                records = [r for r in records if r.get("run_id") == run_id]
            records = sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)
            return records[:limit]
    except Exception as e:
        print(f"[Warning] Failed to read from local compile audit cache: {e}")

    return []


# ---------------------------------------------------------------------------
# Run History — append-only audit log of every pipeline execution attempt.
# One row per execution_node run (success or failure), date-stamped, so past
# runs can be reviewed even after the live checkpoint thread moves on or is
# reset. This is distinct from agent_codebase_memory (which only keeps the
# LATEST code per dataset fingerprint) and from agent_fewshot_memory (which
# logs individual human approve/reject decisions, not full run outcomes).
# ---------------------------------------------------------------------------

LOCAL_RUN_HISTORY_PATH = "./generated/config/agent_run_history.json"


def get_run_history_table_fqn(spark: SparkSession) -> str:
    """Read the run history table path from Spark configuration or catalog settings."""
    try:
        from dbricks_lang_agent.data_platform.spark_utils import load_config
        cfg = load_config()
        catalog = cfg.get("catalog", "databricks_langgraph")
    except Exception:
        catalog = "databricks_langgraph"
    return f"{catalog}.gold.agent_run_history"


def init_run_history_table(spark: SparkSession) -> bool:
    """Initialize the gold.agent_run_history Delta Table in Unity Catalog."""
    fqn = get_run_history_table_fqn(spark)
    print(f"Initializing run history table at: {fqn}...")

    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if is_local:
        os.makedirs(os.path.dirname(LOCAL_RUN_HISTORY_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_RUN_HISTORY_PATH):
            with open(LOCAL_RUN_HISTORY_PATH, "w") as f:
                json.dump([], f)
        return True

    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fqn.split('.')[0]}.gold")

        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                run_id STRING,
                run_date DATE,
                run_timestamp TIMESTAMP,
                pipeline_status STRING,
                active_agent STRING,
                dataset_fingerprint STRING,
                failed_scripts ARRAY<STRING>,
                execution_logs STRING,
                silver_summary STRING,
                gold_summary STRING,
                final_report STRING,
                approved_steps STRING,
                review_comments STRING
            ) USING DELTA
        """)
        return True
    except Exception as e:
        print(f"[Warning] Failed to initialize UC run history table: {e}. Falling back to local file.")
        os.makedirs(os.path.dirname(LOCAL_RUN_HISTORY_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_RUN_HISTORY_PATH):
            with open(LOCAL_RUN_HISTORY_PATH, "w") as f:
                json.dump([], f)
        return False


def log_run(
    spark: SparkSession,
    run_id: str,
    pipeline_status: str,
    active_agent: str,
    dataset_fingerprint: str,
    failed_scripts: List[str],
    execution_logs: Dict[str, Any],
    silver_summary: Dict[str, Any],
    gold_summary: Dict[str, Any],
    final_report: str,
    approved_steps: Dict[str, Any],
    review_comments: str,
) -> None:
    """Append one row to the run history audit log. Always append — never
    overwrite/merge — so every attempt (including failed ones) stays visible
    for auditing, unlike agent_codebase_memory which only keeps the latest."""
    timestamp = datetime.datetime.now()
    run_date = timestamp.date()

    execution_logs_json = json.dumps(execution_logs or {})
    silver_summary_json = json.dumps(silver_summary or {})
    gold_summary_json = json.dumps(gold_summary or {})
    approved_steps_json = json.dumps(approved_steps or {})

    # 1. Always update local JSON cache
    try:
        os.makedirs(os.path.dirname(LOCAL_RUN_HISTORY_PATH), exist_ok=True)
        records = []
        if os.path.exists(LOCAL_RUN_HISTORY_PATH):
            with open(LOCAL_RUN_HISTORY_PATH, "r") as f:
                records = json.load(f)

        records.append({
            "run_id": run_id,
            "run_date": run_date.isoformat(),
            "run_timestamp": timestamp.isoformat(),
            "pipeline_status": pipeline_status,
            "active_agent": active_agent,
            "dataset_fingerprint": dataset_fingerprint,
            "failed_scripts": failed_scripts or [],
            "execution_logs": execution_logs_json,
            "silver_summary": silver_summary_json,
            "gold_summary": gold_summary_json,
            "final_report": final_report,
            "approved_steps": approved_steps_json,
            "review_comments": review_comments,
        })

        with open(LOCAL_RUN_HISTORY_PATH, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to log run history to local JSON cache: {e}")

    # 2. Try Spark/Delta table log
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if is_local:
        return

    try:
        fqn = get_run_history_table_fqn(spark)
        schema = StructType([
            StructField("run_id", StringType(), True),
            StructField("run_date", StringType(), True),  # cast to date below
            StructField("run_timestamp", TimestampType(), True),
            StructField("pipeline_status", StringType(), True),
            StructField("active_agent", StringType(), True),
            StructField("dataset_fingerprint", StringType(), True),
            StructField("failed_scripts", ArrayType(StringType()), True),
            StructField("execution_logs", StringType(), True),
            StructField("silver_summary", StringType(), True),
            StructField("gold_summary", StringType(), True),
            StructField("final_report", StringType(), True),
            StructField("approved_steps", StringType(), True),
            StructField("review_comments", StringType(), True),
        ])

        row_data = [(
            run_id, run_date.isoformat(), timestamp, pipeline_status, active_agent,
            dataset_fingerprint, failed_scripts or [], execution_logs_json,
            silver_summary_json, gold_summary_json, final_report,
            approved_steps_json, review_comments,
        )]
        df = spark.createDataFrame(row_data, schema=schema).withColumn("run_date", F.col("run_date").cast("date"))
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(fqn)
        print(f"Successfully logged run to Delta table '{fqn}'")
    except Exception as e:
        print(f"[Warning] Failed to log run history to Delta table: {e}")


def get_run_history(spark: SparkSession, limit: int = 200) -> List[Dict[str, Any]]:
    """Retrieve recent run history rows, newest first. Used by the dashboard's
    Run History tab. Returns plain dicts (JSON fields left as strings for the
    caller to parse/display as needed)."""
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if not is_local:
        try:
            fqn = get_run_history_table_fqn(spark)
            df = spark.read.table(fqn).orderBy(F.col("run_timestamp").desc()).limit(limit)
            return [r.asDict() for r in df.collect()]
        except Exception as e:
            print(f"[Warning] Failed to read from UC run history table: {e}. Checking local cache.")

    try:
        if os.path.exists(LOCAL_RUN_HISTORY_PATH):
            with open(LOCAL_RUN_HISTORY_PATH, "r") as f:
                records = json.load(f)
            records = sorted(records, key=lambda r: r.get("run_timestamp", ""), reverse=True)
            return records[:limit]
    except Exception as e:
        print(f"[Warning] Failed to read from local run history cache: {e}")

    return []


# ---------------------------------------------------------------------------
# Stage Review Log — append-only audit record of every human approve/reject
# decision made at a review gate, PLUS a full snapshot of the agent output
# that was on screen at the moment of that decision (profiling report, DQ
# report, contracts, DDL/data dictionary, generated code, execution logs /
# final report). This is what powers the dashboard's "Agent Outputs &
# Reviews" audit view: filter by date, open any past run, and see exactly
# what each agent produced and what the reviewer decided/commented — even
# long after the live checkpoint thread has moved on or been reset.
#
# Distinct from agent_fewshot_memory (short resolution text used as
# few-shot context for the agents) and agent_run_history (one row per
# execution_node/Orchestrator attempt only). This table has one row per
# review decision at ANY of the six gates, tagged with pipeline_run_id so
# all stages of the same end-to-end run can be grouped together.
# ---------------------------------------------------------------------------

LOCAL_STAGE_REVIEW_PATH = "./generated/config/agent_stage_review_log.json"


def get_stage_review_table_fqn(spark: SparkSession) -> str:
    """Read the stage review log table path from Spark configuration or catalog settings."""
    try:
        from dbricks_lang_agent.data_platform.spark_utils import load_config
        cfg = load_config()
        catalog = cfg.get("catalog", "databricks_langgraph")
    except Exception:
        catalog = "databricks_langgraph"
    return f"{catalog}.gold.agent_stage_review_log"


def init_stage_review_table(spark: SparkSession) -> bool:
    """Initialize the gold.agent_stage_review_log Delta Table in Unity Catalog."""
    fqn = get_stage_review_table_fqn(spark)
    print(f"Initializing stage review log table at: {fqn}...")

    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if is_local:
        os.makedirs(os.path.dirname(LOCAL_STAGE_REVIEW_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_STAGE_REVIEW_PATH):
            with open(LOCAL_STAGE_REVIEW_PATH, "w") as f:
                json.dump([], f)
        return True

    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fqn.split('.')[0]}.gold")

        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                review_id STRING,
                pipeline_run_id STRING,
                review_date DATE,
                review_timestamp TIMESTAMP,
                stage_key STRING,
                agent_name STRING,
                decision STRING,
                reviewer_comments STRING,
                dataset_fingerprint STRING,
                output_json STRING
            ) USING DELTA
        """)
        return True
    except Exception as e:
        print(f"[Warning] Failed to initialize UC stage review log table: {e}. Falling back to local file.")
        os.makedirs(os.path.dirname(LOCAL_STAGE_REVIEW_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_STAGE_REVIEW_PATH):
            with open(LOCAL_STAGE_REVIEW_PATH, "w") as f:
                json.dump([], f)
        return False


def log_stage_review(
    spark: SparkSession,
    pipeline_run_id: str,
    stage_key: str,
    agent_name: str,
    decision: str,
    reviewer_comments: str,
    output: Dict[str, Any],
    dataset_fingerprint: str = "",
) -> None:
    """Append one row every time a human approves or rejects a review gate.

    `output` should be the full artifact(s) produced by that agent (e.g. the
    profiling report, DQ report, contracts, DDL, generated code, or final
    report) — whatever was visible to the reviewer at decision time. Stored
    as JSON so it can be rendered back in the audit UI. Failures here are
    swallowed (print/warn-only) so audit logging never blocks the pipeline.
    """
    import uuid
    timestamp = datetime.datetime.now()
    review_date = timestamp.date()
    review_id = str(uuid.uuid4())

    try:
        output_json = json.dumps(output or {}, default=str)
    except Exception:
        output_json = "{}"
    # Cap size defensively so a huge code blob can't blow up a single audit row.
    output_json = output_json[:200000]

    # 1. Always update local JSON cache
    try:
        os.makedirs(os.path.dirname(LOCAL_STAGE_REVIEW_PATH), exist_ok=True)
        records = []
        if os.path.exists(LOCAL_STAGE_REVIEW_PATH):
            with open(LOCAL_STAGE_REVIEW_PATH, "r") as f:
                records = json.load(f)

        records.append({
            "review_id": review_id,
            "pipeline_run_id": pipeline_run_id,
            "review_date": review_date.isoformat(),
            "review_timestamp": timestamp.isoformat(),
            "stage_key": stage_key,
            "agent_name": agent_name,
            "decision": decision,
            "reviewer_comments": reviewer_comments or "",
            "dataset_fingerprint": dataset_fingerprint or "",
            "output_json": output_json,
        })

        with open(LOCAL_STAGE_REVIEW_PATH, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to log stage review to local JSON cache: {e}")

    # 2. Try Spark/Delta table log
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if is_local:
        return

    try:
        fqn = get_stage_review_table_fqn(spark)
        schema = StructType([
            StructField("review_id", StringType(), True),
            StructField("pipeline_run_id", StringType(), True),
            StructField("review_date", StringType(), True),  # cast to date below
            StructField("review_timestamp", TimestampType(), True),
            StructField("stage_key", StringType(), True),
            StructField("agent_name", StringType(), True),
            StructField("decision", StringType(), True),
            StructField("reviewer_comments", StringType(), True),
            StructField("dataset_fingerprint", StringType(), True),
            StructField("output_json", StringType(), True),
        ])

        row_data = [(
            review_id, pipeline_run_id, review_date.isoformat(), timestamp, stage_key,
            agent_name, decision, reviewer_comments or "", dataset_fingerprint or "", output_json,
        )]
        df = spark.createDataFrame(row_data, schema=schema).withColumn("review_date", F.col("review_date").cast("date"))
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(fqn)
        print(f"Successfully logged stage review ({stage_key}, {decision}) to Delta table '{fqn}'")
    except Exception as e:
        print(f"[Warning] Failed to log stage review to Delta table: {e}")


def get_stage_reviews(
    spark: SparkSession,
    limit: int = 500,
    pipeline_run_id: str = None,
) -> List[Dict[str, Any]]:
    """Retrieve stage review log rows, newest first, optionally filtered to a
    single pipeline run. Used by the dashboard's 'Agent Outputs & Reviews'
    audit tab. output_json is left as a string for the caller to parse."""
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if not is_local:
        try:
            fqn = get_stage_review_table_fqn(spark)
            df = spark.read.table(fqn)
            if pipeline_run_id:
                df = df.filter(F.col("pipeline_run_id") == pipeline_run_id)
            df = df.orderBy(F.col("review_timestamp").desc()).limit(limit)
            return [r.asDict() for r in df.collect()]
        except Exception as e:
            print(f"[Warning] Failed to read from UC stage review log table: {e}. Checking local cache.")

    try:
        if os.path.exists(LOCAL_STAGE_REVIEW_PATH):
            with open(LOCAL_STAGE_REVIEW_PATH, "r") as f:
                records = json.load(f)
            if pipeline_run_id:
                records = [r for r in records if r.get("pipeline_run_id") == pipeline_run_id]
            records = sorted(records, key=lambda r: r.get("review_timestamp", ""), reverse=True)
            return records[:limit]
    except Exception as e:
        print(f"[Warning] Failed to read from local stage review log cache: {e}")

    return []


def was_previously_approved(
    spark: SparkSession,
    stage_key: str,
    schema_fingerprint: str,
) -> bool:
    """True if a human has already approved THIS EXACT stage output for THIS
    EXACT schema fingerprint at some point in the past (queries
    gold.agent_stage_review_log). Used to decide whether an exact cache hit
    can auto-advance past its review gate instead of pausing for a redundant
    re-approval of content a human has already signed off on.

    Deliberately conservative: only ever returns True for an exact
    (stage_key, schema_fingerprint) match with decision == 'approved'. A
    schema change produces a different fingerprint, so it can never
    accidentally match a stale approval — and content that was only ever
    generated but never actually reviewed (first run at a new fingerprint)
    correctly returns False, so the pipeline still pauses for that first
    real human decision.
    """
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if not is_local:
        try:
            fqn = get_stage_review_table_fqn(spark)
            df = (
                spark.read.table(fqn)
                .filter(
                    (F.col("stage_key") == stage_key)
                    & (F.col("dataset_fingerprint") == schema_fingerprint)
                    & (F.col("decision") == "approved")
                )
                .limit(1)
            )
            return df.count() > 0
        except Exception as e:
            print(f"[Warning] Failed to check prior approval in UC stage review log: {e}. Checking local cache.")

    try:
        if os.path.exists(LOCAL_STAGE_REVIEW_PATH):
            with open(LOCAL_STAGE_REVIEW_PATH, "r") as f:
                records = json.load(f)
            for r in records:
                if (
                    r.get("stage_key") == stage_key
                    and r.get("dataset_fingerprint") == schema_fingerprint
                    and r.get("decision") == "approved"
                ):
                    return True
    except Exception as e:
        print(f"[Warning] Failed to check prior approval in local stage review cache: {e}")

    return False


# ---------------------------------------------------------------------------
# Agent Output Cache — DQ report / Contracts / Modeling (Gold DDL + Data
# Dictionary), keyed by the STRUCTURAL schema fingerprint
# (agents.get_schema_fingerprint — table names + column headers only, no LLM
# content, no row data). This is what lets "generate once, reuse forever
# until the schema actually changes" work: as long as the source tables'
# columns haven't changed, every one of these caches hits and the
# corresponding LLM call is skipped entirely — only Profiler (which
# legitimately reflects new row-level data every run) is exempt from this
# scheme by design.
#
# Each cache keeps the FULL history (append/upsert per fingerprint, never
# delete), so "get the most recent entry regardless of fingerprint" is
# always available to drive the patch-not-regenerate flow when the schema
# DOES change: the node feeds the LLM the last known-good output plus
# what's different now, and asks it to update rather than start from a
# blank page.
# ---------------------------------------------------------------------------

LOCAL_DQ_CACHE_PATH = "./generated/config/agent_dq_cache.json"
LOCAL_CONTRACTS_CACHE_PATH = "./generated/config/agent_contracts_cache.json"
LOCAL_MODELING_CACHE_PATH = "./generated/config/agent_modeling_cache.json"


def _cache_table_fqn(spark: SparkSession, table_name: str) -> str:
    try:
        from dbricks_lang_agent.data_platform.spark_utils import load_config
        cfg = load_config()
        catalog = cfg.get("catalog", "databricks_langgraph")
    except Exception:
        catalog = "databricks_langgraph"
    return f"{catalog}.gold.{table_name}"


def _is_local_spark(spark: SparkSession) -> bool:
    try:
        return spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        return False


# ---- DQ report cache (schema_fingerprint -> dq_report) ----

def get_dq_cache_table_fqn(spark: SparkSession) -> str:
    return _cache_table_fqn(spark, "agent_dq_cache")


def init_dq_cache_table(spark: SparkSession) -> bool:
    fqn = get_dq_cache_table_fqn(spark)
    if _is_local_spark(spark):
        os.makedirs(os.path.dirname(LOCAL_DQ_CACHE_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_DQ_CACHE_PATH):
            with open(LOCAL_DQ_CACHE_PATH, "w") as f:
                json.dump({}, f)
        return True
    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fqn.split('.')[0]}.gold")
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                schema_fingerprint STRING,
                dq_report STRING,
                updated_ts TIMESTAMP
            ) USING DELTA
        """)
        return True
    except Exception as e:
        print(f"[Warning] Failed to initialize UC DQ cache table: {e}. Falling back to local file.")
        os.makedirs(os.path.dirname(LOCAL_DQ_CACHE_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_DQ_CACHE_PATH):
            with open(LOCAL_DQ_CACHE_PATH, "w") as f:
                json.dump({}, f)
        return False


def get_dq_cache(spark: SparkSession, schema_fingerprint: str) -> Optional[str]:
    """Exact-fingerprint lookup. Returns the cached dq_report, or None on a miss."""
    if not _is_local_spark(spark):
        try:
            fqn = get_dq_cache_table_fqn(spark)
            df = spark.read.table(fqn).filter(F.col("schema_fingerprint") == schema_fingerprint).orderBy(F.col("updated_ts").desc()).limit(1)
            rows = df.collect()
            if rows:
                return rows[0]["dq_report"]
            return None
        except Exception as e:
            print(f"[Warning] Failed to read UC DQ cache: {e}. Checking local cache.")
    try:
        if os.path.exists(LOCAL_DQ_CACHE_PATH):
            with open(LOCAL_DQ_CACHE_PATH, "r") as f:
                data = json.load(f)
            entry = data.get(schema_fingerprint)
            return entry.get("dq_report") if entry else None
    except Exception as e:
        print(f"[Warning] Failed to read local DQ cache: {e}")
    return None


def get_latest_dq_cache(spark: SparkSession) -> Optional[Dict[str, Any]]:
    """Most recent entry regardless of fingerprint — used to seed a targeted
    patch when the current schema fingerprint has no exact-match cache."""
    if not _is_local_spark(spark):
        try:
            fqn = get_dq_cache_table_fqn(spark)
            df = spark.read.table(fqn).orderBy(F.col("updated_ts").desc()).limit(1)
            rows = df.collect()
            if rows:
                return {"schema_fingerprint": rows[0]["schema_fingerprint"], "dq_report": rows[0]["dq_report"]}
            return None
        except Exception as e:
            print(f"[Warning] Failed to read latest UC DQ cache: {e}. Checking local cache.")
    try:
        if os.path.exists(LOCAL_DQ_CACHE_PATH):
            with open(LOCAL_DQ_CACHE_PATH, "r") as f:
                data = json.load(f)
            if not data:
                return None
            latest_fp = max(data, key=lambda k: data[k].get("updated_ts", ""))
            return {"schema_fingerprint": latest_fp, "dq_report": data[latest_fp].get("dq_report")}
    except Exception as e:
        print(f"[Warning] Failed to read latest local DQ cache: {e}")
    return None


def upsert_dq_cache(spark: SparkSession, schema_fingerprint: str, dq_report: str) -> None:
    timestamp = datetime.datetime.now()
    try:
        os.makedirs(os.path.dirname(LOCAL_DQ_CACHE_PATH), exist_ok=True)
        data = {}
        if os.path.exists(LOCAL_DQ_CACHE_PATH):
            with open(LOCAL_DQ_CACHE_PATH, "r") as f:
                data = json.load(f)
        data[schema_fingerprint] = {"dq_report": dq_report, "updated_ts": timestamp.isoformat()}
        with open(LOCAL_DQ_CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to upsert local DQ cache: {e}")

    if _is_local_spark(spark):
        return
    try:
        fqn = get_dq_cache_table_fqn(spark)
        schema = StructType([
            StructField("schema_fingerprint", StringType(), True),
            StructField("dq_report", StringType(), True),
            StructField("updated_ts", TimestampType(), True),
        ])
        df = spark.createDataFrame([(schema_fingerprint, dq_report, timestamp)], schema=schema)
        if spark.catalog.tableExists(fqn):
            from delta.tables import DeltaTable
            target = DeltaTable.forName(spark, fqn)
            target.alias("t").merge(
                df.alias("s"), "t.schema_fingerprint = s.schema_fingerprint"
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        else:
            df.write.format("delta").mode("append").saveAsTable(fqn)
        print(f"[Agent Output Cache] Upserted DQ report for fingerprint {schema_fingerprint[:12]}... to '{fqn}'")
    except Exception as e:
        print(f"[Warning] Failed to upsert UC DQ cache: {e}")


# ---- Contracts cache (schema_fingerprint + table_name -> contract_yaml) ----

def get_contracts_cache_table_fqn(spark: SparkSession) -> str:
    return _cache_table_fqn(spark, "agent_contracts_cache")


def init_contracts_cache_table(spark: SparkSession) -> bool:
    fqn = get_contracts_cache_table_fqn(spark)
    if _is_local_spark(spark):
        os.makedirs(os.path.dirname(LOCAL_CONTRACTS_CACHE_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_CONTRACTS_CACHE_PATH):
            with open(LOCAL_CONTRACTS_CACHE_PATH, "w") as f:
                json.dump({}, f)
        return True
    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fqn.split('.')[0]}.gold")
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                schema_fingerprint STRING,
                table_name STRING,
                contract_yaml STRING,
                updated_ts TIMESTAMP
            ) USING DELTA
        """)
        return True
    except Exception as e:
        print(f"[Warning] Failed to initialize UC contracts cache table: {e}. Falling back to local file.")
        os.makedirs(os.path.dirname(LOCAL_CONTRACTS_CACHE_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_CONTRACTS_CACHE_PATH):
            with open(LOCAL_CONTRACTS_CACHE_PATH, "w") as f:
                json.dump({}, f)
        return False


def get_contracts_cache(spark: SparkSession, schema_fingerprint: str) -> Dict[str, str]:
    """Exact-fingerprint lookup. Returns {table_name: contract_yaml}, {} on a miss."""
    if not _is_local_spark(spark):
        try:
            fqn = get_contracts_cache_table_fqn(spark)
            df = spark.read.table(fqn).filter(F.col("schema_fingerprint") == schema_fingerprint)
            return {r["table_name"]: r["contract_yaml"] for r in df.collect()}
        except Exception as e:
            print(f"[Warning] Failed to read UC contracts cache: {e}. Checking local cache.")
    try:
        if os.path.exists(LOCAL_CONTRACTS_CACHE_PATH):
            with open(LOCAL_CONTRACTS_CACHE_PATH, "r") as f:
                data = json.load(f)
            entry = data.get(schema_fingerprint, {})
            return {tbl: v.get("contract_yaml") for tbl, v in entry.items()}
    except Exception as e:
        print(f"[Warning] Failed to read local contracts cache: {e}")
    return {}


def get_latest_contracts_cache(spark: SparkSession) -> Optional[Dict[str, Any]]:
    """Most recent fingerprint's full contract set — seeds the patch flow."""
    if not _is_local_spark(spark):
        try:
            fqn = get_contracts_cache_table_fqn(spark)
            df = spark.read.table(fqn)
            rows = df.collect()
            if not rows:
                return None
            latest_ts = max(r["updated_ts"] for r in rows if r["updated_ts"] is not None)
            latest_fp = next(r["schema_fingerprint"] for r in rows if r["updated_ts"] == latest_ts)
            contracts = {r["table_name"]: r["contract_yaml"] for r in rows if r["schema_fingerprint"] == latest_fp}
            return {"schema_fingerprint": latest_fp, "contracts": contracts}
        except Exception as e:
            print(f"[Warning] Failed to read latest UC contracts cache: {e}. Checking local cache.")
    try:
        if os.path.exists(LOCAL_CONTRACTS_CACHE_PATH):
            with open(LOCAL_CONTRACTS_CACHE_PATH, "r") as f:
                data = json.load(f)
            if not data:
                return None
            def _fp_latest_ts(fp):
                return max((v.get("updated_ts", "") for v in data[fp].values()), default="")
            latest_fp = max(data, key=_fp_latest_ts)
            contracts = {tbl: v.get("contract_yaml") for tbl, v in data[latest_fp].items()}
            return {"schema_fingerprint": latest_fp, "contracts": contracts}
    except Exception as e:
        print(f"[Warning] Failed to read latest local contracts cache: {e}")
    return None


def upsert_contracts_cache(spark: SparkSession, schema_fingerprint: str, contracts: Dict[str, str]) -> None:
    """Replace the full contract set for this fingerprint (one row per table)."""
    timestamp = datetime.datetime.now()
    try:
        os.makedirs(os.path.dirname(LOCAL_CONTRACTS_CACHE_PATH), exist_ok=True)
        data = {}
        if os.path.exists(LOCAL_CONTRACTS_CACHE_PATH):
            with open(LOCAL_CONTRACTS_CACHE_PATH, "r") as f:
                data = json.load(f)
        data[schema_fingerprint] = {
            tbl: {"contract_yaml": yaml_str, "updated_ts": timestamp.isoformat()}
            for tbl, yaml_str in contracts.items()
        }
        with open(LOCAL_CONTRACTS_CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to upsert local contracts cache: {e}")

    if _is_local_spark(spark) or not contracts:
        return
    try:
        fqn = get_contracts_cache_table_fqn(spark)
        schema = StructType([
            StructField("schema_fingerprint", StringType(), True),
            StructField("table_name", StringType(), True),
            StructField("contract_yaml", StringType(), True),
            StructField("updated_ts", TimestampType(), True),
        ])
        row_data = [(schema_fingerprint, tbl, yaml_str, timestamp) for tbl, yaml_str in contracts.items()]
        df = spark.createDataFrame(row_data, schema=schema)
        if spark.catalog.tableExists(fqn):
            from delta.tables import DeltaTable
            target = DeltaTable.forName(spark, fqn)
            target.alias("t").merge(
                df.alias("s"),
                "t.schema_fingerprint = s.schema_fingerprint AND t.table_name = s.table_name"
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        else:
            df.write.format("delta").mode("append").saveAsTable(fqn)
        print(f"[Agent Output Cache] Upserted {len(contracts)} contract(s) for fingerprint {schema_fingerprint[:12]}... to '{fqn}'")
    except Exception as e:
        print(f"[Warning] Failed to upsert UC contracts cache: {e}")


# ---- Modeling cache (schema_fingerprint -> gold_ddl + data_dictionary) ----

def get_modeling_cache_table_fqn(spark: SparkSession) -> str:
    return _cache_table_fqn(spark, "agent_modeling_cache")


def init_modeling_cache_table(spark: SparkSession) -> bool:
    fqn = get_modeling_cache_table_fqn(spark)
    if _is_local_spark(spark):
        os.makedirs(os.path.dirname(LOCAL_MODELING_CACHE_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_MODELING_CACHE_PATH):
            with open(LOCAL_MODELING_CACHE_PATH, "w") as f:
                json.dump({}, f)
        return True
    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fqn.split('.')[0]}.gold")
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                schema_fingerprint STRING,
                gold_ddl STRING,
                data_dictionary STRING,
                updated_ts TIMESTAMP
            ) USING DELTA
        """)
        return True
    except Exception as e:
        print(f"[Warning] Failed to initialize UC modeling cache table: {e}. Falling back to local file.")
        os.makedirs(os.path.dirname(LOCAL_MODELING_CACHE_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_MODELING_CACHE_PATH):
            with open(LOCAL_MODELING_CACHE_PATH, "w") as f:
                json.dump({}, f)
        return False


def get_modeling_cache(spark: SparkSession, schema_fingerprint: str) -> Optional[Dict[str, str]]:
    """Exact-fingerprint lookup. Returns {'gold_ddl':..., 'data_dictionary':...} or None."""
    if not _is_local_spark(spark):
        try:
            fqn = get_modeling_cache_table_fqn(spark)
            df = spark.read.table(fqn).filter(F.col("schema_fingerprint") == schema_fingerprint).orderBy(F.col("updated_ts").desc()).limit(1)
            rows = df.collect()
            if rows:
                return {"gold_ddl": rows[0]["gold_ddl"], "data_dictionary": rows[0]["data_dictionary"]}
            return None
        except Exception as e:
            print(f"[Warning] Failed to read UC modeling cache: {e}. Checking local cache.")
    try:
        if os.path.exists(LOCAL_MODELING_CACHE_PATH):
            with open(LOCAL_MODELING_CACHE_PATH, "r") as f:
                data = json.load(f)
            entry = data.get(schema_fingerprint)
            if entry:
                return {"gold_ddl": entry.get("gold_ddl"), "data_dictionary": entry.get("data_dictionary")}
    except Exception as e:
        print(f"[Warning] Failed to read local modeling cache: {e}")
    return None


def get_latest_modeling_cache(spark: SparkSession) -> Optional[Dict[str, Any]]:
    """Most recent entry regardless of fingerprint — seeds the patch flow."""
    if not _is_local_spark(spark):
        try:
            fqn = get_modeling_cache_table_fqn(spark)
            df = spark.read.table(fqn).orderBy(F.col("updated_ts").desc()).limit(1)
            rows = df.collect()
            if rows:
                return {
                    "schema_fingerprint": rows[0]["schema_fingerprint"],
                    "gold_ddl": rows[0]["gold_ddl"],
                    "data_dictionary": rows[0]["data_dictionary"],
                }
            return None
        except Exception as e:
            print(f"[Warning] Failed to read latest UC modeling cache: {e}. Checking local cache.")
    try:
        if os.path.exists(LOCAL_MODELING_CACHE_PATH):
            with open(LOCAL_MODELING_CACHE_PATH, "r") as f:
                data = json.load(f)
            if not data:
                return None
            latest_fp = max(data, key=lambda k: data[k].get("updated_ts", ""))
            entry = data[latest_fp]
            return {
                "schema_fingerprint": latest_fp,
                "gold_ddl": entry.get("gold_ddl"),
                "data_dictionary": entry.get("data_dictionary"),
            }
    except Exception as e:
        print(f"[Warning] Failed to read latest local modeling cache: {e}")
    return None


def upsert_modeling_cache(spark: SparkSession, schema_fingerprint: str, gold_ddl: str, data_dictionary: str) -> None:
    timestamp = datetime.datetime.now()
    try:
        os.makedirs(os.path.dirname(LOCAL_MODELING_CACHE_PATH), exist_ok=True)
        data = {}
        if os.path.exists(LOCAL_MODELING_CACHE_PATH):
            with open(LOCAL_MODELING_CACHE_PATH, "r") as f:
                data = json.load(f)
        data[schema_fingerprint] = {
            "gold_ddl": gold_ddl, "data_dictionary": data_dictionary, "updated_ts": timestamp.isoformat()
        }
        with open(LOCAL_MODELING_CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to upsert local modeling cache: {e}")

    if _is_local_spark(spark):
        return
    try:
        fqn = get_modeling_cache_table_fqn(spark)
        schema = StructType([
            StructField("schema_fingerprint", StringType(), True),
            StructField("gold_ddl", StringType(), True),
            StructField("data_dictionary", StringType(), True),
            StructField("updated_ts", TimestampType(), True),
        ])
        df = spark.createDataFrame([(schema_fingerprint, gold_ddl, data_dictionary, timestamp)], schema=schema)
        if spark.catalog.tableExists(fqn):
            from delta.tables import DeltaTable
            target = DeltaTable.forName(spark, fqn)
            target.alias("t").merge(
                df.alias("s"), "t.schema_fingerprint = s.schema_fingerprint"
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        else:
            df.write.format("delta").mode("append").saveAsTable(fqn)
        print(f"[Agent Output Cache] Upserted modeling output for fingerprint {schema_fingerprint[:12]}... to '{fqn}'")
    except Exception as e:
        print(f"[Warning] Failed to upsert UC modeling cache: {e}")

    return []

