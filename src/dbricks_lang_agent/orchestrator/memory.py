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


LOCAL_CODEBASE_MEMORY_PATH = "./generated/config/agent_codebase_memory.json"

def get_codebase_table_fqn(spark: SparkSession) -> str:
    """Read the codebase memory table path from Spark configuration or catalog settings."""
    try:
        from dbricks_lang_agent.data_platform.spark_utils import load_config
        cfg = load_config()
        catalog = cfg.get("catalog", "databricks_langgraph")
    except Exception:
        catalog = "databricks_langgraph"
    return f"{catalog}.gold.agent_codebase_memory"


def init_codebase_memory_table(spark: SparkSession) -> bool:
    """Initialize the gold.agent_codebase_memory Delta Table in Unity Catalog."""
    fqn = get_codebase_table_fqn(spark)
    print(f"Initializing codebase memory table at: {fqn}...")
    
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
        
        # Create Delta table
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                dataset_fingerprint STRING,
                bronze_code STRING,
                silver_code STRING,
                gold_code STRING,
                timestamp TIMESTAMP
            ) USING DELTA
        """)
        return True
    except Exception as e:
        print(f"[Warning] Failed to initialize UC codebase memory table: {e}. Falling back to local file.")
        os.makedirs(os.path.dirname(LOCAL_CODEBASE_MEMORY_PATH), exist_ok=True)
        if not os.path.exists(LOCAL_CODEBASE_MEMORY_PATH):
            with open(LOCAL_CODEBASE_MEMORY_PATH, "w") as f:
                json.dump({}, f)
        return False


def get_stored_codebase(spark: SparkSession, fingerprint: str) -> Optional[Dict[str, str]]:
    """Retrieve compiled codebase for the given dataset fingerprint."""
    try:
        is_local = spark.conf.get("spark.master", "").startswith("local")
    except Exception:
        is_local = False

    if not is_local:
        try:
            fqn = get_codebase_table_fqn(spark)
            df = spark.read.table(fqn).filter(
                f"dataset_fingerprint = '{fingerprint}'"
            ).orderBy("timestamp", ascending=False).limit(1)
            
            rows = df.collect()
            if rows:
                return {
                    "bronze_code": rows[0].bronze_code,
                    "silver_code": rows[0].silver_code,
                    "gold_code": rows[0].gold_code
                }
        except Exception as e:
            print(f"[Warning] Failed to read from UC codebase memory: {e}. Checking local cache.")

    # Local fallback
    try:
        if os.path.exists(LOCAL_CODEBASE_MEMORY_PATH):
            with open(LOCAL_CODEBASE_MEMORY_PATH, "r") as f:
                data = json.load(f)
            if fingerprint in data:
                return data[fingerprint]
    except Exception as e:
        print(f"[Warning] Failed to read from local codebase memory: {e}")
        
    return None


def log_codebase(
    spark: SparkSession,
    fingerprint: str,
    bronze_code: str,
    silver_code: str,
    gold_code: str
) -> None:
    """Store successfully compiled codebase in memory."""
    timestamp = datetime.datetime.now()

    # 1. Update local cache
    try:
        os.makedirs(os.path.dirname(LOCAL_CODEBASE_MEMORY_PATH), exist_ok=True)
        data = {}
        if os.path.exists(LOCAL_CODEBASE_MEMORY_PATH):
            with open(LOCAL_CODEBASE_MEMORY_PATH, "r") as f:
                data = json.load(f)

        data[fingerprint] = {
            "bronze_code": bronze_code,
            "silver_code": silver_code,
            "gold_code": gold_code,
            "timestamp": timestamp.isoformat()
        }

        with open(LOCAL_CODEBASE_MEMORY_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to log codebase to local JSON cache: {e}")

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
            StructField("bronze_code", StringType(), True),
            StructField("silver_code", StringType(), True),
            StructField("gold_code", StringType(), True),
            StructField("timestamp", TimestampType(), True)
        ])

        row_data = [(fingerprint, bronze_code, silver_code, gold_code, timestamp)]
        df = spark.createDataFrame(row_data, schema=schema)

        # Merge/Upsert to keep only the latest code for this fingerprint
        if spark.catalog.tableExists(fqn):
            from delta.tables import DeltaTable
            target = DeltaTable.forName(spark, fqn)
            target.alias("t").merge(
                df.alias("s"),
                "t.dataset_fingerprint = s.dataset_fingerprint"
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        else:
            df.write.format("delta").mode("overwrite").saveAsTable(fqn)

        print(f"Successfully logged codebase to Delta table '{fqn}'")
    except Exception as e:
        print(f"[Warning] Failed to log codebase to Delta table: {e}")


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

