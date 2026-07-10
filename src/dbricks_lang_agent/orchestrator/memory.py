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
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, TimestampType

# Local fallback path for development sandbox runs
LOCAL_MEMORY_PATH = "./generated/config/agent_fewshot_memory.json"


def get_memory_table_fqn(spark: SparkSession) -> str:
    """Read the memory table path from Spark configuration or catalog settings."""
    try:
        from dbricks_lang_agent.data_platform.spark_utils import load_config
        cfg = load_config()
        catalog = cfg.get("catalog", "hospitality_catalog")
    except Exception:
        catalog = "hospitality_catalog"
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
            StructField("timestamp", TimestampType(), True)
        ])
        
        row_data = [(dataset_name, issue_type, anomalous_fields, resolution_applied, human_comments, timestamp)]
        df = spark.createDataFrame(row_data, schema=schema)
        df.write.format("delta").mode("append").saveAsTable(fqn)
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
            # Fetch matching history
            df = spark.read.table(fqn).filter(
                f"issue_type = '{issue_type}' OR dataset_name = '{dataset_name}'"
            ).orderBy("timestamp", ascending=False).limit(5)
            
            rows = df.collect()
            for r in rows:
                records.append({
                    "dataset_name": r.dataset_name,
                    "issue_type": r.issue_type,
                    "anomalous_fields": r.anomalous_fields,
                    "resolution_applied": r.resolution_applied,
                    "human_comments": r.human_comments
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
        fields_str = ", ".join(r["anomalous_fields"])
        markdown += (
            f"- **Dataset**: {r['dataset_name']}\n"
            f"  - **Issue Type**: {r['issue_type']}\n"
            f"  - **Affected Fields**: [{fields_str}]\n"
            f"  - **Approved Resolution**: {r['resolution_applied']}\n"
            f"  - **Human Feedback**: \"{r['human_comments'] or 'None'}\"\n"
        )
    return markdown
