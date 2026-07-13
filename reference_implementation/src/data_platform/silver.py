"""
silver.py
=========
Bronze -> Silver transformation + contract validation.
Applies YAML data contracts, standardizes formats, and separates invalid rows into the quarantine layer.
"""
from __future__ import annotations

import os
import json
from typing import Dict, Any, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

from dbricks_lang_agent.data_platform.spark_utils import get_spark, read_table, write_full_overwrite
from dbricks_lang_agent.data_platform.contracts import load_all_contracts, validate_table, summarize_reports

PROCESS_ORDER = ["buyers", "subcontractors", "vessels", "build_slots", "orders", "order_lines"]

BUSINESS_KEYS = {
    "buyers": "external_buyer_id",
    "subcontractors": "external_subcontractor_id",
    "vessels": "external_vessel_id",
    "build_slots": None,  # composite key (handled manually)
    "orders": "external_order_id",
    "order_lines": "line_reference",
}

YN_COLUMNS = {
    "buyers": ["marketing_optin"],
    "vessels": ["has_helideck", "ice_class"],
    "build_slots": ["is_closed"],
    "orders": ["is_owner_order"],
}

DROP_ALWAYS_NULL_COLS = {
    "buyers": [],
    "subcontractors": ["address2"],
    "vessels": ["address2", "address3"],
    "build_slots": [],
    "orders": [],
    "order_lines": [],
}


def _yn_to_bool(df: DataFrame, cols: List[str]) -> DataFrame:
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, F.when(F.col(c) == "Y", True).when(F.col(c) == "N", False).otherwise(None))
    return df


def _dedupe_latest(df: DataFrame, key_col: str, order_col: str = "_ingestion_ts") -> DataFrame:
    w = Window.partitionBy(key_col).orderBy(F.col(order_col).desc())
    return (
        df.withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def _trim_strings(df: DataFrame) -> DataFrame:
    for field in df.schema.fields:
        if str(field.dataType) == "StringType()":
            df = df.withColumn(field.name, F.trim(F.col(field.name)))
    return df


def transform_all(fail_fast_on_hard_breach: bool = True) -> Dict[str, Any]:
    spark = get_spark()
    contracts = load_all_contracts()
    reports = []
    promoted: Dict[str, DataFrame] = {}
    summary: Dict[str, Any] = {"tables": {}, "halted_at": None}

    # Verify contracts are loaded
    if not contracts:
        print("Warning: No contracts found. Schema validation will be skipped.")

    for table in PROCESS_ORDER:
        print(f"Transforming bronze.{table} -> silver.{table}...")
        bronze_df = read_table("bronze", table)

        df = _trim_strings(bronze_df)
        
        # Drop columns that are confirmed to be always null / structurally empty
        drop_cols = [c for c in DROP_ALWAYS_NULL_COLS.get(table, []) if c in df.columns]
        if drop_cols:
            df = df.drop(*drop_cols)

        # Deduplicate keys
        if table == "build_slots":
            w = Window.partitionBy("external_vessel_id", "slot_date").orderBy(F.col("_ingestion_ts").desc())
            df = df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")
        else:
            df = _dedupe_latest(df, BUSINESS_KEYS[table])

        # Validate against YAML contract
        if table in contracts:
            contract = contracts[table]
            # Pass already promoted silver tables as potential parents for referential checks
            clean_df, quarantined_df, report = validate_table(df, contract, promoted)
        else:
            # Fallback when contract is missing (e.g. initial setup)
            clean_df = df
            quarantined_df = spark.createDataFrame([], df.schema)
            report = {
                "table": table, "row_count": df.count(), "clean_count": df.count(),
                "quarantined_count": 0, "promotion_blocked": False, "rule_results": []
            }

        reports.append(report)

        # Apply standardized booleans after checking raw string values in contracts
        clean_df = _yn_to_bool(clean_df, YN_COLUMNS.get(table, []))

        # Write to Quarantine and Silver layers in Unity Catalog
        write_full_overwrite(quarantined_df, "quarantine", table)
        
        clean_df = clean_df.withColumn("_silver_load_ts", F.current_timestamp())
        write_full_overwrite(clean_df, "silver", table)
        
        promoted[table] = read_table("silver", table)

        summary["tables"][table] = {
            "row_count_in": df.count(),
            "row_count_promoted": report["clean_count"],
            "row_count_quarantined": report["quarantined_count"],
            "promotion_blocked": report["promotion_blocked"],
        }

        if report["promotion_blocked"] and fail_fast_on_hard_breach:
            summary["halted_at"] = table
            print(f"!!! Validation failed on hard rules for table '{table}'. Halting pipeline.")
            break

    summary["contract_summary"] = summarize_reports(reports)
    return summary
