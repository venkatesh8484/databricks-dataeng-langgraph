"""
bronze.py
=========
Source/Volume -> Bronze ingestion (UC table write).
Loads CSVs dynamically from the Unity Catalog Volume, appends metadata columns,
and writes them directly as Bronze Delta tables in Unity Catalog.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any

from pyspark.sql import functions as F

from dbricks_lang_agent.data_platform.spark_utils import get_spark, write_full_overwrite
from dbricks_lang_agent.data_platform.profiling import discover_source_tables, load_source


def ingest_all(batch_id: str = None) -> Dict[str, Any]:
    spark = get_spark()
    batch_id = batch_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ingestion_ts = F.current_timestamp()

    discovered = discover_source_tables()
    results = {}
    
    for table_name, csv_filename in discovered.items():
        print(f"Ingesting raw source {csv_filename} -> bronze.{table_name} table...")
        df = load_source(spark, csv_filename)
        
        # Append standard metadata columns
        bronze_df = (
            df.withColumn("_ingestion_ts", ingestion_ts)
            .withColumn("_batch_id", F.lit(batch_id))
            .withColumn("_source_layer", F.lit("raw_volume"))
        )
        
        # Overwrite in UC bronze schema
        fqn = write_full_overwrite(bronze_df, "bronze", table_name)
        results[table_name] = {
            "fqn": fqn,
            "row_count": bronze_df.count(),
            "batch_id": batch_id
        }
        
    return results
