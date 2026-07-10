"""
spark_utils.py
==============
Central Spark bootstrap + storage abstraction for the Databricks Unity Catalog pipeline.
All read/write/merge operations are routed through these functions, writing directly
to Unity Catalog tables (Delta format) rather than file system paths.
"""
from __future__ import annotations

import os
import yaml
from typing import Optional, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

_spark: Optional[SparkSession] = None
_config: Optional[dict] = None


def load_config() -> dict:
    """Load config.yaml from the project root folder."""
    global _config
    if _config is not None:
        return _config

    # Walk up to find config.yaml
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        candidate = os.path.join(cur_dir, "config.yaml")
        if os.path.exists(candidate):
            with open(candidate) as f:
                _config = yaml.safe_load(f)
                return _config
        cur_dir = os.path.dirname(cur_dir)

    # Fallback default configuration
    _config = {
        "catalog": "hospitality_catalog",
        "schemas": {
            "raw": "raw",
            "bronze": "bronze",
            "silver": "silver",
            "gold": "gold"
        },
        "volume_raw_path": "/Volumes/hospitality_catalog/raw/source_volume",
        "llm": {
            "endpoint": "databricks-meta-llama-3-1-70b-instruct",
            "temperature": 0.0
        }
    }
    return _config


def get_spark(app_name: str = "databricks-langgraph-medallion") -> SparkSession:
    """Return the active Databricks SparkSession, or build one if running locally."""
    global _spark
    if _spark is not None:
        return _spark

    try:
        # Check if running in a Databricks Notebook or job
        _spark = SparkSession.getActiveSession()
    except Exception:
        pass

    if _spark is None:
        # Fallback for local development or unit testing
        builder = (
            SparkSession.builder.appName(app_name)
            .master("local[*]")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
            .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
        )
        _spark = builder.getOrCreate()

    try:
        _spark.sparkContext.setLogLevel("ERROR")
    except Exception:
        pass
    return _spark


def get_fqn(schema: str, table: str) -> str:
    """Construct a 3-level Unity Catalog Table Name: `catalog`.`schema`.`table`."""
    cfg = load_config()
    catalog = cfg.get("catalog", "hospitality_catalog")
    schemas = cfg.get("schemas", {})
    resolved_schema = schemas.get(schema, schema)
    return f"`{catalog}`.`{resolved_schema}`.`{table}`"


def table_exists(schema: str, table: str) -> bool:
    """Check if a table exists in Unity Catalog."""
    spark = get_spark()
    fqn = get_fqn(schema, table)
    return spark.catalog.tableExists(fqn)


def read_table(schema: str, table: str) -> DataFrame:
    """Read a table from Unity Catalog."""
    spark = get_spark()
    fqn = get_fqn(schema, table)
    return spark.read.table(fqn)


def write_full_overwrite(df: DataFrame, schema: str, table: str, partition_by: Optional[List[str]] = None) -> str:
    """Write DataFrame as a Delta table in Unity Catalog (full overwrite)."""
    spark = get_spark()
    cfg = load_config()
    catalog = cfg.get("catalog", "hospitality_catalog")
    schemas = cfg.get("schemas", {})
    resolved_schema = schemas.get(schema, schema)

    # Ensure schema exists in Unity Catalog
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{resolved_schema}`")

    fqn = get_fqn(schema, table)
    writer = df.write.format("delta").mode("overwrite")
    if partition_by:
        writer = writer.partitionBy(*partition_by)

    # Allow schema evolution during development full overwrites
    writer.option("overwriteSchema", "true").saveAsTable(fqn)
    return fqn


def merge_upsert(new_df: DataFrame, schema: str, table: str, key_cols: List[str]) -> str:
    """Upsert new_df into target Delta table in Unity Catalog using MERGE INTO."""
    spark = get_spark()
    cfg = load_config()
    catalog = cfg.get("catalog", "hospitality_catalog")
    schemas = cfg.get("schemas", {})
    resolved_schema = schemas.get(schema, schema)

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{resolved_schema}`")
    fqn = get_fqn(schema, table)

    if table_exists(schema, table):
        from delta.tables import DeltaTable
        target = DeltaTable.forName(spark, fqn)
        merge_cond = " AND ".join([f"t.`{k}` = s.`{k}`" for k in key_cols])
        (
            target.alias("t")
            .merge(new_df.alias("s"), merge_cond)
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        new_df.write.format("delta").mode("overwrite").saveAsTable(fqn)
    return fqn


def scd2_merge(
    new_df: DataFrame,
    schema: str,
    table: str,
    business_key: str,
    tracked_cols: List[str],
    surrogate_key_col: str,
    as_of_ts_col: str = "_load_ts",
    initial_load_sentinel_start: str = "1900-01-01 00:00:00",
) -> str:
    """
    Generic slowly changing dimension (SCD) Type 2 merge for Unity Catalog Delta Tables.
    
    Behavior:
      - First load: Creates table, populates surrogate key, sets eff_start_ts to 1900-01-01.
      - Incremental loads: Updates historical versions where hashes changed, inserts new versions.
    """
    spark = get_spark()
    fqn = get_fqn(schema, table)

    # Compute a unique hash of the tracked attributes to detect changes
    row_hash_expr = F.sha2(F.concat_ws("||", *[F.coalesce(F.col(c).cast("string"), F.lit("NULL")) for c in tracked_cols]), 256)
    incoming = new_df.withColumn("row_hash", row_hash_expr)

    if not table_exists(schema, table):
        # Initial Load: set sentinel start date so old facts can match
        w = Window.orderBy(business_key)
        result = (
            incoming
            .withColumn(surrogate_key_col, F.row_number().over(w).cast("long"))
            .withColumn("eff_start_ts", F.lit(initial_load_sentinel_start).cast("timestamp"))
            .withColumn("eff_end_ts", F.lit(None).cast("timestamp"))
            .withColumn("is_current", F.lit(True))
            .drop(as_of_ts_col)
        )
        write_full_overwrite(result, schema, table)
        return fqn

    existing = read_table(schema, table)
    max_sk = existing.agg(F.max(surrogate_key_col)).collect()[0][0] or 0
    current = existing.filter(F.col("is_current") == True)  # noqa: E712

    joined = incoming.alias("inc").join(
        current.select(business_key, "row_hash").alias("cur"),
        on=business_key,
        how="left",
    )
    changed_or_new = joined.filter(
        F.col("cur.row_hash").isNull() | (F.col("inc.row_hash") != F.col("cur.row_hash"))
    ).select("inc.*")

    if not changed_or_new.head(1):
        return fqn  # No updates to process

    keys_changed = changed_or_new.select(business_key).distinct()

    # Close out previous versions for changed keys
    as_of_lookup = changed_or_new.select(business_key, as_of_ts_col).distinct()
    to_close = current.join(keys_changed, on=business_key, how="inner")
    to_close = to_close.join(as_of_lookup, on=business_key, how="left")
    closed = (
        to_close.withColumn("eff_end_ts", F.col(as_of_ts_col))
        .withColumn("is_current", F.lit(False))
        .drop(as_of_ts_col)
    )

    unaffected = existing.join(keys_changed, on=business_key, how="left_anti")

    w = Window.orderBy(business_key)
    new_versions = (
        changed_or_new
        .withColumn(surrogate_key_col, (F.row_number().over(w) + F.lit(max_sk)).cast("long"))
        .withColumn("eff_start_ts", F.col(as_of_ts_col))
        .withColumn("eff_end_ts", F.lit(None).cast("timestamp"))
        .withColumn("is_current", F.lit(True))
        .drop(as_of_ts_col)
    )

    final = unaffected.unionByName(closed, allowMissingColumns=True).unionByName(
        new_versions, allowMissingColumns=True
    )
    write_full_overwrite(final, schema, table)
    return fqn


def build_dim_date(spark: SparkSession, start_date: str, end_date: str) -> DataFrame:
    """Generate a conformed date calendar dimension."""
    df = spark.sql(f"SELECT explode(sequence(to_date('{start_date}'), to_date('{end_date}'), interval 1 day)) AS full_date")
    df = (
        df.withColumn("date_sk", F.date_format("full_date", "yyyyMMdd").cast("int"))
        .withColumn("day_of_month", F.dayofmonth("full_date"))
        .withColumn("day_of_week", F.dayofweek("full_date"))
        .withColumn("day_name", F.date_format("full_date", "EEEE"))
        .withColumn("week_of_year", F.weekofyear("full_date"))
        .withColumn("month_num", F.month("full_date"))
        .withColumn("month_name", F.date_format("full_date", "MMMM"))
        .withColumn("quarter", F.quarter("full_date"))
        .withColumn("year", F.year("full_date"))
        .withColumn("is_weekend", F.col("day_of_week").isin(1, 7))
    )
    return df.select(
        "date_sk", "full_date", "day_of_month", "day_of_week", "day_name",
        "week_of_year", "month_num", "month_name", "quarter", "year", "is_weekend",
    )


def reset_lake(schemas: Optional[List[str]] = None) -> None:
    """Dev/test helper: Drop target schemas in Unity Catalog to start fresh."""
    spark = get_spark()
    cfg = load_config()
    catalog = cfg.get("catalog", "hospitality_catalog")
    schemas = schemas or ["bronze", "silver", "gold"]
    for s in schemas:
        spark.sql(f"DROP SCHEMA IF EXISTS `{catalog}`.`{s}` CASCADE")
