"""
profiling.py
============
Statistical profiling module adapted for Databricks.
Discovers and profiles all *.csv files located inside the configured Unity Catalog Volume,
computing schemas, null rates, cardinality, and duplicate key metrics.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, Any, List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

from .spark_utils import get_spark, load_config

CATEGORICAL_MAX_CARDINALITY = 25
ID_SUFFIX = "_id"
FK_OVERLAP_THRESHOLD = 0.5


def discover_source_tables() -> Dict[str, str]:
    """Discover all CSV source files in the configured Unity Catalog Volume."""
    cfg = load_config()
    vol_path = cfg.get("volume_raw_path", "/Volumes/hospitality_catalog/raw/source_volume")
    
    if not os.path.exists(vol_path):
        # Fallback to local source path for unit testing
        vol_path = os.environ.get("SOURCE_ROOT", "./Source")

    files = sorted(glob.glob(os.path.join(vol_path, "*.csv")))
    tables = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        # Strip raw_ prefix if present
        table_name = stem[4:] if stem.startswith("raw_") else stem
        tables[table_name] = os.path.basename(f)
    return tables


def load_source(spark: SparkSession, filename: str) -> DataFrame:
    """Read a CSV source file from the Unity Catalog Volume as a Spark DataFrame."""
    cfg = load_config()
    vol_path = cfg.get("volume_raw_path", "/Volumes/hospitality_catalog/raw/source_volume")
    
    if not os.path.exists(vol_path):
        vol_path = os.environ.get("SOURCE_ROOT", "./Source")

    path = os.path.join(vol_path, filename)
    return spark.read.option("header", True).option("inferSchema", True).csv(path)


def _null_pct(df: DataFrame, col: str, total: int) -> float:
    if total == 0:
        return 0.0
    n = df.filter(F.col(col).isNull() | (F.col(col).cast("string") == "")).count()
    return round(100.0 * n / total, 2)


def profile_dataframe(df: DataFrame, table_name: str) -> Dict[str, Any]:
    """Build a statistical profile dictionary for a Spark DataFrame."""
    total = df.count()
    profile: Dict[str, Any] = {
        "table": table_name,
        "row_count": total,
        "column_count": len(df.columns),
        "columns": {},
    }
    
    aggs = []
    for field in df.schema.fields:
        col = field.name
        # Null count expression
        aggs.append(F.sum(F.when(F.col(col).isNull() | (F.col(col).cast("string") == ""), 1).otherwise(0)).alias(f"{col}_nulls"))
        # Distinct count expression
        aggs.append(F.countDistinct(F.col(col)).alias(f"{col}_distinct"))
        # Numeric stats expression
        if isinstance(field.dataType, NumericType) or str(field.dataType) in ("IntegerType", "LongType", "DoubleType", "FloatType", "DecimalType"):
            aggs.append(F.min(F.col(col)).alias(f"{col}_min"))
            aggs.append(F.max(F.col(col)).alias(f"{col}_max"))
            aggs.append(F.avg(F.col(col)).alias(f"{col}_avg"))
            aggs.append(F.stddev(F.col(col)).alias(f"{col}_stddev"))

    if aggs:
        stats_row = df.select(*aggs).collect()[0]
        stats_dict = stats_row.asDict()
    else:
        stats_dict = {}

    for field in df.schema.fields:
        col = field.name
        nulls = stats_dict.get(f"{col}_nulls", 0) or 0
        distinct = stats_dict.get(f"{col}_distinct", 0) or 0
        
        col_profile: Dict[str, Any] = {
            "dtype": str(field.dataType),
            "null_pct": round(100.0 * nulls / total, 2) if total > 0 else 0.0,
            "distinct_count": distinct,
        }
        
        if isinstance(field.dataType, NumericType) or str(field.dataType) in ("IntegerType", "LongType", "DoubleType", "FloatType", "DecimalType"):
            col_profile["numeric_stats"] = {
                "min": stats_dict.get(f"{col}_min"),
                "max": stats_dict.get(f"{col}_max"),
                "avg": round(stats_dict.get(f"{col}_avg"), 4) if stats_dict.get(f"{col}_avg") is not None else None,
                "stddev": round(stats_dict.get(f"{col}_stddev"), 4) if stats_dict.get(f"{col}_stddev") is not None else None,
            }
        elif distinct <= CATEGORICAL_MAX_CARDINALITY:
            try:
                vc = df.groupBy(col).count().orderBy(F.desc("count")).limit(CATEGORICAL_MAX_CARDINALITY).collect()
                col_profile["value_counts"] = {str(row[col]): row["count"] for row in vc}
            except Exception:
                pass
        profile["columns"][col] = col_profile
    return profile


def find_candidate_unique_keys(df: DataFrame, total: int) -> List[str]:
    """Find columns with the '_id' suffix that are non-null and distinct (unique)."""
    id_cols = [f.name for f in df.schema.fields if f.name.lower().endswith(ID_SUFFIX)]
    candidates = []
    for c in id_cols:
        null_count = df.filter(F.col(c).isNull()).count()
        distinct_count = df.select(c).distinct().count()
        if null_count == 0 and distinct_count == total:
            candidates.append(c)
    return candidates


def duplicate_key_count(df: DataFrame, key_cols: List[str]) -> int:
    """Calculate duplicate key count for validation checks."""
    total = df.count()
    distinct = df.select(*key_cols).distinct().count()
    return total - distinct


def _collect_value_set(df: DataFrame, col: str, limit: int = 20000) -> set:
    rows = df.select(col).filter(F.col(col).isNotNull()).distinct().limit(limit).collect()
    return {r[0] for r in rows}


def _name_similarity_bonus(child_col: str, parent_table: str, parent_col: str) -> float:
    """Break ties by identifying naming patterns (stems, prefixes)."""
    bonus = 0.0
    if child_col == parent_col:
        bonus += 0.5
    stem = child_col.lower()
    for prefix in ("raw_", "external_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    stem = stem[: -len(ID_SUFFIX)] if stem.endswith(ID_SUFFIX) else stem
    pt = parent_table.lower()
    if stem and (stem in pt or pt in stem or pt.rstrip("s") == stem.rstrip("s")):
        bonus += 0.3
    return bonus


def discover_foreign_keys(dfs: Dict[str, DataFrame], unique_keys: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Determine relationships between tables by comparing column overlaps and names."""
    value_cache: Dict[tuple, set] = {}

    def values_for(table, col):
        key = (table, col)
        if key not in value_cache:
            value_cache[key] = _collect_value_set(dfs[table], col)
        return value_cache[key]

    discovered = []
    for t_name, df in dfs.items():
        id_cols = [f.name for f in df.schema.fields if f.name.lower().endswith(ID_SUFFIX)]
        for col in id_cols:
            if t_name in unique_keys and col in unique_keys[t_name]:
                continue  # skip primary keys

            best_parent_tbl = None
            best_parent_col = None
            best_score = 0.0
            best_overlap = 0.0

            child_values = values_for(t_name, col)
            if not child_values:
                continue

            for p_tbl, p_keys in unique_keys.items():
                if p_tbl == t_name:
                    continue
                for pk in p_keys:
                    parent_values = values_for(p_tbl, pk)
                    if not parent_values:
                        continue
                    overlap_cnt = len(child_values.intersection(parent_values))
                    overlap_pct = overlap_cnt / len(child_values)
                    if overlap_pct >= FK_OVERLAP_THRESHOLD:
                        score = overlap_pct + _name_similarity_bonus(col, p_tbl, pk)
                        if score > best_score:
                            best_score = score
                            best_overlap = overlap_pct
                            best_parent_tbl = p_tbl
                            best_parent_col = pk

            if best_parent_tbl:
                # Calculate the exact orphan count
                parent_keys = dfs[best_parent_tbl].select(F.col(best_parent_col).alias(col)).distinct()
                orphans = df.join(parent_keys, on=col, how="left_anti").filter(F.col(col).isNotNull()).count()
                discovered.append({
                    "table": t_name,
                    "column": col,
                    "parent_table": best_parent_tbl,
                    "parent_column": best_parent_col,
                    "overlap_pct": round(best_overlap * 100.0, 2),
                    "orphan_count": orphans,
                })
    return discovered


def profile_all_sources(output_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute profile checks across all discovered sources."""
    spark = get_spark()
    discovered_tables = discover_source_tables()
    
    dfs = {t_name: load_source(spark, fname) for t_name, fname in discovered_tables.items()}
    total_rows = {t_name: df.count() for t_name, df in dfs.items()}

    # Calculate Candidate Primary Keys
    unique_keys = {t_name: find_candidate_unique_keys(df, total_rows[t_name]) for t_name, df in dfs.items()}

    # Profile DataFrames
    tables_profile = {t_name: profile_dataframe(df, t_name) for t_name, df in dfs.items()}

    # Discover Foreign Keys
    fks = discover_foreign_keys(dfs, unique_keys)

    # Compile Full Profiling Report
    report = {
        "discovered_tables": discovered_tables,
        "candidate_unique_keys": unique_keys,
        "duplicate_keys": {t_name: duplicate_key_count(df, unique_keys[t_name]) if unique_keys[t_name] else 0 for t_name, df in dfs.items()},
        "referential_integrity": fks,
        "tables": tables_profile,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    return report
