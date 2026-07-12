"""
products.py
============
Materializes LLM-proposed "data products" on top of the Gold star schema.

Each product is a single SQL SELECT authored by the DataProductAdvisor agent
(see orchestrator/agents.py::product_advisor_node) and executed here against
the Gold schema, then published as a VIEW or TABLE under a separate `products`
Unity Catalog schema — deliberately kept apart from `gold` so the conformed
dimensional model stays stable/reusable while products stay free to be
denormalized, opinionated, and per-consumer (finance mart, customer 360,
supplier scorecard, etc.).

This module intentionally does NOT go through the DataEngineer's PySpark
codegen + compile-loop machinery — a product definition is one straight
SELECT statement, so it is executed directly via spark.sql() and any error
is surfaced as-is to the caller (the dashboard's "Data Products" tab).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from .spark_utils import get_spark, load_config


def get_products_schema() -> str:
    """Resolve the configured schema name for materialized data products."""
    cfg = load_config()
    return cfg.get("schemas", {}).get("products", "products")


def list_built_products() -> Dict[str, int]:
    """Return {table_name: row_count} for everything currently materialized
    in the products schema. Best-effort — used by the dashboard to show
    status for products that were built in a previous session/app restart."""
    spark = get_spark()
    cfg = load_config()
    catalog = cfg.get("catalog", "databricks_langgraph")
    products_schema = get_products_schema()

    result: Dict[str, int] = {}
    try:
        tables = spark.sql(f"SHOW TABLES IN `{catalog}`.`{products_schema}`").collect()
        for t in tables:
            fqn = f"`{catalog}`.`{products_schema}`.`{t.tableName}`"
            try:
                row_count = spark.sql(f"SELECT COUNT(*) AS c FROM {fqn}").collect()[0]["c"]
            except Exception:
                row_count = None
            result[t.tableName] = row_count
    except Exception:
        pass  # Schema may not exist yet — nothing built so far.
    return result


def build_product(product: Dict[str, Any]) -> Dict[str, Any]:
    """Materialize one advisor-proposed product.

    `product` is one entry from state["product_candidates"]:
    {id, name, description, product_type, source_tables, grain, sql, ...}

    Returns a status dict suitable for state["product_build_status"][product_id]:
    {status, target_fqn, object_kind, row_count, last_built_ts, error}
    """
    spark = get_spark()
    cfg = load_config()
    catalog = cfg.get("catalog", "databricks_langgraph")
    gold_schema = cfg.get("schemas", {}).get("gold", "gold")
    products_schema = get_products_schema()

    product_id = (product or {}).get("id", "")
    sql = (product or {}).get("sql", "").strip().rstrip(";")
    product_type = ((product or {}).get("product_type") or "view").strip().lower()
    object_kind = "TABLE" if product_type == "table" else "VIEW"

    now_iso = datetime.now(timezone.utc).isoformat()

    if not product_id or not sql:
        return {
            "status": "failed",
            "error": "Product definition is missing 'id' or 'sql' — cannot build.",
            "last_built_ts": now_iso,
        }

    try:
        # Ensure the products schema exists.
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{products_schema}`")

        # The advisor authors SQL with bare table names (e.g. `fact_bookings`),
        # assuming Gold is the current schema — set that context here so those
        # references resolve without needing brittle string substitution.
        spark.sql(f"USE CATALOG `{catalog}`")
        spark.sql(f"USE SCHEMA `{gold_schema}`")

        target_fqn = f"`{catalog}`.`{products_schema}`.`{product_id}`"
        spark.sql(f"CREATE OR REPLACE {object_kind} {target_fqn} AS {sql}")

        row_count = None
        try:
            row_count = spark.sql(f"SELECT COUNT(*) AS c FROM {target_fqn}").collect()[0]["c"]
        except Exception:
            pass  # Row count is best-effort; don't fail the build over it.

        return {
            "status": "built",
            "target_fqn": target_fqn.replace("`", ""),
            "object_kind": object_kind,
            "row_count": row_count,
            "last_built_ts": now_iso,
            "error": "",
        }
    except Exception as e:
        return {
            "status": "failed",
            "object_kind": object_kind,
            "error": str(e),
            "last_built_ts": now_iso,
        }
