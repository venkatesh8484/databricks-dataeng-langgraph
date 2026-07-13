"""
gold.py
=======
Silver -> Gold: builds the dimensional star schema tables inside Unity Catalog.
All dimensions are loaded as SCD Type 1 or SCD Type 2, and fact tables resolve dimension keys point-in-time.
"""
from __future__ import annotations

from typing import Dict, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

from dbricks_lang_agent.data_platform.spark_utils import get_spark, read_table, write_full_overwrite, scd2_merge, build_dim_date

DATE_RANGE_START = "2022-01-01"
DATE_RANGE_END = "2025-12-31"


def _date_sk(col):
    return F.date_format(F.col(col), "yyyyMMdd").cast("int")


def build_dim_date_table() -> str:
    spark = get_spark()
    df = build_dim_date(spark, DATE_RANGE_START, DATE_RANGE_END)
    return write_full_overwrite(df, "gold", "dim_date")


def build_dim_channel_table() -> str:
    orders = read_table("silver", "orders")
    combos = orders.select("sales_channel", "channel_group").distinct()
    w = Window.orderBy("sales_channel", "channel_group")
    dim = combos.withColumn("channel_sk", F.row_number().over(w).cast("long"))
    dim = dim.select("channel_sk", "sales_channel", "channel_group")
    return write_full_overwrite(dim, "gold", "dim_channel")


def build_dim_subcontractor_table() -> str:
    """SCD Type 1 - Overwrite dimension table with current data."""
    subcontractors = read_table("silver", "subcontractors")
    w = Window.orderBy("external_subcontractor_id")
    dim = (
        subcontractors.withColumn("subcontractor_sk", F.row_number().over(w).cast("long"))
        .select(
            "subcontractor_sk",
            F.col("external_subcontractor_id").alias("subcontractor_id"),
            "subcontractor_ref",
            "source_system",
            "class_society_code",
            F.col("name").alias("subcontractor_name"),
            "email",
            "telephone",
            F.col("title").alias("contact_title"),
            F.col("forename").alias("contact_forename"),
            F.col("surname").alias("contact_surname"),
            "address1",
            "city",
            "postcode",
            "country",
            "payment_method",
            F.col("_silver_load_ts").alias("last_updated_ts"),
        )
    )
    return write_full_overwrite(dim, "gold", "dim_subcontractor")


def build_dim_buyer_table() -> str:
    """SCD Type 2 Buyer dimension loaded via spark_utils.scd2_merge."""
    buyers = read_table("silver", "buyers")
    tracked_cols = [
        "buyer_ref", "source_system", "email", "title", "forename", "surname",
        "telephone", "mobile", "address1", "address2", "city", "postcode", "country",
        "marketing_optin",
    ]
    incoming = (
        buyers.withColumn("full_name", F.concat_ws(" ", F.col("forename"), F.col("surname")))
        .withColumnRenamed("external_buyer_id", "buyer_id")
        .withColumn("_load_ts", F.col("_silver_load_ts"))
    )
    return scd2_merge(
        incoming,
        schema="gold",
        table="dim_buyer",
        business_key="buyer_id",
        tracked_cols=tracked_cols + ["full_name"],
        surrogate_key_col="buyer_sk",
    )


def build_dim_vessel_table() -> str:
    """SCD Type 2 Vessel dimension loaded via spark_utils.scd2_merge."""
    acc = read_table("silver", "vessels")
    tracked_cols = [
        "vessel_ref", "source_system", "vessel_class", "brand", "name",
        "address1", "city", "postcode", "resort", "area", "region", "country",
        "miles_from_sea", "margin_rate", "value_tier", "value_tier_start",
        "has_helideck", "ice_class", "deck_count", "complement", "rating",
        "contract_type", "withdrawn_date",
    ]
    incoming = (
        acc.withColumnRenamed("external_vessel_id", "vessel_id")
        .withColumnRenamed("name", "vessel_name")
        .withColumn("value_tier_start_date", F.to_date("value_tier_start"))
        .withColumn("is_withdrawn", F.col("withdrawn_date").isNotNull())
        .withColumn("date_created", F.to_date("date_created"))
        .withColumn("date_live", F.to_date("date_live"))
        .withColumn("withdrawn_date", F.to_date("withdrawn_date"))
        .withColumn("_load_ts", F.col("_silver_load_ts"))
    )
    tracked_final = [c if c != "name" else "vessel_name" for c in tracked_cols]
    tracked_final = [c if c != "value_tier_start" else "value_tier_start_date" for c in tracked_final]
    return scd2_merge(
        incoming,
        schema="gold",
        table="dim_vessel",
        business_key="vessel_id",
        tracked_cols=tracked_final,
        surrogate_key_col="vessel_sk",
    )


def _scd2_asof_join(
    fact_df: DataFrame,
    event_ts_col: str,
    dim_table: str,
    fact_key_col: str,
    dim_key_col: str,
    sk_col: str,
) -> DataFrame:
    """Join facts to dimensions point-in-time (as-of event_ts_col)."""
    dim = read_table("gold", dim_table).select(
        F.col(dim_key_col).alias("_dim_bk"), sk_col, "eff_start_ts", "eff_end_ts"
    )
    joined = (
        fact_df.join(
            dim,
            on=(fact_df[fact_key_col] == dim["_dim_bk"])
            & (dim["eff_start_ts"] <= fact_df[event_ts_col])
            & (dim["eff_end_ts"].isNull() | (fact_df[event_ts_col] < dim["eff_end_ts"])),
            how="left",
        )
        .drop("_dim_bk", "eff_start_ts", "eff_end_ts")
    )
    return joined


def build_fact_orders_table() -> str:
    orders = read_table("silver", "orders")
    channel_dim = read_table("gold", "dim_channel")

    b = orders.withColumn("created_ts", F.to_timestamp("created_ts"))
    b = _scd2_asof_join(b, "created_ts", "dim_buyer", "external_buyer_id", "buyer_id", "buyer_sk")
    b = _scd2_asof_join(b, "created_ts", "dim_vessel", "external_vessel_id", "vessel_id", "vessel_sk")

    b = b.join(channel_dim, on=["sales_channel", "channel_group"], how="left")

    w = Window.orderBy("external_order_id")
    fact = (
        b.withColumn("order_sk", F.row_number().over(w).cast("long"))
        .withColumn("date_created_sk", _date_sk("created_ts"))
        .withColumn("date_confirmed_sk", _date_sk("confirmed_ts"))
        .withColumn("date_cancelled_sk", _date_sk("cancelled_ts"))
        .withColumn("build_start_date_sk", _date_sk("build_start"))
        .withColumn("build_end_date_sk", _date_sk("build_end"))
        .withColumn("is_cancelled", F.col("cancelled_ts").isNotNull())
        .withColumn("build_days", F.datediff(F.to_date("build_end"), F.to_date("build_start")))
        .withColumn("margin_eur", F.round(F.col("total_price") - F.col("total_cost"), 2))
        .select(
            "order_sk",
            F.col("external_order_id").alias("order_id"),
            "order_reference",
            "buyer_sk",
            "vessel_sk",
            "channel_sk",
            "date_created_sk",
            "date_confirmed_sk",
            "date_cancelled_sk",
            "build_start_date_sk",
            "build_end_date_sk",
            "source_system",
            "brand",
            "spec_package",
            "is_owner_order",
            "is_cancelled",
            "cancellation_reason",
            "build_days",
            "crew_officers",
            "crew_ratings",
            "crew_cadets",
            F.round(F.col("total_price"), 2).alias("total_price_eur"),
            F.round(F.col("total_cost"), 2).alias("total_cost_eur"),
            "margin_eur",
            F.round(F.col("finance_fee"), 2).alias("finance_fee_eur"),
            F.round(F.col("order_fee"), 2).alias("order_fee_eur"),
            F.round(F.col("cancellation_fee"), 2).alias("cancellation_fee_eur"),
            F.current_timestamp().alias("load_ts"),
        )
    )
    return write_full_overwrite(fact, "gold", "fact_orders", partition_by=["date_created_sk"])


def build_fact_order_lines_table() -> str:
    comps = read_table("silver", "order_lines")
    subcontractor_dim = read_table("gold", "dim_subcontractor").select(
        F.col("subcontractor_id").alias("external_subcontractor_id"), "subcontractor_sk"
    )

    c = comps.withColumn("created_ts", F.to_timestamp("created_ts"))
    c = c.join(subcontractor_dim, on="external_subcontractor_id", how="left")
    c = _scd2_asof_join(c, "created_ts", "dim_vessel", "external_vessel_id", "vessel_id", "vessel_sk")

    w = Window.orderBy("line_reference")
    fact = (
        c.withColumn("line_sk", F.row_number().over(w).cast("long"))
        .withColumn("start_date_sk", _date_sk("start_date"))
        .withColumn("end_date_sk", _date_sk("end_date"))
        .withColumn("date_created_sk", _date_sk("created_ts"))
        .withColumn("date_cancelled_sk", _date_sk("cancelled_ts"))
        .withColumn("is_cancelled", F.col("status") == "Cancelled")
        .withColumn("price_eur", F.round(F.col("price") * F.col("price_fx_rate"), 2))
        .withColumn("cost_eur", F.round(F.col("cost") * F.col("cost_fx_rate"), 2))
        .withColumn("margin_eur", F.round(F.col("price_eur") - F.col("cost_eur"), 2))
        .select(
            "line_sk",
            "line_reference",
            F.col("external_order_id").alias("order_id"),
            "subcontractor_sk",
            "vessel_sk",
            "line_type",
            "line_name",
            "status",
            "is_cancelled",
            "start_date_sk",
            "end_date_sk",
            "date_created_sk",
            "date_cancelled_sk",
            "duration_days",
            F.col("sequence").alias("sequence_no"),
            F.round(F.col("price"), 2).alias("price_original"),
            "price_currency",
            "price_fx_rate",
            "price_eur",
            F.round(F.col("cost"), 2).alias("cost_original"),
            "cost_currency",
            "cost_fx_rate",
            "cost_eur",
            "margin_eur",
            "crew_officers",
            "crew_ratings",
            "crew_cadets",
            F.current_timestamp().alias("load_ts"),
        )
    )
    return write_full_overwrite(fact, "gold", "fact_order_lines", partition_by=["line_type"])


def build_fact_build_slots_table() -> str:
    av = read_table("silver", "build_slots")
    av = av.withColumn("slot_date", F.to_timestamp("slot_date"))
    av = _scd2_asof_join(av, "slot_date", "dim_vessel", "external_vessel_id", "vessel_id", "vessel_sk")

    fact = (
        av.withColumn("date_sk", _date_sk("slot_date"))
        .withColumn(
            "utilization_rate",
            F.when(F.col("allocation") > 0, F.round((F.col("allocation") - F.col("actual_available")) / F.col("allocation"), 4)),
        )
        .select(
            "vessel_sk",
            "date_sk",
            "is_closed",
            "allocation",
            "booked_by_buyer",
            "booked_by_yard",
            "actual_available",
            "potential_available",
            "utilization_rate",
            F.current_timestamp().alias("load_ts"),
        )
    )
    return write_full_overwrite(fact, "gold", "fact_build_slots", partition_by=["date_sk"])


def build_all() -> Dict[str, Any]:
    results = {}
    results["dim_date"] = build_dim_date_table()
    results["dim_channel"] = build_dim_channel_table()
    results["dim_subcontractor"] = build_dim_subcontractor_table()
    results["dim_buyer"] = build_dim_buyer_table()
    results["dim_vessel"] = build_dim_vessel_table()
    results["fact_orders"] = build_fact_orders_table()
    results["fact_order_lines"] = build_fact_order_lines_table()
    results["fact_build_slots"] = build_fact_build_slots_table()

    counts = {}
    for t in results:
        counts[t] = read_table("gold", t).count()
    results["row_counts"] = counts
    return results
