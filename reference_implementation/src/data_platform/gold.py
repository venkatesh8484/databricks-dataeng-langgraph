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
    bookings = read_table("silver", "bookings")
    combos = bookings.select("booking_channel", "channel_group").distinct()
    w = Window.orderBy("booking_channel", "channel_group")
    dim = combos.withColumn("channel_sk", F.row_number().over(w).cast("long"))
    dim = dim.select("channel_sk", "booking_channel", "channel_group")
    return write_full_overwrite(dim, "gold", "dim_channel")


def build_dim_supplier_table() -> str:
    """SCD Type 1 - Overwrite dimension table with current data."""
    suppliers = read_table("silver", "suppliers")
    w = Window.orderBy("external_supplier_id")
    dim = (
        suppliers.withColumn("supplier_sk", F.row_number().over(w).cast("long"))
        .select(
            "supplier_sk",
            F.col("external_supplier_id").alias("supplier_id"),
            "supplier_ref",
            "source_system",
            "abta_code",
            F.col("name").alias("supplier_name"),
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
    return write_full_overwrite(dim, "gold", "dim_supplier")


def build_dim_customer_table() -> str:
    """SCD Type 2 Customer dimension loaded via spark_utils.scd2_merge."""
    customers = read_table("silver", "customers")
    tracked_cols = [
        "customer_ref", "source_system", "email", "title", "forename", "surname",
        "telephone", "mobile", "address1", "address2", "city", "postcode", "country",
        "marketing_optin",
    ]
    incoming = (
        customers.withColumn("full_name", F.concat_ws(" ", F.col("forename"), F.col("surname")))
        .withColumnRenamed("external_customer_id", "customer_id")
        .withColumn("_load_ts", F.col("_silver_load_ts"))
    )
    return scd2_merge(
        incoming,
        schema="gold",
        table="dim_customer",
        business_key="customer_id",
        tracked_cols=tracked_cols + ["full_name"],
        surrogate_key_col="customer_sk",
    )


def build_dim_accommodation_table() -> str:
    """SCD Type 2 Accommodation dimension loaded via spark_utils.scd2_merge."""
    acc = read_table("silver", "accommodations")
    tracked_cols = [
        "accommodation_ref", "source_system", "accommodation_type", "brand", "name",
        "address1", "city", "postcode", "resort", "area", "region", "country",
        "miles_from_sea", "commission_rate", "price_band", "price_band_start",
        "has_pool", "pets_allowed", "bedrooms", "max_occupancy", "rating",
        "contract_type", "withdrawn_date",
    ]
    incoming = (
        acc.withColumnRenamed("external_accommodation_id", "accommodation_id")
        .withColumnRenamed("name", "accommodation_name")
        .withColumn("price_band_start_date", F.to_date("price_band_start"))
        .withColumn("is_withdrawn", F.col("withdrawn_date").isNotNull())
        .withColumn("date_created", F.to_date("date_created"))
        .withColumn("date_live", F.to_date("date_live"))
        .withColumn("withdrawn_date", F.to_date("withdrawn_date"))
        .withColumn("_load_ts", F.col("_silver_load_ts"))
    )
    tracked_final = [c if c != "name" else "accommodation_name" for c in tracked_cols]
    tracked_final = [c if c != "price_band_start" else "price_band_start_date" for c in tracked_final]
    return scd2_merge(
        incoming,
        schema="gold",
        table="dim_accommodation",
        business_key="accommodation_id",
        tracked_cols=tracked_final,
        surrogate_key_col="accommodation_sk",
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


def build_fact_bookings_table() -> str:
    bookings = read_table("silver", "bookings")
    channel_dim = read_table("gold", "dim_channel")

    b = bookings.withColumn("created_ts", F.to_timestamp("created_ts"))
    b = _scd2_asof_join(b, "created_ts", "dim_customer", "external_customer_id", "customer_id", "customer_sk")
    b = _scd2_asof_join(b, "created_ts", "dim_accommodation", "external_accommodation_id", "accommodation_id", "accommodation_sk")

    b = b.join(channel_dim, on=["booking_channel", "channel_group"], how="left")

    w = Window.orderBy("external_booking_id")
    fact = (
        b.withColumn("booking_sk", F.row_number().over(w).cast("long"))
        .withColumn("date_created_sk", _date_sk("created_ts"))
        .withColumn("date_confirmed_sk", _date_sk("confirmed_ts"))
        .withColumn("date_cancelled_sk", _date_sk("cancelled_ts"))
        .withColumn("accommodation_start_date_sk", _date_sk("accommodation_start"))
        .withColumn("accommodation_end_date_sk", _date_sk("accommodation_end"))
        .withColumn("is_cancelled", F.col("cancelled_ts").isNotNull())
        .withColumn("nights", F.datediff(F.to_date("accommodation_end"), F.to_date("accommodation_start")))
        .withColumn("margin_eur", F.round(F.col("total_price") - F.col("total_cost"), 2))
        .select(
            "booking_sk",
            F.col("external_booking_id").alias("booking_id"),
            "booking_reference",
            "customer_sk",
            "accommodation_sk",
            "channel_sk",
            "date_created_sk",
            "date_confirmed_sk",
            "date_cancelled_sk",
            "accommodation_start_date_sk",
            "accommodation_end_date_sk",
            "source_system",
            "brand",
            "brochure",
            "is_owner_booking",
            "is_cancelled",
            "cancellation_reason",
            "nights",
            "adults",
            "children",
            "infants",
            F.round(F.col("total_price"), 2).alias("total_price_eur"),
            F.round(F.col("total_cost"), 2).alias("total_cost_eur"),
            "margin_eur",
            F.round(F.col("credit_card_fee"), 2).alias("credit_card_fee_eur"),
            F.round(F.col("booking_fee"), 2).alias("booking_fee_eur"),
            F.round(F.col("cancellation_fee"), 2).alias("cancellation_fee_eur"),
            F.current_timestamp().alias("load_ts"),
        )
    )
    return write_full_overwrite(fact, "gold", "fact_bookings", partition_by=["date_created_sk"])


def build_fact_booking_components_table() -> str:
    comps = read_table("silver", "booking_components")
    supplier_dim = read_table("gold", "dim_supplier").select(
        F.col("supplier_id").alias("external_supplier_id"), "supplier_sk"
    )

    c = comps.withColumn("created_ts", F.to_timestamp("created_ts"))
    c = c.join(supplier_dim, on="external_supplier_id", how="left")
    c = _scd2_asof_join(c, "created_ts", "dim_accommodation", "external_accommodation_id", "accommodation_id", "accommodation_sk")

    w = Window.orderBy("component_reference")
    fact = (
        c.withColumn("component_sk", F.row_number().over(w).cast("long"))
        .withColumn("start_date_sk", _date_sk("start_date"))
        .withColumn("end_date_sk", _date_sk("end_date"))
        .withColumn("date_created_sk", _date_sk("created_ts"))
        .withColumn("date_cancelled_sk", _date_sk("cancelled_ts"))
        .withColumn("is_cancelled", F.col("status") == "Cancelled")
        .withColumn("price_eur", F.round(F.col("price") * F.col("price_fx_rate"), 2))
        .withColumn("cost_eur", F.round(F.col("cost") * F.col("cost_fx_rate"), 2))
        .withColumn("margin_eur", F.round(F.col("price_eur") - F.col("cost_eur"), 2))
        .select(
            "component_sk",
            "component_reference",
            F.col("external_booking_id").alias("booking_id"),
            "supplier_sk",
            "accommodation_sk",
            "component_type",
            "component_name",
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
            "adults",
            "children",
            "infants",
            F.current_timestamp().alias("load_ts"),
        )
    )
    return write_full_overwrite(fact, "gold", "fact_booking_components", partition_by=["component_type"])


def build_fact_availability_table() -> str:
    av = read_table("silver", "availability")
    av = av.withColumn("availability_date", F.to_timestamp("availability_date"))
    av = _scd2_asof_join(av, "availability_date", "dim_accommodation", "external_accommodation_id", "accommodation_id", "accommodation_sk")

    fact = (
        av.withColumn("date_sk", _date_sk("availability_date"))
        .withColumn(
            "occupancy_rate",
            F.when(F.col("allocation") > 0, F.round((F.col("allocation") - F.col("actual_available")) / F.col("allocation"), 4)),
        )
        .select(
            "accommodation_sk",
            "date_sk",
            "is_closed",
            "allocation",
            "booked_by_customer",
            "booked_by_owner",
            "actual_available",
            "potential_available",
            "occupancy_rate",
            F.current_timestamp().alias("load_ts"),
        )
    )
    return write_full_overwrite(fact, "gold", "fact_availability", partition_by=["date_sk"])


def build_all() -> Dict[str, Any]:
    results = {}
    results["dim_date"] = build_dim_date_table()
    results["dim_channel"] = build_dim_channel_table()
    results["dim_supplier"] = build_dim_supplier_table()
    results["dim_customer"] = build_dim_customer_table()
    results["dim_accommodation"] = build_dim_accommodation_table()
    results["fact_bookings"] = build_fact_bookings_table()
    results["fact_booking_components"] = build_fact_booking_components_table()
    results["fact_availability"] = build_fact_availability_table()

    counts = {}
    for t in results:
        counts[t] = read_table("gold", t).count()
    results["row_counts"] = counts
    return results
