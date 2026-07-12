"""
agents.py
=========
Implements the LangGraph agent node functions.
Connects to Databricks Model Serving endpoints using ChatDatabricks,
handles prompt templating, JSON parsing, and execution of generated Spark ETL code.
"""
from __future__ import annotations

import os
import sys
import json
import yaml
import subprocess
from typing import Dict, Any, List

from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage

from dbricks_lang_agent.data_platform.spark_utils import load_config, get_spark
from dbricks_lang_agent.data_platform import profiling
from dbricks_lang_agent.orchestrator import memory
from .state import AgentState
from . import prompts

# Simple Mock LLM for local test runs in sandbox where Databricks endpoints don't exist
class MockChatModel:
    def __init__(self, node_name: str):
        self.node_name = node_name

    def invoke(self, messages: List[Any]) -> Any:
        class MockResponse:
            def __init__(self, content: str):
                self.content = content
        
        # Determine node name and return dummy but valid structured outputs
        if self.node_name == "dq":
            return MockResponse("# Mock Data Quality Report\nNo critical anomalies found. Schema profiles look stable.")
        elif self.node_name == "contracts":
            contracts_data = {
                "contracts": {
                    "dummy_table": (
                        "table: \"dummy_table\"\n"
                        "business_key: \"id\"\n"
                        "description: \"Mock Table\"\n"
                        "rules:\n"
                        "  not_null:\n"
                        "    severity: \"hard\"\n"
                        "    columns: [\"id\"]\n"
                        "  unique:\n"
                        "    severity: \"hard\"\n"
                        "    columns: [\"id\"]\n"
                    )
                }
            }
            return MockResponse(json.dumps(contracts_data))
        elif self.node_name == "modeling":
            modeling_data = {
                "gold_ddl": "CREATE TABLE IF NOT EXISTS gold.fact_dummy (id LONG);",
                "data_dictionary": "# Mock Data Dictionary\nFact table of dummy items."
            }
            return MockResponse(json.dumps(modeling_data))
        elif self.node_name == "engineering":
            eng_data = {
                "bronze_code": (
                    "from dbricks_lang_agent.data_platform.spark_utils import get_spark, write_full_overwrite\n"
                    "if __name__ == '__main__':\n"
                    "    spark = get_spark()\n"
                    "    df = spark.createDataFrame([(1, 'A'), (2, 'B')], ['id', 'name'])\n"
                    "    write_full_overwrite(df, 'bronze', 'dummy_table')\n"
                    "    print('{\"dummy_table\": {\"row_count\": 2}}')\n"
                ),
                "silver_code": (
                    "from dbricks_lang_agent.data_platform.spark_utils import get_spark, read_table, write_full_overwrite\n"
                    "import json\n"
                    "if __name__ == '__main__':\n"
                    "    spark = get_spark()\n"
                    "    df = read_table('bronze', 'dummy_table')\n"
                    "    write_full_overwrite(df, 'silver', 'dummy_table')\n"
                    "    quarantine = spark.createDataFrame([], df.schema)\n"
                    "    write_full_overwrite(quarantine, 'quarantine', 'dummy_table')\n"
                    "    summary = {\n"
                    "        'tables': {\n"
                    "            'dummy_table': {\n"
                    "                'row_count_in': 2, 'row_count_promoted': 2,\n"
                    "                'row_count_quarantined': 0, 'promotion_blocked': False\n"
                    "            }\n"
                    "        },\n"
                    "        'halted_at': None\n"
                    "    }\n"
                    "    with open('/tmp/silver_summary.json', 'w') as f:\n"
                    "        json.dump(summary, f)\n"
                ),
                "gold_code": (
                    "from dbricks_lang_agent.data_platform.spark_utils import get_spark, read_table, write_full_overwrite\n"
                    "if __name__ == '__main__':\n"
                    "    spark = get_spark()\n"
                    "    df = read_table('silver', 'dummy_table')\n"
                    "    write_full_overwrite(df, 'gold', 'fact_dummy')\n"
                    "    print('Gold load finished!')\n"
                )
            }
            return MockResponse(json.dumps(eng_data))
        elif self.node_name == "product_advisor":
            products_data = {
                "products": [
                    {
                        "id": "customer_360",
                        "name": "Customer 360",
                        "description": "Repeat-booking and marketing segmentation view for CRM/email campaigns.",
                        "product_type": "view",
                        "source_tables": ["dim_customer", "fact_bookings"],
                        "grain": "One row per current customer.",
                        "refresh_frequency": "real-time (view)",
                        "sql": (
                            "SELECT c.customer_sk, c.customer_id, c.full_name, c.email, c.marketing_optin, "
                            "COUNT(b.booking_sk) AS total_bookings, SUM(b.total_price_eur) AS lifetime_value_eur "
                            "FROM dim_customer c LEFT JOIN fact_bookings b ON b.customer_sk = c.customer_sk "
                            "WHERE c.is_current = true "
                            "GROUP BY c.customer_sk, c.customer_id, c.full_name, c.email, c.marketing_optin"
                        ),
                    },
                    {
                        "id": "booking_profitability",
                        "name": "Booking Profitability Mart",
                        "description": "Margin and revenue by brand/channel/time for exec dashboards and finance close.",
                        "product_type": "table",
                        "source_tables": ["fact_bookings", "dim_channel", "dim_date"],
                        "grain": "One row per booking.",
                        "refresh_frequency": "daily",
                        "sql": (
                            "SELECT b.booking_id, b.brand, ch.channel_group, b.total_price_eur, b.total_cost_eur, b.margin_eur "
                            "FROM fact_bookings b LEFT JOIN dim_channel ch ON ch.channel_sk = b.channel_sk"
                        ),
                    },
                ]
            }
            return MockResponse(json.dumps(products_data))
        else:
            return MockResponse("# Mock Orchestrator Report\nPipeline execution finished successfully.")


def get_llm(node_name: str) -> Any:
    """Fetch the Databricks Model Serving LLM or fallback to mock class."""
    if os.environ.get("USE_MOCK_LLM", "false").lower() == "true":
        return MockChatModel(node_name)

    cfg = load_config()
    endpoint = cfg.get("llm", {}).get("endpoint", "databricks-meta-llama-3-1-70b-instruct")
    temperature = cfg.get("llm", {}).get("temperature", 0.0)
    # Without an explicit ceiling, long combined-JSON responses (engineering_node
    # generating bronze+silver+gold in one call is the worst case) can get
    # silently truncated by whatever lower default the endpoint falls back to —
    # producing an incomplete JSON response that then fails to parse. See
    # config.yaml's llm.max_tokens comment for the full explanation.
    max_tokens = cfg.get("llm", {}).get("max_tokens", 8192)

    try:
        # Databricks model serving integration in langchain
        return ChatDatabricks(
            endpoint=endpoint,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"Warning: Failed to load ChatDatabricks endpoint. Falling back to MockChatModel. Error: {e}")
        return MockChatModel(node_name)


def sanitize_json_string(s: str) -> str:
    """Escape raw newlines, tabs, and carriage returns inside double-quoted string values."""
    chars = []
    in_string = False
    escaped = False
    for char in s:
        if char == '"' and not escaped:
            in_string = not in_string
            chars.append(char)
        elif char == '\\' and in_string:
            escaped = not escaped
            chars.append(char)
        else:
            if char == '\n' and in_string:
                chars.append('\\n')
            elif char == '\r' and in_string:
                chars.append('\\r')
            elif char == '\t' and in_string:
                chars.append('\\t')
            else:
                chars.append(char)
            escaped = False
    return "".join(chars)

def parse_json_from_response(content: str, fallback_key: str = None) -> dict:
    """Safely extract and parse JSON block from model response.

    Some models ignore the "return JSON" instruction under a fix/retry prompt and
    instead respond with free-form prose (e.g. a "step-by-step analysis") plus a
    fenced code block containing the actual fix. If every JSON-parsing strategy
    below fails and `fallback_key` is supplied, we make one last attempt to pull
    the first fenced code block out of the response and return it under that key,
    rather than raising and losing the fix entirely.

    IMPORTANT implementation note: the values in this JSON are entire PySpark
    scripts, which are virtually guaranteed to contain their own `{`/`}`
    characters (f-strings, dict literals, `json.dump({...})`, etc.). A naive
    `\\{.*?\\}` / `\\{.*\\}` regex match — the previous approach — gets fooled by
    those embedded braces and either matches too little (non-greedy stops at
    the first `}` it finds, which is usually INSIDE the code, not the JSON
    object's real end) or, in rarer cases, too much. `json.JSONDecoder.raw_decode`
    parses using the actual JSON grammar starting from the first `{`, so braces
    inside string values can never confuse it — it always finds the JSON
    object's true end regardless of what's nested inside the string content.
    """
    import re
    content_str = content.strip()

    def _raw_decode_from_first_brace(text: str):
        start = text.find("{")
        if start == -1:
            return None
        try:
            obj, _end_idx = json.JSONDecoder().raw_decode(text, start)
            return obj
        except Exception as e:
            print(f"[parse_json_from_response] raw_decode attempt failed: {e}")
            return None

    # 1. Try the content of a fenced code block, wherever it appears in the
    # response (models often prefix the fence with prose like "Here are the
    # generated scripts as a JSON object:" — search anywhere, not just at the
    # very start of the response).
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)```", content_str, re.DOTALL)
    if code_block_match:
        sanitized = sanitize_json_string(code_block_match.group(1).strip())
        result = _raw_decode_from_first_brace(sanitized)
        if result is not None:
            return result

    # 2. Try the whole response directly — covers responses with no fence at
    # all, or where the fence regex above didn't match cleanly.
    sanitized_full = sanitize_json_string(content_str)
    result = _raw_decode_from_first_brace(sanitized_full)
    if result is not None:
        return result

    # 3. Last resort: the model may have ignored the JSON-output instruction
    # entirely and replied with prose + a fenced code block instead (common
    # with reasoning-style responses like "Step-by-step analysis... Fixed
    # solution: ```python"). Recover the code rather than discarding a valid
    # fix over a format miss. Only usable when the caller expects a single
    # script back (fallback_key) — a combined multi-script response with no
    # valid JSON structure can't be safely split this way.
    if fallback_key:
        fence_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", content_str, re.DOTALL)
        if fence_match:
            print(
                f"[parse_json_from_response] WARNING: Response was not valid JSON "
                f"(model likely ignored JSON-output instructions). Recovered code "
                f"from a fenced code block instead and mapped it to '{fallback_key}'."
            )
            return {fallback_key: fence_match.group(1).strip()}

    raise ValueError(
        f"Failed to parse JSON from LLM response.\n"
        f"Response content length: {len(content_str)}\n"
        f"Response content snippet (first 1000 chars):\n"
        f"{content_str[:1000]}"
    )


# ---- Node Functions ----

def _is_reset_requested() -> bool:
    """Shared check for the notebook's 'reset_pipeline' dbutils widget. When
    set, every caching node below skips its cache lookup and generates fresh,
    mirroring a full pipeline reset."""
    try:
        dbutils = globals().get("dbutils")
        if dbutils:
            return dbutils.widgets.get("reset_pipeline").lower() == "true"
    except Exception:
        pass
    return False


def profiler_node(state: AgentState) -> Dict[str, Any]:
    """Nodes runs PySpark profiling on raw files and summaries findings."""
    print(">>> [Profiler Agent] Discovering and profiling raw source files...")

    # Run the dynamic data platform profiler
    reports_dir = "/tmp/reports"
    os.makedirs(reports_dir, exist_ok=True)
    report = profiling.profile_all_sources(output_path=os.path.join(reports_dir, "profiling_report.json"))

    discovered = report.get("discovered_tables", {})

    # GUARD: No tables discovered — return diagnostic error WITHOUT calling the LLM.
    # Calling the LLM with empty data causes hallucination (it invents tables/columns).
    if not discovered:
        diagnostics = report.get("discovery_diagnostics", [])
        diag_trail = "\n".join(f"  - {line}" for line in diagnostics) or "  (no diagnostics captured)"
        error_msg = (
            "Profiler BLOCKED: No source tables discovered.\n"
            "Live discovery trail from this run:\n"
            f"{diag_trail}\n"
            "Common causes: (1) volume_raw_path in config.yaml doesn't match the actual UC Volume path. "
            "(2) The app/notebook's identity lacks READ FILES / browse privilege on the volume — "
            "check the Permissions tab on the volume in Catalog Explorer. "
            "(3) DBUtils and SDK listing both raised exceptions — see trail above for the exact error. "
            "(4) If running locally, set SOURCE_ROOT to a directory with .csv files."
        )
        print(f"[Profiler] ABORT: {error_msg}")
        return {
            "discovered_tables": {},
            "profiling_report": {
                "error": error_msg,
                "profiler_narration": "",
                "discovered_tables": {},
                "discovery_diagnostics": diagnostics,
            },
            "profiler_error": error_msg,
            "active_agent": "Profiler",
            "review_comments": "",
            "loop_count": (state.get("loop_count") or 0) + 1,
        }

    unique_keys = report.get("candidate_unique_keys", {})
    duplicate_keys = report.get("duplicate_keys", {})
    ri = report.get("referential_integrity", [])
    tables_profile = report.get("tables", {})

    # Build condensed report including per-column stats (with token budget cap per table)
    tables_summary = {}
    for tbl, profile in tables_profile.items():
        col_summary = {}
        for col_name, col_stat in profile.get("columns", {}).items():
            col_summary[col_name] = {
                "dtype": col_stat.get("dtype"),
                "null_pct": col_stat.get("null_pct"),
                "distinct_count": col_stat.get("distinct_count"),
            }
            if col_stat.get("numeric_stats"):
                col_summary[col_name]["numeric_stats"] = col_stat["numeric_stats"]
            if col_stat.get("value_counts"):
                # Top 5 values only to keep token budget lean
                top5 = dict(list(col_stat["value_counts"].items())[:5])
                col_summary[col_name]["top_values"] = top5
        tables_summary[tbl] = {
            "row_count": profile.get("row_count"),
            "column_count": profile.get("column_count"),
            "columns": col_summary,
        }

    condensed = {
        "discovered_tables": discovered,
        "candidate_primary_keys": unique_keys,
        "duplicate_key_counts": duplicate_keys,
        "referential_integrity": ri,
        "tables": tables_summary,
    }

    # Format message for LLM profiling review
    prompt = f"Raw dataset profiling metrics summary:\n{json.dumps(condensed, indent=2)}\n\nAnalyze and summarize these findings."
    if state.get("review_comments"):
        prompt += f"\n\nHuman feedback on previous profiling report:\n{state['review_comments']}"

    messages = [
        SystemMessage(content=prompts.PROFILER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    llm = get_llm("profiler")
    response = llm.invoke(messages)

    # Save findings narration in state
    report["profiler_narration"] = response.content

    return {
        "discovered_tables": discovered,
        "profiling_report": report,
        "profiler_error": "",
        "active_agent": "Profiler",
        "review_comments": "",
        "loop_count": 0,
    }


def dq_node(state: AgentState) -> Dict[str, Any]:
    """Data Quality Agent Node: Checks for logical and physical data anomalies."""
    print(">>> [Data Quality Agent] Assessing raw source data quality...")

    # GUARD: Cannot run DQ checks without discovered tables — prevents LLM hallucination
    if not state.get("discovered_tables"):
        upstream_error = state.get("profiler_error") or "No tables were discovered by the Profiler."
        blocked_msg = (
            f"\u26a0\ufe0f DQ Agent blocked: {upstream_error}\n\n"
            "**Action required**: Fix the upstream Profiler failure (check Volume path, DBUtils access, "
            "CSV file existence) and re-run the Profiler step before DQ assessment can proceed."
        )
        print(f"[DQ Agent] ABORT: Cannot assess quality — no tables in state.")
        return {
            "dq_report": blocked_msg,
            "active_agent": "DataQualityAgent",
            "review_comments": "",
        }

    # Query few-shot memory database
    spark = get_spark()
    memory.init_memory_table(spark)
    memory.init_dq_cache_table(spark)
    dataset = list(state.get("discovered_tables", {}).keys())[0] if state.get("discovered_tables") else "generic"
    few_shot_context = memory.get_few_shot_context(spark, dataset, "data_quality")

    profiling_report = state.get("profiling_report", {})
    profiling_narration = profiling_report.get("profiler_narration", "")
    profiling_metrics = {k: v for k, v in profiling_report.items() if k != "profiler_narration"}

    # --- Generate-once-per-schema cache ---
    # Only Profiler is expected to differ run-to-run (new row-level data).
    # DQ, Contracts, Modeling and Engineering all key off the same structural
    # schema fingerprint and are reused verbatim across "daily delta" runs
    # where the columns haven't changed — see get_schema_fingerprint().
    schema_fingerprint = get_schema_fingerprint(spark)
    skip_cache = _is_reset_requested() or bool(state.get("review_comments"))

    dq_report = None
    generation_source = "llm_fresh"

    if not skip_cache:
        cached_report = memory.get_dq_cache(spark, schema_fingerprint)
        if cached_report:
            dq_report = cached_report
            generation_source = "cache_reused"
            print(f"[DQ Agent] Exact schema-fingerprint cache hit — reusing prior DQ report, skipping LLM call.")

    if dq_report is None:
        latest_cache = None if skip_cache else memory.get_latest_dq_cache(spark)
        prompt = (
            f"Data Profiler Report Narrative:\n{profiling_narration}\n\n"
            f"Raw Metrics JSON:\n{json.dumps(profiling_metrics, indent=2)}\n\n"
            f"{few_shot_context}"
        )
        if state.get("review_comments"):
            prompt += f"\n\nHuman feedback on previous DQ assessment:\n{state['review_comments']}"
        elif latest_cache and latest_cache.get("dq_report"):
            # Schema changed since the last cached run — ask the LLM to UPDATE
            # the existing report rather than write a new one from scratch.
            generation_source = "llm_patched"
            prompt += (
                f"\n\nIMPORTANT: The source schema has changed since the last time this dataset was assessed. "
                f"Here is the PREVIOUS Data Quality report:\n```\n{latest_cache['dq_report']}\n```\n"
                f"Update this previous report to reflect the current schema/metrics above — preserve findings "
                f"that are still valid, remove/revise findings about columns or tables that no longer exist, "
                f"and add findings for anything new. Do not discard prior context that's still accurate."
            )

        messages = [
            SystemMessage(content=prompts.DQ_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        llm = get_llm("dq")
        response = llm.invoke(messages)
        dq_report = response.content

        try:
            memory.upsert_dq_cache(spark, schema_fingerprint, dq_report)
        except Exception as e_cache:
            print(f"[Warning] Failed to cache DQ report: {e_cache}")

    approved_steps = dict(state.get("approved_steps", {}))
    if generation_source == "cache_reused" and memory.was_previously_approved(spark, "dq", schema_fingerprint):
        approved_steps["dq"] = True
        try:
            memory.log_stage_review(
                spark,
                pipeline_run_id=state.get("pipeline_run_id", ""),
                stage_key="dq",
                agent_name="DataQualityAgent",
                decision="approved",
                reviewer_comments="Auto-approved — schema unchanged, identical output previously approved.",
                output={"dq_report": dq_report},
                dataset_fingerprint=schema_fingerprint,
            )
        except Exception as e_log:
            print(f"[Warning] Failed to log auto-approval for DQ: {e_log}")

    return {
        "dq_report": dq_report,
        "active_agent": "DataQualityAgent",
        "review_comments": "",
        "approved_steps": approved_steps,
        "generation_source": {**(state.get("generation_source") or {}), "dq": generation_source},
    }


def contract_node(state: AgentState) -> Dict[str, Any]:
    """Steward designs and authors YAML schema data contracts."""
    print(">>> [Contract Steward] Authoring YAML schema data contracts...")

    discovered = state.get("discovered_tables", {})

    # GUARD: No tables discovered — return diagnostic error WITHOUT calling the LLM.
    # Without this guard, the LLM receives empty input and hallucinates contracts
    # (e.g., the infamous 'investigation' table from human feedback text).
    if not discovered:
        upstream_error = state.get("profiler_error") or "No tables were discovered by the Profiler."
        review_comments = state.get("review_comments", "")
        diagnostic = (
            f"ContractSteward BLOCKED: Cannot author contracts — no tables in discovered_tables.\n"
            f"Root cause: {upstream_error}\n"
            f"Human feedback received: '{review_comments}'\n"
            f"Action required: Fix the upstream Profiler failure and re-run the Profiler step."
        )
        print(f"[Contract Steward] ABORT: {diagnostic}")
        return {
            "contracts": {},
            "contracts_error": diagnostic,
            "active_agent": "ContractSteward",
            "review_comments": "",
        }

    # Query few-shot memory database
    spark = get_spark()
    memory.init_contracts_cache_table(spark)
    dataset = list(discovered.keys())[0] if discovered else "generic"
    few_shot_context = memory.get_few_shot_context(spark, dataset, "contracts")

    profiling_report = state.get("profiling_report", {})
    profiling_narration = profiling_report.get("profiler_narration", "")
    profiling_metrics = {k: v for k, v in profiling_report.items() if k != "profiler_narration"}

    # Build column schema section — gives LLM exact column names/types to prevent hallucination
    tables_profile = profiling_report.get("tables", {})
    column_schema_lines = []
    for tbl, profile in tables_profile.items():
        column_schema_lines.append(f"\nTable: {tbl}")
        for col_name, col_stat in profile.get("columns", {}).items():
            null_pct = col_stat.get("null_pct", 0)
            dtype = col_stat.get("dtype", "unknown")
            distinct = col_stat.get("distinct_count", "N/A")
            column_schema_lines.append(f"  - {col_name} ({dtype}) null_pct={null_pct}% distinct={distinct}")
    column_schema_str = "\n".join(column_schema_lines) if column_schema_lines else "No column schema available."

    # --- Generate-once-per-schema cache (see dq_node for the full rationale) ---
    schema_fingerprint = get_schema_fingerprint(spark)
    skip_cache = _is_reset_requested() or bool(state.get("review_comments"))

    valid_contracts = None
    generation_source = "llm_fresh"

    if not skip_cache:
        cached_contracts = memory.get_contracts_cache(spark, schema_fingerprint)
        if cached_contracts:
            valid_contracts = cached_contracts
            generation_source = "cache_reused"
            print(f"[Contract Steward] Exact schema-fingerprint cache hit — reusing {len(valid_contracts)} prior contract(s), skipping LLM call.")

    if valid_contracts is None:
        latest_cache = None if skip_cache else memory.get_latest_contracts_cache(spark)
        prompt = (
            f"Discovered Tables: {list(discovered.keys())}\n\n"
            f"Column Schema (use ONLY these exact column names in contracts):\n{column_schema_str}\n\n"
            f"Data Profiling Report:\n{profiling_narration}\n\n"
            f"Data Quality Assessment Report:\n{state.get('dq_report', '')}\n\n"
            f"Raw Metrics:\n{json.dumps({k: v for k, v in profiling_metrics.items() if k != 'tables'}, indent=2)}\n\n"
            f"{few_shot_context}"
        )
        if state.get("review_comments"):
            prompt += f"\n\nHuman feedback on previous contracts draft:\n{state['review_comments']}"
        elif latest_cache and latest_cache.get("contracts"):
            generation_source = "llm_patched"
            prior_contracts_str = "\n".join(
                f"--- PREVIOUS CONTRACT FOR {tbl} ---\n{yaml_str}" for tbl, yaml_str in latest_cache["contracts"].items()
            )
            prompt += (
                f"\n\nIMPORTANT: The source schema has changed since these contracts were last authored. "
                f"Here are the PREVIOUS contracts:\n{prior_contracts_str}\n\n"
                f"Update these contracts to match the current column schema above — preserve rules for "
                f"tables/columns that are unchanged, remove/revise rules for anything that no longer exists, "
                f"and add rules for any new tables/columns. Do not discard prior governance decisions that "
                f"still apply."
            )

        messages = [
            SystemMessage(content=prompts.CONTRACT_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        llm = get_llm("contracts")
        response = llm.invoke(messages)

        parsed = parse_json_from_response(response.content)
        contracts = parsed.get("contracts", {})

        # Write the YAML contracts to disk — validate each one first to prevent downstream crashes
        contracts_dir = "/tmp/generated/config/contracts"
        os.makedirs(contracts_dir, exist_ok=True)
        valid_contracts = {}
        for table, yaml_str in contracts.items():
            try:
                import yaml as _yaml
                parsed_yaml = _yaml.safe_load(yaml_str)
                if not isinstance(parsed_yaml, dict) or "table" not in parsed_yaml:
                    print(f"[Contract Steward] WARNING: YAML for '{table}' is missing 'table' key — skipping.")
                    continue
                if parsed_yaml.get("table") != table:
                    print(f"[Contract Steward] WARNING: YAML table name '{parsed_yaml.get('table')}' != dict key '{table}' — correcting.")
                    parsed_yaml["table"] = table
                with open(os.path.join(contracts_dir, f"{table}.yaml"), "w") as f:
                    f.write(yaml_str)
                valid_contracts[table] = yaml_str
            except Exception as e_yaml:
                print(f"[Contract Steward] WARNING: Invalid YAML for '{table}': {e_yaml} — skipping.")

        if valid_contracts:
            try:
                memory.upsert_contracts_cache(spark, schema_fingerprint, valid_contracts)
            except Exception as e_cache:
                print(f"[Warning] Failed to cache contracts: {e_cache}")
    else:
        # Cache hit — still (re)write the YAML files to disk since downstream
        # nodes/scripts read contracts from /tmp, not directly from state.
        contracts_dir = "/tmp/generated/config/contracts"
        os.makedirs(contracts_dir, exist_ok=True)
        for table, yaml_str in valid_contracts.items():
            try:
                with open(os.path.join(contracts_dir, f"{table}.yaml"), "w") as f:
                    f.write(yaml_str)
            except Exception as e_write:
                print(f"[Warning] Failed to write cached contract for '{table}' to disk: {e_write}")

    approved_steps = dict(state.get("approved_steps", {}))
    if generation_source == "cache_reused" and memory.was_previously_approved(spark, "contracts", schema_fingerprint):
        approved_steps["contracts"] = True
        try:
            memory.log_stage_review(
                spark,
                pipeline_run_id=state.get("pipeline_run_id", ""),
                stage_key="contracts",
                agent_name="ContractSteward",
                decision="approved",
                reviewer_comments="Auto-approved — schema unchanged, identical output previously approved.",
                output={"contracts": valid_contracts},
                dataset_fingerprint=schema_fingerprint,
            )
        except Exception as e_log:
            print(f"[Warning] Failed to log auto-approval for Contracts: {e_log}")

    return {
        "contracts": valid_contracts,
        "contracts_error": "" if valid_contracts else "No valid contracts generated.",
        "active_agent": "ContractSteward",
        "review_comments": "",
        "approved_steps": approved_steps,
        "generation_source": {**(state.get("generation_source") or {}), "contracts": generation_source},
    }


def modeling_node(state: AgentState) -> Dict[str, Any]:
    """Modeler designs Gold-layer dimensions and facts SQL DDL."""
    print(">>> [Dimensional Modeler] Designing Kimball star schema DDL...")

    contracts = state.get("contracts", {})

    # GUARD: Cannot design a model without valid contracts — prevents hallucinated DDL
    if not contracts:
        contracts_error = state.get("contracts_error", "No contracts were generated by ContractSteward.")
        error_msg = (
            f"\u26a0\ufe0f Modeler blocked: {contracts_error}\n"
            "Fix upstream Profiler/Contract failures before attempting dimensional modeling."
        )
        print(f"[Modeler] ABORT: {error_msg}")
        return {
            "gold_ddl": "",
            "data_dictionary": error_msg,
            "active_agent": "DimensionalModeler",
            "review_comments": "",
        }

    profiling_narration = state.get("profiling_report", {}).get("profiler_narration", "")
    contracts_summary = ""
    for table, yaml_str in contracts.items():
        contracts_summary += f"--- CONTRACT FOR {table} ---\n{yaml_str}\n"

    # --- Generate-once-per-schema cache (see dq_node for the full rationale) ---
    spark = get_spark()
    memory.init_modeling_cache_table(spark)
    schema_fingerprint = get_schema_fingerprint(spark)
    skip_cache = _is_reset_requested() or bool(state.get("review_comments"))

    gold_ddl = None
    data_dictionary = None
    generation_source = "llm_fresh"

    if not skip_cache:
        cached_model = memory.get_modeling_cache(spark, schema_fingerprint)
        if cached_model:
            gold_ddl = cached_model.get("gold_ddl", "")
            data_dictionary = cached_model.get("data_dictionary", "")
            generation_source = "cache_reused"
            print(f"[Modeler] Exact schema-fingerprint cache hit — reusing prior Gold DDL + data dictionary, skipping LLM call.")

    if gold_ddl is None:
        latest_cache = None if skip_cache else memory.get_latest_modeling_cache(spark)
        prompt = (
            f"Profiling narrative:\n{profiling_narration}\n\n"
            f"Contracts specifications:\n{contracts_summary}"
        )
        if state.get("review_comments"):
            prompt += f"\n\nHuman feedback on previous DDL draft:\n{state['review_comments']}"
        elif latest_cache and latest_cache.get("gold_ddl"):
            generation_source = "llm_patched"
            prompt += (
                f"\n\nIMPORTANT: The contracts/schema have changed since this star schema was last designed. "
                f"Here is the PREVIOUS Gold DDL:\n```sql\n{latest_cache['gold_ddl']}\n```\n"
                f"And the PREVIOUS Data Dictionary:\n```\n{latest_cache.get('data_dictionary', '')}\n```\n"
                f"Update this existing design to match the current contracts above — keep dimensions/facts that "
                f"are still valid unchanged, add or modify only what the schema change requires. Do not "
                f"redesign the whole star schema from scratch if most of it is still correct."
            )

        messages = [
            SystemMessage(content=prompts.MODELER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        llm = get_llm("modeling")
        response = llm.invoke(messages)

        parsed = parse_json_from_response(response.content)
        gold_ddl = parsed.get("gold_ddl", "")
        data_dictionary = parsed.get("data_dictionary", "")

        if gold_ddl:
            try:
                memory.upsert_modeling_cache(spark, schema_fingerprint, gold_ddl, data_dictionary)
            except Exception as e_cache:
                print(f"[Warning] Failed to cache modeling output: {e_cache}")

    # Write SQL and Markdown DDL files (needed on disk regardless of cache hit/miss —
    # downstream engineering_node's generated scripts don't read these directly, but
    # keeping this mirrors bronze/silver/gold artifact conventions elsewhere)
    ddl_dir = "/tmp/generated/data_model"
    os.makedirs(ddl_dir, exist_ok=True)
    with open(os.path.join(ddl_dir, "gold_ddl.sql"), "w") as f:
        f.write(gold_ddl or "")
    with open(os.path.join(ddl_dir, "data_dictionary.md"), "w") as f:
        f.write(data_dictionary or "")

    approved_steps = dict(state.get("approved_steps", {}))
    if generation_source == "cache_reused" and memory.was_previously_approved(spark, "modeling", schema_fingerprint):
        approved_steps["modeling"] = True
        try:
            memory.log_stage_review(
                spark,
                pipeline_run_id=state.get("pipeline_run_id", ""),
                stage_key="modeling",
                agent_name="DimensionalModeler",
                decision="approved",
                reviewer_comments="Auto-approved — schema unchanged, identical output previously approved.",
                output={"gold_ddl": gold_ddl, "data_dictionary": data_dictionary},
                dataset_fingerprint=schema_fingerprint,
            )
        except Exception as e_log:
            print(f"[Warning] Failed to log auto-approval for Modeling: {e_log}")

    return {
        "gold_ddl": gold_ddl,
        "data_dictionary": data_dictionary,
        "active_agent": "DimensionalModeler",
        "review_comments": "",
        "approved_steps": approved_steps,
        "generation_source": {**(state.get("generation_source") or {}), "modeling": generation_source},
    }


def _sanitize_and_heal_code(code: str) -> str:
    """Auto-inject missing PySpark imports and heal line-continuation/indent issues."""
    if not code:
        return code

    # 1. Replace literal backslash-n sequences with actual newlines
    # (Since we do this first, we can clean up literal '\n' formatting artifacts)
    code = code.replace("\\\\n", "\n").replace("\\n", "\n")

    # 1.5. Fix discover_source_tables() argument signature dynamically
    import re
    code = re.sub(r"\bdiscover_source_tables\s*\([^)]*\)", "discover_source_tables()", code)

    # 1.6. Fix build_dim_date() argument signature dynamically (ensure spark/get_spark() is the first argument if only two are passed)
    code = re.sub(r"\bbuild_dim_date\s*\(\s*([^,)]+\s*,\s*[^,)]+)\s*\)", r"build_dim_date(get_spark(), \1)", code)

    # 1.7. Fix validate_table() unpacking signature dynamically (change clean_df, quarantine_df = validate_table(...) to include third discard)
    code = re.sub(r"(?m)^(\s*)([a-zA-Z0-9_]+)\s*,\s*([a-zA-Z0-9_]+)\s*=\s*validate_table\s*\(", r"\1\2, \3, _ = validate_table(", code)

    # 1.8. Fix load_source with dict key from discover_source_tables
    if "discover_source_tables" in code and "load_source" in code:
        var_match = re.search(r"(\w+)\s*=\s*discover_source_tables\(\)", code)
        if var_match:
            dict_var = var_match.group(1)
            loop_match = re.search(r"for\s+(\w+)\s+in\s+" + dict_var + r"(?:\.keys\(\))?\s*:", code)
            if loop_match:
                loop_var = loop_match.group(1)
                load_pattern = r"\bload_source\(\s*([^,]+)\s*,\s*" + loop_var + r"\s*\)"
                if re.search(load_pattern, code):
                    code = re.sub(
                        r"for\s+" + loop_var + r"\s+in\s+" + dict_var + r"(?:\.keys\(\))?\s*:",
                        f"for {loop_var}, {loop_var}_file in {dict_var}.items():",
                        code
                    )
                    code = re.sub(
                        r"\bload_source\(\s*([^,]+)\s*,\s*" + loop_var + r"\s*\)",
                        r"load_source(\1, " + f"{loop_var}_file)",
                        code
                    )

    # 1.9. Fix spark.sql.functions hallucinations
    code = re.sub(r"\bspark\.sql\.functions\.", "", code)

    # 1.10. Fix applymap call on Spark DataFrame (hallucinated Pandas applymap)
    if "applymap" in code:
        import re
        applymap_match = re.search(r"(?m)^(\s*)(\w+)\s*=\s*\2\.applymap\s*\(.*\)\s*$", code)
        if applymap_match:
            indent = applymap_match.group(1)
            df_var = applymap_match.group(2)
            replacement = (
                f"{indent}# Standardize boolean values safely (only for columns containing only boolean-like values)\n"
                f"{indent}_str_cols = [__c for __c, __t in {df_var}.dtypes if __t == 'string']\n"
                f"{indent}if _str_cols:\n"
                f"{indent}    _aggs = []\n"
                f"{indent}    for __c in _str_cols:\n"
                f"{indent}        _aggs.append(sum(when(~col(__c).isNull() & (col(__c) != '') & lower(col(__c)).isin('yes', 'no', 'y', 'n', 'true', 'false'), 1).otherwise(0)).alias(__c + '_bool'))\n"
                f"{indent}        _aggs.append(sum(when(~col(__c).isNull() & (col(__c) != '') & ~lower(col(__c)).isin('yes', 'no', 'y', 'n', 'true', 'false'), 1).otherwise(0)).alias(__c + '_non_bool'))\n"
                f"{indent}    _counts = {df_var}.select(*_aggs).collect()[0].asDict()\n"
                f"{indent}    for __c in _str_cols:\n"
                f"{indent}        _b = _counts.get(__c + '_bool') or 0\n"
                f"{indent}        _nb = _counts.get(__c + '_non_bool') or 0\n"
                f"{indent}        if _b > 0 and _nb == 0:\n"
                f"{indent}            {df_var} = {df_var}.withColumn(__c, when(lower(col(__c)).isin('yes', 'y', 'true'), lit(True)).when(lower(col(__c)).isin('no', 'n', 'false'), lit(False)).otherwise(None))"
            )
            code = re.sub(r"(?m)^(\s*)" + df_var + r"\s*=\s*" + df_var + r"\.applymap\s*\(.*\)\s*$", replacement, code)

    # 1.11. Fix hallucinated loop-based string-to-boolean mapping that is not type-safe
    if "dtypes" in code and "isin" in code:
        import re
        loop_pattern = r"(?m)^(\s*)for\s+(\w+),\s*(\w+)\s+in\s+(\w+)\.dtypes\s*:\s*\n\1\s+if\s+\3\s*==\s*['\"]string['\"]\s*:\s*\n(?:\1\s+.*\n?)+"
        loop_match = re.search(loop_pattern, code)
        if loop_match:
            indent = loop_match.group(1)
            df_var = loop_match.group(4)
            replacement = (
                f"{indent}# Standardize boolean values safely (only for columns containing only boolean-like values)\n"
                f"{indent}_str_cols = [__c for __c, __t in {df_var}.dtypes if __t == 'string']\n"
                f"{indent}if _str_cols:\n"
                f"{indent}    _aggs = []\n"
                f"{indent}    for __c in _str_cols:\n"
                f"{indent}        _aggs.append(sum(when(~col(__c).isNull() & (col(__c) != '') & lower(col(__c)).isin('yes', 'no', 'y', 'n', 'true', 'false'), 1).otherwise(0)).alias(__c + '_bool'))\n"
                f"{indent}        _aggs.append(sum(when(~col(__c).isNull() & (col(__c) != '') & ~lower(col(__c)).isin('yes', 'no', 'y', 'n', 'true', 'false'), 1).otherwise(0)).alias(__c + '_non_bool'))\n"
                f"{indent}    _counts = {df_var}.select(*_aggs).collect()[0].asDict()\n"
                f"{indent}    for __c in _str_cols:\n"
                f"{indent}        _b = _counts.get(__c + '_bool') or 0\n"
                f"{indent}        _nb = _counts.get(__c + '_non_bool') or 0\n"
                f"{indent}        if _b > 0 and _nb == 0:\n"
                f"{indent}            {df_var} = {df_var}.withColumn(__c, when(lower(col(__c)).isin('yes', 'y', 'true'), lit(True)).when(lower(col(__c)).isin('no', 'n', 'false'), lit(False)).otherwise(None))"
            )
            code = re.sub(loop_pattern, replacement, code)

    # 1.12. Fix scd2_merge return assignment (it returns table name string instead of DataFrame)
    if "scd2_merge" in code:
        import re
        scd2_pattern = r"(?m)^(\s*)(\w+)\s*=\s*scd2_merge\s*\(([^,]+),\s*(['\"][^'\"]+['\"]),\s*(['\"][^'\"]+['\"]),\s*(.*?)\)"
        if re.search(scd2_pattern, code):
            replacement = r"\1scd2_merge(\3, \4, \5, \6)\n\1\2 = read_table(\4, \5)"
            code = re.sub(scd2_pattern, replacement, code)

    # 1.13. Fix typeName() hallucination on string tuple from dtypes
    if "typeName" in code:
        import re
        code = re.sub(
            r"(\w+(?:\[\d+\])?)\.typeName\(\)\s*==\s*['\"](?:StringType|string)['\"]",
            r"\1 == 'string'",
            code
        )

    # 1.14. Fix hallucinated literal "event_ts" column reference. The gold-layer prompt
    # explicitly warns the model that `event_ts` is a placeholder name only and never a
    # real column, but the LLM sometimes emits it verbatim anyway (e.g. in point-in-time
    # SCD2 join conditions). Since we don't know the real per-fact timestamp column
    # ahead of time, infer it structurally: within each function, the real event
    # timestamp column is the one most recently cast via `to_timestamp(...)` before the
    # `event_ts` reference appears (this mirrors the pattern the prompt asks for, e.g.
    # `.withColumn("created_ts", F.to_timestamp("created_ts"))` immediately followed by
    # the point-in-time join). Replace all literal "event_ts"/'event_ts' occurrences in
    # that function with the inferred real column name.
    if re.search(r"""['"]event_ts['"]""", code):
        def_pattern = re.compile(r"^(\s*)def\s+\w+\s*\(")
        ts_assign_pattern = re.compile(r'withColumn\(\s*["\'](\w+)["\']\s*,\s*[\w.]*to_timestamp\(')
        event_ts_pattern = re.compile(r"""(['"])event_ts\1""")

        fixed_lines = []
        current_ts_col = None
        for line in code.split("\n"):
            if def_pattern.match(line):
                current_ts_col = None  # reset inference at each new function boundary
            m_ts = ts_assign_pattern.search(line)
            if m_ts:
                current_ts_col = m_ts.group(1)
            if current_ts_col and event_ts_pattern.search(line):
                line = event_ts_pattern.sub(lambda mm: mm.group(1) + current_ts_col + mm.group(1), line)
            fixed_lines.append(line)
        code = "\n".join(fixed_lines)

    # 2. Inject commonly used PySpark SQL functions if referenced but not imported
    common_funcs = [
        "current_timestamp", "lit", "col", "when", "expr", "coalesce", 
        "to_date", "to_timestamp", "trim", "concat", "substring", 
        "count", "sum", "avg", "min", "max", "year", "month", "dayofmonth", "desc", "asc", "lower"
    ]
    import re
    needed = []
    for func in common_funcs:
        if re.search(r"\b" + func + r"\s*\(", code):
            is_imported = bool(re.search(r"\bimport\s+[^#\n]*\b" + func + r"\b", code)) or \
                          bool(re.search(r"\b" + func + r"\s*=\s*", code))
            if not is_imported:
                needed.append(func)

    if needed:
        import_line = f"from pyspark.sql.functions import {', '.join(needed)}\n"
        code = import_line + code

    # 3. Self-healing compiler loop (resolves line continuation and indentation errors)
    for attempt in range(15):
        try:
            # Attempt to dry-run compile the code
            compile(code, "<string>", "exec")
            print(f"[Sanity Guard] Code compiled successfully on attempt {attempt}.")
            return code
        except SyntaxError as e:
            lines = code.splitlines()
            if e.lineno is None or e.lineno > len(lines):
                break
            err_line = lines[e.lineno - 1]

            # Case A: Unexpected character after line continuation character
            if "unexpected character after line continuation" in e.msg:
                idx = err_line.rfind(chr(92))
                if idx != -1:
                    # Strip off everything after the backslash
                    lines[e.lineno - 1] = err_line[:idx] + chr(92)
                code = "\n".join(lines)
                continue

            # Case B: Unexpected indent (missing a line continuation backslash on the previous line)
            if "unexpected indent" in e.msg:
                if e.lineno > 1:
                    prev_line = lines[e.lineno - 2]
                    if not prev_line.strip().endswith(chr(92)):
                        # Append a backslash line continuation to the previous line
                        lines[e.lineno - 2] = prev_line + " " + chr(92)
                        code = "\n".join(lines)
                        continue
            
            # If we hit any other syntax error, log it and return the code as is to avoid infinite loop
            print(f"[Sanity Guard] Unhandled SyntaxError during healing: {e.msg} at line {e.lineno}")
            break

    return code


def get_schema_fingerprint(spark) -> str:
    """The single, canonical cache key for ALL agent-output caching (DQ report,
    contracts, gold DDL/data dictionary, bronze/silver/gold code).

    Deliberately structural-ONLY: table names + column headers, nothing else.
    No LLM-generated content (contracts text, DDL text, prior reports) feeds
    into this hash, and no row-level data does either. That's what makes it
    stable across "daily delta" runs (new rows, same columns) while still
    invalidating the moment a table is added/removed or a column is
    added/removed/renamed. Every caching node in this file should key off
    THIS function — not a hash of its own LLM output — so re-running any
    single stage never causes a downstream cache miss by itself.
    """
    import hashlib
    import json
    from dbricks_lang_agent.data_platform.profiling import discover_source_tables
    from dbricks_lang_agent.data_platform.spark_utils import load_config

    cfg = load_config()
    vol_path = cfg.get("volume_raw_path", "/Volumes/databricks_langgraph/raw/source_volume")

    source_tables = discover_source_tables()
    fingerprint_data = []

    for table_name, filename in sorted(source_tables.items()):
        path = os.path.join(vol_path, filename)
        try:
            first_line = spark.read.text(path).limit(1).collect()[0][0]
        except Exception as e:
            print(f"[Warning] Failed to read header for {filename} via Spark: {e}")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                except Exception:
                    first_line = ""
            else:
                first_line = ""
        fingerprint_data.append((table_name, first_line))

    return hashlib.sha256(json.dumps(fingerprint_data).encode("utf-8")).hexdigest()


def get_dataset_fingerprint(spark, contracts: Dict[str, str] = None, gold_ddl: str = "") -> str:
    """DEPRECATED as a cache key — kept only for audit-log call sites that want
    a fingerprint reflecting "schema + governance content as of this moment"
    for traceability. Do NOT use this to key a reuse/cache-lookup decision;
    use get_schema_fingerprint() instead. This wraps it and additionally
    folds in a normalized hash of contracts/DDL content, which means it still
    changes on every cosmetic LLM re-generation — that's fine for an audit
    trail annotation, but was the actual bug when this used to be the
    codebase cache key (see git history)."""
    import hashlib
    import json

    schema_fp = get_schema_fingerprint(spark)

    # Include contracts + DDL hash so cache is invalidated when governance rules
    # actually change. IMPORTANT: hash a NORMALIZED/parsed representation, not
    # the raw LLM-generated text. ContractSteward and DimensionalModeler are
    # LLM calls — re-running them for the *same* dataset can (and does)
    # produce YAML/SQL that differs only in comments, key ordering, or
    # whitespace while describing identical rules/schema. Hashing the raw
    # text turned every such cosmetic re-generation into a "new dataset" as
    # far as the codebase cache was concerned, silently discarding proven-good
    # bronze/silver/gold code and forcing a full fresh regeneration — which is
    # exactly the "same dataset, but the script is new every time" failure
    # mode this fingerprint exists to prevent. Normalizing first means the
    # cache only invalidates when the actual contract rules or DDL structure
    # change, not when the LLM just phrases the same thing differently.
    contracts_hash = ""
    if contracts:
        normalized_contracts = {}
        for tbl, contract_yaml in contracts.items():
            try:
                normalized_contracts[tbl] = yaml.safe_load(contract_yaml)
            except Exception:
                # Unparseable — fall back to the raw string for this table only,
                # so one bad YAML doc doesn't crash fingerprinting entirely.
                normalized_contracts[tbl] = contract_yaml
        contracts_hash = hashlib.sha256(
            json.dumps(normalized_contracts, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]

    ddl_hash = ""
    if gold_ddl:
        import re
        # Strip SQL line comments and collapse all whitespace runs to a single
        # space so formatting-only differences don't change the hash.
        ddl_no_comments = re.sub(r"--[^\n]*", "", gold_ddl)
        ddl_normalized = re.sub(r"\s+", " ", ddl_no_comments).strip()
        ddl_hash = hashlib.sha256(ddl_normalized.encode("utf-8")).hexdigest()[:16]

    fingerprint_str = schema_fp + contracts_hash + ddl_hash
    return hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()


def _run_and_verify_script(script_name: str, code_str: str) -> Dict[str, Any]:
    """Write the script to disk and execute it via exec in the same process, returning logs."""
    code_dir = "/tmp/generated/data_platform"
    os.makedirs(code_dir, exist_ok=True)
    path = os.path.join(code_dir, script_name)
    with open(path, "w") as f:
        f.write(code_str)
        
    import io
    import contextlib
    import traceback
    
    stdout_io = io.StringIO()
    stderr_io = io.StringIO()
    exit_code = 0
    
    with contextlib.redirect_stdout(stdout_io):
        with contextlib.redirect_stderr(stderr_io):
            try:
                globals_dict = {
                    "__name__": "__main__",
                    "__file__": path,
                }
                exec(code_str, globals_dict)
            except Exception as e:
                traceback.print_exc()
                exit_code = 1
                
    return {
        "exit_code": exit_code,
        "stdout": stdout_io.getvalue()[:15000],
        "stderr": stderr_io.getvalue()[:15000]
    }


def engineering_node(state: AgentState) -> Dict[str, Any]:
    """Data Engineer writes Bronze, Silver, Gold transformation scripts with compilation-loop healing and persistent memory."""
    print(">>> [Data Engineer] Generates and compiles PySpark Medallion scripts...")
    
    from dbricks_lang_agent.data_platform.spark_utils import get_spark, reset_lake
    from dbricks_lang_agent.orchestrator import memory
    
    spark = get_spark()

    # 1. Initialize codebase memory table + compile audit log, and mint a
    # run id for this engineering_node invocation so every compile/self-heal
    # attempt it makes can be tied back together for auditing.
    memory.init_codebase_memory_table(spark)
    memory.init_compile_audit_table(spark)
    import uuid as _uuid
    compile_run_id = str(_uuid.uuid4())
    
    # GUARD: No contracts — cannot generate code for nonexistent tables
    if not state.get("contracts"):
        contracts_error = state.get("contracts_error", "No contracts available.")
        print(f"[Data Engineer] ABORT: {contracts_error}")
        return {
            "bronze_code": "",
            "silver_code": "",
            "gold_code": "",
            "execution_logs": {},
            "active_agent": "DataEngineer",
            "review_comments": "",
        }

    # 2. Compute fingerprint — the SAME structural schema fingerprint used by
    # dq_node/contract_node/modeling_node (table names + column headers
    # only). Previously this hashed contracts+DDL TEXT, which meant every
    # cosmetic LLM re-generation of contracts/DDL (identical rules, different
    # wording) silently invalidated the entire codebase cache — see git
    # history. Keying off the structural fingerprint closes that gap: code
    # is only regenerated when the schema itself changes.
    fingerprint = get_schema_fingerprint(spark)
    print(f"[Codebase Memory] Computed schema fingerprint: {fingerprint}")

    # 3. Check for stored codebase — PER SCRIPT, not all-or-nothing. A script
    # is only reused/patched if IT SPECIFICALLY has a proven-good cached
    # entry; any script with no cache at all always gets a fresh generation,
    # independent of whether its siblings are cached. This matters because
    # log_script_code() (below, in the compiler loop) persists each script
    # the moment IT compiles clean — so e.g. a working bronze.py stays
    # cached and reused even while silver.py is still being iterated on,
    # instead of the entire bundle being thrown away because one script
    # hasn't compiled yet.
    reset_requested = _is_reset_requested()

    # A prior execution failure for a given script means the cached version
    # (if any) of THAT script is known-bad. Recall the cache and send ONLY
    # the failing script(s) back to the LLM with their traceback for a
    # targeted patch, mirroring the compiler loop's self-heal below. Scripts
    # that are cached and did NOT fail are left untouched.
    prior_execution_logs = state.get("execution_logs") or {}
    had_prior_failure = any(
        (log_info or {}).get("exit_code", 0) != 0
        for log_info in prior_execution_logs.values()
    )
    failed_script_names = {
        script_name for script_name, log_info in prior_execution_logs.items()
        if (log_info or {}).get("exit_code", 0) != 0
    }
    if had_prior_failure:
        print(f"[Codebase Memory] Previous execution attempt failed for {sorted(failed_script_names)} — will patch only the failing script(s), reuse everything else.")

    script_key_map = {
        "bronze.py": "bronze_code",
        "silver.py": "silver_code",
        "gold.py": "gold_code",
    }

    # IMPORTANT: engineering_node is re-entered from TWO different upstream
    # events that both leave review_comments non-empty, but only one of them
    # means "the human rejected the CODE itself":
    #   1. A human explicitly rejected the Engineering review gate — THIS is
    #      "throw everything out and reconsider it against my feedback."
    #   2. execution_review_gate's route_after_execution auto-bounces back
    #      here whenever a script failed at real runtime — the human hasn't
    #      necessarily reviewed the code at all; they may have just rejected
    #      the Orchestrator/report gate (or nothing was rejected — the router
    #      auto-routes back on ANY execution failure, human decision or not).
    # Treating both cases identically used to mean any rejection anywhere
    # downstream wiped out the entire codebase cache and regenerated
    # bronze/silver/gold from scratch — even scripts that never failed and
    # had nothing to do with the rejection. Only case 1 should bypass the
    # cache; case 2 should flow through the existing targeted-patch path
    # below (failed_script_names), which fixes only what's actually broken.
    engineering_explicitly_rejected = state.get("approved_steps", {}).get("engineering") is False

    cached: Dict[str, str] = {}
    if not reset_requested and not engineering_explicitly_rejected:
        cached = memory.get_stored_codebase(spark, fingerprint)
        if cached:
            print(f"[Codebase Memory] Found cached, proven-good code for: {sorted(cached.keys())}")

    contracts_str = "\n".join([f"--- {tbl} contract ---\n{s}" for tbl, s in state["contracts"].items()])
    ddl_str = state["gold_ddl"]

    # Tracks, per script, WHY the code about to enter the compiler loop looks
    # the way it does — 'cache_recalled' (untouched from Unity Catalog),
    # 'llm_targeted_patch' (only this script was regenerated after a prior
    # execution failure), or 'llm_fresh_generated' (no cache exists for it
    # yet). Logged alongside every compile attempt below so cache-vs-
    # regenerate behavior is auditable instead of inferred from stack traces.
    code_source = {"bronze_code": "unknown", "silver_code": "unknown", "gold_code": "unknown"}
    current_by_key = {
        "bronze_code": cached.get("bronze_code", ""),
        "silver_code": cached.get("silver_code", ""),
        "gold_code": cached.get("gold_code", ""),
    }
    for k in current_by_key:
        if cached.get(k):
            code_source[k] = "cache_recalled"

    # Cached scripts whose last real execution failed need a targeted patch.
    scripts_to_patch = [
        script_key_map[s] for s in failed_script_names
        if script_key_map.get(s) and cached.get(script_key_map[s])
    ]
    # Scripts with no cache at all need a full fresh generation.
    scripts_needing_fresh = [k for k in ("bronze_code", "silver_code", "gold_code") if not cached.get(k)]

    llm = get_llm("engineering")

    # 3a. Targeted patch — only for cached scripts whose last execution failed.
    for key_name in scripts_to_patch:
        script_name = next(s for s, k in script_key_map.items() if k == key_name)
        log_info = prior_execution_logs.get(script_name, {}) or {}
        fix_prompt = (
            f"You are a Senior Data Engineer. The generated {script_name} failed during a real "
            f"execution run against production data (not the compile-time sandbox check).\n\n"
            f"Target data contracts:\n{contracts_str}\n\n"
            f"Target Gold DDL star schema:\n{ddl_str}\n\n"
            f"Current failing code for {script_name}:\n```python\n{current_by_key[key_name]}\n```\n\n"
            f"Execution stdout (may contain Spark warnings):\n{log_info.get('stdout', '')[:3000]}\n\n"
            f"Execution stderr / traceback:\n{log_info.get('stderr', '')[:5000]}\n\n"
            + (f"Human feedback / fix comments:\n{state['review_comments']}\n\n" if state.get("review_comments") else "")
            + f"Please analyze the error and output the corrected version of the code. "
            f"Ensure all functions and classes are fully imported and syntactically correct.\n"
            f"Return your corrected code as a JSON object matching this schema:\n"
            f"{{\n"
            f"  \"{key_name}\": \"Corrected PySpark code string\"\n"
            f"}}\n"
        )
        messages = [
            SystemMessage(content=prompts.ENGINEER_SYSTEM_PROMPT),
            HumanMessage(content=fix_prompt)
        ]
        try:
            fix_response = llm.invoke(messages)
            fix_parsed = parse_json_from_response(fix_response.content, fallback_key=key_name)
            current_by_key[key_name] = _sanitize_and_heal_code(fix_parsed.get(key_name, current_by_key[key_name]))
            code_source[key_name] = "llm_targeted_patch"
        except ValueError as e_parse:
            print(f"[Codebase Memory] WARNING: Could not parse targeted fix response for {script_name}: {e_parse}. Keeping cached code as-is; compiler loop below will retry.")
            code_source[key_name] = "cache_recalled_patch_failed"

    # 3b. Fresh generation — only for scripts with no cache at all. The LLM
    # always returns all three scripts in one JSON response (its system
    # prompt requires it), but we only ADOPT the ones that actually needed
    # it — cached/patched scripts from 3a stay untouched even though the LLM
    # also generated something for them in this same response.
    if scripts_needing_fresh:
        if reset_requested or engineering_explicitly_rejected:
            print("[Codebase Memory] Reset requested or Engineering stage explicitly rejected — generating fresh draft for all scripts...")
        else:
            print(f"[Codebase Memory] No cache for: {sorted(scripts_needing_fresh)} — generating fresh...")

        prompt = (
            f"Target data contracts:\n{contracts_str}\n\n"
            f"Target Gold DDL star schema:\n{ddl_str}"
        )
        if state.get("review_comments"):
            prompt += f"\n\nHuman feedback / fix comments:\n{state['review_comments']}"

        if state.get("execution_logs"):
            logs_str = ""
            for script_name, log_info in state["execution_logs"].items():
                if log_info.get("exit_code", 0) != 0:
                    logs_str += (
                        f"\n--- Failure in {script_name} ---\n"
                        f"Exit Code: {log_info.get('exit_code')}\n"
                        f"Stdout snippet:\n{log_info.get('stdout', '')[:1000]}\n"
                        f"Stderr / Traceback:\n{log_info.get('stderr', '')[:2000]}\n"
                    )
            if logs_str:
                prompt += f"\n\nPrevious execution failures during runtime:\n{logs_str}"

        messages = [
            SystemMessage(content=prompts.ENGINEER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        parsed = parse_json_from_response(response.content)

        for key_name in scripts_needing_fresh:
            current_by_key[key_name] = _sanitize_and_heal_code(parsed.get(key_name, ""))
            code_source[key_name] = "llm_fresh_generated"

    bronze = current_by_key["bronze_code"]
    silver = current_by_key["silver_code"]
    gold = current_by_key["gold_code"]

    # 4. Sequential Compilation & Self-Correction Loop
    scripts_info = [
        ("bronze.py", "bronze_code"),
        ("silver.py", "silver_code"),
        ("gold.py", "gold_code")
    ]
    
    current_codes = {
        "bronze_code": bronze,
        "silver_code": silver,
        "gold_code": gold
    }
    
    print("[Compiler Loop] Resetting database schemas for a clean verification run...")
    reset_lake()
    
    contracts_str = "\n".join([f"--- {tbl} contract ---\n{s}" for tbl, s in state["contracts"].items()])
    ddl_str = state["gold_ddl"]
    llm = get_llm("engineering")
    
    verification_logs = {}
    compile_failed = False
    
    for script_name, key_name in scripts_info:
        current_code = current_codes[key_name]
        success = False
        
        for attempt in range(4): # 1 initial + 3 retries
            print(f"[Compiler Loop] Verifying {script_name} (Attempt {attempt + 1})...")
            res = _run_and_verify_script(script_name, current_code)
            verification_logs[script_name] = res

            # Audit every attempt: attempt 0 is tagged with how this script's
            # starting code was sourced (cache/patch/fresh); any retry beyond
            # that is the compiler loop's own self-heal regenerating it.
            try:
                memory.log_compile_attempt(
                    spark,
                    run_id=compile_run_id,
                    dataset_fingerprint=fingerprint,
                    script_name=script_name,
                    attempt_number=attempt + 1,
                    code_source=code_source.get(key_name, "unknown") if attempt == 0 else "llm_self_heal_fix",
                    code=current_code,
                    exit_code=res["exit_code"],
                    stdout=res["stdout"],
                    stderr=res["stderr"],
                )
            except Exception as e_audit:
                print(f"[Compile Audit] WARNING: Failed to log compile attempt for {script_name}: {e_audit}")

            if res["exit_code"] == 0:
                print(f"[Compiler Loop] {script_name} compiled and executed successfully!")
                current_codes[key_name] = current_code
                success = True
                # Persist THIS script immediately — independent of whether its
                # siblings go on to compile. This is what stops a proven-good
                # bronze.py from being thrown away and regenerated just
                # because silver.py is still failing.
                try:
                    memory.log_script_code(spark, fingerprint, key_name, current_code)
                except Exception as e_cache:
                    print(f"[Codebase Memory] WARNING: Failed to cache {script_name}: {e_cache}")
                break
            else:
                print(f"[Compiler Loop] {script_name} failed with exit code {res['exit_code']}")
                if attempt == 3:
                    break
                    
                print(f"[Compiler Loop] Sending error log to Engineering agent to self-heal...")
                fix_prompt = (
                    f"You are a Senior Data Engineer. The generated {script_name} has failed execution.\n\n"
                    f"Target data contracts:\n{contracts_str}\n\n"
                    f"Target Gold DDL star schema:\n{ddl_str}\n\n"
                    f"Current failing code for {script_name}:\n```python\n{current_code}\n```\n\n"
                    f"Execution stdout (may contain Spark warnings):\n{res['stdout'][:3000]}\n\n"
                    f"Execution stderr / traceback:\n{res['stderr'][:5000]}\n\n"
                    f"Please analyze the error and output the corrected version of the code. "
                    f"Ensure all functions and classes are fully imported and syntactically correct.\n"
                    f"Return your corrected code as a JSON object matching this schema:\n"
                    f"{{\n"
                    f"  \"{key_name}\": \"Corrected PySpark code string\"\n"
                    f"}}\n"
                )
                
                messages = [
                    SystemMessage(content=prompts.ENGINEER_SYSTEM_PROMPT),
                    HumanMessage(content=fix_prompt)
                ]

                fix_response = llm.invoke(messages)
                try:
                    fix_parsed = parse_json_from_response(fix_response.content, fallback_key=key_name)
                    current_code = _sanitize_and_heal_code(fix_parsed.get(key_name, ""))
                except ValueError as e_parse:
                    # Even the fenced-code-block fallback couldn't recover anything usable.
                    # Don't let this crash the whole pipeline run — count it as a failed
                    # attempt (current_code is left unchanged) and let the loop retry or
                    # exhaust its attempts and halt cleanly like any other compile failure.
                    print(f"[Compiler Loop] WARNING: Could not parse self-heal response for {script_name}: {e_parse}")

        if not success:
            print(f"[Compiler Loop] CRITICAL: Failed to compile {script_name} after max retries.")
            compile_failed = True
            break

    # Save final code files to generated directory
    bronze = current_codes["bronze_code"]
    silver = current_codes["silver_code"]
    gold = current_codes["gold_code"]
    
    code_dir = "/tmp/generated/data_platform"
    os.makedirs(code_dir, exist_ok=True)
    with open(os.path.join(code_dir, "bronze.py"), "w") as f:
        f.write(bronze)
    with open(os.path.join(code_dir, "silver.py"), "w") as f:
        f.write(silver)
    with open(os.path.join(code_dir, "gold.py"), "w") as f:
        f.write(gold)

    # 5. Codebase memory is now persisted per-script inline in the compiler
    # loop above (memory.log_script_code), the moment each script compiles
    # clean — no bundled all-or-nothing write needed here.
    if compile_failed:
        print("[Codebase Memory] Pipeline has compilation errors — any script(s) that DID compile clean were still cached individually above.")

    # Overall provenance summary for this stage: only "cache_reused" if EVERY
    # script (bronze/silver/gold) came straight from the codebase cache
    # untouched — any LLM involvement (fresh generation or a targeted patch
    # for one script) makes it "llm_patched" so the dashboard/audit trail
    # doesn't claim a fully-untouched reuse when part of it was regenerated.
    if all(v == "cache_recalled" for v in code_source.values()):
        generation_source = "cache_reused"
    elif all(v in ("llm_fresh_generated", "unknown") for v in code_source.values()):
        generation_source = "llm_fresh"
    else:
        generation_source = "llm_patched"

    approved_steps = dict(state.get("approved_steps", {}))
    if generation_source == "cache_reused" and memory.was_previously_approved(spark, "engineering", fingerprint):
        approved_steps["engineering"] = True
        try:
            memory.log_stage_review(
                spark,
                pipeline_run_id=state.get("pipeline_run_id", ""),
                stage_key="engineering",
                agent_name="DataEngineer",
                decision="approved",
                reviewer_comments="Auto-approved — schema unchanged, identical bronze/silver/gold code previously approved.",
                output={"bronze_code": bronze, "silver_code": silver, "gold_code": gold},
                dataset_fingerprint=fingerprint,
            )
        except Exception as e_log:
            print(f"[Warning] Failed to log auto-approval for Engineering: {e_log}")

    return {
        "bronze_code": bronze,
        "silver_code": silver,
        "gold_code": gold,
        "execution_logs": verification_logs,
        "active_agent": "DataEngineer",
        "review_comments": "",
        "approved_steps": approved_steps,
        "generation_source": {**(state.get("generation_source") or {}), "engineering": generation_source},
    }


def execution_node(state: AgentState) -> Dict[str, Any]:
    """Runs the generated scripts and compiles the final orchestrator report."""
    print(">>> [Orchestrator] Executing PySpark scripts on Databricks...")

    # Mint a run id up front (reused below for the run-history row) and set
    # up the compile audit context so each script's REAL execution outcome
    # against production data gets logged alongside its compile-time
    # verification attempts from engineering_node — same dataset_fingerprint,
    # so the two can be cross-referenced.
    import uuid as _uuid
    run_id = str(_uuid.uuid4())
    audit_spark = None
    audit_fingerprint = ""
    try:
        audit_spark = get_spark()
        audit_fingerprint = get_schema_fingerprint(audit_spark)
        memory.init_compile_audit_table(audit_spark)
    except Exception as e:
        print(f"[Compile Audit] WARNING: Could not initialize compile audit context: {e}")

    code_dir = "/tmp/generated/data_platform"
    os.makedirs(code_dir, exist_ok=True)
    
    # Re-create script files on disk in case we are resuming from a checkpoint 
    # where the DataEngineer node was already executed in a previous session
    if state.get("bronze_code"):
        with open(os.path.join(code_dir, "bronze.py"), "w") as f:
            f.write(state["bronze_code"])
    if state.get("silver_code"):
        with open(os.path.join(code_dir, "silver.py"), "w") as f:
            f.write(state["silver_code"])
    if state.get("gold_code"):
        with open(os.path.join(code_dir, "gold.py"), "w") as f:
            f.write(state["gold_code"])

    scripts = ["bronze.py", "silver.py", "gold.py"]
    logs = {}
    
    import io
    import contextlib
    import traceback

    for s in scripts:
        path = os.path.join(code_dir, s)
        print(f"Executing {s}...")
        try:
            with open(path, "r", encoding="utf-8") as f:
                code_content = f.read()
                
            stdout_io = io.StringIO()
            stderr_io = io.StringIO()
            exit_code = 0
            
            with contextlib.redirect_stdout(stdout_io):
                with contextlib.redirect_stderr(stderr_io):
                    try:
                        globals_dict = {
                            "__name__": "__main__",
                            "__file__": path,
                        }
                        exec(code_content, globals_dict)
                    except Exception as e_run:
                        traceback.print_exc()
                        exit_code = 1
                        
            logs[s] = {
                "exit_code": exit_code,
                "stdout": stdout_io.getvalue()[:15000],
                "stderr": stderr_io.getvalue()[:15000]
            }

            # Safety net independent of whether the LLM-generated script actually
            # raises on a halted Silver run (see ENGINEER_SYSTEM_PROMPT): if
            # silver.py "succeeded" (exit_code 0) but its own summary reports a
            # halt, treat that as a failure here too, so gold.py never runs
            # against a Silver layer with tables missing after the halt point.
            # Without this, gold.py fails downstream with a confusing
            # TABLE_OR_VIEW_NOT_FOUND instead of the real cause being visible.
            if s == "silver.py" and exit_code == 0 and os.path.exists("/tmp/silver_summary.json"):
                try:
                    with open("/tmp/silver_summary.json") as f_summary:
                        _silver_summary_check = json.load(f_summary)
                    halted_at = _silver_summary_check.get("halted_at")
                    if halted_at:
                        exit_code = 1
                        halt_msg = (
                            f"silver.py reported halted_at='{halted_at}' in silver_summary.json "
                            "(a hard contract-rule failure rate was exceeded for that table) even "
                            "though the script itself did not raise. Treating this as a failed "
                            "execution to prevent gold.py from running against an incomplete Silver layer."
                        )
                        print(f"!!! {halt_msg}")
                        logs[s]["stderr"] = (logs[s]["stderr"] or "") + f"\n[Orchestrator Safety Net] {halt_msg}"
                        logs[s]["exit_code"] = exit_code
                except Exception as e_halt_check:
                    print(f"[Warning] Could not inspect silver_summary.json for halted_at: {e_halt_check}")

            if audit_spark is not None:
                try:
                    memory.log_compile_attempt(
                        audit_spark,
                        run_id=run_id,
                        dataset_fingerprint=audit_fingerprint,
                        script_name=s,
                        attempt_number=1,
                        code_source="execution_node_real_run",
                        code=code_content,
                        exit_code=exit_code,
                        stdout=logs[s]["stdout"],
                        stderr=logs[s]["stderr"],
                    )
                except Exception as e_audit:
                    print(f"[Compile Audit] WARNING: Failed to log execution attempt for {s}: {e_audit}")

            if exit_code != 0:
                print(f"!!! Script {s} failed with exit code {exit_code}")
                # Stop executing downstream scripts if upstream fails
                break
        except Exception as e:
            logs[s] = {"exit_code": -1, "stdout": "", "stderr": str(e)}
            print(f"!!! Execution of {s} threw exception: {e}")
            break

    # Read silver and gold summary reports if written by the executed scripts
    silver_summary = {}
    gold_summary = {}
    
    if os.path.exists("/tmp/silver_summary.json"):
        try:
            with open("/tmp/silver_summary.json") as f:
                silver_summary = json.load(f)
        except Exception:
            pass
            
    if os.path.exists("/tmp/gold_summary.json"):
        try:
            with open("/tmp/gold_summary.json") as f:
                gold_summary = json.load(f)
        except Exception:
            pass

    # Compile Final Orchestrator Report
    # Detect failures so the LLM prompt explicitly notes unavailable data
    failed_scripts = [s for s, r in logs.items() if r.get("exit_code", 0) != 0]
    pipeline_status = "HALTED" if failed_scripts else "COMPLETED"

    failure_details = ""
    if failed_scripts:
        for s in failed_scripts:
            failure_details += (
                f"\n--- FAILED SCRIPT: {s} ---\n"
                f"stdout:\n{logs[s].get('stdout', '')[:2000]}\n"
                f"stderr:\n{logs[s].get('stderr', '')[:3000]}\n"
            )

    prompt = (
        f"Pipeline status: {pipeline_status}\n"
        f"Script execution logs:\n{json.dumps(logs, indent=2)}\n\n"
        f"Silver Validation summary (empty if silver.py failed):\n{json.dumps(silver_summary, indent=2)}\n\n"
        f"Gold schema summary (empty if gold.py failed):\n{json.dumps(gold_summary, indent=2)}\n\n"
        + (f"FAILURE DETAILS:\n{failure_details}" if failure_details else "")
    )
    
    messages = [
        SystemMessage(content=prompts.ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    llm = get_llm("orchestrator")
    response = llm.invoke(messages)
    
    final_report = response.content

    # Save final report to generated directory
    reports_dir = "./generated/reports"
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "final_run_report.md"), "w") as f:
        f.write(final_report)

    # Append this attempt to the run history audit log (Unity Catalog) so it
    # stays reviewable even after the live checkpoint thread moves on, loops
    # back to self-heal, or is reset. Every execution_node run gets its own
    # row — success or failure — unlike agent_codebase_memory which only
    # keeps the latest code per fingerprint.
    try:
        spark = get_spark()
        fingerprint = get_schema_fingerprint(spark)
        memory.init_run_history_table(spark)
        memory.log_run(
            spark,
            run_id=run_id,
            pipeline_status=pipeline_status,
            active_agent="Orchestrator",
            dataset_fingerprint=fingerprint,
            failed_scripts=failed_scripts,
            execution_logs=logs,
            silver_summary=silver_summary,
            gold_summary=gold_summary,
            final_report=final_report,
            approved_steps=state.get("approved_steps", {}),
            review_comments=state.get("review_comments", ""),
        )
    except Exception as e:
        print(f"[Warning] Failed to log run to audit history: {e}")

    return {
        "execution_logs": logs,
        "silver_summary": silver_summary,
        "gold_summary": gold_summary,
        "final_report": final_report,
        "active_agent": "Orchestrator",
        "review_comments": "",
        "last_run_id": run_id,
    }


def product_advisor_node(state: AgentState) -> Dict[str, Any]:
    """Data Product Advisor: inspects the completed Gold star schema (DDL + data
    dictionary + row counts) and proposes candidate downstream data products —
    purpose-built marts/views for a specific consumer (finance, marketing,
    supplier ops, revenue management, ML) — built ON TOP of Gold.

    Unlike the six sequential pipeline agents, this one is intentionally NOT
    wired into the interrupt_before HITL chain in graph.py. Gold is a stable,
    already-approved asset by the time this runs, so there's no draft to
    approve/reject here — it's an on-demand analysis triggered from the
    dashboard's "Data Products" tab, any time after Gold has been built.
    """
    print(">>> [Data Product Advisor] Analyzing Gold layer for data product opportunities...")

    gold_ddl = state.get("gold_ddl", "")
    data_dictionary = state.get("data_dictionary", "")
    gold_summary = state.get("gold_summary", {}) or {}

    # GUARD: No Gold DDL means Gold hasn't been designed/built yet — nothing to analyze.
    if not gold_ddl:
        error_msg = (
            "Data Product Advisor blocked: no Gold DDL found in pipeline state. "
            "Run the pipeline through a completed Gold build (Modeler → Engineer → Orchestrator stages) first."
        )
        print(f"[Data Product Advisor] ABORT: {error_msg}")
        return {
            "product_candidates": [],
            "product_advisor_error": error_msg,
            "active_agent": "DataProductAdvisor",
        }

    row_counts = gold_summary.get("row_counts", {})
    prompt = (
        f"Gold Star Schema DDL:\n{gold_ddl}\n\n"
        f"Data Dictionary:\n{data_dictionary}\n\n"
        f"Current Gold table row counts:\n{json.dumps(row_counts, indent=2)}"
    )

    messages = [
        SystemMessage(content=prompts.PRODUCT_ADVISOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    llm = get_llm("product_advisor")
    response = llm.invoke(messages)

    parsed = parse_json_from_response(response.content)
    products = parsed.get("products", [])

    return {
        "product_candidates": products,
        "product_advisor_error": "",
        "active_agent": "DataProductAdvisor",
    }


# ---- Talk to Data Chatbot Prompts ----

CHAT_INTENT_CLASSIFIER_PROMPT = """
You are an intent classifier for a data-platform chatbot.
Given the user's question, classify it into EXACTLY ONE of these intents (output only the intent word, nothing else):

  data_stats        – asks for general statistical summaries of the dataset (e.g. averages, distributions, trends)
  schema            – asks about table structures, column names, data types, or field definitions
  quality           – asks about data quality, null rates, anomalies, or validation issues
  count             – asks for row counts, record totals, or volume information
  sample            – asks for sample values, example records, or value distributions / frequencies
  pipeline_status   – asks about the current state of the pipeline, which agent ran last, or what step is next
  pipeline_control  – user wants to START, RUN, RESUME, PROCEED with, APPROVE, CONFIRM, or TRIGGER the pipeline
  general           – any other question about the dataset, contracts, DDL, reports, or anything not covered above

Respond with ONLY the single intent word. No punctuation. No explanation.
"""

CHAT_ANSWER_PROMPT = """
You are a friendly, expert Data Analyst assistant embedded in a Medallion Pipeline Control Center.
Your only knowledge source is the context provided below — do NOT make up facts or numbers.

IMPORTANT FORMATTING RULES:
- Reply in plain Markdown only (bullet points, bold, tables).
- Do NOT output any HTML tags, XML tags, or raw code.
- Do NOT output </div>, <div>, or any other HTML.
- Keep responses concise — under 300 words unless a table genuinely requires more space.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION:
{question}

Instructions:
- Answer the question conversationally and concisely using ONLY the data in the CONTEXT above.
- Use bullet points or short tables where they improve clarity.
- If the context does not contain enough information to fully answer the question, say so honestly
  and suggest what the user can do (e.g. "Run the pipeline to generate profiling data first").
- Never reveal raw JSON blobs or internal field names directly — translate them into plain English.
- When citing numbers, be precise (use exact figures from the context, not approximations).
"""

# Module-level cache dict – keyed by a hash of the volume path + mtime so the
# profile is automatically invalidated when new source files land in the Volume.
_profiling_cache: Dict[str, Any] = {}


def _profiling_cache_key() -> str:
    """Return a lightweight cache key for the profiling report.

    Uses the volume path from config so that different environments naturally
    produce different keys. A full file-mtime hash is avoided here because
    listing the Volume inside a cache-key function would add Spark overhead
    on every chat turn. Instead, the Streamlit UI offers a 'Clear Cache'
    button that calls clear_profiling_cache() directly.
    """
    cfg = load_config()
    return cfg.get("volume_raw_path", "default")


def clear_profiling_cache() -> None:
    """Evict the cached profiling report so the next chat query re-runs Spark."""
    _profiling_cache.clear()


def _get_or_run_profiling() -> Dict[str, Any]:
    """Return a cached profiling report or run a fresh profiling pass.

    Results are cached in the module-level ``_profiling_cache`` dict so that
    repeated chat queries within the same Python process (i.e. the same
    Streamlit app server instance) do not re-run Spark.
    """
    key = _profiling_cache_key()
    if key not in _profiling_cache:
        print("[Talk to Data] Running profiling to answer data query...")
        report = profiling.profile_all_sources()
        _profiling_cache[key] = report
    return _profiling_cache[key]


def _classify_intent(question: str) -> str:
    """Use the LLM to classify the user's question into one of 8 intents."""
    llm = get_llm("chat")
    messages = [
        SystemMessage(content=CHAT_INTENT_CLASSIFIER_PROMPT),
        HumanMessage(content=question)
    ]
    try:
        response = llm.invoke(messages)
        intent = response.content.strip().lower().split()[0]
        valid_intents = {
            "data_stats", "schema", "quality", "count", "sample",
            "pipeline_status", "pipeline_control", "general"
        }
        return intent if intent in valid_intents else "general"
    except Exception as e:
        print(f"[Talk to Data] Intent classification failed ({e}), defaulting to 'general'.")
        return "general"


def _build_context(intent: str, state: AgentState, profiling_report: Dict[str, Any]) -> str:
    """Build a rich natural-language context string for the answer LLM.

    The context is tailored to the classified intent so the LLM only receives
    data that is actually relevant to the question, keeping the prompt lean.
    """
    parts: List[str] = []

    # --- Profiling-backed context (data_stats / schema / quality / count / sample) ---
    profiling_intents = {"data_stats", "schema", "quality", "count", "sample"}
    if intent in profiling_intents and profiling_report:
        discovered = profiling_report.get("discovered_tables", {})
        tables_profile = profiling_report.get("tables", {})
        unique_keys = profiling_report.get("candidate_unique_keys", {})
        dup_keys = profiling_report.get("duplicate_keys", {})
        ri = profiling_report.get("referential_integrity", [])

        # Discovered tables summary
        if discovered:
            lines = [f"  • {tbl} ← {fname}" for tbl, fname in discovered.items()]
            parts.append("DISCOVERED TABLES:\n" + "\n".join(lines))

        # Per-table stats
        for tbl, profile in tables_profile.items():
            row_count = profile.get("row_count", "N/A")
            col_count = profile.get("column_count", "N/A")
            cols_info = []
            for col_name, col_stat in profile.get("columns", {}).items():
                dtype = col_stat.get("dtype", "unknown")
                null_pct = col_stat.get("null_pct", 0)
                distinct = col_stat.get("distinct_count", "N/A")
                num_stats = col_stat.get("numeric_stats")
                val_counts = col_stat.get("value_counts")

                col_line = f"    - {col_name} ({dtype}): {null_pct}% null, {distinct} distinct values"
                if num_stats:
                    col_line += f" | min={num_stats.get('min')}, max={num_stats.get('max')}, avg={num_stats.get('avg')}"
                if val_counts:
                    top_vals = ", ".join([f"'{k}' ({v})" for k, v in list(val_counts.items())[:5]])
                    col_line += f" | top values: {top_vals}"
                cols_info.append(col_line)

            tbl_summary = (
                f"TABLE: {tbl}\n"
                f"  Rows: {row_count} | Columns: {col_count}\n"
                f"  Primary key candidates: {unique_keys.get(tbl, 'none found')}\n"
                f"  Duplicate key count: {dup_keys.get(tbl, 0)}\n"
                f"  Column details:\n" + "\n".join(cols_info)
            )
            parts.append(tbl_summary)

        # Referential integrity
        if ri:
            ri_lines = [
                f"  • {r['table']}.{r['column']} → {r['parent_table']}.{r['parent_column']} "
                f"(overlap: {r['overlap_pct']}%, orphans: {r['orphan_count']})"
                for r in ri
            ]
            parts.append("REFERENTIAL INTEGRITY:\n" + "\n".join(ri_lines))

        # Profiler narrative (if already available from a prior pipeline run)
        narration = state.get("profiling_report", {}).get("profiler_narration", "")
        if narration:
            parts.append(f"PROFILER NARRATIVE:\n{narration[:1500]}")

        # DQ report if available
        dq = state.get("dq_report", "")
        if dq and intent == "quality":
            parts.append(f"DATA QUALITY REPORT:\n{dq[:2000]}")

    # --- Pipeline control context ---
    if intent == "pipeline_control":
        active_agent = state.get("active_agent", "Not started")
        approved = state.get("approved_steps", {})
        next_gate = None
        # Map active_agent to the gate it is waiting at
        gate_map = {
            "Profiler": ("profile_review_gate", "profile"),
            "DataQuality": ("data_quality_review_gate", "dq"),
            "Contracts": ("contracts_review_gate", "contracts"),
            "DimensionalModeler": ("modeling_review_gate", "modeling"),
            "DataEngineer": ("engineering_review_gate", "engineering"),
            "Orchestrator": ("execution_review_gate", "execution"),
        }
        if active_agent and active_agent in gate_map:
            gate, step = gate_map[active_agent]
            already_approved = approved.get(step, "") == "approved"
            next_gate = None if already_approved else {"gate": gate, "step": step}
        parts.append(
            f"PIPELINE CONTROL CONTEXT:\n"
            f"  Last active agent: {active_agent}\n"
            f"  Approved steps: {json.dumps(approved)}\n"
            f"  Next pending gate: {next_gate}\n"
            f"  To proceed: approve the gate listed in next_pending_gate."
        )

    # --- Pipeline-state context (status questions) ---
    if intent == "pipeline_status":
        active_agent = state.get("active_agent", "Not started")
        approved = state.get("approved_steps", {})
        parts.append(
            f"PIPELINE STATUS:\n"
            f"  Last active agent: {active_agent}\n"
            f"  Approved steps: {json.dumps(approved)}\n"
            f"  Review comments: {state.get('review_comments', 'None')}"
        )

    # --- General / contracts / DDL context ---
    if intent == "general":
        if state.get("contracts"):
            contracts_summary = "\n".join(
                [f"  {tbl}:\n{yaml_str[:400]}" for tbl, yaml_str in state["contracts"].items()]
            )
            parts.append(f"DATA CONTRACTS (YAML):\n{contracts_summary}")
        if state.get("gold_ddl"):
            parts.append(f"GOLD STAR SCHEMA DDL:\n{state['gold_ddl'][:1000]}")
        if state.get("data_dictionary"):
            parts.append(f"DATA DICTIONARY:\n{state['data_dictionary'][:1000]}")
        if state.get("final_report"):
            parts.append(f"FINAL PIPELINE REPORT:\n{state['final_report'][:1500]}")

    if not parts:
        parts.append("No pipeline data available yet. The pipeline has not been run or has not produced output for this query type.")

    return "\n\n".join(parts)


def chat_with_data_agent(
    question: str,
    state: AgentState,
    history: List[Dict[str, str]],
) -> tuple:
    """Answer a natural-language question about the dataset.

    Parameters
    ----------
    question:
        The user's latest question.
    state:
        Current LangGraph AgentState (read-only).
    history:
        List of previous ``{"role": "user"|"assistant", "content": str}`` dicts
        (most recent last). Used to provide few-shot conversational context.

    Returns
    -------
    answer : str
        The LLM-generated answer in plain Markdown.
    profiling_triggered : bool
        ``True`` when a fresh Spark profiling job was triggered.
    pipeline_action : dict | None
        When the user wants to start/approve a pipeline step, this dict contains
        ``{"gate": ..., "step": ..., "description": ...}`` for the UI to surface
        a confirmation button. ``None`` if no pipeline action is required.
    """
    # 1. Classify intent
    intent = _classify_intent(question)
    print(f"[Talk to Data] Classified intent: '{intent}' for question: {question[:80]}")

    # 2. Conditionally run profiling
    profiling_report: Dict[str, Any] = {}
    profiling_triggered = False
    profiling_intents = {"data_stats", "schema", "quality", "count", "sample"}

    if intent in profiling_intents:
        cached_before = bool(_profiling_cache)
        profiling_report = _get_or_run_profiling()
        profiling_triggered = not cached_before

    # 3. Handle pipeline_control intent directly — no LLM answer generation needed
    if intent == "pipeline_control":
        active_agent = state.get("active_agent", None)
        approved = state.get("approved_steps", {})
        gate_map = {
            "Profiler":          ("profile_review_gate",      "profile"),
            "DataQuality":       ("data_quality_review_gate", "dq"),
            "Contracts":         ("contracts_review_gate",    "contracts"),
            "DimensionalModeler":("modeling_review_gate",     "modeling"),
            "DataEngineer":      ("engineering_review_gate",  "engineering"),
            "Orchestrator":      ("execution_review_gate",    "execution"),
        }
        if not active_agent:
            # Pipeline hasn't started yet — offer to launch it from scratch
            action = {
                "gate": "__start__",
                "step": "pipeline_start",
                "description": "Launch the Medallion Pipeline from scratch (Profiler → DQ → Contracts → Modeler → Engineer → Executor)."
            }
            return (
                "The pipeline hasn't started yet. I can kick it off for you right now! "
                "Click **▶ Yes, Proceed** below to launch the full Medallion Pipeline from the beginning.",
                False,
                action
            )
        if active_agent not in gate_map:
            return (
                f"The pipeline is currently at agent **{active_agent}** but I don't know "
                "which gate to approve. Please use the **Action Center (HITL)** tab directly.",
                False,
                None
            )
        gate, step = gate_map[active_agent]
        if approved.get(step) == "approved":
            return (
                f"The **{step}** step has already been approved. "
                "The pipeline should be advancing automatically. "
                "Check the **Ingestion Monitoring** tab for live status.",
                False,
                None
            )
        # Surface a confirmation button in the UI
        action = {
            "gate": gate,
            "step": step,
            "description": f"Approve the **{step.title()}** step (current agent: {active_agent}) and advance the pipeline."
        }
        return (
            f"I can approve the **{step.title()}** step and advance the pipeline from **{active_agent}**. "
            f"A confirmation button will appear below — click **\u25b6 Yes, Proceed** to continue.",
            False,
            action
        )

    # 4. Build context string
    context = _build_context(intent, state, profiling_report)

    # 5. Build history string (last 10 turns)
    history_lines = []
    for turn in history[-10:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {turn['content']}")
    history_str = "\n".join(history_lines) if history_lines else "No prior conversation."

    # 6. Call answer LLM
    answer_prompt = CHAT_ANSWER_PROMPT.format(
        context=context,
        history=history_str,
        question=question,
    )
    llm = get_llm("chat")
    messages = [HumanMessage(content=answer_prompt)]
    try:
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        answer = f"\u26a0\ufe0f I encountered an error while generating your answer: {e}"

    return answer, profiling_triggered, None
