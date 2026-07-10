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
        else:
            return MockResponse("# Mock Orchestrator Report\nPipeline execution finished successfully.")


def get_llm(node_name: str) -> Any:
    """Fetch the Databricks Model Serving LLM or fallback to mock class."""
    if os.environ.get("USE_MOCK_LLM", "false").lower() == "true":
        return MockChatModel(node_name)

    cfg = load_config()
    endpoint = cfg.get("llm", {}).get("endpoint", "databricks-meta-llama-3-1-70b-instruct")
    temperature = cfg.get("llm", {}).get("temperature", 0.0)

    try:
        # Databricks model serving integration in langchain
        return ChatDatabricks(
            endpoint=endpoint,
            temperature=temperature,
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

def parse_json_from_response(content: str) -> dict:
    """Safely extract and parse JSON block from model response."""
    import re
    content_str = content.strip()
    
    # 1. Try to find json code block first
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_str, re.DOTALL)
    if code_block_match:
        try:
            sanitized = sanitize_json_string(code_block_match.group(1).strip())
            return json.loads(sanitized)
        except Exception:
            pass
            
    # 2. Find the first '{' and the last '}' (greedy match)
    brace_match = re.search(r"(\{.*\})", content_str, re.DOTALL)
    if brace_match:
        try:
            sanitized = sanitize_json_string(brace_match.group(1).strip())
            return json.loads(sanitized)
        except Exception:
            pass
            
    # 3. Fallback to parsing the stripped content directly
    cleaned = content_str
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        sanitized = sanitize_json_string(cleaned)
        return json.loads(sanitized)
    except Exception as e_direct:
        raise ValueError(
            f"Failed to parse JSON from LLM response.\n"
            f"Parser error: {e_direct}\n"
            f"Response content length: {len(content_str)}\n"
            f"Response content snippet (first 1000 chars):\n"
            f"{content_str[:1000]}"
        ) from e_direct


# ---- Node Functions ----

def profiler_node(state: AgentState) -> Dict[str, Any]:
    """Nodes runs PySpark profiling on raw files and summaries findings."""
    print(">>> [Profiler Agent] Discovering and profiling raw source files...")
    
    # Run the dynamic data platform profiler
    reports_dir = "/tmp/reports"
    os.makedirs(reports_dir, exist_ok=True)
    report = profiling.profile_all_sources(output_path=os.path.join(reports_dir, "profiling_report.json"))
    
    # Prepare a condensed profiling report for the LLM to read
    discovered = report.get("discovered_tables", {})
    unique_keys = report.get("candidate_unique_keys", {})
    duplicate_keys = report.get("duplicate_keys", {})
    ri = report.get("referential_integrity", [])
    
    condensed = {
        "discovered_tables": discovered,
        "candidate_primary_keys": unique_keys,
        "duplicate_key_counts": duplicate_keys,
        "referential_integrity": ri
    }

    # Format message for LLM profiling review
    prompt = f"Raw dataset profiling metrics summary:\n{json.dumps(condensed, indent=2)}\n\nAnalyze and summarize these findings."
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
        "active_agent": "Profiler",
        "review_comments": ""
    }


def dq_node(state: AgentState) -> Dict[str, Any]:
    """Data Quality Agent Node: Checks for logical and physical data anomalies."""
    print(">>> [Data Quality Agent] Assessing raw source data quality...")
    
    # Query few-shot memory database
    spark = get_spark()
    memory.init_memory_table(spark)
    dataset = list(state.get("discovered_tables", {}).keys())[0] if state.get("discovered_tables") else "generic"
    few_shot_context = memory.get_few_shot_context(spark, dataset, "data_quality")
    
    profiling_report = state.get("profiling_report", {})
    profiling_narration = profiling_report.get("profiler_narration", "")
    profiling_metrics = {k: v for k, v in profiling_report.items() if k != "profiler_narration"}
    
    prompt = (
        f"Data Profiler Report Narrative:\n{profiling_narration}\n\n"
        f"Raw Metrics JSON:\n{json.dumps({k: v for k, v in profiling_metrics.items() if k != 'tables'}, indent=2)}\n\n"
        f"{few_shot_context}"
    )
    if state.get("review_comments"):
        prompt += f"\n\nHuman feedback on previous DQ assessment:\n{state['review_comments']}"
        
    messages = [
        SystemMessage(content=prompts.DQ_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    llm = get_llm("dq")
    response = llm.invoke(messages)
    
    return {
        "dq_report": response.content,
        "active_agent": "DataQualityAgent",
        "review_comments": ""
    }


def contract_node(state: AgentState) -> Dict[str, Any]:
    """Steward designs and authors YAML schema data contracts."""
    print(">>> [Contract Steward] Authoring YAML schema data contracts...")
    
    # Query few-shot memory database
    spark = get_spark()
    dataset = list(state.get("discovered_tables", {}).keys())[0] if state.get("discovered_tables") else "generic"
    few_shot_context = memory.get_few_shot_context(spark, dataset, "contracts")
    
    profiling_report = state.get("profiling_report", {})
    profiling_narration = profiling_report.get("profiler_narration", "")
    profiling_metrics = {k: v for k, v in profiling_report.items() if k != "profiler_narration"}
    
    prompt = (
        f"Data Profiling Report:\n{profiling_narration}\n\n"
        f"Data Quality Assessment Report:\n{state.get('dq_report', '')}\n\n"
        f"Raw Metrics:\n{json.dumps({k: v for k, v in profiling_metrics.items() if k != 'tables'}, indent=2)}\n\n"
        f"{few_shot_context}"
    )
    if state.get("review_comments"):
        prompt += f"\n\nHuman feedback on previous contracts draft:\n{state['review_comments']}"

    messages = [
        SystemMessage(content=prompts.CONTRACT_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    llm = get_llm("contracts")
    response = llm.invoke(messages)
    
    parsed = parse_json_from_response(response.content)
    contracts = parsed.get("contracts", {})
    
    # Write the YAML contracts to disk so that they are visible in generated/config/contracts
    contracts_dir = "/tmp/generated/config/contracts"
    os.makedirs(contracts_dir, exist_ok=True)
    for table, yaml_str in contracts.items():
        with open(os.path.join(contracts_dir, f"{table}.yaml"), "w") as f:
            f.write(yaml_str)

    return {
        "contracts": contracts,
        "active_agent": "ContractSteward",
        "review_comments": ""
    }


def modeling_node(state: AgentState) -> Dict[str, Any]:
    """Modeler designs Gold-layer dimensions and facts SQL DDL."""
    print(">>> [Dimensional Modeler] Designing Kimball star schema DDL...")
    
    profiling_narration = state["profiling_report"].get("profiler_narration", "")
    contracts_summary = ""
    for table, yaml_str in state["contracts"].items():
        contracts_summary += f"--- CONTRACT FOR {table} ---\n{yaml_str}\n"

    prompt = (
        f"Profiling narrative:\n{profiling_narration}\n\n"
        f"Contracts specifications:\n{contracts_summary}"
    )
    if state.get("review_comments"):
        prompt += f"\n\nHuman feedback on previous DDL draft:\n{state['review_comments']}"

    messages = [
        SystemMessage(content=prompts.MODELER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    llm = get_llm("modeling")
    response = llm.invoke(messages)
    
    parsed = parse_json_from_response(response.content)
    gold_ddl = parsed.get("gold_ddl", "")
    data_dictionary = parsed.get("data_dictionary", "")
    
    # Write SQL and Markdown DDL files
    ddl_dir = "/tmp/generated/data_model"
    os.makedirs(ddl_dir, exist_ok=True)
    with open(os.path.join(ddl_dir, "gold_ddl.sql"), "w") as f:
        f.write(gold_ddl)
    with open(os.path.join(ddl_dir, "data_dictionary.md"), "w") as f:
        f.write(data_dictionary)

    return {
        "gold_ddl": gold_ddl,
        "data_dictionary": data_dictionary,
        "active_agent": "DimensionalModeler",
        "review_comments": ""
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
                f"{indent}    _aggs = [sum(when(~col(__c).isNull() & (col(__c) != '') & ~lower(col(__c)).isin('yes', 'no', 'y', 'n', 'true', 'false'), 1).otherwise(0)).alias(__c) for __c in _str_cols]\n"
                f"{indent}    _counts = {df_var}.select(*_aggs).collect()[0].asDict()\n"
                f"{indent}    for __c, __count in _counts.items():\n"
                f"{indent}        if __count == 0:\n"
                f"{indent}            {df_var} = {df_var}.withColumn(__c, when(lower(col(__c)).isin('yes', 'y', 'true'), lit(True)).when(lower(col(__c)).isin('no', 'n', 'false'), lit(False)).otherwise(None))"
            )
            code = re.sub(r"(?m)^(\s*)" + df_var + r"\s*=\s*" + df_var + r"\.applymap\s*\(.*\)\s*$", replacement, code)

    # 1.11. Fix hallucinated loop-based string-to-boolean mapping that is not type-safe
    if "dtypes" in code and "isin" in code:
        import re
        loop_pattern = r"(?m)^(\s*)for\s+(\w+),\s*(\w+)\s+in\s+(\w+)\.dtypes\s*:\s*\n\1\s+if\s+\3\s*==\s*['\"]string['\"]\s*:\s*\n\1\s+\s*\4\s*=\s*\4\.withColumn\(\s*\2\s*,\s*when\(col\(\s*\2\s*\)\.isin\(.*?\).*?\)"
        loop_match = re.search(loop_pattern, code, re.DOTALL)
        if loop_match:
            indent = loop_match.group(1)
            df_var = loop_match.group(4)
            replacement = (
                f"{indent}# Standardize boolean values safely (only for columns containing only boolean-like values)\n"
                f"{indent}_str_cols = [__c for __c, __t in {df_var}.dtypes if __t == 'string']\n"
                f"{indent}if _str_cols:\n"
                f"{indent}    _aggs = [sum(when(~col(__c).isNull() & (col(__c) != '') & ~lower(col(__c)).isin('yes', 'no', 'y', 'n', 'true', 'false'), 1).otherwise(0)).alias(__c) for __c in _str_cols]\n"
                f"{indent}    _counts = {df_var}.select(*_aggs).collect()[0].asDict()\n"
                f"{indent}    for __c, __count in _counts.items():\n"
                f"{indent}        if __count == 0:\n"
                f"{indent}            {df_var} = {df_var}.withColumn(__c, when(lower(col(__c)).isin('yes', 'y', 'true'), lit(True)).when(lower(col(__c)).isin('no', 'n', 'false'), lit(False)).otherwise(None))"
            )
            code = re.sub(loop_pattern, replacement, code, flags=re.DOTALL)

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


def engineering_node(state: AgentState) -> Dict[str, Any]:
    """Data Engineer writes Bronze, Silver, Gold transformation scripts."""
    print(">>> [Data Engineer] Generates PySpark Medallion scripts...")
    
    contracts_str = "\n".join([f"--- {tbl} contract ---\n{s}" for tbl, s in state["contracts"].items()])
    ddl_str = state["gold_ddl"]
    
    prompt = (
        f"Target data contracts:\n{contracts_str}\n\n"
        f"Target Gold DDL star schema:\n{ddl_str}"
    )
    if state.get("review_comments"):
        prompt += f"\n\nHuman feedback / fix comments:\n{state['review_comments']}"
    if state.get("execution_logs"):
        prompt += f"\n\nLast execution logs and errors (if any):\n{json.dumps(state['execution_logs'], indent=2)}"

    messages = [
        SystemMessage(content=prompts.ENGINEER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    llm = get_llm("engineering")
    response = llm.invoke(messages)
    
    parsed = parse_json_from_response(response.content)
    bronze = _sanitize_and_heal_code(parsed.get("bronze_code", ""))
    silver = _sanitize_and_heal_code(parsed.get("silver_code", ""))
    gold = _sanitize_and_heal_code(parsed.get("gold_code", ""))

    # Save code files to generated directory
    code_dir = "/tmp/generated/data_platform"
    os.makedirs(code_dir, exist_ok=True)
    with open(os.path.join(code_dir, "bronze.py"), "w") as f:
        f.write(bronze)
    with open(os.path.join(code_dir, "silver.py"), "w") as f:
        f.write(silver)
    with open(os.path.join(code_dir, "gold.py"), "w") as f:
        f.write(gold)

    return {
        "bronze_code": bronze,
        "silver_code": silver,
        "gold_code": gold,
        "active_agent": "DataEngineer",
        "review_comments": ""
    }


def execution_node(state: AgentState) -> Dict[str, Any]:
    """Runs the generated scripts and compiles the final orchestrator report."""
    print(">>> [Orchestrator] Executing PySpark scripts on Databricks...")
    
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
    
    # Set PYTHONPATH relative to execution
    env = os.environ.copy()
    # Resolve the src directory relative to this file's location to be robust across environments
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    env["PYTHONPATH"] = src_dir + os.pathsep + project_root + os.pathsep + env.get("PYTHONPATH", "")

    for s in scripts:
        path = os.path.join(code_dir, s)
        print(f"Executing {s}...")
        try:
            res = subprocess.run(
                 [sys.executable, path],
                 capture_output=True,
                 text=True,
                 env=env,
                 timeout=300
             )
            logs[s] = {
                "exit_code": res.returncode,
                "stdout": res.stdout[:15000],
                "stderr": res.stderr[:15000]
            }
            if res.returncode != 0:
                print(f"!!! Script {s} failed with exit code {res.returncode}")
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
    prompt = (
        f"Script execution logs:\n{json.dumps(logs, indent=2)}\n\n"
        f"Silver Validation summary:\n{json.dumps(silver_summary, indent=2)}\n\n"
        f"Gold schema summary:\n{json.dumps(gold_summary, indent=2)}"
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

    return {
        "execution_logs": logs,
        "silver_summary": silver_summary,
        "gold_summary": gold_summary,
        "final_report": final_report,
        "active_agent": "Orchestrator",
        "review_comments": ""
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
            return (
                "The pipeline hasn't been started yet. Please use the **Action Center** tab "
                "to kick off the first run, or ask me to start it and I'll prompt you to confirm.",
                False,
                None
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
