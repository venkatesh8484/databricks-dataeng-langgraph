"""
agents.py
=========
Implements the LangGraph agent node functions.
Connects to Databricks Model Serving endpoints using ChatDatabricks,
handles prompt templating, JSON parsing, and execution of generated Spark ETL code.
"""
from __future__ import annotations

import os
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
    contracts_dir = "./generated/config/contracts"
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
    ddl_dir = "./generated/data_model"
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
    bronze = parsed.get("bronze_code", "")
    silver = parsed.get("silver_code", "")
    gold = parsed.get("gold_code", "")

    # Save code files to generated directory
    code_dir = "./generated/data_platform"
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
    
    code_dir = "./generated/data_platform"
    scripts = ["bronze.py", "silver.py", "gold.py"]
    logs = {}
    
    # Set PYTHONPATH relative to execution
    env = os.environ.copy()
    project_root = os.path.abspath(os.getcwd())
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    for s in scripts:
        path = os.path.join(code_dir, s)
        print(f"Executing {s}...")
        try:
            res = subprocess.run(
                ["python3", path],
                capture_output=True,
                text=True,
                env=env,
                timeout=300
            )
            logs[s] = {
                "exit_code": res.returncode,
                "stdout": res.stdout[-4000:],
                "stderr": res.stderr[-4000:]
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
