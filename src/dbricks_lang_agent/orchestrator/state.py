"""
state.py
========
Defines the LangGraph AgentState representing the pipeline metadata,
generated artifacts, execution logs, and Human-in-the-Loop approval flags.
"""
from __future__ import annotations

from typing import Dict, Any, List, TypedDict, Optional

class AgentState(TypedDict):
    # Metadata and discovered sources
    discovered_tables: Dict[str, str]   # table_name -> CSV filename

    # Reports and schemas
    profiling_report: Dict[str, Any]    # JSON profiling metrics
    contracts: Dict[str, str]           # table_name -> YAML contract string
    gold_ddl: str                       # Gold star schema DDL (SQL)
    data_dictionary: str                # Data dictionary (markdown)

    # Generated code files
    bronze_code: str                    # PySpark code for Bronze ingest
    silver_code: str                    # PySpark code for Silver transform
    gold_code: str                      # PySpark code for Gold schema build

    # Execution results
    execution_logs: Dict[str, Any]      # exit_code, stdout, stderr per script
    silver_summary: Dict[str, Any]      # promoting/quarantine counts from silver
    gold_summary: Dict[str, Any]        # gold loading details
    final_report: str                   # Executive run report (markdown)
    dq_report: str                      # Data Quality Assessment report (markdown)

    # Human-in-the-Loop review and orchestration state
    active_agent: str                   # Name of the currently executing agent node
    review_comments: str                # Human feedback entered during rejection
    approved_steps: Dict[str, bool]     # Step approvals: 'profile', 'dq', 'contracts', 'modeling', 'engineering', 'report'

    # Diagnostic / error state fields — set when an agent cannot proceed
    profiler_error: str                 # Set by profiler_node when no tables discovered
    contracts_error: str                # Set by contract_node when it cannot generate contracts

    # LangGraph routing variables
    loop_count: int                     # Tracks consecutive agent rejections — capped at MAX_AGENT_RETRIES to prevent infinite loops
