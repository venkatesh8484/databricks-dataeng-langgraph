"""
graph.py
========
Wires up the LangGraph StateGraph, defines conditional routing paths,
and configures memory checkpointing to enable sequential breakpoints.
Uses explicit review gate nodes to prevent tight infinite loop routing.
"""
from __future__ import annotations

from typing import Dict, Any, Literal, Iterator, Optional
import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from dbricks_lang_agent.data_platform.spark_utils import load_config

class PureSqliteSaver(BaseCheckpointSaver):
    """Pure Python SQLite Checkpointer with zero binary dependencies, safe for Databricks Apps containers."""
    def __init__(self, conn: sqlite3.Connection):
        super().__init__(serde=JsonPlusSerializer())
        self.conn = conn
        self._create_tables()

    def _create_tables(self):
        # Connection is in autocommit mode (isolation_level=None), so we
        # manage the DDL transaction explicitly to keep it atomic.
        cursor = self.conn.cursor()
        cursor.execute("BEGIN")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    parent_checkpoint_id TEXT,
                    checkpoint_format TEXT,
                    checkpoint_bytes BLOB,
                    metadata_format TEXT,
                    metadata_bytes BLOB,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    task_id TEXT,
                    idx INTEGER,
                    channel TEXT,
                    value_format TEXT,
                    value_bytes BLOB,
                    PRIMARY KEY (thread_id, checkpoint_id, task_id, idx)
                )
            """)
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        cursor = self.conn.cursor()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        if checkpoint_id:
            cursor.execute(
                "SELECT parent_checkpoint_id, checkpoint_format, checkpoint_bytes, metadata_format, metadata_bytes FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
                (thread_id, checkpoint_id)
            )
        else:
            cursor.execute(
                "SELECT parent_checkpoint_id, checkpoint_format, checkpoint_bytes, metadata_format, metadata_bytes, checkpoint_id FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id,)
            )
            
        row = cursor.fetchone()
        if not row:
            return None
            
        if checkpoint_id:
            parent_id, cp_fmt, cp_bytes, meta_fmt, meta_bytes = row
            curr_id = checkpoint_id
        else:
            parent_id, cp_fmt, cp_bytes, meta_fmt, meta_bytes, curr_id = row
            
        checkpoint = self.serde.loads_typed((cp_fmt, cp_bytes))
        metadata = self.serde.loads_typed((meta_fmt, meta_bytes))
        
        cursor.execute(
            "SELECT task_id, channel, value_format, value_bytes FROM writes WHERE thread_id = ? AND checkpoint_id = ?",
            (thread_id, curr_id)
        )
        pending_writes = []
        for task_id, channel, val_fmt, val_bytes in cursor.fetchall():
            value = self.serde.loads_typed((val_fmt, val_bytes))
            pending_writes.append((task_id, channel, value))
            
        config_out = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": curr_id
            }
        }
        
        parent_config = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": parent_id
                }
            }
            
        return CheckpointTuple(
            config=config_out,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes
        )

    def put(self, config: dict, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: dict) -> dict:
        cursor = self.conn.cursor()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")
        
        cp_fmt, cp_bytes = self.serde.dumps_typed(checkpoint)
        meta_fmt, meta_bytes = self.serde.dumps_typed(metadata)
        
        # Connection is in autocommit mode — each statement commits immediately.
        cursor.execute(
            "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_id, parent_checkpoint_id, checkpoint_format, checkpoint_bytes, metadata_format, metadata_bytes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (thread_id, checkpoint_id, parent_id, cp_fmt, cp_bytes, meta_fmt, meta_bytes)
        )
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id
            }
        }

    def put_writes(self, config: dict, writes: list, task_id: str) -> None:
        cursor = self.conn.cursor()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        
        for idx, (channel, value) in enumerate(writes):
            val_fmt, val_bytes = self.serde.dumps_typed(value)
            cursor.execute(
                "INSERT OR REPLACE INTO writes (thread_id, checkpoint_id, task_id, idx, channel, value_format, value_bytes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (thread_id, checkpoint_id, task_id, idx, channel, val_fmt, val_bytes)
            )
        # No explicit commit needed — autocommit mode is active.

    def list(self, config: dict, *, before: dict = None, limit: int = None) -> Iterator[CheckpointTuple]:
        cursor = self.conn.cursor()
        thread_id = config["configurable"]["thread_id"]
        query = "SELECT checkpoint_id, parent_checkpoint_id, checkpoint_format, checkpoint_bytes, metadata_format, metadata_bytes FROM checkpoints WHERE thread_id = ?"
        params = [thread_id]
        
        if before:
            query += " AND checkpoint_id < ?"
            params.append(before["configurable"]["checkpoint_id"])
            
        query += " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
            
        cursor.execute(query, tuple(params))
        for row in cursor.fetchall():
            curr_id, parent_id, cp_fmt, cp_bytes, meta_fmt, meta_bytes = row
            checkpoint = self.serde.loads_typed((cp_fmt, cp_bytes))
            metadata = self.serde.loads_typed((meta_fmt, meta_bytes))
            
            parent_config = None
            if parent_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": parent_id
                    }
                }
                
            yield CheckpointTuple(
                config={"configurable": {"thread_id": thread_id, "checkpoint_id": curr_id}},
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config
            )

from .state import AgentState
from .agents import (
    profiler_node,
    dq_node,
    contract_node,
    modeling_node,
    engineering_node,
    execution_node
)

# ---- Review Gate Nodes (Breakpoints are placed before these nodes) ----

def profile_review_gate(state: AgentState) -> Dict[str, Any]:
    return {"active_agent": "Profiler"}


def data_quality_review_gate(state: AgentState) -> Dict[str, Any]:
    return {"active_agent": "DataQualityAgent"}


def contracts_review_gate(state: AgentState) -> Dict[str, Any]:
    return {"active_agent": "ContractSteward"}


def modeling_review_gate(state: AgentState) -> Dict[str, Any]:
    return {"active_agent": "DimensionalModeler"}


def engineering_review_gate(state: AgentState) -> Dict[str, Any]:
    return {"active_agent": "DataEngineer"}


def execution_review_gate(state: AgentState) -> Dict[str, Any]:
    return {"active_agent": "Orchestrator"}


# ---- Conditional Routing Functions ----

def route_after_profiler(state: AgentState) -> str:
    """Route after profile_review_gate node runs."""
    approvals = state.get("approved_steps", {})
    if approvals.get("profile") is True:
        return "data_quality"
    return "profiler"  # Route back to profiler if rejected/unapproved


def route_after_dq(state: AgentState) -> str:
    """Route after data_quality_review_gate node runs."""
    approvals = state.get("approved_steps", {})
    if approvals.get("dq") is True:
        return "contracts"
    return "data_quality"  # Route back to data_quality node if rejected


def route_after_contracts(state: AgentState) -> str:
    """Route after contracts_review_gate node runs."""
    approvals = state.get("approved_steps", {})
    if approvals.get("contracts") is True:
        return "modeling"
    return "contracts"  # Route back to contract node if rejected


def route_after_modeling(state: AgentState) -> str:
    """Route after modeling_review_gate node runs."""
    approvals = state.get("approved_steps", {})
    if approvals.get("modeling") is True:
        return "engineering"
    return "modeling"  # Route back to modeling node if rejected


def route_after_engineering(state: AgentState) -> str:
    """Route after engineering_review_gate node runs."""
    approvals = state.get("approved_steps", {})
    if approvals.get("engineering") is True:
        return "execution"
    return "engineering"  # Route back to engineering node if rejected


def route_after_execution(state: AgentState) -> str:
    """Route after execution_review_gate node runs."""
    approvals = state.get("approved_steps", {})
    if approvals.get("report") is True:
        return END
    
    # Check if there are execution failures, route back to engineering to auto-fix
    # or route back if human rejects the final run report.
    logs = state.get("execution_logs", {})
    has_failure = any(l.get("exit_code", 0) != 0 for l in logs.values())
    if has_failure or approvals.get("report") is False:
        return "engineering"
        
    return END


def get_checkpoint_db_path() -> str:
    """Return the local checkpoint database path (using local disk to prevent SQLite network lock errors)."""
    return "/tmp/checkpoint.db"


# ---- Graph Compilation ----

def create_pipeline_graph():
    """Compile the LangGraph state machine with memory checkpointers."""
    workflow = StateGraph(AgentState)

    # Register Agent Nodes
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("data_quality", dq_node)
    workflow.add_node("contracts", contract_node)
    workflow.add_node("modeling", modeling_node)
    workflow.add_node("engineering", engineering_node)
    workflow.add_node("execution", execution_node)

    # Register Gate Nodes
    workflow.add_node("profile_review_gate", profile_review_gate)
    workflow.add_node("data_quality_review_gate", data_quality_review_gate)
    workflow.add_node("contracts_review_gate", contracts_review_gate)
    workflow.add_node("modeling_review_gate", modeling_review_gate)
    workflow.add_node("engineering_review_gate", engineering_review_gate)
    workflow.add_node("execution_review_gate", execution_review_gate)

    # Wire Edges
    workflow.add_edge(START, "profiler")
    
    # Profiler -> Gate -> Data Quality
    workflow.add_edge("profiler", "profile_review_gate")
    workflow.add_conditional_edges(
        "profile_review_gate",
        route_after_profiler,
        {
            "data_quality": "data_quality",
            "profiler": "profiler"
        }
    )

    # Data Quality -> Gate -> Contracts
    workflow.add_edge("data_quality", "data_quality_review_gate")
    workflow.add_conditional_edges(
        "data_quality_review_gate",
        route_after_dq,
        {
            "contracts": "contracts",
            "data_quality": "data_quality"
        }
    )

    # Contracts -> Gate -> Modeling
    workflow.add_edge("contracts", "contracts_review_gate")
    workflow.add_conditional_edges(
        "contracts_review_gate",
        route_after_contracts,
        {
            "modeling": "modeling",
            "contracts": "contracts"
        }
    )

    # Modeling -> Gate -> Engineering
    workflow.add_edge("modeling", "modeling_review_gate")
    workflow.add_conditional_edges(
        "modeling_review_gate",
        route_after_modeling,
        {
            "engineering": "engineering",
            "modeling": "modeling"
        }
    )

    # Engineering -> Gate -> Execution
    workflow.add_edge("engineering", "engineering_review_gate")
    workflow.add_conditional_edges(
        "engineering_review_gate",
        route_after_engineering,
        {
            "execution": "execution",
            "engineering": "engineering"
        }
    )

    # Execution -> Gate -> End
    workflow.add_edge("execution", "execution_review_gate")
    workflow.add_conditional_edges(
        "execution_review_gate",
        route_after_execution,
        {
            "engineering": "engineering",
            "__end__": END
        }
    )

    # Setup Persistent Sqlite Checkpointer for sharing state between notebook and dashboard app
    db_path = get_checkpoint_db_path()

    print(f"[Info] LangGraph Checkpointer using Sqlite database: {db_path}")

    # Ensure the file is writable before opening (it may have been synced from
    # Volume with restrictive permissions, causing "readonly database" errors).
    if os.path.exists(db_path):
        try:
            os.chmod(db_path, 0o666)
        except OSError:
            pass

    # Open in read-write-create mode via URI. isolation_level=None puts the
    # connection in autocommit mode so LangGraph's put() + put_writes() calls
    # never hit "cannot start a transaction within a transaction".
    conn = sqlite3.connect(
        f"file:{db_path}?mode=rwc",
        uri=True,
        check_same_thread=False,
        isolation_level=None,
    )
    # WAL journal mode allows concurrent readers while a write is in progress,
    # which is important when the notebook and dashboard share the same DB file.
    conn.execute("PRAGMA journal_mode=WAL")
    memory = PureSqliteSaver(conn)


    # Compile the graph. Breakpoints are placed BEFORE each review gate node,
    # forcing the graph to pause execution and yield control back to the notebook runner.
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=[
            "profile_review_gate", 
            "data_quality_review_gate",
            "contracts_review_gate", 
            "modeling_review_gate", 
            "engineering_review_gate", 
            "execution_review_gate"
        ]
    )
    
    return app
