"""
graph.py
========
Wires up the LangGraph StateGraph, defines conditional routing paths,
and configures memory checkpointing to enable sequential breakpoints.
"""
from __future__ import annotations

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .agents import (
    profiler_node,
    contract_node,
    modeling_node,
    engineering_node,
    execution_node
)

# ---- Conditional Routing Functions ----

def route_after_profiler(state: AgentState) -> str:
    """Route after profiler node runs."""
    approvals = state.get("approved_steps", {})
    if approvals.get("profile") is True:
        return "contracts"
    return "profiler"  # Route back to profiler if rejected/unapproved


def route_after_contracts(state: AgentState) -> str:
    """Route after contract node runs."""
    approvals = state.get("approved_steps", {})
    if approvals.get("contracts") is True:
        return "modeling"
    return "contracts"  # Route back to contract node if rejected


def route_after_modeling(state: AgentState) -> str:
    """Route after modeling node runs."""
    approvals = state.get("approved_steps", {})
    if approvals.get("modeling") is True:
        return "engineering"
    return "modeling"  # Route back to modeling node if rejected


def route_after_engineering(state: AgentState) -> str:
    """Route after engineering node runs."""
    approvals = state.get("approved_steps", {})
    if approvals.get("engineering") is True:
        return "execution"
    return "engineering"  # Route back to engineering node if rejected


def route_after_execution(state: AgentState) -> str:
    """Route after execution node runs."""
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


# ---- Graph Compilation ----

def create_pipeline_graph():
    """Compile the LangGraph state machine with memory checkpointers."""
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("contracts", contract_node)
    workflow.add_node("modeling", modeling_node)
    workflow.add_node("engineering", engineering_node)
    workflow.add_node("execution", execution_node)

    # Wire Edges
    workflow.add_edge(START, "profiler")

    workflow.add_conditional_edges(
        "profiler",
        route_after_profiler,
        {
            "contracts": "contracts",
            "profiler": "profiler"
        }
    )

    workflow.add_conditional_edges(
        "contracts",
        route_after_contracts,
        {
            "modeling": "modeling",
            "contracts": "contracts"
        }
    )

    workflow.add_conditional_edges(
        "modeling",
        route_after_modeling,
        {
            "engineering": "engineering",
            "modeling": "modeling"
        }
    )

    workflow.add_conditional_edges(
        "engineering",
        route_after_engineering,
        {
            "execution": "execution",
            "engineering": "engineering"
        }
    )

    workflow.add_conditional_edges(
        "execution",
        route_after_execution,
        {
            "engineering": "engineering",
            "end": END
        }
    )

    # Setup Memory for Human-In-The-Loop Checkpointing
    memory = MemorySaver()

    # Compile the graph. Breakpoints are placed BEFORE each node transitions
    # (except the first node) to allow review of the previous node's state additions.
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["contracts", "modeling", "engineering", "execution"]
    )
    
    return app
