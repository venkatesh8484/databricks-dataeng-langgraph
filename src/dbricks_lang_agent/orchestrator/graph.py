"""
graph.py
========
Wires up the LangGraph StateGraph, defines conditional routing paths,
and configures memory checkpointing to enable sequential breakpoints.
Uses explicit review gate nodes to prevent tight infinite loop routing.
"""
from __future__ import annotations

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

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
            "end": END
        }
    )

    # Setup Memory for Human-In-The-Loop Checkpointing
    memory = MemorySaver()

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
