# Databricks notebook source
# Medallion Pipeline Agent Runner
#
# Click "Run All" to start the pipeline. The execution will pause at key breakpoints
# for you to review agent outputs.
#
# COMMAND ----------

# MAGIC %md
# MAGIC # Databricks + LangGraph Medallion Pipeline
# MAGIC Execution runs sequentially and halts for Human-in-the-Loop review before each layer promotion.

# COMMAND ----------

# MAGIC %pip install langgraph>=0.1.0 langchain>=0.2.0 langchain-community>=0.2.0 databricks-sdk>=0.28.0 pyyaml>=6.0 typing-extensions>=4.13.0 databricks-langchain

# COMMAND ----------

import os
import sys
import json

# Ensure project root is on Python path
sys.path.append(os.path.abspath("src"))

from dbricks_lang_agent.orchestrator.graph import create_pipeline_graph
from dbricks_lang_agent.orchestrator.state import AgentState

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup Databricks Notebook Widgets
# MAGIC We use widgets to handle approvals/rejections and capture review comments.

# COMMAND ----------

dbutils.widgets.dropdown("hitl_action", "Approve", ["Approve", "Reject"], "Human Action")
dbutils.widgets.text("hitl_feedback", "", "Review Feedback")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize the LangGraph State Machine
# MAGIC Setup the thread context and compile the graph.

# COMMAND ----------

app = create_pipeline_graph()
thread_id = "medallion_pipeline_run"
config = {"configurable": {"thread_id": thread_id}}

# COMMAND ----------

def get_current_node(state) -> str:
    if not state.next:
        return "FINISHED"
    return state.next[0]

def print_artifacts(state_values):
    agent = state_values.get("active_agent", "None")
    print(f"===========================================================")
    print(f"LAST ACTIVE AGENT: {agent}")
    print(f"===========================================================\n")
    
    if agent == "Profiler":
        print("### Dynamic Source Profiling Report Narrative:")
        print(state_values.get("profiling_report", {}).get("profiler_narration", "No report narrative found."))
        print("\n### Inferred Schemas & Table Grains:")
        print(json.dumps(state_values.get("discovered_tables", {}), indent=2))
        
    elif agent == "ContractSteward":
        print("### Authored YAML Data Contracts:")
        contracts = state_values.get("contracts", {})
        for tbl, contract_yaml in contracts.items():
            print(f"--- Contract: {tbl} ---")
            print(contract_yaml)
            print("-" * 30)
            
    elif agent == "DimensionalModeler":
        print("### Gold Star Schema DDL SQL:")
        print(state_values.get("gold_ddl", ""))
        print("\n### Data Dictionary MD:")
        print(state_values.get("data_dictionary", ""))
        
    elif agent == "DataEngineer":
        print("### Generated PySpark Code Blocks:")
        print("\n--- 1. Bronze Ingestion Code (`bronze.py`) ---")
        print(state_values.get("bronze_code", ""))
        print("\n--- 2. Silver Transformation Code (`silver.py`) ---")
        print(state_values.get("silver_code", ""))
        print("\n--- 3. Gold Dimensional Code (`gold.py`) ---")
        print(state_values.get("gold_code", ""))
        
    elif agent == "Orchestrator":
        print("### Script Execution Logs:")
        print(json.dumps(state_values.get("execution_logs", {}), indent=2))
        print("\n### Final Executive Run Report:")
        print(state_values.get("final_report", ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run / Resume Pipeline Graph
# MAGIC Execute this cell to start or resume the state machine. If it hits an approval breakpoint,
# MAGIC it will pause and print the details for your review.

# COMMAND ----------

# Get active state
state = app.get_state(config)

if not state.values:
    # First Kickoff
    print("Initiating pipeline run: Starting Profiler Agent...")
    initial_input = {
        "approved_steps": {},
        "loop_count": 0,
        "active_agent": "Start",
        "review_comments": ""
    }
    events = app.stream(initial_input, config, stream_mode="values")
    for event in events:
        state = app.get_state(config)
else:
    # Resume after human feedback
    action = dbutils.widgets.get("hitl_action")
    feedback = dbutils.widgets.get("hitl_feedback")
    
    current_node = get_current_node(state)
    active_agent = state.values.get("active_agent")
    
    # Map next node to corresponding step approval flag
    step_mapping = {
        "contracts": "profile",
        "modeling": "contracts",
        "engineering": "modeling",
        "execution": "engineering",
        "report": "report" # Final report approval
    }
    
    step_key = step_mapping.get(current_node)
    
    if step_key:
        approvals = dict(state.values.get("approved_steps", {}))
        
        if action == "Approve":
            print(f"Review APPROVED for step '{step_key}'. Resuming pipeline execution...")
            approvals[step_key] = True
            comments = ""
        else:
            print(f"Review REJECTED with comments. Re-routing back to agent...")
            approvals[step_key] = False
            comments = feedback
            
        # Update state with human action
        app.update_state(
            config,
            {
                "approved_steps": approvals,
                "review_comments": comments
            }
        )
        
        # Reset the widgets inputs for safety
        dbutils.widgets.remove("hitl_feedback")
        dbutils.widgets.text("hitl_feedback", "", "Review Feedback")
        
        # Resume graph execution
        events = app.stream(None, config, stream_mode="values")
        for event in events:
            pass
            
    else:
        print("Pipeline is already finished or in an invalid state.")

# Re-read state after run/resume
state = app.get_state(config)
next_node = get_current_node(state)

if next_node == "FINISHED":
    print("\n===========================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("===========================================================")
    print_artifacts(state.values)
else:
    print(f"\n[PAUSED] Pipeline halted at breakpoint BEFORE running node: '{next_node}'")
    print("Review the outputs generated by the previous agent below.")
    print_artifacts(state.values)
    print("\n-----------------------------------------------------------")
    print("INSTRUCTIONS FOR APPROVAL:")
    print(f"1. Check the inputs/metrics displayed above.")
    print(f"2. Set the 'Human Action' widget in the top panel to 'Approve' or 'Reject'.")
    print(f"3. (Optional) Provide comments in 'Review Feedback' if rejecting.")
    print(f"4. Re-run this cell to resume execution.")
