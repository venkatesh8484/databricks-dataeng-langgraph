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

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import json

# Ensure project root is on Python path
sys.path.append(os.path.abspath("src"))

# Force reload custom package modules to prevent Databricks caching old code in memory
for mod in list(sys.modules.keys()):
    if mod.startswith("dbricks_lang_agent"):
        del sys.modules[mod]

from dbricks_lang_agent.orchestrator.graph import create_pipeline_graph
from dbricks_lang_agent.orchestrator.state import AgentState

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup Databricks Notebook Widgets
# MAGIC We use widgets to handle approvals/rejections and capture review comments.

# COMMAND ----------

dbutils.widgets.dropdown("hitl_action", "Approve", ["Approve", "Reject"], "Human Action")
dbutils.widgets.text("hitl_feedback", "", "Review Feedback")
dbutils.widgets.dropdown("reset_pipeline", "False", ["True", "False"], "Reset Pipeline (Start Fresh)")

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
        
    elif agent == "DataQualityAgent":
        print("### Data Quality Assessment Report:")
        print(state_values.get("dq_report", "No DQ report found."))
        
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
# Sync database from Unity Catalog Volume to local SSD (/tmp/checkpoint.db) to prevent network database locking errors
def sync_db_from_volume():
    """Copy checkpoint.db from UC Volume to local /tmp using Python native I/O.
    Avoids dbutils.fs.cp which is blocked for file:/tmp/ paths on Serverless compute.
    """
    try:
        from dbricks_lang_agent.data_platform.spark_utils import load_config
        import shutil
        cfg = load_config()
        catalog = cfg.get("catalog", "hospitality_catalog")
        raw_volume = cfg.get("raw_volume", "raw/source_volume")
        volume_db = os.path.join(cfg.get("volume_raw_path", f"/Volumes/{catalog}/{raw_volume}"), "checkpoint.db")
        local_db = "/tmp/checkpoint.db"

        if os.path.exists(volume_db):
            shutil.copy2(volume_db, local_db)
            print(f"[Info] Synced checkpoint database from Volume to local disk: {local_db}")
        else:
            print("[Info] No existing checkpoint database found in volume. Starting fresh.")
    except Exception as e_sync:
        print(f"[Warning] Local checkpoint sync from volume failed: {e_sync}")

def sync_db_to_volume():
    """Copy checkpoint.db from local /tmp back to UC Volume using Python native I/O.
    Avoids dbutils.fs.cp which is blocked for file:/tmp/ paths on Serverless compute.
    Falls back to Databricks SDK files API if direct POSIX copy fails.
    """
    try:
        from dbricks_lang_agent.data_platform.spark_utils import load_config
        import shutil
        cfg = load_config()
        catalog = cfg.get("catalog", "hospitality_catalog")
        raw_volume = cfg.get("raw_volume", "raw/source_volume")
        volume_db = os.path.join(cfg.get("volume_raw_path", f"/Volumes/{catalog}/{raw_volume}"), "checkpoint.db")
        local_db = "/tmp/checkpoint.db"

        if not os.path.exists(local_db):
            print("[Info] No local checkpoint database to sync.")
            return

        # Primary: direct POSIX copy to /Volumes/ mount (works on Notebooks)
        try:
            os.makedirs(os.path.dirname(volume_db), exist_ok=True)
            shutil.copy2(local_db, volume_db)
            print(f"[Info] Synced checkpoint database back to Volume: {volume_db}")
            return
        except Exception as e_posix:
            print(f"[Info] Direct POSIX copy failed ({e_posix}), falling back to Databricks SDK...")

        # Fallback: Databricks Files API (works in all environments)
        try:
            import io
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            with open(local_db, "rb") as f:
                file_data = f.read()
            w.files.upload(volume_db, io.BytesIO(file_data), overwrite=True)
            print(f"[Info] Synced checkpoint database to Volume via SDK: {volume_db}")
        except Exception as e_sdk:
            print(f"[Warning] Checkpoint database sync to volume failed (SDK): {e_sdk}")
    except Exception as e_sync:
        print(f"[Warning] Checkpoint database sync to volume failed: {e_sync}")

# Check if reset is requested
try:
    reset_requested = dbutils.widgets.get("reset_pipeline") == "True"
    if reset_requested:
        print("[Reset] Wiping out local and Volume checkpoint databases to start a fresh run...")
        local_db = "/tmp/checkpoint.db"
        if os.path.exists(local_db):
            os.remove(local_db)
            
        from dbricks_lang_agent.data_platform.spark_utils import load_config
        cfg = load_config()
        catalog = cfg.get("catalog", "hospitality_catalog")
        raw_volume = cfg.get("raw_volume", "raw/source_volume")
        volume_db = os.path.join(cfg.get("volume_raw_path", f"/Volumes/{catalog}/{raw_volume}"), "checkpoint.db")
        if os.path.exists(volume_db):
            os.remove(volume_db)
            print(f"[Reset] Deleted Volume database file: {volume_db}")
            
        # Try deleting via Workspace SDK as fallback
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            w.files.delete(volume_db)
        except Exception:
            pass
            
        # Reset widget back to False for safety
        dbutils.widgets.remove("reset_pipeline")
        dbutils.widgets.dropdown("reset_pipeline", "False", ["True", "False"], "Reset Pipeline (Start Fresh)")
except Exception as e_reset:
    print(f"[Warning] Reset check skipped: {e_reset}")

# Get active state
sync_db_from_volume()
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
    sync_db_to_volume()
else:
    # Resume after human feedback
    action = dbutils.widgets.get("hitl_action")
    feedback = dbutils.widgets.get("hitl_feedback")
    
    current_node = get_current_node(state)
    active_agent = state.values.get("active_agent")
    
    # Map next node to corresponding step approval flag
    step_mapping = {
        "profile_review_gate": "profile",
        "data_quality_review_gate": "dq",
        "contracts_review_gate": "contracts",
        "modeling_review_gate": "modeling",
        "engineering_review_gate": "engineering",
        "execution_review_gate": "report"
    }
    
    step_key = step_mapping.get(current_node)
    
    if step_key:
        approvals = dict(state.values.get("approved_steps", {}))
        
        if action == "Approve":
            print(f"Review APPROVED for step '{step_key}'. Resuming pipeline execution...")
            approvals[step_key] = True
            comments = ""
            try:
                from dbricks_lang_agent.orchestrator import memory
                from dbricks_lang_agent.data_platform.spark_utils import get_spark
                spark = get_spark()
                dataset = list(state.values.get("discovered_tables", {}).keys())[0] if state.values.get("discovered_tables") else "generic"
                issue_type = "data_quality" if step_key == "dq" else step_key
                resolution = f"Approved {step_key} design"
                memory.log_approval(spark, dataset, issue_type, [], resolution, feedback)
            except Exception as e:
                print(f"[Warning] Failed to log approval to few-shot memory: {e}")
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
        sync_db_to_volume()
            
    else:
        if current_node != "FINISHED":
            print(f"Resuming execution from node '{current_node}'...")
            events = app.stream(None, config, stream_mode="values")
            for event in events:
                pass
            sync_db_to_volume()
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
    print("1. Open your Streamlit Control Center app (playground).")
    print("2. Go to the 'Action Center (HITL)' tab to review the generated outputs.")
    print("3. Select 'Approve' or 'Reject', provide comments, and click 'Submit & Resume Pipeline'.")
    print("4. Once submitted, the app will automatically resume the pipeline execution.")
    print("\n*(Note: You can also choose to use the widgets above and re-run this cell as a fallback).*")
