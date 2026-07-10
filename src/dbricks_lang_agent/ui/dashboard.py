"""
dashboard.py
============
Streamlit dashboard for monitoring the Medallion Pipeline and acting as the 
Human-in-the-Loop approval client. Designed to run natively inside Databricks.
"""
from __future__ import annotations

import os
import sys
import json
import yaml
import streamlit as st
import pandas as pd
from datetime import datetime

# Add project root to sys path
sys.path.append(os.path.abspath("src"))

from dbricks_lang_agent.data_platform.spark_utils import get_spark, load_config
from dbricks_lang_agent.orchestrator.graph import create_pipeline_graph, get_checkpoint_db_path
from dbricks_lang_agent.orchestrator import memory
from databricks.sdk import WorkspaceClient
import io

def sync_db_from_volume():
    """Download checkpoint.db from the shared UC Volume using Databricks SDK (FUSE is not mounted in App container)."""
    # Always sync if DATABRICKS_APP_NAME is set, meaning we are inside the App environment
    if os.environ.get("DATABRICKS_APP_NAME"):
        try:
            cfg = load_config()
            catalog = cfg.get("catalog", "hospitality_catalog")
            raw_volume = cfg.get("raw_volume", "raw/source_volume")
            # In Databricks, volume raw path can also be configured directly
            volume_db_path = os.path.join(cfg.get("volume_raw_path", f"/Volumes/{catalog}/{raw_volume}"), "checkpoint.db")
            local_db_path = get_checkpoint_db_path()
            
            w = WorkspaceClient()
            try:
                response = w.files.download(volume_db_path)
                with open(local_db_path, "wb") as f:
                    f.write(response.contents.read())
                print(f"[Debug] Successfully downloaded volume checkpoint to {local_db_path}")
            except Exception as e_dl:
                print(f"[Debug] Could not download checkpoint (might not exist yet): {e_dl}")
        except Exception as e:
            print(f"[Warning] Failed to sync db from volume: {e}")

def sync_db_to_volume():
    """Upload checkpoint.db to the shared UC Volume using Databricks SDK."""
    if os.environ.get("DATABRICKS_APP_NAME"):
        try:
            cfg = load_config()
            catalog = cfg.get("catalog", "hospitality_catalog")
            raw_volume = cfg.get("raw_volume", "raw/source_volume")
            volume_db_path = os.path.join(cfg.get("volume_raw_path", f"/Volumes/{catalog}/{raw_volume}"), "checkpoint.db")
            local_db_path = get_checkpoint_db_path()
            
            w = WorkspaceClient()
            with open(local_db_path, "rb") as f:
                file_data = f.read()
            w.files.upload(volume_db_path, io.BytesIO(file_data), overwrite=True)
            print(f"[Debug] Successfully uploaded local checkpoint to volume.")
        except Exception as e:
            print(f"[Warning] Failed to sync db to volume: {e}")

# Page configuration for premium visuals
st.set_page_config(
    page_title="Medallion Agent Control Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: #888888;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background-color: #1e1e1e;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #333333;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #FF4B4B;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #888888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0.5rem;
        }
        
        .agent-badge {
            background-color: #2e2e2e;
            border: 1px solid #FF4B4B;
            color: #FF8F8F;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- Initialize Spark & Graph -----------------

@st.cache_resource
def get_spark_session():
    return get_spark()

@st.cache_resource
def get_graph():
    """Build the graph once and cache it. The SQLite connection is reused across reruns.
    Call refresh_graph_db() after syncing from volume to ensure the checkpointer reads fresh data.
    """
    return create_pipeline_graph()

def refresh_graph_checkpoint():
    """Close and reopen the SQLite connection so the cached graph reads the freshly synced file."""
    try:
        app_instance = get_graph()
        # Access the underlying SQLite connection via the checkpointer
        checkpointer = app_instance.checkpointer
        if hasattr(checkpointer, 'conn'):
            db_path = checkpointer.conn.database if hasattr(checkpointer.conn, 'database') else get_checkpoint_db_path()
            checkpointer.conn.close()
            import sqlite3
            checkpointer.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    except Exception as e:
        print(f"[Warning] Could not refresh graph DB connection: {e}")

spark = get_spark_session()

# Always sync checkpoint from Volume on every page load so state is current
sync_db_from_volume()
refresh_graph_checkpoint()

app = get_graph()
thread_id = "medallion_pipeline_run"
config = {"configurable": {"thread_id": thread_id}}

# ----------------- Dashboard Layout -----------------

st.markdown('<p class="main-title">🤖 Medallion Agent Control Center</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Unity Catalog Governed Multi-Agent Data Engineering Ingestion & Transformation</p>', unsafe_allow_html=True)

# Fetch Current Graph State
state = app.get_state(config)

# Sidebar with Status Overview
st.sidebar.markdown("### Ingestion Settings")
catalog_config = load_config()
st.sidebar.info(f"**Unity Catalog**: {catalog_config.get('catalog', 'hospitality_catalog')}\n\n**Raw Volume**: {catalog_config.get('raw_volume', 'raw/source_volume')}")

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    sync_db_from_volume()
    refresh_graph_checkpoint()
    st.rerun()

# Layout Tabs
tab1, tab2, tab3 = st.tabs(["📊 Ingestion Monitoring", "📥 Action Center (HITL)", "🧠 Agent Memories"])

# ----------------- Tab 1: Ingestion Monitoring -----------------
with tab1:
    st.markdown("### 📈 Ingestion Pipeline Metrics")
    
    # Try fetching table statistics from Unity Catalog
    try:
        catalog = catalog_config.get("catalog", "hospitality_catalog")
        # List of candidate schemas to profile counts
        schemas = ["raw", "bronze", "silver", "gold"]
        counts = {}
        for s in schemas:
            tbl_count = 0
            row_count = 0
            try:
                tables_df = spark.sql(f"SHOW TABLES IN {catalog}.{s}")
                tables = [r.tableName for r in tables_df.collect()]
                tbl_count = len(tables)
                for t in tables:
                    row_count += spark.sql(f"SELECT COUNT(*) as cnt FROM {catalog}.{s}.{t}").collect()[0].cnt
            except Exception:
                pass
            counts[s] = (tbl_count, row_count)
            
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{counts["raw"][1]}</div><div class="metric-label">Raw Source Files Rows</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{counts["bronze"][1]}</div><div class="metric-label">Bronze Delta Tables Rows ({counts["bronze"][0]} tables)</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{counts["silver"][1]}</div><div class="metric-label">Silver Validated Rows ({counts["silver"][0]} tables)</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{counts["gold"][1]}</div><div class="metric-label">Gold Star Schema Rows ({counts["gold"][0]} tables)</div></div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.warning(f"Unable to query live table metrics: {e}")
        
    st.markdown("---")
    st.markdown("### 📝 Pipeline Run Details")
    if not state.values:
        st.write("No active pipeline execution logs found in checkpointer. Start the run from the notebook.")
    else:
        st.write(f"**Current State Pointer**: `{state.next[0] if state.next else 'FINISHED'}`")
        st.write(f"**Last Executing Agent**: `{state.values.get('active_agent', 'None')}`")
        
        # Display execution logs if they exist
        logs = state.values.get("execution_logs", {})
        if logs:
            st.markdown("#### ETL Executions stdout/stderr Logs")
            for script_name, log_info in logs.items():
                with st.expander(f"Script: {script_name} (Exit Code: {log_info.get('exit_code')})"):
                    st.code(log_info.get("stdout") or "No stdout output.")
                    if log_info.get("stderr"):
                        st.error(log_info.get("stderr"))
        else:
            st.info("No ETL script execution logs generated yet. Awaiting agent schema design and coding steps.")

# ----------------- Tab 2: Action Center (HITL) -----------------
with tab2:
    st.markdown("### 📥 Human-in-the-Loop Pending Approvals")
    
    if not state.next:
        st.success("🎉 Pipeline is completely finished! No actions pending.")
    else:
        current_node = state.next[0]
        active_agent = state.values.get("active_agent", "None")
        
        st.warning(f"⚠️ **Action Required**: Pipeline is suspended BEFORE running node: **`{current_node}`**")
        st.markdown(f"Review the artifacts generated by the previous agent (**{active_agent}**) below:")
        
        # Define step mappings for user actions
        step_mapping = {
            "profile_review_gate": "profile",
            "data_quality_review_gate": "dq",
            "contracts_review_gate": "contracts",
            "modeling_review_gate": "modeling",
            "engineering_review_gate": "engineering",
            "execution_review_gate": "report"
        }
        step_key = step_mapping.get(current_node)
        
        # Render specific artifact content based on what node is paused
        if active_agent == "Profiler":
            st.markdown("### Inferred Source Table Schemas:")
            st.json(state.values.get("discovered_tables", {}))
            st.markdown("### Dynamic Profiling Report:")
            st.markdown(state.values.get("profiling_report", {}).get("profiler_narration", "No report narrative found."))
            
        elif active_agent == "DataQualityAgent":
            st.markdown("### Data Quality Assessment Report:")
            st.markdown(state.values.get("dq_report", "No DQ report found."))
            
        elif active_agent == "ContractSteward":
            st.markdown("### Generated YAML Data Contracts:")
            contracts = state.values.get("contracts", {})
            for tbl, contract_yaml in contracts.items():
                st.subheader(f"Table Contract: {tbl}")
                st.code(contract_yaml, language="yaml")
                
        elif active_agent == "DimensionalModeler":
            st.markdown("### Inferred Star Schema SQL DDL:")
            st.code(state.values.get("gold_ddl", ""), language="sql")
            st.markdown("### Star Schema Data Dictionary:")
            st.markdown(state.values.get("data_dictionary", ""))
            
        elif active_agent == "DataEngineer":
            st.markdown("### Generated PySpark Code Blocks:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Bronze Ingest (`bronze.py`)**")
                st.code(state.values.get("bronze_code", ""), language="python")
            with col2:
                st.markdown("**Silver Validate (`silver.py`)**")
                st.code(state.values.get("silver_code", ""), language="python")
            with col3:
                st.markdown("**Gold Load (`gold.py`)**")
                st.code(state.values.get("gold_code", ""), language="python")
                
        elif active_agent == "Orchestrator":
            st.markdown("### Executed Script Logs:")
            st.json(state.values.get("execution_logs", {}))
            st.markdown("### Final Run Summary Report:")
            st.markdown(state.values.get("final_report", ""))

        st.markdown("---")
        st.markdown("#### Submit Decision")
        
        # Approve/Reject controls
        action = st.radio("Review Decision", ["Approve", "Reject"], index=0, horizontal=True)
        feedback = st.text_area("Review Feedback / Comments", placeholder="Enter comments or instructions if rejecting, or additional context...")
        
        if st.button("Submit & Resume Pipeline"):
            if step_key:
                approvals = dict(state.values.get("approved_steps", {}))
                
                if action == "Approve":
                    st.success(f"Approving step '{step_key}'...")
                    approvals[step_key] = True
                    comments = ""
                    
                    # Log approval to memory Delta Table
                    try:
                        dataset = list(state.values.get("discovered_tables", {}).keys())[0] if state.values.get("discovered_tables") else "generic"
                        issue_type = "data_quality" if step_key == "dq" else step_key
                        resolution = f"Approved {step_key} design"
                        memory.log_approval(spark, dataset, issue_type, [], resolution, feedback)
                    except Exception as e:
                        st.warning(f"Unable to log memory to table: {e}")
                else:
                    st.error(f"Rejecting step '{step_key}'...")
                    approvals[step_key] = False
                    comments = feedback
                    
                # Update graph checkpointer state
                app.update_state(
                    config,
                    {
                        "approved_steps": approvals,
                        "review_comments": comments
                    }
                )
                
                # Resume execution in background thread / stream
                with st.spinner("Resuming pipeline execution with new state context... Please wait..."):
                    events = app.stream(None, config, stream_mode="values")
                    for event in events:
                        pass
                
                # Sync state back to Unity Catalog Volume
                sync_db_to_volume()
                
                st.success("Pipeline resumed! Refreshing dashboard...")
                st.rerun()
            else:
                st.error("Invalid state. Unable to route step mapping approvals.")

# ----------------- Tab 3: Agent Memories -----------------
with tab3:
    st.markdown("### 🧠 Agent Memory Catalog")
    st.write("Below are the historical decisions logged into the few-shot memory system. The agents read this memory before executing steps to reuse your past resolutions.")
    
    try:
        catalog = catalog_config.get("catalog", "hospitality_catalog")
        # Try displaying from Delta table
        try:
            is_local = spark.conf.get("spark.master", "").startswith("local")
        except Exception:
            is_local = False
        
        if not is_local:
            fqn = f"{catalog}.gold.agent_fewshot_memory"
            df = spark.read.table(fqn).orderBy("timestamp", ascending=False)
            pandas_df = df.toPandas()
            if len(pandas_df) > 0:
                st.dataframe(pandas_df, use_container_width=True)
            else:
                st.info("Few-shot memory table is empty. Approvals will appear here once submitted.")
        else:
            # Local fallback json file
            if os.path.exists(memory.LOCAL_MEMORY_PATH):
                with open(memory.LOCAL_MEMORY_PATH, "r") as f:
                    records = json.load(f)
                if records:
                    st.dataframe(pd.DataFrame(records), use_container_width=True)
                else:
                    st.info("Local memory cache is empty.")
            else:
                st.info("Local memory cache has not been created yet.")
                
    except Exception as e:
        st.warning(f"Could not load memory catalog view: {e}")
