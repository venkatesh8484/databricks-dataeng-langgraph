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

def get_volume_db_path() -> str:
    """Return the Volume POSIX path for the checkpoint database.
    Both the notebook and the app can access /Volumes/ paths directly.
    """
    try:
        cfg = load_config()
        volume_raw_path = cfg.get("volume_raw_path", "/Volumes/databricks_langgraph/raw/source_volume")
        return os.path.join(volume_raw_path, "checkpoint.db")
    except Exception:
        return "/Volumes/databricks_langgraph/raw/source_volume/checkpoint.db"

def copy_file_binary(src, dst):
    """Copy file contents chunk by chunk without copying metadata or permissions.
    Deletes the target file if it already exists to bypass read-only attribute constraints.
    """
    if os.path.exists(dst):
        try:
            os.chmod(dst, 0o666)
        except Exception:
            pass
        try:
            os.remove(dst)
        except Exception:
            pass
    with open(src, "rb") as f_src:
        with open(dst, "wb") as f_dst:
            f_dst.write(f_src.read())

def sync_db_from_volume():
    """Sync checkpoint from Volume to local /tmp as a fallback for environments
    where the Volume POSIX path is not directly accessible.
    Tries POSIX access first; falls back to Databricks SDK download.
    """
    if "sync_logs" not in st.session_state:
        st.session_state["sync_logs"] = []
    
    volume_db = get_volume_db_path()
    local_db = get_checkpoint_db_path()
    st.session_state["sync_logs"].append(f"Starting sync from volume: {volume_db} to local: {local_db}")

    # Primary: POSIX copy (works when /Volumes/ is mounted)
    try:
        if os.path.exists(volume_db):
            copy_file_binary(volume_db, local_db)
            try:
                os.chmod(local_db, 0o666)
            except Exception:
                pass
            msg = f"Synced checkpoint from Volume (POSIX): {volume_db} → {local_db} (Size: {os.path.getsize(local_db)} bytes)"
            st.session_state["sync_logs"].append(msg)
            print(f"[Info] {msg}")
            return
        else:
            st.session_state["sync_logs"].append(f"POSIX file does not exist: {volume_db}")
    except Exception as e_posix:
        st.session_state["sync_logs"].append(f"POSIX Volume access failed: {e_posix}")
        print(f"[Info] POSIX Volume access failed ({e_posix}), trying SDK...")

    # Fallback: Databricks SDK Files API
    try:
        w = WorkspaceClient()
        response = w.files.download(volume_db)
        if os.path.exists(local_db):
            try:
                os.chmod(local_db, 0o666)
            except Exception:
                pass
            try:
                os.remove(local_db)
            except Exception:
                pass
        with open(local_db, "wb") as f:
            f.write(response.contents.read())
        try:
            os.chmod(local_db, 0o666)
        except Exception:
            pass
        msg = f"Synced checkpoint via SDK to {local_db} (Size: {os.path.getsize(local_db)} bytes)"
        st.session_state["sync_logs"].append(msg)
        print(f"[Info] {msg}")
    except Exception as e_sdk:
        st.session_state["sync_logs"].append(f"SDK sync failed: {e_sdk}")
        print(f"[Warning] Could not sync checkpoint: {e_sdk}")

def sync_db_to_volume():
    """Upload local checkpoint back to Volume after app updates state."""
    if "sync_logs" not in st.session_state:
        st.session_state["sync_logs"] = []
        
    volume_db = get_volume_db_path()
    local_db = get_checkpoint_db_path()
    st.session_state["sync_logs"].append(f"Starting sync to volume: {local_db} to {volume_db}")

    if not os.path.exists(local_db):
        st.session_state["sync_logs"].append(f"Local DB does not exist: {local_db}")
        return

    # Primary: POSIX copy
    try:
        os.makedirs(os.path.dirname(volume_db), exist_ok=True)
        copy_file_binary(local_db, volume_db)
        msg = f"Synced checkpoint to Volume (POSIX): {local_db} → {volume_db}"
        st.session_state["sync_logs"].append(msg)
        print(f"[Info] {msg}")
        return
    except Exception as e_posix:
        st.session_state["sync_logs"].append(f"POSIX write failed: {e_posix}")
        print(f"[Info] POSIX write failed ({e_posix}), trying SDK...")

    # Fallback: SDK upload
    try:
        w = WorkspaceClient()
        with open(local_db, "rb") as f:
            file_data = f.read()
        w.files.upload(volume_db, io.BytesIO(file_data), overwrite=True)
        st.session_state["sync_logs"].append("Synced checkpoint to Volume via SDK.")
        print("[Info] Synced checkpoint to Volume via SDK.")
    except Exception as e_sdk:
        st.session_state["sync_logs"].append(f"SDK upload failed: {e_sdk}")
        print(f"[Warning] Could not upload checkpoint: {e_sdk}")

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
        
        /* Force light theme colors on Streamlit main container */
        [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
        }
        
        /* Style the sidebar */
        [data-testid="stSidebar"] {
            background-color: #f7f9fa !important;
            border-right: 1px solid #e0e0e0 !important;
        }

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            color: #1a1a1a !important;
        }
        
        /* Headers styling */
        h1, h2, h3, h4, h5, h6, .stMarkdown p {
            color: #2c3e50 !important;
        }
        
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #1b3a4b 0%, #2c5e7a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: #5a6b7c;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }
        
        /* Light themed metric cards */
        .metric-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 4px 10px rgba(0,0,0,0.03);
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.4rem;
            font-weight: 800;
            color: #1b3a4b;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #5a6b7c;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 0.5rem;
            font-weight: 600;
        }
        
        .agent-badge {
            background-color: #f0f4f8;
            border: 1px solid #2c3e50;
            color: #2c3e50;
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

def get_or_create_graph():
    """Create a fresh LangGraph instance per Streamlit session, reading directly
    from the Volume checkpoint file so state is always current.
    Uses st.session_state to avoid recreating on every widget interaction.
    """
    if "pipeline_app" not in st.session_state:
        # Sync latest checkpoint from Volume to local /tmp first
        sync_db_from_volume()
        st.session_state["pipeline_app"] = create_pipeline_graph()
    return st.session_state["pipeline_app"]

def refresh_graph_checkpoint():
    """Force re-sync from Volume and recreate the graph connection."""
    sync_db_from_volume()
    # Clear cached graph so next call to get_or_create_graph() rebuilds with fresh DB
    if "pipeline_app" in st.session_state:
        del st.session_state["pipeline_app"]

spark = get_spark_session()
app = get_or_create_graph()
thread_id = "medallion_pipeline_run"
config = {"configurable": {"thread_id": thread_id}}

# ----------------- Dashboard Layout -----------------

st.markdown('<p class="main-title">🤖 Medallion Agent Control Center</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Unity Catalog Governed Multi-Agent Data Engineering Ingestion & Transformation</p>', unsafe_allow_html=True)

# Fetch Current Graph State
state = app.get_state(config)

# ----------------- Pipeline Lineage Status Bar -----------------
stages = [
    {"name": "1. Profiler", "gate": "profile_review_gate"},
    {"name": "2. Data Quality", "gate": "data_quality_review_gate"},
    {"name": "3. Contracts", "gate": "contracts_review_gate"},
    {"name": "4. Modeler", "gate": "modeling_review_gate"},
    {"name": "5. Engineer", "gate": "engineering_review_gate"},
    {"name": "6. Orchestrator", "gate": "execution_review_gate"}
]

# Calculate stage statuses: completed (Green), review (Amber), pending (Grey)
stage_statuses = []
if not state.values:
    stage_statuses = ["pending"] * len(stages)
else:
    current_gate = state.next[0] if state.next else None
    if not current_gate:
        # Check if the pipeline has finished
        if "final_report" in state.values:
            stage_statuses = ["completed"] * len(stages)
        else:
            stage_statuses = ["pending"] * len(stages)
    else:
        # Find index of current gate
        gate_idx = -1
        for idx, s in enumerate(stages):
            if s["gate"] == current_gate:
                gate_idx = idx
                break
        for idx, s in enumerate(stages):
            if idx < gate_idx:
                stage_statuses.append("completed")
            elif idx == gate_idx:
                stage_statuses.append("review")
            else:
                stage_statuses.append("pending")

# Render progress lineage columns
st.markdown("<h4 style='color: #2c3e50; margin-bottom: 12px;'>🗺️ Agent Lineage & Progress Flow</h4>", unsafe_allow_html=True)
cols = st.columns(len(stages))
for i, stage in enumerate(stages):
    status = stage_statuses[i]
    if status == "completed":
        bg_color = "#e8f5e9"
        border_color = "#81c784"
        text_color = "#2e7d32"
        badge = "🟢 Completed"
    elif status == "review":
        bg_color = "#fff8e1"
        border_color = "#ffb74d"
        text_color = "#ef6c00"
        badge = "🟡 In Review"
    else:
        bg_color = "#f5f5f5"
        border_color = "#e0e0e0"
        text_color = "#757575"
        badge = "⚪ Pending"

    with cols[i]:
        st.markdown(f"""
            <div style="
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 10px;
                padding: 12px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.02);
                margin-bottom: 20px;
            ">
                <div style="font-size: 0.75rem; font-weight: bold; color: {text_color}; text-transform: uppercase; margin-bottom: 6px;">
                    {badge}
                </div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #2c3e50;">
                    {stage["name"]}
                </div>
            </div>
        """, unsafe_allow_html=True)

# Sidebar with Status Overview
st.sidebar.markdown("### Ingestion Settings")
catalog_config = load_config()
st.sidebar.info(f"**Unity Catalog**: {catalog_config.get('catalog', 'hospitality_catalog')}\n\n**Raw Volume**: {catalog_config.get('raw_volume', 'raw/source_volume')}")

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    sync_db_from_volume()
    refresh_graph_checkpoint()
    st.rerun()

# Reset button
if st.sidebar.button("🗑️ Reset Pipeline / Start Fresh", type="secondary"):
    local_db = get_checkpoint_db_path()
    volume_db = get_volume_db_path()
    
    # 1. Delete local checkpoint DB
    if os.path.exists(local_db):
        try:
            os.remove(local_db)
        except Exception:
            pass
            
    # 2. Delete Volume checkpoint DB
    if os.path.exists(volume_db):
        try:
            os.remove(volume_db)
        except Exception:
            pass
            
    # 3. Try deleting via Workspace SDK
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient(profile="venkatesh8484")
        w.files.delete(volume_db)
    except Exception:
        pass
        
    st.session_state["sync_logs"] = ["Pipeline reset. Database deleted from local disk and UC Volume."]
    refresh_graph_checkpoint()
    st.rerun()

# Diagnostics & Health check in sidebar
with st.sidebar.expander("🛠️ Diagnostics & Health Check"):
    local_db = get_checkpoint_db_path()
    volume_db = get_volume_db_path()
    
    st.markdown("**Checkpoints Sync Status:**")
    st.write(f"Volume file exists: `{os.path.exists(volume_db)}`")
    if os.path.exists(volume_db):
        st.write(f"Volume size: `{os.path.getsize(volume_db)} bytes`")
        
    st.write(f"Local file exists: `{os.path.exists(local_db)}`")
    if os.path.exists(local_db):
        st.write(f"Local size: `{os.path.getsize(local_db)} bytes`")
        
        # Query local SQLite
        try:
            import sqlite3
            conn = sqlite3.connect(local_db)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]
            st.write(f"Tables: `{tables}`")
            if "checkpoints" in tables:
                cursor.execute("SELECT thread_id, checkpoint_id, parent_checkpoint_id FROM checkpoints")
                cps = cursor.fetchall()
                st.write(f"Checkpoints count: `{len(cps)}`")
                if cps:
                    st.write("Latest checkpoints:")
                    st.dataframe(pd.DataFrame(cps, columns=["thread_id", "checkpoint_id", "parent_id"]))
            conn.close()
        except Exception as e_db:
            st.error(f"SQLite check failed: {e_db}")

    st.markdown("**Graph state debug:**")
    try:
        st.write(f"State Pointer `state.next`: `{state.next}`")
        st.write(f"Values present: `{bool(state.values)}`")
        if state.values:
            st.write(f"Active Agent: `{state.values.get('active_agent')}`")
    except Exception as e_state:
        st.error(f"State fetch failed: {e_state}")

    st.markdown("**Sync Logs:**")
    if "sync_logs" in st.session_state:
        for log in st.session_state["sync_logs"][-15:]:
            st.text(log)
    else:
        st.text("No sync logs yet.")

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
            try:
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
            except Exception as e_click:
                # Catch and ignore the LangGraph end-of-execution KeyError
                if isinstance(e_click, KeyError) and "__end__" in str(e_click):
                    # Sync state and rerun since the graph finished successfully
                    try:
                        sync_db_to_volume()
                    except Exception:
                        pass
                    st.success("Pipeline successfully finished! Refreshing dashboard...")
                    st.rerun()
                else:
                    st.error("Execution error during Submit & Resume:")
                    st.exception(e_click)

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
