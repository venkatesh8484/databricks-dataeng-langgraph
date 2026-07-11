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
from dbricks_lang_agent.orchestrator.agents import chat_with_data_agent, clear_profiling_cache
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

# Enterprise Design System CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700&display=swap');

        /* ── Base ────────────────────────────────────────────────── */
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        [data-testid="stMain"] {
            background-color: #F8FAFC !important;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            -webkit-font-smoothing: antialiased;
        }

        /* ── Global text — all paragraph/list/label content must be readable ── */
        p, label,
        .stMarkdown p, .stMarkdown li, .stMarkdown ol, .stMarkdown ul,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: #1E293B !important;
            font-size: 0.925rem !important;
            line-height: 1.6 !important;
        }

        /* ── Chat message containers ─────────────────────────────── */
        /* Ensure all text inside native st.chat_message() is visible */
        [data-testid="stChatMessage"] { background: transparent !important; }
        [data-testid="stChatMessageContent"] { background: transparent !important; }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] div,
        [data-testid="stChatMessage"] .stMarkdown p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            color: #1E293B !important;
            font-size: 0.9rem !important;
        }
        /* Caption text inside chat stays muted */
        [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p,
        [data-testid="stChatMessage"] small {
            color: #64748B !important;
            font-size: 0.78rem !important;
        }

        /* Code blocks: let Streamlit's own theme handle colours/background.
           Only set the monospace font; do NOT override color or background-color
           or syntax highlighting will be stripped and it renders as plain text. */
        code, pre,
        [data-testid="stCode"] code,
        .stCodeBlock code {
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
        }

        /* Ensure horizontal + vertical scroll on code blocks */
        [data-testid="stCode"] pre,
        .stCodeBlock pre {
            overflow: auto !important;
            max-height: 600px !important;
            white-space: pre !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            color: #0F172A !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }

        /* ── Sidebar ─────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            background-color: transparent !important;
            color: #334155 !important;
            font-size: 0.875rem !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #0F172A !important;
            font-size: 0.875rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
        }

        /* ── Page header ─────────────────────────────────────────── */
        .main-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0F172A;
            letter-spacing: -0.03em;
            margin-bottom: 2px;
            line-height: 1.2;
        }
        .subtitle {
            font-size: 0.875rem;
            color: #64748B;
            font-weight: 400;
            margin-bottom: 1.25rem;
            letter-spacing: 0;
        }

        /* ── Tabs ────────────────────────────────────────────────── */
        button[data-baseweb="tab"] {
            color: #64748B !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            opacity: 1 !important;
            letter-spacing: 0 !important;
            padding: 10px 16px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #4F46E5 !important;
            border-bottom: 2px solid #4F46E5 !important;
            font-weight: 600 !important;
        }
        button[data-baseweb="tab"]:hover {
            color: #1E293B !important;
            background: #F1F5F9 !important;
        }

        /* ── Alerts ──────────────────────────────────────────────── */
        .stAlert, [data-testid="stNotification"], [data-testid="stAlert"] {
            background-color: #EFF6FF !important;
            border: 1px solid #BFDBFE !important;
            border-left: 3px solid #3B82F6 !important;
            border-radius: 6px !important;
        }
        .stAlert p, [data-testid="stNotification"] p, [data-testid="stAlert"] p,
        .stAlert div, [data-testid="stNotification"] div {
            color: #1E40AF !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
        }

        /* ── Buttons ─────────────────────────────────────────────── */
        div[data-testid="stButton"] button {
            background-color: #FFFFFF !important;
            color: #374151 !important;
            border: 1px solid #D1D5DB !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
            transition: all 0.15s ease !important;
            letter-spacing: 0 !important;
        }
        div[data-testid="stButton"] button:hover {
            background-color: #F9FAFB !important;
            border-color: #9CA3AF !important;
            color: #111827 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
        }

        /* ── Metric cards ────────────────────────────────────────── */
        .metric-card {
            background: #FFFFFF;
            padding: 1.25rem 1.5rem;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
            text-align: center;
        }
        .metric-value {
            font-size: 1.875rem;
            font-weight: 700;
            color: #0F172A;
            letter-spacing: -0.03em;
            line-height: 1.2;
        }
        .metric-label {
            font-size: 0.7rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-top: 6px;
            font-weight: 600;
        }

        /* ── Agent badge ─────────────────────────────────────────── */
        .agent-badge {
            background-color: #F1F5F9;
            border: 1px solid #CBD5E1;
            color: #475569;
            padding: 2px 10px;
            border-radius: 100px;
            font-weight: 500;
            display: inline-block;
            font-size: 0.78rem;
            letter-spacing: 0.01em;
        }

        /* ── Talk to Data Chat UI ────────────────────────────────── */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            padding: 16px 4px;
            max-height: 520px;
            overflow-y: auto;
            scroll-behavior: smooth;
        }
        .chat-container::-webkit-scrollbar { width: 4px; }
        .chat-container::-webkit-scrollbar-track { background: transparent; }
        .chat-container::-webkit-scrollbar-thumb {
            background: #CBD5E1;
            border-radius: 4px;
        }

        .chat-bubble-wrapper {
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }
        .chat-bubble-wrapper.user { flex-direction: row-reverse; }

        .chat-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            flex-shrink: 0;
            margin-top: 2px;
        }
        .chat-avatar.bot {
            background: #4F46E5;
            color: #fff;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0;
        }
        .chat-avatar.user {
            background: #0F172A;
            color: #fff;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .chat-bubble {
            max-width: 72%;
            padding: 11px 15px;
            border-radius: 12px;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        .chat-bubble.bot {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-bottom-left-radius: 3px;
            color: #1E293B;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .chat-bubble.user {
            background: #4F46E5;
            color: #FFFFFF;
            border-bottom-right-radius: 3px;
        }

        .chat-profiling-badge {
            font-size: 0.72rem;
            color: #64748B;
            margin-top: 5px;
            padding-left: 44px;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .chat-empty-state {
            text-align: center;
            padding: 48px 24px;
        }
        .chat-empty-state .chat-empty-icon {
            font-size: 2.5rem;
            margin-bottom: 10px;
            opacity: 0.4;
        }
        .chat-empty-state p {
            font-size: 0.9rem !important;
            color: #94A3B8 !important;
        }

        .chat-input-area {
            border-top: 1px solid #E2E8F0;
            padding-top: 16px;
            margin-top: 4px;
        }

        .chat-suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 4px;
        }
        .chat-suggestions span {
            display: inline-block;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            color: #475569;
            border-radius: 6px;
            padding: 5px 12px;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            transition: border-color 0.15s, color 0.15s;
        }
        .chat-suggestions span:hover {
            border-color: #4F46E5;
            color: #4F46E5;
        }

        /* ── Streamlit chat input (st.chat_input dark pill) ────────────────── */
        [data-testid="stChatInput"] textarea {
            background: #1E293B !important;
            color: #F1F5F9 !important;
            border-radius: 8px !important;
            font-size: 0.9rem !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #94A3B8 !important;
        }
        [data-testid="stChatInput"] button {
            background: #4F46E5 !important;
            color: #FFFFFF !important;
            border-radius: 6px !important;
        }

        /* ── Streamlit form text input override ─────────────────────────────── */
        [data-testid="stTextInput"] input {
            border-radius: 6px !important;
            border: 1px solid #D1D5DB !important;
            font-size: 0.9rem !important;
            color: #0F172A !important;
            background: #FFFFFF !important;
            padding: 10px 14px !important;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: #4F46E5 !important;
            box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
            outline: none !important;
        }
        [data-testid="stTextInput"] input::placeholder { color: #94A3B8 !important; }

        /* ── Caption / small text ──────────────────────────────────────────── */
        [data-testid="stCaptionContainer"] p,
        .stCaption p {
            color: #64748B !important;
            font-size: 0.8rem !important;
        }

        /* ── Alert type text colours ───────────────────────────────────────── */
        .stSuccess [data-testid="stMarkdownContainer"] p  { color: #065F46 !important; }
        .stWarning [data-testid="stMarkdownContainer"] p  { color: #92400E !important; }
        .stError   [data-testid="stMarkdownContainer"] p  { color: #991B1B !important; }
        .stInfo    [data-testid="stMarkdownContainer"] p  { color: #1E40AF !important; }
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

st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:2px;">
  <div style="width:36px; height:36px; background:#4F46E5; border-radius:8px; display:flex;
              align-items:center; justify-content:center; font-size:1.1rem; flex-shrink:0;">🤖</div>
  <p class="main-title" style="margin:0;">Medallion Agent Control Center</p>
</div>
<p class="subtitle" style="padding-left:48px;">Unity Catalog · Multi-Agent Data Engineering · LangGraph Orchestration</p>
""", unsafe_allow_html=True)

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
st.markdown("""
<div style="display:flex; align-items:center; gap:8px; margin-bottom:14px;">
  <span style="font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em;
               color:#64748B;">Pipeline Progress</span>
  <div style="flex:1; height:1px; background:#E2E8F0;"></div>
</div>
""", unsafe_allow_html=True)
cols = st.columns(len(stages))
for i, stage in enumerate(stages):
    status = stage_statuses[i]
    if status == "completed":
        accent      = "#059669"   # emerald
        badge_bg    = "#ECFDF5"
        badge_color = "#065F46"
        badge_text  = "Completed"
        dot         = "&#9679;"   # filled circle
        name_color  = "#0F172A"
    elif status == "review":
        accent      = "#D97706"   # amber
        badge_bg    = "#FFFBEB"
        badge_color = "#92400E"
        badge_text  = "In Review"
        dot         = "&#9679;"
        name_color  = "#0F172A"
    else:
        accent      = "#CBD5E1"   # slate
        badge_bg    = "#F8FAFC"
        badge_color = "#94A3B8"
        badge_text  = "Pending"
        dot         = "&#9675;"   # open circle
        name_color  = "#94A3B8"

    with cols[i]:
        st.markdown(f"""
            <div style="
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-top: 3px solid {accent};
                border-radius: 8px;
                padding: 12px 10px 10px;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                margin-bottom: 16px;
            ">
                <div style="
                    display:inline-flex; align-items:center; gap:4px;
                    background:{badge_bg}; color:{badge_color};
                    font-size:0.65rem; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.06em;
                    padding:2px 7px; border-radius:100px;
                    margin-bottom:7px;
                ">
                    <span style="color:{accent}; font-size:0.55rem;">{dot}</span>
                    {badge_text}
                </div>
                <div style="font-size:0.82rem; font-weight:600; color:{name_color}; letter-spacing:-0.01em;">
                    {stage["name"]}
                </div>
            </div>
        """, unsafe_allow_html=True)

# Sidebar with Status Overview
st.sidebar.markdown("### Ingestion Settings")
catalog_config = load_config()
st.sidebar.info(f"**Unity Catalog**: {catalog_config.get('catalog', 'databricks_langgraph')}\n\n**Raw Volume**: {catalog_config.get('raw_volume', 'raw/source_volume')}")

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
tab0, tab1, tab2, tab3 = st.tabs([
    "💬 Talk to Data",
    "📊 Ingestion Monitoring",
    "📥 Action Center (HITL)",
    "🧠 Agent Memories"
])

# ----------------- Tab 0: Talk to Data Chatbot -----------------
with tab0:
    # Initialise session-state keys
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "pending_pipeline_action" not in st.session_state:
        st.session_state["pending_pipeline_action"] = None

    # ---- Header ----
    st.markdown("#### 💬 Talk to Data")
    st.caption(
        "Ask anything about your dataset — row counts, schemas, data quality, "
        "pipeline status, and more. I can also start or advance the pipeline for you."
    )

    # ---- Quick-start suggestion buttons ----
    suggestions = [
        "How many rows are in my dataset?",
        "What are the column names and data types?",
        "Are there any data quality issues?",
        "What is the current pipeline status?",
        "Start the pipeline",
    ]
    btn_cols = st.columns(len(suggestions))
    for idx, sug in enumerate(suggestions):
        with btn_cols[idx]:
            if st.button(sug, key=f"sug_{idx}", use_container_width=True):
                st.session_state["_chat_prefill"] = sug

    st.divider()

    # ---- Render chat history using native st.chat_message ----
    history = st.session_state["chat_history"]
    if not history:
        st.info("No conversation yet. Type a question below or click a suggestion above to get started!")
    else:
        for turn in history:
            role = turn["role"]
            content = turn["content"]
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(content)
                if turn.get("profiling_triggered"):
                    st.caption("ℹ️ Fresh Spark profiling was run to answer this question.")

    # ---- Pipeline action confirmation button ----
    pending = st.session_state.get("pending_pipeline_action")
    if pending:
        st.warning(
            f"⚡ **Pipeline Action Ready:** {pending.get('description', 'Proceed with next pipeline step?')}"
        )
        proceed_col, cancel_col, _ = st.columns([1, 1, 4])
        with proceed_col:
            if st.button("▶ Yes, Proceed", key="pipeline_proceed_btn", type="primary"):
                try:
                    gate = pending.get("gate")
                    step = pending.get("step")

                    # Ensure the local SQLite file is writable before any write
                    local_db = get_checkpoint_db_path()
                    if os.path.exists(local_db):
                        try:
                            os.chmod(local_db, 0o666)
                        except Exception:
                            pass

                    # Build updated approvals — use boolean True so routing
                    # functions (which check `is True`) correctly advance
                    current_state = app.get_state(config)
                    current_approved = dict(current_state.values.get("approved_steps", {}))
                    current_approved[step] = True  # Must be boolean, not string

                    app.update_state(
                        config,
                        {"approved_steps": current_approved}
                    )

                    # Stream the graph forward (same as HITL Action Center)
                    with st.spinner("Resuming pipeline… this may take a moment."):
                        try:
                            events = app.stream(None, config, stream_mode="values")
                            for event in events:
                                pass
                        except KeyError as e_key:
                            if "__end__" not in str(e_key):
                                raise

                    # Persist to Volume
                    sync_db_to_volume()

                    msg = (
                        f"✅ The **{step.title()}** step is approved and the pipeline has resumed. "
                        "Switch to the **Ingestion Monitoring** tab to track progress."
                    )
                    st.session_state["chat_history"].append({
                        "role": "assistant", "content": msg, "profiling_triggered": False
                    })
                    st.session_state["pending_pipeline_action"] = None
                    refresh_graph_checkpoint()
                    st.success(msg)
                    st.rerun()
                except Exception as e_proceed:
                    st.error(f"Failed to trigger pipeline: {e_proceed}")
                    st.exception(e_proceed)
        with cancel_col:
            if st.button("✕ Cancel", key="pipeline_cancel_btn"):
                st.session_state["pending_pipeline_action"] = None
                st.rerun()

    # ---- Chat input (native — no HTML injection risk) ----
    prefill = st.session_state.pop("_chat_prefill", None)
    user_input = st.chat_input(
        "Ask anything about your data, or say 'start the pipeline'…",
        key="chat_input_native"
    )
    # Use prefill if user clicked a suggestion button
    question_to_process = prefill or user_input

    if question_to_process:
        question = question_to_process.strip()

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state["chat_history"].append({
            "role": "user", "content": question, "profiling_triggered": False
        })

        # Call the agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    current_state = app.get_state(config)
                    state_values = current_state.values if current_state.values else {}

                    answer, prof_triggered, pipeline_action = chat_with_data_agent(
                        question=question,
                        state=state_values,
                        history=st.session_state["chat_history"][:-1],
                    )
                except Exception as e_chat:
                    answer = f"⚠️ Sorry, I ran into an error: {e_chat}"
                    prof_triggered = False
                    pipeline_action = None

            st.markdown(answer)
            if prof_triggered:
                st.caption("ℹ️ Fresh Spark profiling was run to answer this question.")

        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "profiling_triggered": prof_triggered
        })

        # Store any pipeline action for the confirmation button above
        if pipeline_action:
            st.session_state["pending_pipeline_action"] = pipeline_action

        st.rerun()

    # ---- Utility controls ----
    st.markdown("---")
    ctrl_col1, ctrl_col2, _ = st.columns([1, 1, 4])
    with ctrl_col1:
        if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
            st.session_state["chat_history"] = []
            st.session_state["pending_pipeline_action"] = None
            st.rerun()
    with ctrl_col2:
        if st.button("♻️ Reset Profile Cache", key="reset_profile_cache_btn"):
            clear_profiling_cache()
            st.success("Profiling cache cleared.")


# ----------------- Tab 1: Ingestion Monitoring -----------------
with tab1:
    st.markdown("### 📈 Ingestion Pipeline Metrics")
    
    # Try fetching table statistics from Unity Catalog
    try:
        catalog = catalog_config.get("catalog", "databricks_langgraph")
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
            st.caption("Each tab shows the full generated script. Scroll horizontally/vertically inside the code block.")
            bronze_code = state.values.get("bronze_code", "")
            silver_code = state.values.get("silver_code", "")
            gold_code   = state.values.get("gold_code", "")
            code_tab1, code_tab2, code_tab3 = st.tabs([
                "🥉 Bronze — bronze.py",
                "🥈 Silver — silver.py",
                "🥇 Gold — gold.py",
            ])
            with code_tab1:
                if bronze_code:
                    st.code(bronze_code, language="python", line_numbers=True)
                else:
                    st.info("Bronze code not yet generated.")
            with code_tab2:
                if silver_code:
                    st.code(silver_code, language="python", line_numbers=True)
                else:
                    st.info("Silver code not yet generated.")
            with code_tab3:
                if gold_code:
                    st.code(gold_code, language="python", line_numbers=True)
                else:
                    st.info("Gold code not yet generated.")
                
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
        catalog = catalog_config.get("catalog", "databricks_langgraph")
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
