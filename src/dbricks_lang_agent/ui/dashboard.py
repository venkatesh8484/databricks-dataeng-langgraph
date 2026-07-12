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
import uuid
import streamlit as st
import pandas as pd
from datetime import datetime

# Add project root to sys path
sys.path.append(os.path.abspath("src"))

from dbricks_lang_agent.data_platform.spark_utils import get_spark, load_config
from dbricks_lang_agent.orchestrator.graph import create_pipeline_graph, get_checkpoint_db_path, resume_with_autopilot
from dbricks_lang_agent.orchestrator import memory
from dbricks_lang_agent.orchestrator.agents import (
    chat_with_data_agent, clear_profiling_cache, get_schema_fingerprint, product_advisor_node,
)
from dbricks_lang_agent.data_platform import products as products_module
from databricks.sdk import WorkspaceClient
import io

# ----------------- Shared Stage Metadata -----------------
# Single source of truth for the six pipeline stages: display name, the gate
# node LangGraph pauses at, the step key used in `approved_steps`, and the
# agent name stamped into `active_agent` while that stage is in review. Used
# by the workflow canvas, the Stage Inspector review panel, and the
# clickable "View Output" viewer so all three stay in sync.
STAGE_DEFS = [
    {"name": "1. Profiler",      "gate": "profile_review_gate",     "step_key": "profile",     "agent_name": "Profiler",           "node": "profiler"},
    {"name": "2. Data Quality",  "gate": "data_quality_review_gate","step_key": "dq",          "agent_name": "DataQualityAgent",   "node": "data_quality"},
    {"name": "3. Contracts",     "gate": "contracts_review_gate",   "step_key": "contracts",   "agent_name": "ContractSteward",    "node": "contracts"},
    {"name": "4. Modeler",       "gate": "modeling_review_gate",    "step_key": "modeling",    "agent_name": "DimensionalModeler", "node": "modeling"},
    {"name": "5. Engineer",      "gate": "engineering_review_gate", "step_key": "engineering", "agent_name": "DataEngineer",       "node": "engineering"},
    {"name": "6. Orchestrator",  "gate": "execution_review_gate",   "step_key": "report",       "agent_name": "Orchestrator",       "node": "execution"},
]


def get_stage_artifacts(step_key: str, state_values: dict) -> dict:
    """Pull out just the fields relevant to one stage from the (cumulative)
    LangGraph state. Used both for on-screen rendering and for the audit
    snapshot written to gold.agent_stage_review_log."""
    state_values = state_values or {}
    if step_key == "profile":
        return {
            "discovered_tables": state_values.get("discovered_tables", {}),
            "profiling_report": state_values.get("profiling_report", {}),
            "profiler_error": state_values.get("profiler_error", ""),
        }
    elif step_key == "dq":
        return {"dq_report": state_values.get("dq_report", "")}
    elif step_key == "contracts":
        return {
            "contracts": state_values.get("contracts", {}),
            "contracts_error": state_values.get("contracts_error", ""),
        }
    elif step_key == "modeling":
        return {
            "gold_ddl": state_values.get("gold_ddl", ""),
            "data_dictionary": state_values.get("data_dictionary", ""),
        }
    elif step_key == "engineering":
        return {
            "bronze_code": state_values.get("bronze_code", ""),
            "silver_code": state_values.get("silver_code", ""),
            "gold_code": state_values.get("gold_code", ""),
        }
    elif step_key == "report":
        return {
            "execution_logs": state_values.get("execution_logs", {}),
            "final_report": state_values.get("final_report", ""),
        }
    return {}


def render_agent_output(agent_name: str, state_values: dict) -> None:
    """Render the artifacts produced by one agent stage. Pulled out of the
    Stage Inspector review flow so the SAME rendering can be reused by the
    clickable stage viewer (which shows a stage's output at any point,
    regardless of which gate is currently active) and by the audit tab
    (which replays a past review's output snapshot)."""
    state_values = state_values or {}

    # Provenance badge: was this generated fresh by the LLM this run, patched
    # from a prior version because the schema changed, or reused untouched
    # from the Unity Catalog cache because nothing changed? Only Profiler is
    # exempt (it always reflects the current run's actual data).
    step_key_for_agent = next((s["step_key"] for s in STAGE_DEFS if s["agent_name"] == agent_name), None)
    gen_source = (state_values.get("generation_source") or {}).get(step_key_for_agent) if step_key_for_agent else None
    if gen_source and agent_name != "Profiler":
        badge = {
            "cache_reused": "♻️ Reused from Unity Catalog cache — schema unchanged, no LLM call made.",
            "llm_patched": "🔧 Updated from a prior version — schema changed since it was last generated.",
            "llm_fresh": "✨ Freshly generated by the LLM.",
        }.get(gen_source)
        if badge:
            st.caption(badge)

    if agent_name == "Profiler":
        profiler_error = state_values.get("profiler_error", "")
        if profiler_error:
            st.error(profiler_error)
            diagnostics = state_values.get("profiling_report", {}).get("discovery_diagnostics", [])
            if diagnostics:
                with st.expander("Raw discovery trail (DBUtils / SDK / POSIX attempts)", expanded=True):
                    st.code("\n".join(diagnostics), language="text")
        st.markdown("### Inferred Source Table Schemas:")
        st.json(state_values.get("discovered_tables", {}))
        st.markdown("### Dynamic Profiling Report:")
        st.markdown(state_values.get("profiling_report", {}).get("profiler_narration", "No report narrative found."))

    elif agent_name == "DataQualityAgent":
        st.markdown("### Data Quality Assessment Report:")
        st.markdown(state_values.get("dq_report", "No DQ report found."))

    elif agent_name == "ContractSteward":
        contracts_error = state_values.get("contracts_error", "")
        if contracts_error:
            st.error(contracts_error)
        st.markdown("### Generated YAML Data Contracts:")
        contracts = state_values.get("contracts", {})
        if not contracts:
            if not contracts_error:
                st.info("No contracts were generated and no error was recorded — this looks like an unexpected empty result. Check the app logs for '[Contract Steward]' entries.")
        for tbl, contract_yaml in contracts.items():
            st.subheader(f"Table Contract: {tbl}")
            st.code(contract_yaml, language="yaml")

    elif agent_name == "DimensionalModeler":
        st.markdown("### Inferred Star Schema SQL DDL:")
        st.code(state_values.get("gold_ddl", ""), language="sql")
        st.markdown("### Star Schema Data Dictionary:")
        st.markdown(state_values.get("data_dictionary", ""))

    elif agent_name == "DataEngineer":
        st.markdown("### Generated PySpark Code Blocks:")
        st.caption("Each tab shows the full generated script. Scroll horizontally/vertically inside the code block.")
        bronze_code = state_values.get("bronze_code", "")
        silver_code = state_values.get("silver_code", "")
        gold_code   = state_values.get("gold_code", "")
        code_tab1, code_tab2, code_tab3 = st.tabs([
            "bronze.py",
            "silver.py",
            "gold.py",
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

    elif agent_name == "Orchestrator":
        st.markdown("### Executed Script Logs:")
        st.json(state_values.get("execution_logs", {}))
        st.markdown("### Final Run Summary Report:")
        st.markdown(state_values.get("final_report", ""))
    else:
        st.info("No output available yet for this stage — it hasn't run in the current pipeline thread.")

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
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design System ("Ledger" light theme) ─────────────────────────────────
# One accent (#0F62FE) on warm-neutral surfaces; soft semantic chips for
# status; no gradients, no emoji chrome. CSS is layered deliberately:
# base text rules never target bare <span>/<div>, so custom components
# (chips, canvas, inspector) keep their own colors without specificity wars.
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg: #F7F8FA;
            --surface: #FFFFFF;
            --border: #E5E7EB;
            --border-strong: #D1D5DB;
            --ink: #111827;
            --ink-2: #4B5563;
            --ink-3: #9CA3AF;
            --accent: #0F62FE;
            --accent-hover: #0043CE;
            --accent-soft: #EDF5FF;
            --ok: #16A34A;   --ok-soft: #F0FDF4;   --ok-ink: #166534;   --ok-line: #BBE7C9;
            --warn: #D97706; --warn-soft: #FFFBEB; --warn-ink: #92400E; --warn-line: #F5D9A8;
            --err: #DC2626;  --err-soft: #FEF2F2;  --err-ink: #991B1B;  --err-line: #F3B6B6;
            --shadow-sm: 0 1px 2px rgba(17,24,39,0.05);
            --shadow-md: 0 2px 8px rgba(17,24,39,0.07);
        }

        /* ── Base ──────────────────────────────────────────────────────── */
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        [data-testid="stMain"] { background-color: var(--bg) !important; }
        [data-testid="stHeader"] { background: transparent !important; }
        #MainMenu, footer { visibility: hidden; }
        .block-container { padding-top: 1.0rem !important; max-width: 1440px; }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            -webkit-font-smoothing: antialiased;
        }

        /* Text: paragraphs/lists/labels only — never bare span/div */
        p, li, label,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: var(--ink);
            font-size: 0.92rem;
            line-height: 1.6;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            color: var(--ink) !important;
            font-weight: 650 !important;
            letter-spacing: -0.02em !important;
        }
        hr { border-color: var(--border) !important; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }

        /* ── Command bar ───────────────────────────────────────────────── */
        .cmdbar {
            display: flex; justify-content: space-between; align-items: center;
            gap: 14px; flex-wrap: wrap;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 14px;
            box-shadow: var(--shadow-sm);
        }
        .cmdbar-left { display: flex; align-items: center; gap: 12px; }
        .cmd-logo {
            width: 36px; height: 36px; border-radius: 9px; flex-shrink: 0;
            background: var(--accent); color: #FFFFFF;
            font-size: 1rem; font-weight: 700;
            display: flex; align-items: center; justify-content: center;
        }
        .cmd-title { color: var(--ink); font-size: 1.08rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2; }
        .cmd-sub { color: var(--ink-3); font-size: 0.78rem; margin-top: 2px; }
        .cmdbar-right { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }

        /* ── Chips (soft bg + dark ink — contrast-safe everywhere) ─────── */
        .chip {
            display: inline-flex; align-items: center; gap: 7px;
            background: #F3F4F6; border: 1px solid var(--border);
            color: #374151; font-size: 0.74rem; font-weight: 600;
            padding: 4px 11px; border-radius: 100px;
            letter-spacing: 0.01em; white-space: nowrap;
        }
        .chip .chip-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ink-3); flex-shrink: 0; }
        .chip.ok   { background: var(--ok-soft);   border-color: var(--ok-line);   color: var(--ok-ink); }
        .chip.ok .chip-dot { background: var(--ok); }
        .chip.warn { background: var(--warn-soft); border-color: var(--warn-line); color: var(--warn-ink); }
        .chip.warn .chip-dot { background: var(--warn); }
        .chip.err  { background: var(--err-soft);  border-color: var(--err-line);  color: var(--err-ink); }
        .chip.err .chip-dot { background: var(--err); }
        .chip.neutral { }

        /* ── Panel labels & canvas card ────────────────────────────────── */
        .panel-label {
            font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.1em; color: var(--ink-3);
            margin: 10px 0 8px 2px;
        }
        .canvas-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 16px 6px;
            box-shadow: var(--shadow-sm);
            margin-bottom: 14px;
            overflow-x: auto;
        }

        /* ── Stage Inspector ───────────────────────────────────────────── */
        .insp-head {
            display: flex; justify-content: space-between; align-items: center;
            gap: 10px; flex-wrap: wrap; margin-bottom: 6px;
        }
        .insp-title { color: var(--ink); font-size: 1.05rem; font-weight: 700; letter-spacing: -0.02em; }
        .insp-agent {
            color: var(--ink-3); font-size: 0.76rem; font-weight: 600;
            margin-left: 8px; font-family: 'JetBrains Mono', monospace;
        }
        .insp-chips { display: flex; gap: 6px; flex-wrap: wrap; }

        /* ── Radio → segmented pills (stage selector, approve/reject) ──── */
        [data-testid="stRadio"] div[role="radiogroup"] { gap: 6px; flex-wrap: wrap; }
        [data-testid="stRadio"] label {
            background: var(--surface);
            border: 1px solid var(--border-strong);
            border-radius: 100px;
            padding: 5px 14px !important;
            cursor: pointer;
            transition: border-color .12s, background .12s;
            margin: 0 !important;
        }
        [data-testid="stRadio"] label > div:first-child { display: none; }
        [data-testid="stRadio"] label p {
            font-size: 0.82rem !important; font-weight: 600 !important;
            color: var(--ink-2) !important; line-height: 1.3 !important;
        }
        [data-testid="stRadio"] label:hover { border-color: var(--accent); }
        [data-testid="stRadio"] label:has(input:checked) {
            background: var(--accent-soft) !important;
            border-color: var(--accent) !important;
        }
        [data-testid="stRadio"] label:has(input:checked) p { color: var(--accent-hover) !important; }

        /* ── Sidebar (light) ───────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background-color: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { font-size: 0.84rem; }
        .side-brand {
            display: flex; align-items: center; gap: 10px;
            padding: 6px 0 12px; margin-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }
        .side-brand-logo {
            width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0;
            background: var(--accent); color: #FFFFFF;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 0.85rem;
        }
        .side-brand-name { color: var(--ink); font-weight: 700; font-size: 0.92rem; letter-spacing: -0.01em; }
        .side-brand-sub { color: var(--ink-3); font-size: 0.7rem; }
        .side-section {
            font-size: 0.64rem; font-weight: 700; letter-spacing: 0.1em;
            text-transform: uppercase; color: var(--ink-3); margin: 14px 0 6px;
        }
        .side-kv {
            display: flex; justify-content: space-between; align-items: center; gap: 8px;
            padding: 7px 10px; border-radius: 8px; margin-bottom: 6px;
            background: #F3F4F6; border: 1px solid var(--border);
        }
        .side-kv span { color: var(--ink-3); font-size: 0.7rem; font-weight: 600; }
        .side-kv code {
            color: var(--ink); background: transparent;
            font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }

        /* ── Tabs (segmented) ──────────────────────────────────────────── */
        div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
            background: #F1F3F5; padding: 4px; border-radius: 10px;
            gap: 2px; width: fit-content; max-width: 100%;
        }
        button[data-baseweb="tab"] {
            background: transparent !important;
            border-radius: 7px !important;
            color: var(--ink-2) !important;
            font-size: 0.85rem !important; font-weight: 600 !important;
            padding: 7px 15px !important;
            border-bottom: none !important;
            opacity: 1 !important; letter-spacing: 0 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: var(--surface) !important;
            color: var(--ink) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        button[data-baseweb="tab"]:hover { color: var(--ink) !important; }
        div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }

        /* ── Buttons ───────────────────────────────────────────────────── */
        [data-testid="stButton"] > button,
        [data-testid="stFormSubmitButton"] > button {
            border-radius: 8px !important;
            font-weight: 600 !important; font-size: 0.84rem !important;
            letter-spacing: 0 !important;
            transition: all 0.12s ease !important;
        }
        [data-testid="stButton"] > button[kind="secondary"],
        [data-testid="stFormSubmitButton"] > button[kind="secondary"] {
            background: var(--surface) !important;
            color: var(--ink-2) !important;
            border: 1px solid var(--border-strong) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        [data-testid="stButton"] > button[kind="secondary"]:hover,
        [data-testid="stFormSubmitButton"] > button[kind="secondary"]:hover {
            border-color: var(--accent) !important;
            color: var(--accent-hover) !important;
            background: var(--accent-soft) !important;
        }
        [data-testid="stButton"] > button[kind="primary"],
        [data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: var(--accent) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--accent) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        [data-testid="stButton"] > button[kind="primary"]:hover,
        [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
            background: var(--accent-hover) !important;
            border-color: var(--accent-hover) !important;
        }

        /* ── Alerts ────────────────────────────────────────────────────── */
        div[data-testid="stAlert"], .stAlert {
            border-radius: 10px !important;
            border: 1px solid #C6E0F5 !important;
            border-left: 3px solid #0284C7 !important;
            background: #F0F9FF !important;
            box-shadow: none !important;
        }
        div[data-testid="stAlert"] p, .stAlert p {
            font-size: 0.86rem !important; font-weight: 500 !important;
            color: #075985 !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
            background: var(--ok-soft) !important;
            border-color: var(--ok-line) !important; border-left-color: var(--ok) !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) p { color: var(--ok-ink) !important; }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
            background: var(--warn-soft) !important;
            border-color: var(--warn-line) !important; border-left-color: var(--warn) !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) p { color: var(--warn-ink) !important; }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
            background: var(--err-soft) !important;
            border-color: var(--err-line) !important; border-left-color: var(--err) !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) p { color: var(--err-ink) !important; }
        .stSuccess [data-testid="stMarkdownContainer"] p { color: var(--ok-ink) !important; }
        .stWarning [data-testid="stMarkdownContainer"] p { color: var(--warn-ink) !important; }
        .stError   [data-testid="stMarkdownContainer"] p { color: var(--err-ink) !important; }

        /* ── Metric cards ──────────────────────────────────────────────── */
        .metric-card {
            background: var(--surface);
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }
        .metric-kicker {
            font-size: 0.64rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.09em; margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.65rem; font-weight: 700; color: var(--ink);
            letter-spacing: -0.03em; line-height: 1.15;
            font-variant-numeric: tabular-nums;
        }
        .metric-label { font-size: 0.73rem; color: var(--ink-2); margin-top: 4px; font-weight: 500; }

        .agent-badge {
            background: var(--accent-soft);
            border: 1px solid #C3DBFC;
            color: var(--accent-hover);
            padding: 2px 10px; border-radius: 100px;
            font-weight: 600; display: inline-block;
            font-size: 0.74rem; letter-spacing: 0.02em;
        }

        /* ── Expanders / containers / dataframes ───────────────────────── */
        [data-testid="stExpander"] details {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
        }
        [data-testid="stExpander"] summary { font-weight: 600 !important; color: var(--ink) !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            background: var(--surface);
            box-shadow: var(--shadow-sm);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
            background: var(--surface);
        }

        /* ── Chat ──────────────────────────────────────────────────────── */
        [data-testid="stChatMessage"] {
            background: var(--surface) !important;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px 16px;
            box-shadow: var(--shadow-sm);
        }
        [data-testid="stChatMessageContent"] { background: transparent !important; }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li { color: var(--ink) !important; font-size: 0.9rem !important; }
        [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p {
            color: var(--ink-3) !important; font-size: 0.76rem !important;
        }
        [data-testid="stChatInput"] {
            border: 1px solid var(--border-strong) !important;
            border-radius: 12px !important;
            background: var(--surface) !important;
        }
        [data-testid="stChatInput"] textarea {
            background: var(--surface) !important;
            color: var(--ink) !important;
            font-size: 0.9rem !important;
        }
        [data-testid="stChatInput"] textarea::placeholder { color: var(--ink-3) !important; }
        [data-testid="stChatInput"] button {
            background: var(--accent) !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
        }

        /* ── Inputs ────────────────────────────────────────────────────── */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            border-radius: 8px !important;
            border: 1px solid var(--border-strong) !important;
            font-size: 0.9rem !important;
            color: var(--ink) !important;
            background: var(--surface) !important;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(15,98,254,0.12) !important;
            outline: none !important;
        }
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder { color: var(--ink-3) !important; }
        [data-baseweb="select"] > div {
            border-radius: 8px !important;
            border-color: var(--border-strong) !important;
            background: var(--surface) !important;
        }

        [data-testid="stCaptionContainer"] p, .stCaption p {
            color: var(--ink-2) !important; font-size: 0.78rem !important;
        }

        /* ── Code ──────────────────────────────────────────────────────── */
        /* Only the monospace font — Streamlit's theme keeps syntax colors. */
        code, pre,
        [data-testid="stCode"] code,
        .stCodeBlock code {
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
        }
        [data-testid="stCode"] pre,
        .stCodeBlock pre {
            overflow: auto !important;
            max-height: 600px !important;
            white-space: pre !important;
            border-radius: 10px !important;
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

def get_or_assign_run_id() -> str:
    """Return the current pipeline's stable run ID, assigning one lazily if
    this thread doesn't have one yet (e.g. a pipeline started before this
    field existed). Every stage's audit row in gold.agent_stage_review_log
    is tagged with this ID so all six stages of one end-to-end run can be
    grouped together in the audit UI."""
    current_state = app.get_state(config)
    existing = (current_state.values or {}).get("pipeline_run_id")
    if existing:
        return existing
    new_run_id = str(uuid.uuid4())
    try:
        app.update_state(config, {"pipeline_run_id": new_run_id})
    except Exception as e:
        print(f"[Warning] Failed to persist pipeline_run_id: {e}")
    return new_run_id

def rollback_to_node(target_node: str) -> bool:
    """Roll back the LangGraph state checkpoints to the point before target_node runs."""
    try:
        # If targeting the first node and no checkpoints exist, or we want to start fresh:
        if target_node == "profiler":
            local_db = get_checkpoint_db_path()
            has_checkpoints = False
            if os.path.exists(local_db):
                import sqlite3
                conn = sqlite3.connect(local_db)
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT COUNT(*) FROM checkpoints")
                    if cursor.fetchone()[0] > 0:
                        has_checkpoints = True
                except Exception:
                    pass
                conn.close()
            
            if not has_checkpoints:
                volume_db = get_volume_db_path()
                for db_path in [local_db, volume_db]:
                    if os.path.exists(db_path):
                        try:
                            os.remove(db_path)
                        except Exception:
                            pass
                sync_db_to_volume()
                return True

        history = list(app.get_state_history(config))
        target_checkpoint_id = None
        
        # Search from newest to oldest for a checkpoint where target_node is about to run next
        for h in history:
            if h.next and target_node in h.next:
                target_checkpoint_id = h.config["configurable"]["checkpoint_id"]
                break
                
        # If not found in h.next, check if the first checkpoint can be used for 'profiler'
        if not target_checkpoint_id and target_node == "profiler":
            target_checkpoint_id = history[-1].config["configurable"]["checkpoint_id"] if history else None

        if target_checkpoint_id:
            local_db = get_checkpoint_db_path()
            if os.path.exists(local_db):
                import sqlite3
                conn = sqlite3.connect(local_db)
                cursor = conn.cursor()
                
                # Get all checkpoint IDs created after our target checkpoint
                newer_ids = []
                for h in history:
                    cid = h.config["configurable"]["checkpoint_id"]
                    if cid == target_checkpoint_id:
                        break
                    newer_ids.append(cid)
                
                if newer_ids:
                    placeholders = ",".join(["?"] * len(newer_ids))
                    cursor.execute(f"DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id IN ({placeholders})", [thread_id] + newer_ids)
                    cursor.execute(f"DELETE FROM writes WHERE thread_id = ? AND checkpoint_id IN ({placeholders})", [thread_id] + newer_ids)
                    conn.commit()
                conn.close()
                
                # Sync back to Volume to notify downstream notebook or app containers
                sync_db_to_volume()
                return True
    except Exception as e:
        print(f"[Warning] Rollback to {target_node} failed: {e}")
    return False

# ----------------- Dashboard Layout -----------------

# Fetch current graph state once for the whole page
state = app.get_state(config)
stages = STAGE_DEFS

# ── Stage statuses: completed / review / pending / failed ────────────────
stage_statuses = []
if not state.values:
    stage_statuses = ["pending"] * len(stages)
else:
    _sg_gate = state.next[0] if state.next else None
    if not _sg_gate:
        stage_statuses = ["completed"] * len(stages) if state.values.get("final_report") else ["pending"] * len(stages)
    else:
        _gate_idx = next((i for i, s in enumerate(stages) if s["gate"] == _sg_gate), -1)
        for _idx in range(len(stages)):
            stage_statuses.append("completed" if _idx < _gate_idx else ("review" if _idx == _gate_idx else "pending"))

_exec_logs_cv = (state.values or {}).get("execution_logs", {}) if state.values else {}
if any((l or {}).get("exit_code", 0) != 0 for l in _exec_logs_cv.values()) and stage_statuses[-1] != "completed":
    stage_statuses[-1] = "failed"

current_gate_key = state.next[0] if state.next else None
current_step_key = next((s["step_key"] for s in STAGE_DEFS if s["gate"] == current_gate_key), None)
_gen_src_cv = (state.values or {}).get("generation_source", {}) if state.values else {}

# ── Command bar ──────────────────────────────────────────────────────────
if not state.values:
    _status_cls, _status_label = "neutral", "Not started"
elif current_step_key:
    _stage_nm = next(s["name"].split(". ", 1)[-1] for s in stages if s["step_key"] == current_step_key)
    _status_cls, _status_label = "warn", f"Awaiting review &mdash; {_stage_nm}"
elif state.values.get("final_report"):
    _status_cls, _status_label = "ok", "Pipeline completed"
else:
    _status_cls, _status_label = "neutral", "Idle"

_n_reused = sum(1 for v in _gen_src_cv.values() if v == "cache_reused")
_reuse_chip = (
    f'<span class="chip ok">{_n_reused}/{len(_gen_src_cv)} stages reused from cache</span>'
    if _gen_src_cv and _n_reused else ""
)

_cfg_cb = load_config()
# Build the right-side chips as ONE flat, unindented string. Streamlit runs HTML
# through a markdown parser even with unsafe_allow_html=True: when _reuse_chip is
# empty it leaves a blank line, and the following 4-space-indented <span> is then
# parsed as an indented code block and shown as literal text. Joining the chips
# on a single line with no leading whitespace avoids that entirely.
_right_chips = "".join(
    c for c in [
        f'<span class="chip {_status_cls}"><span class="chip-dot"></span>{_status_label}</span>',
        _reuse_chip,
        f'<span class="chip neutral">{_cfg_cb.get("catalog", "databricks_langgraph")}</span>',
    ] if c
)
st.markdown(f"""
<div class="cmdbar">
  <div class="cmdbar-left">
    <div class="cmd-logo">M</div>
    <div>
      <div class="cmd-title">Medallion Agent Control Center</div>
      <div class="cmd-sub">Multi-agent data engineering on Databricks &mdash; Bronze / Silver / Gold</div>
    </div>
  </div>
  <div class="cmdbar-right">{_right_chips}</div>
</div>
""", unsafe_allow_html=True)


# ── Agent workflow & lineage canvas (SVG) ────────────────────────────────
def _workflow_canvas_svg(stages, statuses, gen_src) -> str:
    """Render the six-agent workflow and the medallion data-lineage lane as
    one self-contained SVG: status-tinted agent nodes with provenance tags
    (CACHED / PATCHED / LLM), animated flow on completed edges, a pulsing
    ring on the stage awaiting review, and dashed drops showing which agent
    feeds which data layer."""
    W, H = 1160, 268
    NW, NH, Y = 158, 64, 26
    C = {
        "completed": {"bar": "#16A34A", "ring": "#BBE7C9", "ink": "#166534", "sub": "Completed"},
        "review":    {"bar": "#D97706", "ring": "#F5D9A8", "ink": "#92400E", "sub": "Needs review"},
        "pending":   {"bar": "#D1D5DB", "ring": "#E5E7EB", "ink": "#9CA3AF", "sub": "Queued"},
        "failed":    {"bar": "#DC2626", "ring": "#F3B6B6", "ink": "#991B1B", "sub": "Failed"},
    }
    PROV = {"cache_reused": ("CACHED", "#16A34A"), "llm_patched": ("PATCHED", "#D97706"), "llm_fresh": ("LLM", "#0F62FE")}
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:Inter,sans-serif;min-width:900px;">']
    parts.append(
        "<style>"
        ".wf-edge{stroke:#D1D5DB;stroke-width:2;fill:none;}"
        ".wf-edge.done{stroke:#16A34A;stroke-dasharray:6 5;animation:wfdash 1s linear infinite;}"
        ".wf-drop{stroke:#CBD5E1;stroke-width:1.5;stroke-dasharray:3 4;fill:none;}"
        "@keyframes wfdash{to{stroke-dashoffset:-11;}}"
        ".wf-pulse{animation:wfpulse 1.8s ease-out infinite;}"
        "@keyframes wfpulse{0%{opacity:.55;}55%{opacity:.12;}100%{opacity:.55;}}"
        "</style>"
    )
    parts.append('<defs><marker id="wfarr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#9CA3AF"/></marker></defs>')
    xs = [20 + i * ((W - 40 - NW) / 5.0) for i in range(6)]
    midy = Y + NH / 2
    for i in range(5):
        cls = "wf-edge done" if statuses[i] == "completed" else "wf-edge"
        parts.append(f'<line class="{cls}" x1="{xs[i] + NW:.0f}" y1="{midy:.0f}" x2="{xs[i + 1]:.0f}" y2="{midy:.0f}"/>')
    for i, stg in enumerate(stages):
        c = C[statuses[i]]
        x = xs[i]
        name = stg["name"].split(". ", 1)[-1]
        if statuses[i] == "review":
            parts.append(f'<rect class="wf-pulse" x="{x - 4:.0f}" y="{Y - 4}" width="{NW + 8}" height="{NH + 8}" rx="13" fill="none" stroke="{c["bar"]}" stroke-width="2"/>')
        parts.append(f'<rect x="{x:.0f}" y="{Y}" width="{NW}" height="{NH}" rx="10" fill="#FFFFFF" stroke="{c["ring"]}" stroke-width="1.5"/>')
        parts.append(f'<rect x="{x:.0f}" y="{Y}" width="4" height="{NH}" rx="2" fill="{c["bar"]}"/>')
        parts.append(f'<text x="{x + 14:.0f}" y="{Y + 26}" font-size="13" font-weight="650" fill="#111827">{name}</text>')
        parts.append(f'<text x="{x + 14:.0f}" y="{Y + 45}" font-size="10.5" font-weight="500" fill="{c["ink"]}">{c["sub"]}</text>')
        prov = (gen_src or {}).get(stg["step_key"])
        if prov in PROV:
            lbl, col = PROV[prov]
            bw = 10 + 6.2 * len(lbl)
            parts.append(f'<rect x="{x + NW - bw - 7:.0f}" y="{Y + NH - 21}" width="{bw:.0f}" height="14" rx="7" fill="{col}" opacity="0.13"/>')
            parts.append(f'<text x="{x + NW - 7 - bw / 2:.0f}" y="{Y + NH - 10.5}" font-size="8.5" font-weight="700" fill="{col}" text-anchor="middle" letter-spacing="0.5">{lbl}</text>')
    LY = 196
    lane = [("RAW VOLUME", "#6B7280"), ("BRONZE", "#A16207"), ("SILVER", "#64748B"), ("GOLD", "#CA8A04"), ("DATA PRODUCTS", "#0F62FE")]
    LW, LH = 140, 32
    lxs = [20 + j * ((W - 40 - LW) / 4.0) for j in range(5)]

    def _drop(i_node, j_lane):
        parts.append(
            f'<path class="wf-drop" d="M {xs[i_node] + NW / 2:.0f} {Y + NH} '
            f'C {xs[i_node] + NW / 2:.0f} {Y + NH + 42}, {lxs[j_lane] + LW / 2:.0f} {LY - 38}, {lxs[j_lane] + LW / 2:.0f} {LY}"/>'
        )

    _drop(0, 0); _drop(4, 1); _drop(4, 2); _drop(4, 3); _drop(5, 4)
    parts.append(f'<text x="20" y="{LY - 12}" font-size="9.5" font-weight="700" fill="#9CA3AF" letter-spacing="1.4">DATA LINEAGE</text>')
    for j, (lbl, col) in enumerate(lane):
        lx = lxs[j]
        parts.append(f'<rect x="{lx:.0f}" y="{LY}" width="{LW}" height="{LH}" rx="16" fill="{col}" opacity="0.08"/>')
        parts.append(f'<rect x="{lx:.0f}" y="{LY}" width="{LW}" height="{LH}" rx="16" fill="none" stroke="{col}" opacity="0.4"/>')
        parts.append(f'<circle cx="{lx + 18:.0f}" cy="{LY + 16}" r="4" fill="{col}"/>')
        parts.append(f'<text x="{lx + 30:.0f}" y="{LY + 20}" font-size="10.5" font-weight="700" fill="#374151" letter-spacing="0.6">{lbl}</text>')
        if j < 4:
            parts.append(f'<line class="wf-edge" x1="{lx + LW:.0f}" y1="{LY + 16}" x2="{lxs[j + 1]:.0f}" y2="{LY + 16}" marker-end="url(#wfarr)"/>')
    parts.append("</svg>")
    return "".join(parts)


st.markdown('<div class="panel-label">Agent Workflow &amp; Lineage</div>', unsafe_allow_html=True)
st.markdown('<div class="canvas-card">' + _workflow_canvas_svg(stages, stage_statuses, _gen_src_cv) + "</div>", unsafe_allow_html=True)

# ── Stage Inspector ──────────────────────────────────────────────────────
# One panel replaces the old per-stage button grid and the Action Center
# tab: pick a stage, see its output + provenance, and — when it's the stage
# the pipeline is paused on — approve/reject right here. Restart-from-stage
# lives under Advanced.
st.markdown('<div class="panel-label">Stage Inspector</div>', unsafe_allow_html=True)

_GLYPH = {"completed": "✓", "review": "◉", "pending": "○", "failed": "✕"}
_step_keys = [s["step_key"] for s in stages]
_label_by_key = {
    s["step_key"]: f"{_GLYPH[stage_statuses[i]]} {s['name'].split('. ', 1)[-1]}"
    for i, s in enumerate(stages)
}
_default_idx = _step_keys.index(current_step_key) if current_step_key in _step_keys else 0
selected_step = st.radio(
    "Stage", _step_keys, index=_default_idx, horizontal=True,
    format_func=lambda k: _label_by_key[k], label_visibility="collapsed",
    key="inspector_stage",
)
sel_idx = _step_keys.index(selected_step)
sel_stage = stages[sel_idx]
sel_status = stage_statuses[sel_idx]
sel_name = sel_stage["name"].split(". ", 1)[-1]

with st.container(border=True):
    _pill_cls, _pill_lbl = {
        "completed": ("ok", "Completed"),
        "review": ("warn", "Awaiting review"),
        "pending": ("neutral", "Queued"),
        "failed": ("err", "Failed"),
    }[sel_status]
    _prov_chip = {
        "cache_reused": '<span class="chip ok">Reused from Unity Catalog cache &mdash; no LLM call</span>',
        "llm_patched": '<span class="chip warn">Patched from prior version (schema changed)</span>',
        "llm_fresh": '<span class="chip neutral">Freshly generated by the LLM</span>',
    }.get(_gen_src_cv.get(selected_step), "")
    st.markdown(f"""
    <div class="insp-head">
      <div><span class="insp-title">{sel_name}</span><span class="insp-agent">{sel_stage["agent_name"]}</span></div>
      <div class="insp-chips"><span class="chip {_pill_cls}"><span class="chip-dot"></span>{_pill_lbl}</span>{_prov_chip}</div>
    </div>
    """, unsafe_allow_html=True)

    if not state.values:
        st.info("No pipeline state yet. Start the pipeline from the Talk to Data tab (say “start the pipeline”) or from the notebook.")
    else:
        render_agent_output(sel_stage["agent_name"], state.values)

    # ── Review decision — only for the stage the pipeline is paused on ──
    if state.next and selected_step == current_step_key and state.values:
        st.markdown("---")
        st.markdown("##### Review decision")
        st.caption(f"The pipeline is paused at **{sel_name}** and needs a human decision before it can continue.")
        _decision = st.radio(
            "Decision", ["Approve", "Reject"], index=0, horizontal=True,
            key="inspector_decision", label_visibility="collapsed",
        )
        _feedback = st.text_area(
            "Feedback / comments",
            placeholder="Optional context if approving — concrete instructions for the agent if rejecting…",
            key="inspector_feedback",
        )
        if st.button("Submit & resume pipeline", type="primary", key="inspector_submit"):
            try:
                approvals = dict(state.values.get("approved_steps", {}))
                _agent_for_log = state.values.get("active_agent", sel_stage["agent_name"])
                _dataset = list(state.values.get("discovered_tables", {}).keys())[0] if state.values.get("discovered_tables") else "generic"
                _issue_type = "data_quality" if selected_step == "dq" else selected_step

                if _decision == "Approve":
                    approvals[selected_step] = True
                    _comments = ""
                    try:
                        memory.log_approval(spark, _dataset, _issue_type, [], f"Approved {selected_step} design", _feedback)
                    except Exception as e:
                        st.warning(f"Unable to log memory to table: {e}")
                else:
                    approvals[selected_step] = False
                    _comments = _feedback
                    try:
                        memory.log_rejection(spark, _dataset, _issue_type, _feedback, selected_step)
                    except Exception as e:
                        st.warning(f"Unable to log rejection to memory table: {e}")

                # Full audit snapshot to gold.agent_stage_review_log
                try:
                    run_id_for_audit = get_or_assign_run_id()
                    try:
                        # Must be the same structural schema fingerprint the caching
                        # nodes compute — that's what lets was_previously_approved()
                        # find THIS decision and auto-advance future cache-hit runs.
                        fingerprint = get_schema_fingerprint(spark)
                    except Exception:
                        fingerprint = ""
                    stage_output = get_stage_artifacts(selected_step, state.values)
                    memory.init_stage_review_table(spark)
                    memory.log_stage_review(
                        spark,
                        pipeline_run_id=run_id_for_audit,
                        stage_key=selected_step,
                        agent_name=_agent_for_log,
                        decision="approved" if _decision == "Approve" else "rejected",
                        reviewer_comments=_feedback,
                        output=stage_output,
                        dataset_fingerprint=fingerprint,
                    )
                except Exception as e:
                    st.warning(f"Unable to log stage output to audit table: {e}")

                app.update_state(config, {"approved_steps": approvals, "review_comments": _comments})
                with st.spinner("Resuming pipeline… later cache-hit stages that were already approved will cascade automatically."):
                    resume_with_autopilot(app, config)
                sync_db_to_volume()
                st.success("Pipeline resumed.")
                st.rerun()
            except Exception as e_click:
                if isinstance(e_click, KeyError) and "__end__" in str(e_click):
                    try:
                        sync_db_to_volume()
                    except Exception:
                        pass
                    st.success("Pipeline finished. Refreshing…")
                    st.rerun()
                else:
                    st.error("Execution error during submit & resume:")
                    st.exception(e_click)

    # ── Advanced: restart the pipeline from this stage ───────────────────
    with st.expander("Advanced — restart pipeline from this stage"):
        st.caption(
            "Rolls the checkpoint back to just before this agent and re-executes it. "
            "Later stages re-run too, reusing Unity Catalog caches wherever the source "
            "schema is unchanged."
        )
        _gate_to_node = {
            "profile_review_gate": "profiler",
            "data_quality_review_gate": "data_quality",
            "contracts_review_gate": "contracts",
            "modeling_review_gate": "modeling",
            "engineering_review_gate": "engineering",
            "execution_review_gate": "execution",
        }
        if st.button(f"Restart from {sel_name}", key=f"restart_{sel_stage['gate']}"):
            _target_node = _gate_to_node.get(sel_stage["gate"])
            with st.spinner(f"Rolling back and running '{sel_name}'…"):
                if rollback_to_node(_target_node):
                    refresh_graph_checkpoint()
                    _app_rr = get_or_create_graph()
                    try:
                        # Intentionally NOT injecting a fresh pipeline_run_id: rolling
                        # back within the same run keeps its run ID; a new one is only
                        # assigned lazily after a full Reset.
                        _inputs = {} if _target_node == "profiler" else None
                        resume_with_autopilot(_app_rr, config, initial_input=_inputs)
                    except Exception as e_stream:
                        if not (isinstance(e_stream, KeyError) and "__end__" in str(e_stream)):
                            st.error(f"Stream execution error: {e_stream}")
                    sync_db_to_volume()
                    refresh_graph_checkpoint()
                    st.success(f"'{sel_name}' execution completed.")
                    st.rerun()
                else:
                    st.error("Cannot restart: no historical checkpoint found for this stage.")

if not state.next and state.values and state.values.get("final_report"):
    st.success("Pipeline is completely finished — no review pending.")

# Sidebar: brand, environment, actions
catalog_config = load_config()
st.sidebar.markdown("""
<div class="side-brand">
  <div class="side-brand-logo">&#9670;</div>
  <div>
    <div class="side-brand-name">Medallion Agents</div>
    <div class="side-brand-sub">Pipeline Control Center</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="side-section">Environment</div>', unsafe_allow_html=True)
_llm_ep = ((catalog_config.get("llm") or {}).get("endpoint") or "&mdash;")
st.sidebar.markdown(f"""
<div class="side-kv"><span>Catalog</span><code>{catalog_config.get("catalog", "databricks_langgraph")}</code></div>
<div class="side-kv"><span>Raw Volume</span><code>{os.path.basename(catalog_config.get("volume_raw_path", "source_volume"))}</code></div>
<div class="side-kv"><span>LLM</span><code>{_llm_ep[:26]}</code></div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="side-section">Actions</div>', unsafe_allow_html=True)

# Refresh button
if st.sidebar.button("Refresh data"):
    sync_db_from_volume()
    refresh_graph_checkpoint()
    st.rerun()

# Reset button
if st.sidebar.button("Reset pipeline / start fresh", type="secondary"):
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
        w = WorkspaceClient()  # default auth (App service principal / local profile)
        w.files.delete(volume_db)
    except Exception:
        pass

    # 4. Clear the agent OUTPUT caches (codebase / dq / contracts / modeling /
    #    stage approvals). Deleting only the checkpoint above resets the graph
    #    state but leaves these caches intact, so the Engineer stage would
    #    otherwise recall the SAME (possibly broken) cached scripts and it would
    #    look like the reset did nothing. This is the piece that makes "start
    #    fresh" actually regenerate code.
    cache_logs = []
    try:
        cache_logs = memory.reset_agent_caches(spark)
    except Exception as e_cache:
        cache_logs = [f"Cache reset failed: {e_cache}"]

    st.session_state["sync_logs"] = (
        ["Pipeline reset. Checkpoint deleted from local disk and UC Volume."]
        + ["Agent output caches cleared:"]
        + [f"  • {line}" for line in cache_logs]
    )
    refresh_graph_checkpoint()
    st.rerun()

# Diagnostics & Health check in sidebar
with st.sidebar.expander("Diagnostics & health"):
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

# Layout Tabs — approvals live in the Stage Inspector above, so there is
# no separate Action Center tab anymore.
tab0, tab1, tab3, tab4 = st.tabs([
    "Talk to Data",
    "Observability",
    "Audit Trail",
    "Data Products",
])

# ----------------- Tab 0: Talk to Data Chatbot -----------------
with tab0:
    # Initialise session-state keys
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "pending_pipeline_action" not in st.session_state:
        st.session_state["pending_pipeline_action"] = None

    # ---- Header ----
    st.markdown("#### Talk to Data")
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

                    if gate == "__start__":
                        # ── Cold start: kick off a fresh pipeline run ────────
                        # resume_with_autopilot cascades straight through any stage
                        # that turns out to be an exact schema-fingerprint cache hit
                        # already approved on a prior run (e.g. re-running against the
                        # same dataset after a Reset) — only genuinely new/changed
                        # stages, plus Profiler every time, will actually pause here.
                        new_pipeline_run_id = str(uuid.uuid4())
                        with st.spinner("🚀 Launching pipeline from scratch… this may take a few minutes."):
                            try:
                                resume_with_autopilot(app, config, initial_input={"pipeline_run_id": new_pipeline_run_id})
                            except KeyError as e_key:
                                if "__end__" not in str(e_key):
                                    raise

                        sync_db_to_volume()

                        msg = (
                            "🚀 **Pipeline launched!** The Profiler agent is now running. "
                            "Use the **Stage Inspector** at the top of the page to approve each stage as it completes."
                        )
                        st.session_state["chat_history"].append({
                            "role": "assistant", "content": msg, "profiling_triggered": False
                        })
                        st.session_state["pending_pipeline_action"] = None
                        refresh_graph_checkpoint()
                        st.success(msg)
                        st.rerun()
                    else:
                        # ── Mid-run: approve the current gate ───────────────
                        # Build updated approvals — use boolean True so routing
                        # functions (which check `is True`) correctly advance
                        current_state = app.get_state(config)
                        current_approved = dict(current_state.values.get("approved_steps", {}))
                        current_approved[step] = True  # Must be boolean, not string

                        app.update_state(
                            config,
                            {"approved_steps": current_approved}
                        )

                        # Stream the graph forward — auto-cascades through any later
                        # cache-hit-and-previously-approved gate too.
                        with st.spinner("Resuming pipeline… this may take a moment."):
                            try:
                                resume_with_autopilot(app, config)
                            except KeyError as e_key:
                                if "__end__" not in str(e_key):
                                    raise

                        # Persist to Volume
                        sync_db_to_volume()

                        msg = (
                            f"✅ The **{step.title()}** step is approved and the pipeline has resumed. "
                            "Track progress in the **Observability** tab."
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
        if st.button("Clear chat", key="clear_chat_btn"):
            st.session_state["chat_history"] = []
            st.session_state["pending_pipeline_action"] = None
            st.rerun()
    with ctrl_col2:
        if st.button("Reset profile cache", key="reset_profile_cache_btn"):
            clear_profiling_cache()
            st.success("Profiling cache cleared.")


# ----------------- Tab 1: Observability -----------------
with tab1:
    st.markdown("### Ingestion Pipeline Metrics")
    
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
            st.markdown(f'<div class="metric-card" style="border-top:3px solid #0284C7;"><div class="metric-kicker" style="color:#0284C7;">Raw Layer</div><div class="metric-value">{counts["raw"][1]:,}</div><div class="metric-label">source rows staged in Volume</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card" style="border-top:3px solid #B45309;"><div class="metric-kicker" style="color:#B45309;">Bronze Layer</div><div class="metric-value">{counts["bronze"][1]:,}</div><div class="metric-label">rows across {counts["bronze"][0]} Delta tables</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card" style="border-top:3px solid #64748B;"><div class="metric-kicker" style="color:#64748B;">Silver Layer</div><div class="metric-value">{counts["silver"][1]:,}</div><div class="metric-label">validated rows across {counts["silver"][0]} tables</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card" style="border-top:3px solid #B7791F;"><div class="metric-kicker" style="color:#B7791F;">Gold Layer</div><div class="metric-value">{counts["gold"][1]:,}</div><div class="metric-label">star-schema rows across {counts["gold"][0]} tables</div></div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.warning(f"Unable to query live table metrics: {e}")
        
    st.markdown("---")
    st.markdown("### Pipeline Run Details")
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

# ----------------- Tab 3: Run History -----------------
with tab3:
    st.markdown("### Run History & Review Decisions")
    st.caption("Every pipeline execution attempt and every human review decision, kept for auditing — even after the live thread moves on or is reset.")

    hist_subtab, decisions_subtab, outputs_subtab = st.tabs([
        "Pipeline Runs", "Review Decisions", "Agent Outputs & Reviews"
    ])

    # ---- Sub-tab: Pipeline Run History (gold.agent_run_history) ----
    with hist_subtab:
        try:
            runs = memory.get_run_history(spark, limit=500)
        except Exception as e:
            runs = []
            st.warning(f"Could not load run history: {e}")

        if not runs:
            st.info("No pipeline runs logged yet. Run history is recorded automatically every time the Orchestrator executes the ETL scripts.")
        else:
            runs_df = pd.DataFrame(runs)
            runs_df["run_timestamp"] = pd.to_datetime(runs_df["run_timestamp"])
            runs_df["run_date"] = pd.to_datetime(runs_df["run_date"]).dt.date

            # ---- Filters ----
            f1, f2, f3 = st.columns([1.3, 1, 1.7])
            with f1:
                min_date, max_date = runs_df["run_date"].min(), runs_df["run_date"].max()
                date_range = st.date_input(
                    "Date range", value=(min_date, max_date),
                    min_value=min_date, max_value=max_date, key="run_hist_date_range"
                )
            with f2:
                status_options = sorted(runs_df["pipeline_status"].dropna().unique().tolist())
                status_filter = st.multiselect("Status", status_options, default=status_options, key="run_hist_status")
            with f3:
                search_text = st.text_input("Search (run ID or dataset fingerprint)", key="run_hist_search")

            filtered = runs_df.copy()
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_d, end_d = date_range
                filtered = filtered[(filtered["run_date"] >= start_d) & (filtered["run_date"] <= end_d)]
            if status_filter:
                filtered = filtered[filtered["pipeline_status"].isin(status_filter)]
            if search_text:
                mask = (
                    filtered["run_id"].str.contains(search_text, case=False, na=False)
                    | filtered["dataset_fingerprint"].str.contains(search_text, case=False, na=False)
                )
                filtered = filtered[mask]

            st.caption(f"Showing {len(filtered)} of {len(runs_df)} runs.")

            display_cols = ["run_timestamp", "pipeline_status", "active_agent", "failed_scripts", "dataset_fingerprint", "run_id"]
            st.dataframe(
                filtered[display_cols].sort_values("run_timestamp", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "run_timestamp": st.column_config.DatetimeColumn("Run Time"),
                    "pipeline_status": "Status",
                    "active_agent": "Agent",
                    "failed_scripts": "Failed Scripts",
                    "dataset_fingerprint": "Fingerprint",
                    "run_id": "Run ID",
                },
            )

            if len(filtered) > 0:
                st.markdown("#### Inspect a Run")
                options = filtered.sort_values("run_timestamp", ascending=False)["run_id"].tolist()
                labels = {
                    r: f"{ts} · {status} · {rid[:8]}"
                    for r, ts, status, rid in zip(
                        options,
                        filtered.set_index("run_id").loc[options, "run_timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
                        filtered.set_index("run_id").loc[options, "pipeline_status"],
                        options,
                    )
                }
                selected_run_id = st.selectbox(
                    "Select a run", options, format_func=lambda r: labels.get(r, r), key="run_hist_selected_run"
                )
                run_row = filtered[filtered["run_id"] == selected_run_id].iloc[0]

                status_color = "🟢" if run_row["pipeline_status"] == "COMPLETED" else "🔴"
                st.markdown(f"**{status_color} {run_row['pipeline_status']}** — `{run_row['run_id']}` — {run_row['run_timestamp']}")

                with st.expander("Final Run Report", expanded=True):
                    st.markdown(run_row.get("final_report") or "No report captured for this run.")

                with st.expander("Script Execution Logs (stdout/stderr)"):
                    try:
                        exec_logs = json.loads(run_row.get("execution_logs") or "{}")
                    except Exception:
                        exec_logs = {}
                    if exec_logs:
                        for script_name, log_info in exec_logs.items():
                            st.markdown(f"**{script_name}** — exit code `{log_info.get('exit_code')}`")
                            st.code(log_info.get("stdout") or "No stdout output.")
                            if log_info.get("stderr"):
                                st.error(log_info.get("stderr"))
                    else:
                        st.info("No execution logs captured.")

                with st.expander("Silver / Gold Summaries"):
                    try:
                        silver_s = json.loads(run_row.get("silver_summary") or "{}")
                    except Exception:
                        silver_s = {}
                    try:
                        gold_s = json.loads(run_row.get("gold_summary") or "{}")
                    except Exception:
                        gold_s = {}
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.markdown("**Silver Summary**")
                        st.json(silver_s or {"info": "empty"})
                    with sc2:
                        st.markdown("**Gold Summary**")
                        st.json(gold_s or {"info": "empty"})

                with st.expander("Approvals in effect at time of run"):
                    try:
                        approved = json.loads(run_row.get("approved_steps") or "{}")
                    except Exception:
                        approved = {}
                    st.json(approved or {"info": "empty"})
                    if run_row.get("review_comments"):
                        st.markdown(f"**Review comments that triggered this attempt:** {run_row['review_comments']}")

    # ---- Sub-tab: Review Decisions (gold.agent_fewshot_memory) ----
    with decisions_subtab:
        st.caption("Human approve/reject decisions made at each review gate. The agents also read this history as few-shot context for future runs.")

        try:
            catalog = catalog_config.get("catalog", "databricks_langgraph")
            try:
                is_local = spark.conf.get("spark.master", "").startswith("local")
            except Exception:
                is_local = False

            if not is_local:
                fqn = f"{catalog}.gold.agent_fewshot_memory"
                pandas_df = spark.read.table(fqn).orderBy("timestamp", ascending=False).toPandas()
            elif os.path.exists(memory.LOCAL_MEMORY_PATH):
                with open(memory.LOCAL_MEMORY_PATH, "r") as f:
                    records = json.load(f)
                pandas_df = pd.DataFrame(records)
            else:
                pandas_df = pd.DataFrame()

            if len(pandas_df) == 0:
                st.info("No review decisions logged yet. They'll appear here once you approve or reject a step in the Stage Inspector.")
            else:
                if "decision" not in pandas_df.columns:
                    # Backward-compat: infer from resolution text if reading an older table/cache
                    pandas_df["decision"] = pandas_df["resolution_applied"].str.contains("Rejected", case=False, na=False).map(
                        {True: "rejected", False: "approved"}
                    )
                pandas_df["timestamp"] = pd.to_datetime(pandas_df["timestamp"])
                pandas_df["decision_date"] = pandas_df["timestamp"].dt.date

                d1, d2, d3 = st.columns([1.3, 1, 1.7])
                with d1:
                    dmin, dmax = pandas_df["decision_date"].min(), pandas_df["decision_date"].max()
                    decision_date_range = st.date_input(
                        "Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="decisions_date_range"
                    )
                with d2:
                    decision_filter = st.multiselect(
                        "Decision", ["approved", "rejected"], default=["approved", "rejected"], key="decisions_type"
                    )
                with d3:
                    step_options = sorted(pandas_df["issue_type"].dropna().unique().tolist())
                    step_filter = st.multiselect("Step", step_options, default=step_options, key="decisions_step")

                filtered_d = pandas_df.copy()
                if isinstance(decision_date_range, tuple) and len(decision_date_range) == 2:
                    sd, ed = decision_date_range
                    filtered_d = filtered_d[(filtered_d["decision_date"] >= sd) & (filtered_d["decision_date"] <= ed)]
                if decision_filter:
                    filtered_d = filtered_d[filtered_d["decision"].isin(decision_filter)]
                if step_filter:
                    filtered_d = filtered_d[filtered_d["issue_type"].isin(step_filter)]

                st.caption(f"Showing {len(filtered_d)} of {len(pandas_df)} decisions.")
                display_cols_d = ["timestamp", "decision", "issue_type", "dataset_name", "human_comments", "resolution_applied"]
                display_cols_d = [c for c in display_cols_d if c in filtered_d.columns]
                st.dataframe(
                    filtered_d[display_cols_d].sort_values("timestamp", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "timestamp": st.column_config.DatetimeColumn("When"),
                        "decision": "Decision",
                        "issue_type": "Step",
                        "dataset_name": "Dataset",
                        "human_comments": "Comments",
                        "resolution_applied": "Resolution",
                    },
                )

        except Exception as e:
            st.warning(f"Could not load review decisions: {e}")

    # ---- Sub-tab: Agent Outputs & Reviews (gold.agent_stage_review_log) ----
    with outputs_subtab:
        st.caption(
            "Full audit trail: what each agent produced at every review gate, the human "
            "decision, and any review comments — stored in Unity Catalog and filterable by date."
        )

        try:
            stage_reviews = memory.get_stage_reviews(spark, limit=1000)
        except Exception as e:
            stage_reviews = []
            st.warning(f"Could not load agent output audit log: {e}")

        if not stage_reviews:
            st.info(
                "No agent outputs logged yet. Every time you approve or reject a step in the "
                "Stage Inspector, that stage's full output is captured here."
            )
        else:
            audit_df = pd.DataFrame(stage_reviews)
            audit_df["review_timestamp"] = pd.to_datetime(audit_df["review_timestamp"])
            audit_df["review_date"] = pd.to_datetime(audit_df["review_date"]).dt.date

            agent_name_by_step = {s["step_key"]: s["agent_name"] for s in STAGE_DEFS}
            stage_label_by_step = {s["step_key"]: s["name"] for s in STAGE_DEFS}

            a1, a2, a3, a4 = st.columns([1.3, 1, 1, 1.7])
            with a1:
                amin, amax = audit_df["review_date"].min(), audit_df["review_date"].max()
                audit_date_range = st.date_input(
                    "Date range", value=(amin, amax), min_value=amin, max_value=amax, key="audit_date_range"
                )
            with a2:
                audit_decision_filter = st.multiselect(
                    "Decision", ["approved", "rejected"], default=["approved", "rejected"], key="audit_decision"
                )
            with a3:
                audit_stage_options = sorted(audit_df["stage_key"].dropna().unique().tolist())
                audit_stage_filter = st.multiselect(
                    "Stage", audit_stage_options,
                    default=audit_stage_options,
                    format_func=lambda k: stage_label_by_step.get(k, k),
                    key="audit_stage",
                )
            with a4:
                audit_run_filter = st.text_input(
                    "Filter by Run ID (optional, exact or partial)", key="audit_run_id_search"
                )

            filtered_a = audit_df.copy()
            if isinstance(audit_date_range, tuple) and len(audit_date_range) == 2:
                asd, aed = audit_date_range
                filtered_a = filtered_a[(filtered_a["review_date"] >= asd) & (filtered_a["review_date"] <= aed)]
            if audit_decision_filter:
                filtered_a = filtered_a[filtered_a["decision"].isin(audit_decision_filter)]
            if audit_stage_filter:
                filtered_a = filtered_a[filtered_a["stage_key"].isin(audit_stage_filter)]
            if audit_run_filter:
                filtered_a = filtered_a[filtered_a["pipeline_run_id"].str.contains(audit_run_filter, case=False, na=False)]

            filtered_a = filtered_a.sort_values("review_timestamp", ascending=False)
            st.caption(f"Showing {len(filtered_a)} of {len(audit_df)} reviewed outputs.")

            summary_cols = ["review_timestamp", "decision", "stage_key", "agent_name", "reviewer_comments", "pipeline_run_id"]
            summary_cols = [c for c in summary_cols if c in filtered_a.columns]
            st.dataframe(
                filtered_a[summary_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "review_timestamp": st.column_config.DatetimeColumn("When"),
                    "decision": "Decision",
                    "stage_key": "Stage",
                    "agent_name": "Agent",
                    "reviewer_comments": "Comments",
                    "pipeline_run_id": "Run ID",
                },
            )

            st.markdown("#### Inspect a Reviewed Output")
            if len(filtered_a) > 0:
                row_options = filtered_a.index.tolist()
                row_labels = {
                    idx: (
                        f"{r['review_timestamp'].strftime('%Y-%m-%d %H:%M')} · "
                        f"{'✅' if r['decision'] == 'approved' else '🚫'} {r['decision']} · "
                        f"{stage_label_by_step.get(r['stage_key'], r['stage_key'])} · "
                        f"run {str(r.get('pipeline_run_id') or '')[:8]}"
                    )
                    for idx, r in filtered_a.iterrows()
                }
                selected_idx = st.selectbox(
                    "Select a reviewed output", row_options, format_func=lambda i: row_labels.get(i, str(i)),
                    key="audit_selected_row",
                )
                sel_row = filtered_a.loc[selected_idx]

                decision_icon = "✅" if sel_row["decision"] == "approved" else "🚫"
                st.markdown(
                    f"**{decision_icon} {sel_row['decision'].upper()}** — "
                    f"{stage_label_by_step.get(sel_row['stage_key'], sel_row['stage_key'])} "
                    f"(**{sel_row['agent_name']}**) — {sel_row['review_timestamp']}"
                )
                if sel_row.get("dataset_fingerprint"):
                    st.caption(f"Dataset fingerprint: `{sel_row['dataset_fingerprint']}` · Run ID: `{sel_row.get('pipeline_run_id', '')}`")

                if sel_row.get("reviewer_comments"):
                    st.markdown(f"**Reviewer comments:** {sel_row['reviewer_comments']}")
                else:
                    st.caption("No reviewer comments were entered for this decision.")

                with st.expander("Agent Output at Time of Review", expanded=True):
                    try:
                        output_dict = json.loads(sel_row.get("output_json") or "{}")
                    except Exception:
                        output_dict = {}
                    agent_name_for_render = sel_row.get("agent_name") or agent_name_by_step.get(sel_row["stage_key"])
                    if output_dict:
                        render_agent_output(agent_name_for_render, output_dict)
                    else:
                        st.info("No output snapshot was captured for this review.")
            else:
                st.info("No reviewed outputs match the current filters.")

# ----------------- Tab 4: Data Products -----------------
with tab4:
    st.markdown("### Data Products")
    st.caption(
        "The Data Product Advisor analyzes your completed Gold star schema and proposes purpose-built "
        "marts/views on top of it — a Customer 360, a booking profitability mart, a supplier scorecard, etc. "
        "Review a proposal, then build it with one click. Each product is materialized into its own "
        "`products` schema in Unity Catalog, kept separate from the conformed Gold model."
    )

    gold_ready = bool(state.values.get("gold_ddl")) if state.values else False

    if not gold_ready:
        st.info(
            "The Gold star schema hasn't been designed/built yet. Run the pipeline through the "
            "Modeler → Engineer → Orchestrator stages (see the **Stage Inspector** at the top of the page) before "
            "analyzing data product opportunities."
        )
    else:
        adv_col1, adv_col2 = st.columns([1, 3])
        with adv_col1:
            analyze_clicked = st.button(
                "Analyze Gold for data products", type="primary",
                use_container_width=True, key="analyze_products_btn",
            )
        with adv_col2:
            existing_candidates = state.values.get("product_candidates", []) if state.values else []
            if existing_candidates:
                st.caption(
                    f"{len(existing_candidates)} proposed product(s) from the last analysis. "
                    "Re-run anytime — Gold rarely changes shape between runs."
                )

        if analyze_clicked:
            with st.spinner("🧠 Data Product Advisor is analyzing the Gold star schema..."):
                try:
                    advisor_result = product_advisor_node(state.values or {})
                    app.update_state(config, advisor_result)
                    sync_db_to_volume()
                    refresh_graph_checkpoint()
                    if advisor_result.get("product_advisor_error"):
                        st.warning(advisor_result["product_advisor_error"])
                    else:
                        st.success(f"Proposed {len(advisor_result.get('product_candidates', []))} candidate data product(s).")
                    st.rerun()
                except Exception as e_adv:
                    st.error(f"Data Product Advisor failed: {e_adv}")
                    st.exception(e_adv)

        st.markdown("---")

        candidates = state.values.get("product_candidates", []) if state.values else []
        build_status = state.values.get("product_build_status", {}) if state.values else {}
        advisor_error = state.values.get("product_advisor_error", "") if state.values else ""

        if advisor_error:
            st.warning(advisor_error)

        if not candidates:
            st.info("No data products proposed yet. Click **🔍 Analyze Gold for Data Products** above to get started.")
        else:
            for product in candidates:
                product_id = product.get("id", "unknown")
                status_info = build_status.get(product_id, {})
                status = status_info.get("status", "not_built")

                with st.container(border=True):
                    head_col, badge_col = st.columns([4, 1])
                    with head_col:
                        st.markdown(f"#### {product.get('name', product_id)}")
                        st.caption(product.get("description", ""))
                    with badge_col:
                        type_label = (product.get("product_type") or "view").upper()
                        st.markdown(
                            f'<div style="text-align:right;"><span class="agent-badge">{type_label}</span></div>',
                            unsafe_allow_html=True,
                        )

                    meta_col1, meta_col2, meta_col3 = st.columns(3)
                    with meta_col1:
                        st.markdown(f"**Grain:** {product.get('grain', 'N/A')}")
                    with meta_col2:
                        st.markdown(f"**Source tables:** {', '.join(product.get('source_tables', [])) or 'N/A'}")
                    with meta_col3:
                        st.markdown(f"**Refresh:** {product.get('refresh_frequency', 'N/A')}")

                    with st.expander("View generated SQL"):
                        st.code(product.get("sql", "-- no SQL generated --"), language="sql")

                    build_col, status_col = st.columns([1, 3])
                    with build_col:
                        build_clicked = st.button(
                            "Build", key=f"build_{product_id}", use_container_width=True
                        )
                    with status_col:
                        if status == "built":
                            st.success(
                                f"✅ Built as `{status_info.get('target_fqn', '')}` "
                                f"({status_info.get('row_count', 'N/A')} rows) — "
                                f"last built {status_info.get('last_built_ts', '')}"
                            )
                        elif status == "failed":
                            st.error(f"❌ Build failed: {status_info.get('error', 'Unknown error')}")
                        else:
                            st.caption("Not built yet.")

                    if build_clicked:
                        with st.spinner(f"Building `{product_id}`..."):
                            try:
                                build_result = products_module.build_product(product)
                                updated_status = dict(build_status)
                                updated_status[product_id] = build_result
                                app.update_state(config, {"product_build_status": updated_status})
                                sync_db_to_volume()
                                refresh_graph_checkpoint()
                                if build_result.get("status") == "built":
                                    st.success(f"Built `{product_id}` successfully!")
                                else:
                                    st.error(f"Build failed: {build_result.get('error')}")
                                st.rerun()
                            except Exception as e_build:
                                st.error(f"Unexpected error building `{product_id}`: {e_build}")
                                st.exception(e_build)
