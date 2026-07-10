# Databricks + LangGraph Medallion Pipeline Agent Framework

A general-purpose multi-agent pipeline framework built using **LangGraph** and deployed on **Databricks**, governed by **Unity Catalog**. 

This codebase replaces CrewAI with a robust LangGraph state machine that incorporates sequential **Human-in-the-Loop (HITL)** validation gates at every stage of the pipeline (Profiling → Contracts → Modeling → Code Gen → Run & Report).

---

## Architecture Overview

The pipeline utilizes 5 distinct agent personas to move raw data through a **Bronze → Silver → Gold** medallion lifecycle:

1. **Profiler Agent**: Inspects raw data files inside the Unity Catalog Volume, discovering schemas, tables, duplicate business keys, and table relations.
2. **Contract Steward**: Designs rules (`not_null`, `unique`, `allowed_values`, `range`, `referential_integrity`) written to YAML contract files.
3. **Data Modeler**: Designs Kimball-style dimensions (SCD Type 1 vs SCD Type 2) and fact tables, producing SQL DDL.
4. **Data Engineer**: Automatically writes PySpark code for Bronze, Silver, and Gold transitions.
5. **Orchestrator**: Executes the transformation scripts on Databricks and produces an executive final run report.

### Human-In-The-Loop (HITL) Breakpoints

At each stage, the execution halts. A human operator reviews the current state variables (e.g. profiling report, contracts YAML, modeling DDL, generated python code) directly in the Databricks notebook, approving or rejecting with feedback:

```
[Agent Node] ──> [State Pause (Breakpoint)] ──> [Human Review Gate (Approve / Reject)]
                                                      │              │
                                                      │ (Approve)    │ (Reject + Feedback)
                                                      ▼              ▼
                                                [Next Agent]   [Re-route to Agent]
```

---

## Directory Structure

*   `config.yaml`: Global configurations for Catalog, schemas, Volume paths, and the LLM endpoint.
*   `Medallion_Pipeline_Notebook.py`: Databricks Notebook script that imports the graph and runs it interactively with HTML/Widget inputs.
*   `src/data_platform/`: Shared data infrastructure.
    *   `spark_utils.py`: Databricks-specific read, write, upsert, and Slowly Changing Dimension Type 2 (`scd2_merge`) helpers.
    *   `contracts.py`: Unity Catalog-compatible data contract check engine.
    *   `profiling.py`: Statistical profiling tool reading from Unity Catalog Volumes.
*   `src/orchestrator/`: LangGraph implementation.
    *   `state.py`: Defines the execution state and review variables.
    *   `prompts.py`: General-purpose system prompts for the agents.
    *   `agents.py`: Agent initializations calling Databricks Model Serving LLMs.
    *   `graph.py`: Builds the state graph, registers nodes, and configures conditional edges.
*   `reference_implementation/`: Manual reference implementation of the pipeline for Databricks.

---

## Setup & Deployment on Databricks

### 1. Prerequisites
*   A Databricks Workspace with **Unity Catalog** enabled.
*   A running cluster with a Databricks Runtime (DBR 14.x+ recommended for modern Python/PySpark features).
*   A Unity Catalog **Volume** containing your raw CSV files (e.g., `/Volumes/hospitality_catalog/raw/source_volume/`).
*   A **Model Serving Endpoint** running a foundation model (e.g., `databricks-meta-llama-3-1-70b-instruct`).

### 2. Configure Git & Databricks Repos
1.  Push this folder (`dbricks-lang-agent`) to a git repository (GitHub, GitLab, etc.).
2.  In Databricks, navigate to **Workspace** → **Repos** → **Add Repo**.
3.  Enter your Repository URL and clone it into the workspace.

### 3. Setup Configuration
Update `config.yaml` with your target Catalog, schemas, volume path, and Model Serving endpoint. Make sure the schemas (`bronze`, `silver`, `gold`) exist in your Catalog, or the orchestrator will automatically create them.

### 4. Running the Pipeline
Open `Medallion_Pipeline_Notebook.py` inside your Databricks Workspace and click **Run All**.
The notebook will:
1. Initialize the LangGraph pipeline state.
2. Run the Profiler Agent.
3. Pause and present a widget interface for you to inspect findings, approve, or reject.
4. Step sequentially through each agent and Human-in-the-Loop checkpoint.
