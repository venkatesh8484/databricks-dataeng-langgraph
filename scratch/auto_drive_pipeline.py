import time
import os
import sqlite3
import configparser
import shutil
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

def get_current_node_from_local_db():
    local_path = "checkpoint_inspect_final.db"
    tmp_path = "/tmp/checkpoint.db"
    if not os.path.exists(local_path):
        return "UNKNOWN"
    try:
        # Copy to /tmp/checkpoint.db so create_pipeline_graph reads it
        shutil.copyfile(local_path, tmp_path)
        
        from dbricks_lang_agent.orchestrator.graph import create_pipeline_graph
        app = create_pipeline_graph()
        config = {"configurable": {"thread_id": "medallion_pipeline_run"}}
        state = app.get_state(config)
        
        if not state.next:
            return "FINISHED"
        return state.next[0]
    except Exception as e:
        print("Failed to read current node using LangGraph API:", e)
        return "UNKNOWN"

def download_checkpoint(w):
    path = "/Volumes/databricks_langgraph/raw/source_volume/checkpoint.db"
    local_path = "checkpoint_inspect_final.db"
    try:
        res = w.files.download(path)
        with open(local_path, "wb") as lf:
            lf.write(res.contents.read())
        return True
    except Exception as e:
        print(f"[Warning] Failed to download checkpoint: {e}")
        return False

def main():
    print("Connecting to Databricks Workspace...")
    cfg_path = os.path.expanduser("~/.databrickscfg")
    config = configparser.ConfigParser()
    config.read(cfg_path)
    profile = "venkatesh8484"
    if profile not in config.sections():
        profile = "DEFAULT"
    host = config.get(profile, "host").strip('"').strip("'")
    token = config.get(profile, "token").strip('"').strip("'")
    w = WorkspaceClient(host=host, token=token)
    
    notebook_path = "/Workspace/Users/venkatesh8484@gmail.com/.bundle/dbricks-lang-agent/dev/files/Medallion_Pipeline_Notebook"
    
    # Download the initial checkpoint to set up local DB
    download_checkpoint(w)
    
    step_num = 1
    while True:
        current_node = get_current_node_from_local_db()
        print(f"\n==================================================")
        print(f"STEP {step_num}: Current Node = {current_node}")
        print(f"==================================================")
        
        if current_node == "FINISHED" or current_node == "UNKNOWN":
            if current_node == "FINISHED":
                print("Pipeline execution is fully FINISHED! Exiting auto-drive.")
            else:
                print("Could not retrieve current node, breaking.")
            break
            
        print(f"Submitting Approval to resume from '{current_node}'...")
        try:
            run_waiter = w.jobs.submit(
                run_name=f"auto-drive-step-{step_num}-{int(time.time())}",
                tasks=[
                    jobs.SubmitTask(
                        task_key="notebook_run",
                        notebook_task=jobs.NotebookTask(
                            notebook_path=notebook_path,
                            base_parameters={
                                "hitl_action": "Approve",
                                "hitl_comment": f"Auto-approved at step {step_num}"
                            }
                        )
                    )
                ]
            )
            print(f"Run submitted! Run ID: {run_waiter.run_id}")
            result = run_waiter.result()
            print(f"Run finished with State: {result.state.life_cycle_state.value} | {result.state.result_state.value}")
            
            # Download the updated checkpoint
            download_checkpoint(w)
            
            # Parse outputs if they are in the task run output
            if result.tasks:
                task_run_id = result.tasks[0].run_id
                try:
                    export_resp = w.jobs.export_run(run_id=task_run_id)
                    html_content = export_resp.views[0].content
                    import base64
                    import urllib.parse
                    import json
                    pattern = r"var __DATABRICKS_NOTEBOOK_MODEL = '([^']*)';"
                    match = re.search(pattern, html_content)
                    if match:
                        encoded_str = match.group(1)
                        decoded_bytes = base64.b64decode(encoded_str)
                        decoded_base64 = decoded_bytes.decode('utf-8')
                        decoded_str = urllib.parse.unquote(decoded_base64)
                        model = json.loads(decoded_str)
                        for cmd in model.get("commands", []):
                            results = cmd.get("results")
                            if results:
                                data = results.get("data", [])
                                ans = []
                                if isinstance(data, list):
                                    for d in data:
                                        if isinstance(d, dict) and d.get("type") == "ansi":
                                            ans.append(d.get("data", ""))
                                output_str = "".join(ans).strip()
                                if output_str:
                                    print(f"[Run Output]: {output_str[-1500:]}")
                except Exception as ex:
                    print(f"Warning: Failed to fetch task outputs: {ex}")
            
        except Exception as e:
            print(f"Failed to submit or run notebook at step {step_num}: {e}")
            break
            
        step_num += 1
        time.sleep(2)

if __name__ == "__main__":
    import re
    main()
