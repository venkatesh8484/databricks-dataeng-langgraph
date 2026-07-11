import time
import os
import configparser
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

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
    print(f"Submitting one-time notebook run for: {notebook_path}...")
    
    try:
        run_waiter = w.jobs.submit(
            run_name=f"agent-pipeline-compiler-loop-run-{int(time.time())}",
            tasks=[
                jobs.SubmitTask(
                    task_key="notebook_run",
                    notebook_task=jobs.NotebookTask(
                        notebook_path=notebook_path,
                        base_parameters={
                            "reset_pipeline": "True",
                            "hitl_action": "Approve",
                            "hitl_comment": "Run with compiler self-correcting loop and codebase memory"
                        }
                    )
                )
            ]
        )
        
        print(f"Run submitted! Run ID: {run_waiter.run_id}")
        print("Waiting for run to complete/breakpoint... (this can take 3-5 minutes because of compiler loops)")
        
        result = run_waiter.result()
        state = result.state
        life_cycle_state = getattr(state.life_cycle_state, "value", str(state.life_cycle_state))
        result_state = getattr(state.result_state, "value", str(state.result_state))
        
        print(f"Run finished with Life Cycle State: {life_cycle_state} | Result State: {result_state}")
        
        if result.tasks:
            task_run_id = result.tasks[0].run_id
            run_output = w.jobs.get_run_output(run_id=task_run_id)
            print("\n=== Notebook Output / Logs ===")
            if hasattr(run_output, "notebook_output") and run_output.notebook_output:
                print(run_output.notebook_output.result)
            else:
                print(run_output.logs)
                if run_output.error:
                    print(f"Error: {run_output.error}")
    except Exception as e:
        print(f"Failed to submit or run notebook: {e}")

if __name__ == "__main__":
    main()
