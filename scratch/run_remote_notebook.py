import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

def main():
    print("Connecting to Databricks Workspace...")
    w = WorkspaceClient(profile="venkatesh8484")
    
    # List active clusters
    print("Finding active clusters...")
    cluster_id = None
    for cluster in w.clusters.list():
        # Prefer running clusters
        state_str = getattr(cluster.state, "value", str(cluster.state))
        print(f"  Cluster Name: {cluster.cluster_name} | ID: {cluster.cluster_id} | State: {state_str}")
        if "RUNNING" in state_str:
            cluster_id = cluster.cluster_id
            print(f"  Selected running cluster: '{cluster.cluster_name}' ({cluster_id})")
            break
            
    if not cluster_id:
        # Fallback to the first cluster in the list if none are running
        clusters = list(w.clusters.list())
        if clusters:
            cluster_id = clusters[0].cluster_id
            print(f"  No running clusters found. Selected first available cluster: '{clusters[0].cluster_name}' ({cluster_id})")
        else:
            print("  No clusters found in workspace! Cannot run notebook.")
            return

    notebook_path = "/Workspace/Users/venkatesh8484@gmail.com/.bundle/dbricks-lang-agent/dev/files/Medallion_Pipeline_Notebook"
    print(f"Submitting one-time notebook run for: {notebook_path} on cluster: {cluster_id}...")
    
    try:
        # Submit a one-time job run
        run_waiter = w.jobs.submit(
            run_name=f"agent-notebook-run-{int(time.time())}",
            tasks=[
                jobs.SubmitTask(
                    task_key="notebook_run",
                    notebook_task=jobs.NotebookTask(
                        notebook_path=notebook_path,
                        # Pass widget values to resume execution
                        base_parameters={
                            "hitl_action": "Approve",
                            "hitl_comment": "Resumed from agent script"
                        }
                    ),
                    existing_cluster_id=cluster_id
                )
            ]
        )
        
        print(f"Run submitted successfully! Run ID: {run_waiter.run_id}")
        print("Waiting for run to complete... (this may take a few minutes)")
        
        result = run_waiter.result()
        state = result.state
        life_cycle_state = getattr(state.life_cycle_state, "value", str(state.life_cycle_state))
        result_state = getattr(state.result_state, "value", str(state.result_state))
        
        print(f"Run completed with Life Cycle State: {life_cycle_state} | Result State: {result_state}")
        
        # Get run output / logs
        # First task run
        if result.tasks:
            task_run_id = result.tasks[0].run_id
            run_output = w.jobs.get_run_output(run_id=task_run_id)
            print("\n=== Notebook Output ===")
            if hasattr(run_output, "notebook_output") and run_output.notebook_output:
                print(run_output.notebook_output.result)
            else:
                print("No output returned or execution failed.")
                if hasattr(run_output, "error") and run_output.error:
                    print(f"Error: {run_output.error}")
    except Exception as e:
        print(f"Failed to submit or run notebook: {e}")

if __name__ == "__main__":
    main()
