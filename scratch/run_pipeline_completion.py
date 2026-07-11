import os
import sys

# Ensure src is on Python path
sys.path.append(os.path.abspath("src"))

from dbricks_lang_agent.orchestrator.graph import create_pipeline_graph
from dbricks_lang_agent.ui.dashboard import sync_db_from_volume, sync_db_to_volume

def main():
    print("Syncing checkpoint database from Volume...")
    sync_db_from_volume()
    
    print("Initializing LangGraph state machine...")
    app = create_pipeline_graph()
    thread_id = "medallion_pipeline_run"
    config = {"configurable": {"thread_id": thread_id}}
    
    state = app.get_state(config)
    print(f"Current State Pointer: {state.next}")
    if not state.next:
        print("Pipeline is already completed!")
        return
        
    current_node = state.next[0]
    print(f"Current Node: {current_node}")
    
    # Get current approvals
    approvals = dict(state.values.get("approved_steps", {}))
    
    # We are at execution_review_gate, which matches 'report' approval
    step_key = "report"
    print(f"Approving step '{step_key}'...")
    approvals[step_key] = True
    
    # Update approvals in graph
    app.update_state(
        config,
        {
            "approved_steps": approvals,
            "review_comments": ""
        }
    )
    
    print("Resuming execution...")
    try:
        events = app.stream(None, config, stream_mode="values")
        for event in events:
            # Print state updates during execution
            active_agent = event.get("active_agent", "None")
            print(f"  [Active Agent]: {active_agent}")
    except Exception as e:
        print(f"Execution finished or interrupted: {e}")
        
    print("Syncing updated checkpoint back to Volume...")
    sync_db_to_volume()
    
    # Verification check
    final_state = app.get_state(config)
    print("=== FINAL STATUS ===")
    print(f"State Pointer: {final_state.next}")
    print(f"Pipeline finished: {not final_state.next}")

if __name__ == "__main__":
    main()
