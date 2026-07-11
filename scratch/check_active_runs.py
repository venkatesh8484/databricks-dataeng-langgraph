import os
import configparser
from databricks.sdk import WorkspaceClient

def main():
    cfg_path = os.path.expanduser("~/.databrickscfg")
    config = configparser.ConfigParser()
    config.read(cfg_path)
    profile = "venkatesh8484"
    if profile not in config.sections():
        profile = "DEFAULT"
    host = config.get(profile, "host").strip('"').strip("'")
    token = config.get(profile, "token").strip('"').strip("'")
    w = WorkspaceClient(host=host, token=token)
    
    print("Listing recent job runs...")
    try:
        # List runs for the job or overall runs
        runs = list(w.jobs.list_runs(limit=10))
        for r in runs:
            state = r.state
            life_cycle = getattr(state.life_cycle_state, "value", str(state.life_cycle_state))
            result = getattr(state.result_state, "value", str(state.result_state)) if state.result_state else "NONE"
            print(f"Run ID: {r.run_id} | Name: {r.run_name} | State: {life_cycle} / {result} | Started: {r.start_time}")
    except Exception as e:
        print("Failed to list runs:", e)

if __name__ == "__main__":
    main()
