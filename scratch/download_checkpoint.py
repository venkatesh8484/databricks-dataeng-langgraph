import os
import sqlite3
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
    
    path = "/Volumes/databricks_langgraph/raw/source_volume/checkpoint.db"
    local_path = "checkpoint_inspect_final.db"
    
    print(f"Downloading remote checkpoint {path} to local {local_path}...")
    try:
        res = w.files.download(path)
        with open(local_path, "wb") as lf:
            lf.write(res.contents.read())
        print("Download succeeded!")
    except Exception as e:
        print("Download failed:", e)
        return
        
    print("\nReading sqlite checkpoint data...")
    try:
        conn = sqlite3.connect(local_path)
        cursor = conn.cursor()
        cursor.execute("SELECT checkpoint_format, checkpoint_bytes FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
            serde = JsonPlusSerializer()
            checkpoint = serde.loads_typed((row[0], row[1]))
            print("\n=== Graph State Values ===")
            values = checkpoint.get("channel_values", {})
            
            # Print active agent and steps
            print(f"Active Agent: {values.get('active_agent')}")
            print(f"Approved Steps: {values.get('approved_steps')}")
            print("\n=== Silver Code ===")
            print(values.get("silver_code", ""))
            
            # Print execution logs
            print("\n=== Execution Logs ===")
            import json
            logs = values.get("execution_logs", {})
            if isinstance(logs, str):
                try:
                    logs = json.loads(logs)
                except Exception:
                    pass
            print(json.dumps(logs, indent=2))
            
            # Check if codebase was stored in database
            print("\nQuerying agent_codebase_memory in gold schema...")
            warehouses = list(w.warehouses.list())
            wh_id = warehouses[0].id
            
            sql_check = "SELECT dataset_fingerprint, timestamp FROM databricks_langgraph.gold.agent_codebase_memory"
            try:
                stmt_res = w.statement_execution.execute_statement(
                    warehouse_id=wh_id,
                    statement=sql_check
                )
                print(f"Query Status: {stmt_res.status.state}")
                if stmt_res.result and stmt_res.result.data_array:
                    print("Found codebase memory entries:")
                    for row in stmt_res.result.data_array:
                        print(f"  Fingerprint: {row[0]} | Saved At: {row[1]}")
                else:
                    print("No entries in codebase memory.")
            except Exception as e:
                print(f"Failed to query UC codebase memory: {e}")
        else:
            print("No checkpoints found in SQLite.")
    except Exception as e:
        print("Failed to inspect SQLite:", e)

if __name__ == "__main__":
    main()
