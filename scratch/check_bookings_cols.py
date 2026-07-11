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
    warehouses = list(w.warehouses.list())
    wh_id = warehouses[0].id
    
    # Let's query columns of bookings, booking_components, availability
    tables = ["bookings", "booking_components", "availability"]
    for t in tables:
        sql = f"SELECT * FROM databricks_langgraph.bronze.{t} LIMIT 1"
        try:
            res = w.statement_execution.execute_statement(
                warehouse_id=wh_id,
                statement=sql
            )
            print(f"=== Table: {t} ===")
            if res.status.state.value == "SUCCEEDED":
                columns = [col.name for col in res.result.schema.columns]
                print(f"Columns: {columns}")
            else:
                print(f"Failed: {res.status.error}")
        except Exception as e:
            print(f"Error querying {t}: {e}")

if __name__ == "__main__":
    main()
