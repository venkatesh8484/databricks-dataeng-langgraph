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
    
    # Query row counts of the CSV files
    csv_files = [
        "raw_accommodations.csv",
        "raw_availability.csv",
        "raw_booking_components.csv",
        "raw_bookings.csv",
        "raw_customers.csv",
        "raw_suppliers.csv"
    ]
    
    for f in csv_files:
        path = f"/Volumes/databricks_langgraph/raw/source_volume/{f}"
        sql = f"SELECT COUNT(*) as cnt FROM read_files('{path}', format => 'csv', header => true)"
        try:
            res = w.statement_execution.execute_statement(
                warehouse_id=wh_id,
                statement=sql
            )
            # Fetch results
            print(f"File: {f}")
            if res.status.state.value == "SUCCEEDED":
                # Get the count from data
                row = res.result.data_array[0]
                print(f"  Count: {row[0]}")
            else:
                print(f"  Failed: {res.status.error}")
        except Exception as e:
            print(f"Error querying {f}: {e}")

if __name__ == "__main__":
    main()
