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
    
    catalog = "databricks_langgraph"
    schemas = ["silver", "quarantine", "gold"]
    tables = [
        "accommodations",
        "availability",
        "booking_components",
        "bookings",
        "customers",
        "suppliers",
        "dim_date",
        "dim_channel",
        "dim_customer",
        "dim_accommodation",
        "dim_supplier",
        "fact_bookings",
        "fact_booking_components",
        "fact_availability"
    ]
    
    statements = []
    for s in schemas:
        for t in tables:
            statements.append(f"DROP TABLE IF EXISTS {catalog}.{s}.{t}")
            
    print(f"Connecting to SQL Warehouse {wh_id}...")
    for sql in statements:
        print(f"Executing: {sql}")
        try:
            res = w.statement_execution.execute_statement(
                warehouse_id=wh_id,
                statement=sql
            )
            print(f"  Status: {res.status.state}")
        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    main()
