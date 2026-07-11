import os
import configparser
from databricks.sdk import WorkspaceClient

def main():
    print("Reading databrickscfg...")
    cfg_path = os.path.expanduser("~/.databrickscfg")
    
    host = None
    token = None
    
    if os.path.exists(cfg_path):
        config = configparser.ConfigParser()
        config.read(cfg_path)
        
        # Try finding the venkatesh8484 profile, otherwise default
        profile = "venkatesh8484"
        if profile not in config.sections():
            profile = "DEFAULT"
            
        if profile in config.sections() or profile == "DEFAULT":
            host = config.get(profile, "host", fallback=None)
            token = config.get(profile, "token", fallback=None)
            print(f"Loaded credentials from profile: {profile}")
            # Clean trailing/leading spaces or quotes
            if host: host = host.strip('"').strip("'")
            if token: token = token.strip('"').strip("'")
    
    if not host or not token:
        # Fallback to config.yaml if databrickscfg doesn't have it
        print("Could not load from databrickscfg. Please supply token/host.")
        return
        
    print(f"Connecting to host: {host}")
    w = WorkspaceClient(host=host, token=token)
    
    app_client_id = "c02759ea-57f1-44e2-adb0-10dd9eb7913f"
    catalog = "databricks_langgraph"
    
    statements = [
        f"GRANT USE CATALOG ON CATALOG {catalog} TO `{app_client_id}`",
        f"GRANT USE SCHEMA, ALL PRIVILEGES ON SCHEMA {catalog}.raw TO `{app_client_id}`",
        f"GRANT USE SCHEMA, ALL PRIVILEGES ON SCHEMA {catalog}.bronze TO `{app_client_id}`",
        f"GRANT USE SCHEMA, ALL PRIVILEGES ON SCHEMA {catalog}.silver TO `{app_client_id}`",
        f"GRANT USE SCHEMA, ALL PRIVILEGES ON SCHEMA {catalog}.gold TO `{app_client_id}`",
        f"GRANT ALL PRIVILEGES ON VOLUME {catalog}.raw.source_volume TO `{app_client_id}`"
    ]
    
    # Let's execute using the Workspace Statement Execution API
    # Find a SQL warehouse first
    print("Listing warehouses...")
    warehouses = list(w.warehouses.list())
    if not warehouses:
        print("No SQL warehouses found in the workspace!")
        return
        
    wh_id = warehouses[0].id
    print(f"Using SQL Warehouse: {wh_id} ({warehouses[0].name})")
    
    for sql in statements:
        print(f"Executing: {sql}")
        try:
            res = w.statement_execution.execute_statement(
                warehouse_id=wh_id,
                statement=sql
            )
            print(f"Success! Status: {res.status.state}")
        except Exception as e:
            print(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
