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
    
    path = "/Volumes/databricks_langgraph/raw/source_volume/raw_accommodations.csv"
    print(f"Downloading first 1000 bytes of {path}...")
    try:
        res = w.files.download(path)
        content = res.contents.read(1000).decode("utf-8")
        print("=== Content ===")
        print(content)
    except Exception as e:
        print(f"Failed to read file: {e}")

if __name__ == "__main__":
    main()
