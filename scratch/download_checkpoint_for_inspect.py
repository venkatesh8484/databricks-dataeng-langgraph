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

if __name__ == "__main__":
    main()
