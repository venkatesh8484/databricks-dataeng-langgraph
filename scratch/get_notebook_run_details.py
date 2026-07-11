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
    
    run_id = 604117192163841
    print(f"Exporting run {run_id}...")
    try:
        run = w.jobs.get_run(run_id=run_id)
        print(f"Run state: {run.state.life_cycle_state} | {run.state.result_state}")
        
        if run.tasks:
            task_run_id = run.tasks[0].run_id
            print(f"Task run ID: {task_run_id}")
            export_resp = w.jobs.export_run(run_id=task_run_id)
            html_content = export_resp.views[0].content
            
            # Save to /tmp/run_output.html
            local_html = "./scratch/run_output.html"
            with open(local_html, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"Exported successfully to {local_html}!")
            
            # Parse HTML to find print outputs
            # Standard Databricks notebook prints are enclosed in specific divs or pre tags
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            print("\n=== Extracted Text/Output from Notebook View ===")
            text_blocks = soup.find_all("pre")
            for b in text_blocks:
                print(b.text[:2000])
                print("-" * 40)
    except Exception as e:
        print("Failed to get run details:", e)

if __name__ == "__main__":
    main()
