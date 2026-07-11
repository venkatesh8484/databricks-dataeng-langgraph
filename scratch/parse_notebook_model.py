import re
import urllib.parse
import json
import base64
import os

def main():
    html_path = "./scratch/run_output.html"
    if not os.path.exists(html_path):
        print(f"File {html_path} does not exist!")
        return
        
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    pattern = r"var __DATABRICKS_NOTEBOOK_MODEL = '([^']*)';"
    match = re.search(pattern, content)
    if not match:
        print("Could not find __DATABRICKS_NOTEBOOK_MODEL in the HTML.")
        return
        
    encoded_str = match.group(1)
    
    # 1. Base64 decode
    try:
        decoded_bytes = base64.b64decode(encoded_str)
        decoded_base64 = decoded_bytes.decode('utf-8')
    except Exception as e:
        print("Failed base64 decode:", e)
        return
        
    # 2. URL decode
    decoded_str = urllib.parse.unquote(decoded_base64)
    
    try:
        model = json.loads(decoded_str)
        print("Successfully loaded notebook model!")
        commands = model.get("commands", [])
        print(f"Number of commands/cells: {len(commands)}")
        
        for idx, cmd in enumerate(commands):
            # Print cell output/results
            results = cmd.get("results")
            if results:
                data = results.get("data", [])
                ans = []
                if isinstance(data, list):
                    for d in data:
                        if isinstance(d, dict) and d.get("type") == "ansi":
                            ans.append(d.get("data", ""))
                
                output_str = "".join(ans).strip()
                if output_str:
                    print(f"\n--- Cell {idx + 1} Output ---")
                    print(output_str)
                    print("-" * 60)
            
            error = cmd.get("error")
            if error:
                print(f"\nError in Cell {idx + 1}: {error}")
                print(f"Error Details: {cmd.get('errorDetails')}")
                
    except Exception as e:
        print("Failed to parse decoded JSON:", e)

if __name__ == "__main__":
    main()
