import sqlite3
import pickle
import json

def main():
    conn = sqlite3.connect("./checkpoint.db")
    cursor = conn.cursor()
    
    # Query latest checkpoint_bytes
    cursor.execute("SELECT thread_id, checkpoint_id, checkpoint_bytes, metadata_bytes FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1")
    row = cursor.fetchone()
    if row:
        thread_id, cp_id, cp_bytes, meta_bytes = row
        print(f"Checkpoint ID: {cp_id} (Thread: {thread_id})")
        
        # In newer LangGraph, checkpoint_bytes could be JSON or pickle
        state_data = None
        # Try JSON
        try:
            state_data = json.loads(cp_bytes.decode('utf-8'))
            print("Decoded as JSON successfully!")
        except Exception:
            pass
            
        # Try pickle
        if state_data is None:
            try:
                state_data = pickle.loads(cp_bytes)
                print("Decoded as Pickle successfully!")
            except Exception as e:
                print("Pickle decode failed:", e)
                
        if state_data:
            # Look at channel values
            print("Checkpoint keys:", state_data.keys())
            
            # channel_values holds the actual variables of AgentState
            values = state_data.get("channel_values", {})
            print("Channel values keys:", values.keys())
            
            # Let's inspect the gold_ddl and data_dictionary
            if "gold_ddl" in values:
                # The channel values in LangGraph can be stored as a string, list of strings, or dictionary.
                # Let's see what type and content they have.
                val = values["gold_ddl"]
                print(f"gold_ddl type: {type(val)}")
                print("=== gold_ddl content ===")
                print(val)
                
            if "data_dictionary" in values:
                val = values["data_dictionary"]
                print(f"data_dictionary type: {type(val)}")
                print("=== data_dictionary content ===")
                print(val)
                
            # If they are empty, let's print all key-value contents in channel_values
            print("\nAll values in channels:")
            for k, v in values.items():
                print(f"  {k}: {repr(v)[:250]}")
                
            # Check what's in metadata
            try:
                meta = json.loads(meta_bytes.decode('utf-8'))
                print("Metadata:", meta)
            except Exception:
                try:
                    meta = pickle.loads(meta_bytes)
                    print("Metadata (Pickle):", meta)
                except Exception:
                    pass
        else:
            print("Could not deserialize checkpoint bytes.")
    else:
        print("No checkpoints found.")
        
    conn.close()

if __name__ == "__main__":
    main()
