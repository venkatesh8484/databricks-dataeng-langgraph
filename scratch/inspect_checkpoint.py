import sqlite3
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

def main():
    conn = sqlite3.connect("checkpoint_inspect.db")
    cursor = conn.cursor()
    
    # List all checkpoints
    print("=== Checkpoints in DB ===")
    cursor.execute("SELECT thread_id, checkpoint_id, parent_checkpoint_id, metadata_bytes FROM checkpoints ORDER BY checkpoint_id DESC")
    rows = cursor.fetchall()
    
    serde = JsonPlusSerializer()
    
    for r in rows:
        thread_id, cp_id, parent_id, meta_bytes = r
        try:
            meta = serde.loads(meta_bytes)
        except Exception:
            meta = {}
        # Print checkpoint details
        step = meta.get("step", "unknown")
        source = meta.get("source", "unknown")
        print(f"Thread: {thread_id} | Checkpoint ID: {cp_id} | Parent: {parent_id} | Step: {step} | Source: {source}")

    # Inspect the latest checkpoint values
    if rows:
        latest_cp_id = rows[0][1]
        thread_id = rows[0][0]
        cursor.execute("SELECT checkpoint_format, checkpoint_bytes FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?", (thread_id, latest_cp_id))
        cp_row = cursor.fetchone()
        if cp_row:
            cp_fmt, cp_bytes = cp_row
            cp_data = serde.loads_typed((cp_fmt, cp_bytes))
            channel_values = cp_data.get("channel_values", {})
            print(f"\n=== Latest Checkpoint ({latest_cp_id}) Channels ===")
            
            # Print state keys
            keys = sorted(channel_values.keys())
            print(f"Keys in state: {keys}")
            
            # Print interesting keys
            for k in ["active_agent", "approved_steps", "review_comments", "execution_logs"]:
                if k in channel_values:
                    val = channel_values[k]
                    # Print values
                    print(f"  {k}: {val}")
                    
            if "final_report" in channel_values:
                print("\n  final_report (snippet):")
                print(str(channel_values["final_report"])[:1000])

if __name__ == "__main__":
    main()
