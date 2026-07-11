import sqlite3

def decode_msgpack(data):
    # Try importing msgpack
    try:
        import msgpack
        return msgpack.unpackb(data)
    except Exception as e:
        return f"Msgpack decode failed: {e}"

def main():
    conn = sqlite3.connect("./checkpoint.db")
    cursor = conn.cursor()
    
    # Query latest values for gold_ddl and data_dictionary in writes table
    channels = ["gold_ddl", "data_dictionary", "review_comments", "approved_steps"]
    
    for channel in channels:
        cursor.execute(
            "SELECT checkpoint_id, value_bytes FROM writes WHERE channel=? ORDER BY checkpoint_id DESC LIMIT 1",
            (channel,)
        )
        row = cursor.fetchone()
        print(f"\n=== Channel: {channel} ===")
        if row:
            cp_id, val_bytes = row
            print(f"Checkpoint ID: {cp_id}")
            decoded = decode_msgpack(val_bytes)
            print(f"Decoded type: {type(decoded)}")
            print("Decoded content:")
            print(decoded)
        else:
            print("No writes found for this channel.")
            
    conn.close()

if __name__ == "__main__":
    main()
