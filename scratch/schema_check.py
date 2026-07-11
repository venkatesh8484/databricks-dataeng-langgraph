import sqlite3

def main():
    conn = sqlite3.connect("./checkpoint.db")
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id, checkpoint_id, checkpoint_format, length(checkpoint_bytes), metadata_format FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1")
    row = cursor.fetchone()
    print("Checkpoint row:", row)
    
    # Let's list some rows in writes table
    cursor.execute("SELECT thread_id, checkpoint_id, channel, value_format, length(value_bytes) FROM writes ORDER BY checkpoint_id DESC LIMIT 10")
    print("Writes rows:")
    for w_row in cursor.fetchall():
        print(f"  {w_row}")
        
    conn.close()

if __name__ == "__main__":
    main()
