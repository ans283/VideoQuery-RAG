import sqlite3
# 1. Connect to the database file
conn = sqlite3.connect("chat_memory.db")
cursor = conn.cursor()

# 2. Run the SQL command
cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    session_id INTEGER,
    sender TEXT,
    text TEXT
)
""")

# 3. Insert a simulated conversation (Session 1)
cursor.execute("INSERT INTO messages (session_id, sender, text) VALUES (1, 'user', 'Hello')")
cursor.execute("INSERT INTO messages (session_id, sender, text) VALUES (1, 'ai', 'Hi there!')")

# 4. Save the changes to the database
conn.commit()

# 5. Retrieve the messages
cursor.execute("SELECT * FROM messages")
rows = cursor.fetchall()

# 6. Loop through the results and print them cleanly
print("--- Raw Data from SQLite ---")
print(rows)

print("\n--- Clean Chat Log ---")
for row in rows:
    sender = row[1]
    text = row[2]
    print(f"{sender.upper()}: {text}")

# 7. Close the connection when done
conn.close()