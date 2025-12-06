import sqlite3
import os

# Delete old database completely
if os.path.exists('media_library.db'):
    os.remove('media_library.db')
if os.path.exists('media_library.db-journal'):
    os.remove('media_library.db-journal')

# Create new database with correct schema
conn = sqlite3.connect('media_library.db')
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE media_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT NOT NULL,
        file_type TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        embedding_index INTEGER,
        compression_mode TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(file_hash, compression_mode)
    )
""")

conn.commit()
conn.close()
print("✓ Database recreated successfully!")