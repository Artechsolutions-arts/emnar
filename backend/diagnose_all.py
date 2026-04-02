from sqlalchemy import create_engine, text
import os

db_url = "postgresql://postgres:Artech%40707@localhost:5432/onyx_db"
engine = create_engine(db_url)

rag_tables = ["admin_uploads", "user_uploads", "documents", "chunks", "embeddings", "departments", "users"]

print("Table Counts:")
with engine.connect() as conn:
    for t in rag_tables:
        res = conn.execute(text(f"SELECT count(*) FROM {t}"))
        count = res.scalar()
        print(f"- {t}: {count}")

# Check for specific User and Department used in rag_upload.py
print("\nChecking Context:")
with engine.connect() as conn:
    res = conn.execute(text("SELECT name FROM departments WHERE name = 'Default'"))
    print(f"- Default Dept: {'Exists' if res.scalar() else 'Missing'}")
    res = conn.execute(text("SELECT email FROM users WHERE email = 'system@onyx.app'"))
    print(f"- System User: {'Exists' if res.scalar() else 'Missing'}")
