import urllib.parse
from sqlalchemy import create_engine, text

def alter_user_table():
    password = urllib.parse.quote_plus("Artech@707")
    DATABASE_URL = f"postgresql://postgres:{password}@localhost:5432/onyx_db"
    engine = create_engine(DATABASE_URL)
    
    with engine.begin() as conn:
        try:
            conn.execute(text("ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE"))
            print("Successfully added is_deleted to user table")
        except Exception as e:
            print(f"Error altering table: {e}")
            
if __name__ == "__main__":
    alter_user_table()
