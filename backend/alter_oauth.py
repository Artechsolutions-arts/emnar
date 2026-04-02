import urllib.parse
from sqlalchemy import create_engine, text

def alter_other_tables():
    password = urllib.parse.quote_plus("Artech@707")
    DATABASE_URL = f"postgresql://postgres:{password}@localhost:5432/onyx_db"
    engine = create_engine(DATABASE_URL)
    
    with engine.begin() as conn:
        try:
            conn.execute(text("ALTER TABLE oauth_account ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE"))
            print("Successfully added is_deleted to oauth_account table")
        except Exception as e:
            print(f"Error altering table: {e}")
            
if __name__ == "__main__":
    alter_other_tables()
