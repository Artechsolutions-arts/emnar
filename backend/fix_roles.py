import urllib.parse
from sqlalchemy import create_engine, text

def fix_user_roles():
    password = urllib.parse.quote_plus("Artech@707")
    DATABASE_URL = f"postgresql://postgres:{password}@localhost:5432/onyx_db"
    engine = create_engine(DATABASE_URL)
    
    with engine.begin() as conn:
        # Check current roles
        result = conn.execute(text("SELECT email, role FROM \"user\""))
        rows = result.fetchall()
        print("Current users and roles:")
        for row in rows:
            print(f"  {row[0]}: '{row[1]}'")
        
        # Fix any lowercase 'admin' -> 'ADMIN'
        updated = conn.execute(
            text("UPDATE \"user\" SET role = 'ADMIN' WHERE role = 'admin'")
        )
        print(f"\nFixed {updated.rowcount} row(s) with lowercase 'admin' role.")
        
        # Verify
        result = conn.execute(text("SELECT email, role FROM \"user\""))
        rows = result.fetchall()
        print("\nUpdated users and roles:")
        for row in rows:
            print(f"  {row[0]}: '{row[1]}'")

if __name__ == "__main__":
    fix_user_roles()
