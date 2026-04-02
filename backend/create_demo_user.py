import sys
import uuid
import urllib.parse
from datetime import datetime, timezone

sys.path.append("d:/AI_ML/onyx/backend")

from fastapi_users.password import PasswordHelper
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def create_demo_user():
    # Setup standard engine directly
    password = urllib.parse.quote_plus("Artech@707")
    DATABASE_URL = f"postgresql://postgres:{password}@localhost:5432/onyx_db"
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    
    with Session() as session:
        email = "admin@example.com"
        password = "admin123"
        
        # Check if user already exists
        existing_user = session.execute(
            text("SELECT id FROM \"user\" WHERE email = :email"),
            {"email": email}
        ).fetchone()
        
        hashed_pw = PasswordHelper().hash(password)
        
        if existing_user:
            print(f"User {email} already exists.")
            session.execute(
                text("UPDATE \"user\" SET role = 'ADMIN', is_active = true, hashed_password = :hashed_pw WHERE email = :email"),
                {"email": email, "hashed_pw": hashed_pw}
            )
            session.commit()
            print("Reset existing user credentials.")
            return

        user_id = str(uuid.uuid4())
        
        session.execute(
            text("""
                INSERT INTO "user" (id, email, hashed_password, is_active, is_superuser, is_verified, role)
                VALUES (:id, :email, :hashed_password, :is_active, :is_superuser, :is_verified, :role)
            """),
            {
                "id": user_id,
                "email": email,
                "hashed_password": hashed_pw,
                "is_active": True,
                "is_superuser": True,
                "is_verified": True,
                "role": "ADMIN"
            }
        )
        try:
            session.commit()
            print(f"Demo user {email} created successfully.")
        except Exception as e:
            session.rollback()
            print(f"Failed to create user: {e}")

if __name__ == "__main__":
    create_demo_user()
