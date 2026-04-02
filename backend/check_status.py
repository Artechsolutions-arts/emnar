from sqlalchemy import select, create_engine
from sqlalchemy.orm import Session
from onyx.db.custom_rag_models import AdminUpload
import urllib.parse
import os
from dotenv import load_dotenv

load_dotenv()

PG_PASS = os.getenv("POSTGRES_PASSWORD", "Artech@707")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_DB = os.getenv("POSTGRES_DB", "onyx_db")

encoded_pass = urllib.parse.quote_plus(PG_PASS)
url = f"postgresql://{PG_USER}:{encoded_pass}@{PG_HOST}:{PG_PORT}/{PG_DB}"
engine = create_engine(url)

with Session(engine) as session:
    uploads = session.execute(select(AdminUpload).order_by(AdminUpload.created_at.desc())).scalars().all()
    print(f"\n{'FILE NAME':<30} | {'STATUS':<15} | {'ERROR MESSAGE'}")
    print("-" * 140)
    for u in uploads:
        err = u.error_message or "None"
        print(f"{u.file_name[:30]:<30} | {u.processing_status:<15} | {err}")
