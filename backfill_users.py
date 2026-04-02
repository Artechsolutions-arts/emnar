from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from onyx.db.models import User
from onyx.auth.schemas import CompanyEnum, Department, UserStatusEnum
from onyx.configs.app_configs import POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

def backfill():
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        users = session.query(User).all()
        for u in users:
            # Update only if null
            if u.company is None:
                u.company = CompanyEnum.VIRCHOW
            if u.department is None:
                u.department = Department.QA
            if u.status is None:
                u.status = UserStatusEnum.ACTIVE
        session.commit()
    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    backfill()
