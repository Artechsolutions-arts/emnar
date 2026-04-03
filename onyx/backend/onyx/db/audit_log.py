from sqlalchemy.orm import Session
from onyx.db.models import AuditLog
from uuid import UUID
import json

def log_action(db_session: Session, action: str, details: dict | str | None = None, user_id: UUID | None = None) -> None:
    if isinstance(details, dict):
        details_str = json.dumps(details)
    else:
        details_str = details
        
    audit_log = AuditLog(
        user_id=user_id,
        action=action,
        details=details_str
    )
    db_session.add(audit_log)
    db_session.commit()
