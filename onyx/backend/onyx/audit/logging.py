from onyx.db.models import AuditLog
from sqlalchemy.orm import Session
from uuid import UUID

def log_action(
    db_session: Session,
    user_id: UUID | None,
    action: str,
    resource_type: str,
    resource_id: str | None = None
) -> None:
    audit_log = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id
    )
    db_session.add(audit_log)
    db_session.commit()
