from typing import Any
from onyx.db.models import User
from onyx.auth.schemas import UserRole
from sqlalchemy import or_
from onyx.db.models import DocumentChunk

def get_data_access_filter(current_user: User) -> Any:
    # superadmin -> full access
    if current_user.role == UserRole.SUPERADMIN:
        return True # Or empty filter depending on SQLAlchemy usage

    # admin -> only same department
    if current_user.role == UserRole.ADMIN:
        return DocumentChunk.department == current_user.department

    # user -> own uploads or department historical data
    # department historical data: data_type == 'historical' and department == current_user.department
    return or_(
        DocumentChunk.user_id == current_user.id,
        (DocumentChunk.department == current_user.department) & (DocumentChunk.data_type == 'historical')
    )
