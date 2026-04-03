import uuid
from enum import Enum
from typing import Any

from fastapi_users import schemas
from typing_extensions import override


class UserRole(str, Enum):
    """
    User roles
    - Basic can't perform any admin actions
    - Admin can perform all admin actions
    - Curator can perform admin actions for
        groups they are curators of
    - Global Curator can perform admin actions
        for all groups they are a member of
    - Limited can access a limited set of basic api endpoints
    - External permissioned users that have been picked up during the external permissions sync process but don't have a web login
    """

    LIMITED = "limited"
    BASIC = "basic"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"
    USER = "user"
    CURATOR = "curator"
    GLOBAL_CURATOR = "global_curator"
    EXT_PERM_USER = "ext_perm_user"

    def is_web_login(self) -> bool:
        return self not in [
            UserRole.EXT_PERM_USER,
        ]

class Department(str, Enum):
    QA = "QA"
    PRODUCTION = "Production"
    ACCOUNTS = "Accounts"
    SALES = "Sales"


class UserStatusEnum(str, Enum):
    ACTIVE = "active"
    HOLD = "hold"
    TERMINATED = "terminated"

class CompanyEnum(str, Enum):
    VIRCHOW = "Virchow"
    EMNAR = "Emnar"



class UserRead(schemas.BaseUser[uuid.UUID]):
    role: UserRole
    department: Department | None = None
    company: CompanyEnum | None = CompanyEnum.VIRCHOW
    personal_name: str | None = None
    status: UserStatusEnum | None = UserStatusEnum.ACTIVE


class UserCreate(schemas.BaseUserCreate):
    role: UserRole = UserRole.BASIC
    department: Department | None = None
    company: CompanyEnum | None = CompanyEnum.VIRCHOW
    personal_name: str | None = None
    status: UserStatusEnum | None = UserStatusEnum.ACTIVE
    tenant_id: str | None = None
    # Captcha token for cloud signup protection (optional, only used when captcha is enabled)
    # Excluded from create_update_dict so it never reaches the DB layer
    captcha_token: str | None = None

    @override
    def create_update_dict(self) -> dict[str, Any]:
        d = super().create_update_dict()
        d.pop("captcha_token", None)
        # Ensure our custom fields are always included
        if self.department is not None:
            d["department"] = self.department
        if self.company is not None:
            d["company"] = self.company
        if self.status is not None:
            d["status"] = self.status
        if self.personal_name is not None:
            d["personal_name"] = self.personal_name
        return d

    @override
    def create_update_dict_superuser(self) -> dict[str, Any]:
        d = super().create_update_dict_superuser()
        d.pop("captcha_token", None)
        return d


class UserUpdateWithRole(schemas.BaseUserUpdate):
    role: UserRole


class UserUpdate(schemas.BaseUserUpdate):
    """
    Role updates are not allowed through the user update endpoint for security reasons
    Role changes should be handled through a separate, admin-only process
    """


class AuthBackend(str, Enum):
    REDIS = "redis"
    POSTGRES = "postgres"
    JWT = "jwt"
