from fastapi import APIRouter
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError

admin_router = APIRouter(prefix="/admin/kg")

@admin_router.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def disabled_kg_router(path_name: str) -> None:
    raise OnyxError(OnyxErrorCode.FORBIDDEN, "Knowledge Graph features are disabled")
