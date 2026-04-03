from fastapi import APIRouter
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError

admin_router = APIRouter(prefix="/admin/opensearch-migration")

@admin_router.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def disabled_opensearch_migration_router(path_name: str) -> None:
    raise OnyxError(OnyxErrorCode.FORBIDDEN, "OpenSearch migration features are disabled")
