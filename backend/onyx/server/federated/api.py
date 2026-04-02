from fastapi import APIRouter
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError

router = APIRouter(prefix="/federated")

@router.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def disabled_federated_router(path_name: str) -> None:
    raise OnyxError(OnyxErrorCode.FORBIDDEN, "Federated features are disabled")
