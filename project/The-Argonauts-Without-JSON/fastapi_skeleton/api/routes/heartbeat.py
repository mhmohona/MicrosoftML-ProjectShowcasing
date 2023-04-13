
from fastapi import APIRouter

from fastapi_skeleton.models.heartbeat import HearbeatResult

router = APIRouter()


@router.get("/heartbeat", response_model=HearbeatResult, name="heartbeat")
def get_hearbeat() -> HearbeatResult:
    return HearbeatResult(is_alive=True)
