

from fastapi import APIRouter

from fastapi_skeleton.api.routes import heartbeat, playlist

api_router = APIRouter()
api_router.include_router(heartbeat.router, tags=["health"], prefix="/health")
api_router.include_router(playlist.router, tags=[
                          "playlist"], prefix="/model")
