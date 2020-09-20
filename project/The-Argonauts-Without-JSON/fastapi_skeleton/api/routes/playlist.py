from fastapi import APIRouter, Depends
from starlette.requests import Request

from fastapi_skeleton.core import security
#todo create payload
from fastapi_skeleton.models.prediction import PlayListResult
# add model for request


router = APIRouter()

# we need to finish hookup the model request and respone
@router.post("/playlist", response_model=PlayListResult, name="playlist")
def post_playlist(
    
)
