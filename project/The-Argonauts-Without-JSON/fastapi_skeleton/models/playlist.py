

from pydantic import BaseModel
from fastapi_skeleton.models.payload import (song)

class PlayListResult(BaseModel):
    #songs: list[song]
    song: str
