

from pydantic import BaseModel
from fastapi_skeleton.models.payload import (song)

class MoodListResult(BaseModel):
    songs: str