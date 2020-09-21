from pydantic import BaseModel


class baseclass(BaseModel):
    name: str
    danceability: float
    acousticness: float
    duration: int
    energy: float
    explicit: float
    instrumentalness: float
    key: int
    liveness: float
    loudness: float
    mode: float
    popularity: int
    releasedate: int
    speechiness: float
    tempo: float
    valence: float
    year: int