from fastapi import APIRouter

from stream_backend.api.v1 import stream

api_router = APIRouter()
api_router.include_router(stream.router, prefix="/stream", tags=["stream"])
