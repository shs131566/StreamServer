import argparse
import sys

import fastapi
import uvicorn
from loguru import logger
from starlette.middleware.cors import CORSMiddleware

from stream_backend.api.v1.api import api_router
from stream_backend.config import settings

app = fastapi.FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)
app.include_router(api_router, prefix=settings.API_V1_STR)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main():
    logger.remove(0)
    logger.add(sys.stderr, level=settings.LOGGING_LEVEL)

    parser = argparse.ArgumentParser()

    parser.add_argument("--host", help="The host to run the server", default="0.0.0.0")
    parser.add_argument(
        "--port", help="The port to run the server", type=int, default=8080
    )
    args = parser.parse_args()

    uvicorn.run(
        app, host=args.host, port=args.port, ws_ping_interval=25, ws_ping_timeout=100
    )


if __name__ == "__main__":
    main()
