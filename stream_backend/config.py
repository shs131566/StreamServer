from typing import List

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "stream-server"
    API_V1_STR: str = "/api/v1"
    VAD_MODEL_PATH: str = "assets/silero_vad.jit"
    TRITON_SERVER_URL: str = "localhost"
    TRITON_SERVER_PORT: str = "8001"

    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"


settings = Settings()
