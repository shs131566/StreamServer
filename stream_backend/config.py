from typing import ClassVar, Dict, List

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "stream-server"
    API_V1_STR: str = "/api/v1"
    VAD_MODEL_PATH: str = "assets/silero_vad.jit"
    TRITON_SERVER_URL: str = "localhost"
    TRITON_SERVER_PORT: str = "8001"
    WHISPER_MODEL_NAME: str = "whisper"
    TRANSLATE_MODEL_NAME: str = "nmt"
    AUDIO_SAMPLING_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_SAMPLE_WIDTH: int = 2
    TRANSCRIBE_CHUNK_SIZE: int = (
        AUDIO_SAMPLING_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH * 0.3
    )
    OVERLAPPING_TRANSCRIBE_CHUNK_SIZE: int = (
        AUDIO_SAMPLING_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH * 0.1
    )
    LANGUAGE_DICT: ClassVar[Dict[str, str]] = {"ko": "ko_KR", "en": "en_XX"}
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    LOGGING_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"


settings = Settings()
