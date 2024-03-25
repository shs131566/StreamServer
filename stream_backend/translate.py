import asyncio
import json

from fastapi import WebSocket
from loguru import logger

from stream_backend.config import settings
from stream_backend.triton_client import TritonClient


async def translate(
    transcript_queue: asyncio.Queue,
    triton_client: TritonClient,
    websocket: WebSocket,
    src_lang: str,
    tgt_lang: str,
):
    while True:
        message_id, transcript = await transcript_queue.get()
        translation = triton_client.translate(
            transcript,
            src_lang=settings.LANGUAGE_DICT[src_lang],
            tgt_lang=settings.LANGUAGE_DICT[tgt_lang],
        )

        message_dict = {
            "language": tgt_lang,
            "message_id": f"{message_id:05}",
            "transcript": None,
            "translate": translation,
        }

        await websocket.send_text(json.dumps(message_dict))
