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
        message_id, transcript, out_language = await transcript_queue.get()

        if src_lang == None and tgt_lang == None:
            src_lang = out_language
            tgt_lang = next(
                settings.LANGUAGE_DICT[key]
                for key in settings.LANGUAGE_DICT
                if key != out_language
            )

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
