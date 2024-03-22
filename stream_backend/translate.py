import asyncio

from fastapi import WebSocket
from loguru import logger

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
        translate = triton_client.translate(
            transcript,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

        await websocket.send_text(f"EN:{message_id:05}: {translate}")
