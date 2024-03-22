import asyncio

from fastapi import WebSocket

from stream_backend.triton_client import TritonClient


async def translate(
    transcript_queue: asyncio.Queue, triton_client: TritonClient, websocket: WebSocket
):
    NotImplemented
