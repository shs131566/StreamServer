import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from stream_backend.config import settings
from stream_backend.transcribe import (
    overlap_transcribe,
    process_vad,
    speech_detect,
    transcribe,
)
from stream_backend.translate import translate
from stream_backend.triton_client import TritonClient
from stream_backend.voice_activity_detector import VoiceActivityDetect

router = APIRouter()


@router.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket, translate_flag: bool = False):
    await websocket.accept()
    logger.info(
        f"WebSocket accepted from {websocket.client.host}:{websocket.client.port}"
    )

    vad = VoiceActivityDetect()
    triton_client = TritonClient()
    audio_bytes_queue = asyncio.Queue()
    vad_queue = asyncio.Queue()
    speech_queue = asyncio.Queue()
    transcript_queue = asyncio.Queue()

    vad_task = asyncio.create_task(
        process_vad(audio_bytes_queue=audio_bytes_queue, vad_queue=vad_queue, vad=vad)
    )
    speech_detect_task = asyncio.create_task(
        speech_detect(vad_queue=vad_queue, speech_queue=speech_queue)
    )
    transcribe_task = asyncio.create_task(
        transcribe(
            speech_queue=speech_queue,
            triton_client=triton_client,
            transcript_queue=transcribe_task,
            websocket=websocket,
        )
    )
    if translate_flag:
        tranlate_task = asyncio.create_task(
            translate(
                transcript_queue=transcript_queue,
                triton_client=triton_client,
                websocket=websocket,
            )
        )
    accumulated_data = b""
    try:
        vad = VoiceActivityDetect()
        while True:
            data = await websocket.receive_bytes()
            accumulated_data += data

            while len(accumulated_data) >= settings.TRANSCRIBE_CHUNK_SIZE:
                chunk, accumulated_data = (
                    accumulated_data[: settings.TRANSCRIBE_CHUNK_SIZE],
                    accumulated_data[settings.TRANSCRIBE_CHUNK_SIZE :],
                )
                await audio_bytes_queue.put(chunk)
    except WebSocketDisconnect:
        logger.info(
            f"WebSocket disconnected from {websocket.client.host}:{websocket.client.port}"
        )
    except Exception as e:
        logger.error(f"Unexpected error occurred: {type(e).__name__}: {e}")
    finally:
        vad_task.cancel()
        speech_detect_task.cancel()
        transcribe_task.cancel()
        if translate_flag:
            transcribe_task.cancel()


@router.websocket("/overlap")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(
        f"WebSocket accepted from {websocket.client.host}:{websocket.client.port}"
    )

    vad = VoiceActivityDetect(min_silence_duration_ms=50)
    triton_client = TritonClient()
    audio_bytes_queue = asyncio.Queue()

    overlap_transcribe_task = asyncio.create_task(
        overlap_transcribe(audio_bytes_queue, vad, websocket, triton_client)
    )

    accumulated_data = b""
    try:
        vad = VoiceActivityDetect()
        while True:
            data = await websocket.receive_bytes()
            accumulated_data += data

            while len(accumulated_data) >= settings.OVERLAPPING_TRANSCRIBE_CHUNK_SIZE:
                chunk, accumulated_data = (
                    accumulated_data[: settings.OVERLAPPING_TRANSCRIBE_CHUNK_SIZE],
                    accumulated_data[settings.OVERLAPPING_TRANSCRIBE_CHUNK_SIZE :],
                )
                await audio_bytes_queue.put(chunk)

    except WebSocketDisconnect:
        logger.info(
            f"WebSocket disconnected from {websocket.client.host}:{websocket.client.port}"
        )
    except Exception as e:
        logger.error(f"Unexpected error occurred: {type(e).__name__}: {e}")
    finally:
        overlap_transcribe_task.cancel()
