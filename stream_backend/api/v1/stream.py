import asyncio
import io

import numpy as np
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from scipy.io.wavfile import write

from stream_backend.triton_client import TritonClient
from stream_backend.voice_activity_detector import VoiceActivityDetect

router = APIRouter()


async def transcribe(speech_queue: asyncio.Queue, triton_client: TritonClient):
    while True:
        audio = await speech_queue.get()
        combined_audio = audio
        logger.critical(audio)
        transcript, repetition = triton_client.transcribe(combined_audio, language="ko")

        # while repetition:
        #     next_audio_data = await speech_queue.get()
        #     combined_audio.append(next_audio_data)
        #     result = await loop.run_in_executor(None, triton_client.transcribe, combined_audio, "ko")

        logger.critical(transcript)


async def speech_detect(vad_queue: asyncio.Queue, speech_queue: asyncio.Queue):
    audio_buffer = []
    start_time = None
    previous_frame = None

    while True:
        audio_float32, vad_result = await vad_queue.get()

        if vad_result is not None:
            if "start" in vad_result:
                start_time = vad_result["start"]
                if previous_frame is not None:
                    audio_buffer.append(previous_frame)
            elif "end" in vad_result and start_time is not None:
                audio_buffer.append(audio_float32)
                audio_segment = np.concatenate(audio_buffer, axis=0)
                # write(f"{start_time}.wav", 16000, audio_segment)
                await speech_queue.put(audio_segment)
                audio_buffer = []
                start_time = None
            previous_frame = None
        else:
            if start_time is not None:
                audio_buffer.append(audio_float32)
            previous_frame = audio_float32


async def process_vad(
    audio_bytes_queue: asyncio.Queue,
    vad_queue: asyncio.Queue,
    websocket: WebSocket,
    vad: VoiceActivityDetect,
):
    while True:
        try:
            audio_bytes = await audio_bytes_queue.get()
            audio_float32 = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                / np.iinfo(np.int16).max
            )
            vad_result = vad.iterator(audio_float32, return_seconds=True)
            await websocket.send_text(f"Processed audio data of length {vad_result}")
            await vad_queue.put((audio_float32, vad_result))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(e)


@router.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(
        f"WebSocket accepted from {websocket.client.host}:{websocket.client.port}"
    )

    vad = VoiceActivityDetect()
    triton_client = TritonClient(url="localhost:8001")
    audio_bytes_queue = asyncio.Queue()
    vad_queue = asyncio.Queue()
    speech_queue = asyncio.Queue()

    vad_task = asyncio.create_task(
        process_vad(audio_bytes_queue, vad_queue, websocket, vad)
    )
    speech_detect_task = asyncio.create_task(speech_detect(vad_queue, speech_queue))
    transcribe_task = asyncio.create_task(transcribe(speech_queue, triton_client))
    try:
        vad = VoiceActivityDetect()
        while True:
            audio_data = await websocket.receive_bytes()
            await audio_bytes_queue.put(audio_data)
    except WebSocketDisconnect:
        logger.info(
            f"WebSocket disconnected from {websocket.client.host}:{websocket.client.port}"
        )
    except Exception as e:
        logger.error(f"Unexpected error occurred: {type(e).__name__}: {e}")
    finally:
        vad_task.cancel()
        transcribe_task.cancel()