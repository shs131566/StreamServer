import asyncio

import numpy as np
from fastapi import WebSocket
from loguru import logger

from stream_backend.config import settings
from stream_backend.triton_client import TritonClient
from stream_backend.voice_activity_detector import VoiceActivityDetect


async def overlap_transcribe(
    overlap_speech_queue: asyncio, websocket: WebSocket, triton_client: TritonClient
):
    while True:
        duration, message_id, audio = await overlap_speech_queue.get()
        transcript, repetition = triton_client.transcribe(audio, language="ko")
        if not repetition:
            await websocket.send_text(f"{message_id:05}: {transcript}")


async def overlap_speech_collect(
    vad_queue: asyncio.Queue,
    overlap_speech_queue: asyncio.Queue,
    speech_queue: asyncio.Queue,
):
    accumulating = False
    accumulated_audio = []

    message_id = 0
    while True:
        audio_float32, vad_result = await vad_queue.get()

        if vad_result is not None:
            if "start" in vad_result:
                accumulating = True
                accumulated_audio = [audio_float32]

            elif "end" in vad_result and accumulating:
                accumulated_audio.append(audio_float32)
                accumulated_duration = (
                    len(accumulated_audio)
                    * settings.OVERLAPPING_TRANSCRIBE_CHUNK_SIZE
                    / settings.AUDIO_SAMPLING_RATE
                )

                if accumulated_duration > 2.0:
                    speech = np.concatenate(accumulated_audio, axis=0, dtype=np.float32)
                    await overlap_speech_queue.put(
                        (accumulated_duration, message_id, speech)
                    )
                    accumulating = False
                    accumulated_audio = []
                    message_id += 1

            elif vad_result is "speak":
                accumulated_audio.append(audio_float32)
                accumulated_duration = (
                    len(accumulated_audio)
                    * settings.OVERLAPPING_TRANSCRIBE_CHUNK_SIZE
                    / settings.AUDIO_SAMPLING_RATE
                )
                if accumulated_duration % 2 == 0:
                    speech = np.concatenate(accumulated_audio, axis=0, dtype=np.float32)
                    await overlap_speech_queue.put(
                        (accumulated_duration, message_id, speech)
                    )
                    await speech_queue.put((message_id, speech))
                elif accumulated_duration > 15:
                    speech = np.concatenate(accumulated_audio, axis=0, dtype=np.float32)
                    await overlap_speech_queue.put(
                        (accumulated_duration, message_id, speech)
                    )
                    await speech_queue.put((message_id, speech))
                    accumulated_audio = []
                    message_id += 1

        else:
            if accumulating:
                accumulated_audio.append(audio_float32)

            accumulated_duration = (
                len(accumulated_audio)
                * settings.OVERLAPPING_TRANSCRIBE_CHUNK_SIZE
                / settings.AUDIO_SAMPLING_RATE
            )

            if accumulated_duration > 2.0:
                speech = np.concatenate(accumulated_audio, axis=0, dtype=np.float32)
                await overlap_speech_queue.put(
                    (accumulated_duration, message_id, speech)
                )


async def overlap_vad(
    audio_bytes_queue: asyncio.Queue, vad_queue: asyncio.Queue, vad: VoiceActivityDetect
):
    is_speaking = False
    while True:
        try:
            audio_bytes = await audio_bytes_queue.get()
            audio_float32 = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                / np.iinfo(np.int16).max
            )
            vad_result = vad.iterator(audio_float32, return_seconds=True)

            if vad_result is None:
                if is_speaking:
                    vad_result = "speak"
            elif "start" in vad_result:
                is_speaking = True
            elif "end" in vad_result:
                is_speaking = False

            await vad_queue.put((audio_float32, vad_result))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(
                f"Unexpected error occurred on transcribe: {type(e).__name__}: {e}"
            )
