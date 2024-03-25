import asyncio
import json
from enum import IntEnum

import numpy as np
from fastapi import WebSocket
from loguru import logger

from stream_backend.config import settings
from stream_backend.triton_client import InferenceServerException, TritonClient
from stream_backend.voice_activity_detector import VoiceActivityDetect


class OverlapStatus(IntEnum):
    OVERLAP = 1
    END_OF_SPEECH = 2
    SPEAKING = 3


async def overlap_transcribe(
    overlap_speech_queue: asyncio.Queue,
    websocket: WebSocket,
    triton_client: TritonClient,
    transcript_queue: asyncio.Queue,
    language: str = "ko",
):
    previous_transcript = ""
    concat_flag = False
    while True:
        status, message_id, audio = await overlap_speech_queue.get()

        try:
            transcript, repetition = triton_client.transcribe(
                audio, language=language, client_timeout=10
            )
        except InferenceServerException as e:
            logger.info(e)
            continue

        if not repetition:
            if status == OverlapStatus.OVERLAP:
                transcript = (
                    previous_transcript + " " + transcript
                    if concat_flag
                    else transcript
                )
                message_dict = {
                    "language": "KO",
                    "message_id": f"{message_id:05}",
                    "transcript": transcript,
                    "translate": None,
                }
                await websocket.send_text(json.dumps(message_dict))

            elif status == OverlapStatus.END_OF_SPEECH:
                transcript = previous_transcript + " " + transcript
                message_dict = {
                    "language": "KO",
                    "message_id": f"{message_id:05}",
                    "transcript": transcript,
                    "translate": None,
                }
                await websocket.send_text(json.dumps(message_dict))
                await transcript_queue.put((message_id, transcript))
                concat_flag = False
                previous_transcript = ""

            elif status == OverlapStatus.SPEAKING:
                transcript = previous_transcript + " " + transcript
                previous_transcript = transcript
                concat_flag = True
                message_dict = {
                    "language": "KO",
                    "message_id": f"{message_id:05}",
                    "transcript": transcript,
                    "translate": None,
                }
                await websocket.send_text(json.dumps(message_dict))


async def overlap_speech_collect(
    vad_queue: asyncio.Queue,
    overlap_speech_queue: asyncio.Queue,
    speech_queue: asyncio.Queue,  # Not used
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
                        (OverlapStatus.END_OF_SPEECH, message_id, speech)
                    )
                    message_id += 1
                    accumulating = False
                    accumulated_audio = []

                else:
                    accumulating = False
                    accumulated_audio = []

            elif vad_result == "speak":
                accumulated_audio.append(audio_float32)
                accumulated_duration = (
                    len(accumulated_audio)
                    * settings.OVERLAPPING_TRANSCRIBE_CHUNK_SIZE
                    / settings.AUDIO_SAMPLING_RATE
                )
                if accumulated_duration % 2 == 0:
                    speech = np.concatenate(accumulated_audio, axis=0, dtype=np.float32)
                    await overlap_speech_queue.put(
                        (OverlapStatus.OVERLAP, message_id, speech)
                    )
                    await speech_queue.put((message_id, speech))
                elif accumulated_duration > 10:
                    speech = np.concatenate(accumulated_audio, axis=0, dtype=np.float32)
                    await overlap_speech_queue.put(
                        (OverlapStatus.SPEAKING, message_id, speech)
                    )
                    await speech_queue.put((message_id, speech))
                    accumulated_audio = []

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
