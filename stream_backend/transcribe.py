import asyncio
from uuid import uuid4

import numpy as np
from fastapi import WebSocket
from loguru import logger

from stream_backend.config import settings
from stream_backend.triton_client import TritonClient
from stream_backend.voice_activity_detector import VoiceActivityDetect


async def transcribe(
    speech_queue: asyncio.Queue,
    triton_client: TritonClient,
    transcript_queue: asyncio.Queue,
    websocket: WebSocket,
):
    message_id = 0
    while True:
        audio = await speech_queue.get()
        combined_audio = audio
        if len(combined_audio) / settings.AUDIO_SAMPLING_RATE < 1.0:
            logger.info("Short audio is ignored")
            continue
        logger.info(
            f"Send audio data {len(combined_audio)/settings.AUDIO_SAMPLING_RATE}s to Whisper"
        )
        transcript, repetition = triton_client.transcribe(combined_audio, language="ko")

        while repetition:
            audio = await speech_queue.get()
            combined_audio.extend(audio)
            logger.info(
                f"Send audio data {len(combined_audio)/settings.AUDIO_SAMPLING_RATE}s to Whisper"
            )
            transcript, repetition = triton_client.transcribe(
                combined_audio, language="ko"
            )

        await websocket.send_text(f"{message_id:05}: {transcript}")
        transcript_queue.put({"id": message_id, "transcript": transcript})
        message_id += 1


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
                audio_segment = np.concatenate(audio_buffer, axis=0, dtype=np.float32)
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
            await vad_queue.put((audio_float32, vad_result))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(
                f"Unexpected error occurred on transcribe: {type(e).__name__}: {e}"
            )


async def overlap_transcribe(
    audio_bytes_queue: asyncio.Queue,
    vad: VoiceActivityDetect,
    websocket: WebSocket,
    triton_client: TritonClient,
):
    is_speaking = False
    combined_audio = []
    message_id = 0

    while True:
        try:
            audio_bytes = await audio_bytes_queue.get()
            audio_float32 = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                / np.iinfo(np.int16).max
            )
            vad_result = vad.iterator(audio_float32, return_seconds=True)

            if vad_result is not None:
                if "start" in vad_result and not is_speaking:
                    is_speaking = True
                    combined_audio = []

                if is_speaking:
                    combined_audio.append(audio_float32)

                if "end" in vad_result and is_speaking:
                    if len(combined_audio) > 3:
                        is_speaking = False

            elif is_speaking:
                combined_audio.append(audio_float32)

            logger.info(f"{is_speaking}: {len(combined_audio)}, {vad_result}")
            if combined_audio:
                accumulated_length_seconds = (
                    len(combined_audio)
                    * settings.OVERLAPPING_TRANSCRIBE_CHUNK_SIZE
                    / settings.AUDIO_SAMPLING_RATE
                    / settings.AUDIO_CHANNELS
                    / settings.AUDIO_SAMPLE_WIDTH
                )

                combined_audio_float32 = np.concatenate(combined_audio)
                logger.info(f"Send audio data {accumulated_length_seconds}s to Whisper")
                transcript, repetition = triton_client.transcribe(
                    combined_audio_float32, language="ko"
                )

                if not repetition:
                    await websocket.send_text(f"{message_id:05}: {transcript}")

                    if is_speaking and accumulated_length_seconds > 10:
                        combined_audio = []
                        message_id += 1

        except Exception as e:
            logger.error(
                f"Unexpected error occurred on transcribe: {type(e).__name__}: {e}"
            )
