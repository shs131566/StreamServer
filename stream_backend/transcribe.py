import asyncio
import json
from uuid import uuid4

import numpy as np
from fastapi import WebSocket
from loguru import logger

from stream_backend.config import settings
from stream_backend.triton_client import InferenceServerException, TritonClient
from stream_backend.voice_activity_detector import VoiceActivityDetect


async def transcribe(
    speech_queue: asyncio.Queue,
    triton_client: TritonClient,
    transcript_queue: asyncio.Queue,
    websocket: WebSocket,
    language: str = None,
):
    message_id = 0

    while True:
        try:
            audio = await speech_queue.get()
            combined_audio = audio
            if len(combined_audio) / settings.AUDIO_SAMPLING_RATE < 1.0:
                logger.info("transcribe_queue: short audio is ignored")
                continue
            logger.info(
                f"transcribe_queue: send audio data {len(combined_audio)/settings.AUDIO_SAMPLING_RATE}s to Whisper"
            )

            transcript, repetition, out_language = triton_client.transcribe(
                combined_audio, language=language, inference_type="transcribe"
            )

            while repetition:
                audio = await speech_queue.get()
                combined_audio.extend(audio)
                logger.info(
                    f"transcribe_queue: repetition occured resend audio data {len(combined_audio)/settings.AUDIO_SAMPLING_RATE}s to Whisper"
                )
                transcript, repetition, out_language = triton_client.transcribe(
                    combined_audio, language=language, inference_type="transcribe"
                )

            logger.success(f"transcribe_queue: whisper inference success")
            logger.debug(
                f"transcript: {transcript['text']}, repetiton: {repetition} spoken_language: {out_language}"
            )

            if transcript["text"].strip() == "":
                logger.info(f"transcribe_queue: empty transcript, skipping")
                continue

            message_dict = {
                "language": out_language,
                "message_id": f"{message_id:05}",
                "transcript": transcript["text"],
                "translate": None,
            }

            await websocket.send_text(json.dumps(message_dict))

            await transcript_queue.put((message_id, out_language, transcript["text"]))

            message_id += 1
        except InferenceServerException as e:
            logger.error(f"speech_detect_queue: InferenceServerException occurred: {e}")
            if triton_client.check_model_status(model_name=settings.WHISPER_MODEL_NAME):
                logger.error(f"speech_detect_queue: Whisper model is not ready")
            continue
        except asyncio.CancelledError:
            logger.warning("speech_detect_queue: speech detect task cancelled.")
            break
        except Exception as e:
            logger.error(
                f"An error occurred in transcribe_queue: {type(e).__name__}: {e}"
            )
            continue


async def speech_detect(vad_queue: asyncio.Queue, speech_queue: asyncio.Queue):
    audio_buffer = []
    start_time = None
    previous_frame = None

    while True:
        try:
            audio_float32, vad_result = await vad_queue.get()
            logger.debug(
                f"speech_detect_queue: received audio_float32 of length {len(audio_float32)}, vad_result {vad_result}"
            )

            if vad_result is not None:
                if "start" in vad_result:
                    start_time = vad_result["start"]
                    if previous_frame is not None:
                        audio_buffer.append(previous_frame)
                    logger.info(f"speech_detect_queue: {vad_result}")
                    logger.debug(
                        "speech_detect_queue: detected start of speech, appending previous frame to audio buffer."
                    )
                elif "end" in vad_result and start_time is not None:
                    audio_buffer.append(audio_float32)
                    audio_segment = np.concatenate(
                        audio_buffer, axis=0, dtype=np.float32
                    )
                    logger.info(f"speech_detect_queue: {vad_result}")
                    logger.info(
                        f"speech_detect_queue: pushing audio segment of length {len(audio_segment)/settings.AUDIO_SAMPLING_RATE} to speech queue."
                    )
                    await speech_queue.put(audio_segment)
                    audio_buffer = []
                    start_time = None
                    logger.debug(
                        "speech_detect_queue: detected end of speech, appended audio segment to speech queue and reset variables."
                    )
                previous_frame = None
            else:
                if start_time is not None:
                    audio_buffer.append(audio_float32)
                    logger.debug(
                        "speech_detect_queue: appended audio frame to buffer if speech has started."
                    )
                previous_frame = audio_float32
        except asyncio.CancelledError:
            logger.warning("speech_detect_queue: speech detect task cancelled.")
            break
        except Exception as e:
            logger.error(
                f"Unexpected error occurred on speech_detect_queue: {type(e).__name__}: {e}"
            )


async def process_vad(
    audio_bytes_queue: asyncio.Queue,
    vad_queue: asyncio.Queue,
    vad: VoiceActivityDetect,
):
    while True:
        try:
            audio_bytes = await audio_bytes_queue.get()
            logger.debug(f"vad_queue: received audio data: {len(audio_bytes)} bytes")
            audio_float32 = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                / np.iinfo(np.int16).max
            )
            vad_result = vad.iterator(audio_float32, return_seconds=True)
            logger.debug(f"vad_queue: result {vad_result}")
            await vad_queue.put((audio_float32, vad_result))
        except asyncio.CancelledError:
            logger.warning("vad_queue: process vad task cancelled.")
            break
        except Exception as e:
            logger.error(
                f"Unexpected error occurred on vad_queue: {type(e).__name__}: {e}"
            )
