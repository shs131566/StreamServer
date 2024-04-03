import asyncio
import json
from datetime import datetime
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
    timeout: float = 2.0,
):
    previous_transcript = ""
    concat_flag = False
    while True:
        try:
            status, message_id, audio = await overlap_speech_queue.get()
            logger.info(
                f"overlap_transcribe_queue: received status {status}, message_id {message_id}, and {len(audio)/settings.AUDIO_SAMPLING_RATE}s audio."
            )

            if status == OverlapStatus.OVERLAP:
                start_time = datetime.now()
                transcript, repetition, out_language = triton_client.transcribe(
                    audio, language=language, client_timeout=timeout
                )

                logger.success(
                    f"overlap_transcribe_queue: whisper inference success {len(audio)/settings.AUDIO_SAMPLING_RATE}s audio completed in {(datetime.now()-start_time).total_seconds()}s"
                )
                logger.debug(
                    f"overlap_transcribe_queue: repetition: {repetition}, spoken_language: {out_language}"
                )

                if not repetition:
                    transcript = (
                        previous_transcript + " " + transcript
                        if concat_flag
                        else transcript
                    )

                    message_dict = {
                        "language": out_language,
                        "message_id": f"{message_id:05}",
                        "transcript": transcript,
                        "translate": None,
                    }
                    logger.info(f"overlapping")
                    await websocket.send_text(json.dumps(message_dict))

            elif status == OverlapStatus.END_OF_SPEECH:
                start_time = datetime.now()
                transcript, repetition, out_language = triton_client.transcribe(
                    audio, language=language, client_timeout=timeout
                )

                logger.success(
                    f"overlap_transcribe_queue: whisper inference success {len(audio)/settings.AUDIO_SAMPLING_RATE}s audio completed in {(datetime.now()-start_time).total_seconds()}s"
                )
                logger.debug(
                    f"overlap_transcribe_queue: repetition: {repetition}, spoken_language: {out_language}"
                )

                if not repetition:
                    transcript = previous_transcript + " " + transcript

                    message_dict = {
                        "language": out_language,
                        "message_id": f"{message_id:05}",
                        "transcript": transcript,
                        "translate": None,
                    }
                    logger.info(f"end of speech")
                    await websocket.send_text(json.dumps(message_dict))
                    logger.debug(
                        f"message_id {message_id}, spoken language {out_language}, transcript {transcript} push to transcript_queue."
                    )
                    await transcript_queue.put((message_id, out_language, transcript))
                    concat_flag = False
                    previous_transcript = ""

            elif status == OverlapStatus.SPEAKING:
                start_time = datetime.now()
                transcript, repetition, out_language = triton_client.transcribe(
                    audio, language=language, client_timeout=timeout
                )

                logger.success(
                    f"overlap_transcribe_queue: whisper inference success {len(audio)/settings.AUDIO_SAMPLING_RATE}s audio completed in {(datetime.now()-start_time).total_seconds()}s"
                )
                logger.debug(
                    f"overlap_transcribe_queue: repetition: {repetition}, spoken_language: {out_language}"
                )
                if not repetition:
                    transcript = previous_transcript + " " + transcript
                    previous_transcript = transcript
                    concat_flag = True
                    message_dict = {
                        "language": out_language,
                        "message_id": f"{message_id:05}",
                        "transcript": transcript,
                        "translate": None,
                    }
                    logger.info(f"speaking")
                    await websocket.send_text(json.dumps(message_dict))
        except InferenceServerException as e:
            logger.error(
                f"overlap_transcribe_queue: InferenceServerException occurred: {e}"
            )
            logger.warning(
                f"previous transcript: {previous_transcript}, concat_flag: {concat_flag}, message_id: {message_id}, status: {status}"
            )
            if status == OverlapStatus.END_OF_SPEECH:
                await transcript_queue.put((message_id, out_language, transcript))
                concat_flag = False
                previous_transcript = ""
            elif status == OverlapStatus.SPEAKING:
                previous_transcript = transcript + " "
                concat_flag = True
            continue
        except asyncio.CancelledError:
            logger.warning("overlap_transcribe_queue: Task was cancelled.")
            break
        except Exception as e:
            logger.error(
                f"Unexpected error occurred in overlap_transcribe_queue: {type(e).__name__}: {e}"
            )


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
        logger.debug(f"overlap_speech_collect_queue: received vad_result {vad_result}")

        if vad_result is not None:
            if "start" in vad_result:
                accumulating = True
                accumulated_audio = [audio_float32]
                logger.debug("overlap_speech_collect_queue: Start of speech detected.")
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
                    logger.debug(
                        "overlap_speech_collect_queue: End of speech detected."
                    )
                    message_id += 1
                    accumulating = False
                    accumulated_audio = []

                else:
                    accumulating = False
                    accumulated_audio = [audio_float32]

            elif vad_result == "speak":
                accumulated_audio.append(audio_float32)
                accumulated_duration = (
                    len(accumulated_audio)
                    * settings.OVERLAPPING_TRANSCRIBE_CHUNK_SIZE
                    / settings.AUDIO_SAMPLING_RATE
                    / settings.AUDIO_SAMPLE_WIDTH
                )
                if accumulated_duration % 4 == 0:
                    speech = np.concatenate(accumulated_audio, axis=0, dtype=np.float32)
                    logger.debug(
                        f"overlap_speech_collect_queue: accumulate speech {accumulated_duration}s push to speech_queue."
                    )
                    await overlap_speech_queue.put(
                        (OverlapStatus.OVERLAP, message_id, speech)
                    )
                    logger.info(
                        f"overlap_speech_collect_queue: {accumulated_duration}s of speech {message_id} of status {OverlapStatus.OVERLAP} push to overlap_speech_queue."
                    )
                    # await speech_queue.put((message_id, speech))
                elif accumulated_duration > 10:
                    speech = np.concatenate(accumulated_audio, axis=0, dtype=np.float32)
                    logger.debug(
                        f"overlap_speech_collect_queue: accumulated speech {accumulated_duration}s push to speech_queue."
                    )
                    await overlap_speech_queue.put(
                        (OverlapStatus.SPEAKING, message_id, speech)
                    )
                    logger.info(
                        f"overlap_speech_collect_queue: {accumulated_duration}s of speech {message_id} of status {OverlapStatus.SPEAKING} push to overlap_speech_queue."
                    )
                    # await speech_queue.put((message_id, speech))
                    accumulated_audio = [audio_float32]
                    logger.debug(
                        f"overlap_speech_collect_queue: accumulated speech is too long, reset."
                    )

        else:
            if accumulating:
                accumulated_audio.append(audio_float32)

            accumulated_duration = (
                len(accumulated_audio)
                * settings.OVERLAPPING_TRANSCRIBE_CHUNK_SIZE
                / settings.AUDIO_SAMPLING_RATE
                / settings.AUDIO_SAMPLE_WIDTH
            )

            if accumulated_duration > 2.0:
                speech = np.concatenate(accumulated_audio, axis=0, dtype=np.float32)
                logger.debug(
                    f"overlap_speech_collect_queue: accumulate silence {accumulated_duration}s push to speech_queue."
                )
                await overlap_speech_queue.put(
                    (accumulated_duration, message_id, speech)
                )
                logger.info(
                    f"overlap_speech_collect_queue: {accumulated_duration}s of silence {message_id} push to overlap_speech_queue."
                )


async def overlap_vad(
    audio_bytes_queue: asyncio.Queue, vad_queue: asyncio.Queue, vad: VoiceActivityDetect
):
    is_speaking = False
    while True:
        try:
            audio_bytes = await audio_bytes_queue.get()
            logger.debug(f"overlap_vad_queue: Received {len(audio_bytes)} audio bytes.")
            audio_float32 = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                / np.iinfo(np.int16).max
            )
            vad_result = vad.iterator(audio_float32, return_seconds=True)

            if vad_result is None:
                if is_speaking:
                    vad_result = "speak"
                    logger.debug("overlap_vad_queue: Detected continuous speech.")
            elif "start" in vad_result:
                is_speaking = True
                logger.info(f"overlap_vad_queue: {vad_result}")
                logger.debug("overlap_vad_queue: Detected start of speech.")
            elif "end" in vad_result:
                is_speaking = False
                logger.info(f"overlap_vad_queue: {vad_result}")
                logger.debug("overlap_vad_queue: Detected end of speech.")

            await vad_queue.put((audio_float32, vad_result))
        except asyncio.CancelledError:
            logger.warning("overlap_vad_queue: Task was cancelled.")
            break
        except Exception as e:
            logger.error(
                f"Unexpected error occurred in overlap_vad_queue: {type(e).__name__}: {e}"
            )
