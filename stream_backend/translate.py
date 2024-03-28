import asyncio
import json

from fastapi import WebSocket
from loguru import logger

from stream_backend.config import settings
from stream_backend.triton_client import InferenceServerException, TritonClient


async def translate(
    transcript_queue: asyncio.Queue,
    triton_client: TritonClient,
    websocket: WebSocket,
    src_lang: str = None,
    tgt_lang: str = None,
):
    while True:
        try:
            message_id, out_language, transcript = await transcript_queue.get()
            logger.info(
                f"translate_queue: received message_id {message_id}, spoken language {out_language}, {transcript}."
            )
            if src_lang == None and tgt_lang == None:
                to_translate = next(
                    key for key in settings.LANGUAGE_DICT if key != out_language
                )

            logger.info(f"translate_queue: tanslate {out_language} to {to_translate}.")
            translation = triton_client.translate(
                transcript,
                src_lang=settings.LANGUAGE_DICT[out_language],
                tgt_lang=settings.LANGUAGE_DICT[to_translate],
            )
            logger.success(
                f"translate_queue: {message_id:05} {transcript} \t translate to \t {translation}"
            )
            message_dict = {
                "language": tgt_lang,
                "message_id": f"{message_id:05}",
                "transcript": None,
                "translate": translation,
            }

            await websocket.send_text(json.dumps(message_dict))
        except InferenceServerException as e:
            logger.error(f"translate_queue: InferenceServerException occurred: {e}")
            continue
        except asyncio.CancelledError:
            logger.warning("translate_queue: Task was cancelled.")
            break
        except Exception as e:
            logger.error(
                f"Unexpected error occurred in translate_queue: {type(e).__name__}: {e}"
            )
