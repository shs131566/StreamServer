import json

import numpy as np
from loguru import logger
from tritonclient.grpc import InferenceServerClient, InferInput

from stream_backend.config import settings


class TritonClient:
    def __init__(self, url: str = None):
        self.url = (
            f"{settings.TRITON_SERVER_URL}:{settings.TRITON_SERVER_PORT}"
            if url is None
            else url
        )
        self.triton_client = InferenceServerClient(url=self.url, verbose=True)

    def transcribe(
        self,
        audio: np.ndarray,
        language: str,
        sample_rate: int = 16000,
        inference_type: str = "streaming",
        model_name: str = "whisper",
    ):
        audio_input = InferInput(name="audio", shape=audio.shape, datatype="FP32")
        sr_input = InferInput(name="sample_rate", shape=[1], datatype="INT32")
        language_input = InferInput(name="language", shape=[1], datatype="BYTES")
        inference_type_input = InferInput(name="type", shape=[1], datatype="BYTES")

        audio_input.set_data_from_numpy(audio)
        sr_input.set_data_from_numpy(np.array([sample_rate], dtype=np.int32))
        language_input.set_data_from_numpy(
            np.array([language.encode("utf-8")], dtype=object)
        )
        inference_type.set_data_from_numpy(
            np.array([inference_type_input.encode("utf-8")], dtype=object)
        )

        result = self.triton_client.infer(
            model_name=model_name,
            inputs=[audio_input, sr_input, language_input, inference_type],
        )

        logger.info(result)
        return (
            json.loads(result.as_numpy("transcription")[0]),
            result.as_numpy("repetition")[0],
        )
