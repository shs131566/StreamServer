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
        self.triton_client = InferenceServerClient(url=self.url)

    def transcribe(
        self,
        audio: np.ndarray,
        language: str,
        sample_rate: int = settings.AUDIO_SAMPLING_RATE,
        inference_type: str = "streaming",
        model_name: str = "whisper",
    ):
        audio = audio.reshape(1, -1)
        audio_input = InferInput(name="audio", shape=audio.shape, datatype="FP32")
        sr_input = InferInput(name="sample_rate", shape=[1], datatype="INT32")
        language_input = InferInput(name="language", shape=[1], datatype="BYTES")
        inference_type_input = InferInput(name="type", shape=[1], datatype="BYTES")

        audio_input.set_data_from_numpy(audio)
        sr_input.set_data_from_numpy(np.array([sample_rate], dtype=np.int32))
        language_input.set_data_from_numpy(
            np.array([language.encode("utf-8")], dtype=object)
        )
        inference_type_input.set_data_from_numpy(
            np.array([inference_type.encode("utf-8")], dtype=object)
        )
        result = self.triton_client.infer(
            model_name=model_name,
            inputs=[audio_input, sr_input, language_input, inference_type_input],
        )

        return (
            json.loads(result.as_numpy("transcription")[0]),
            result.as_numpy("repetition")[0],
        )

    def translate(
        self,
        transcript: str,
        src_lang: str,
        tgt_lang: str,
        model_name: str = "nmt",
    ):
        # TODO: 현재는 triton server config가 query인데, 추후 수정 요망
        transcript_input = InferInput(name="query", shape=[1], datatype="BYTES")
        src_lang_input = InferInput(name="src_lang", shape=[1], datatype="BYTES")
        tgt_lang_input = InferInput(name="tgt_lang", shape=[1], datatype="BYTES")

        transcript_input.set_data_from_numpy(
            np.array([transcript.encode("utf-8")], dtype=object)
        )
        src_lang_input.set_data_from_numpy(
            np.array([src_lang.encode("utf-8")], dtype=object)
        )
        tgt_lang_input.set_data_from_numpy(
            np.array([tgt_lang.encode("utf-8")], dtype=object)
        )

        result = self.triton_client.infer(
            model_name=model_name,
            inputs=[transcript_input, src_lang_input, tgt_lang_input],
        )

        return json.loads(result.as_numpy("translated_txt")[0])[0]["translated"]
