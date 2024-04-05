import json
from typing import List

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import whisper
from loguru import logger


class TritonPythonModel:
    def initialize(self, args):
        self.device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        if self.device == "cuda":
            self.device = f"{self.device}:{args['model_instance_device_id']}"

        logger.info(f"Using device {self.device}")
        self.model = whisper.load_model(
            "/models/whisper/1/large-v3.pt",
            device=self.device,
        )

    def execute_inference(self, request, inference_type, language, input_audio):
        audio_tensor = torch.tensor(whisper.pad_or_trim(input_audio.flatten())).to(
            self.device
        )
        mel = whisper.log_mel_spectrogram(audio_tensor, n_mels=128)
        if language == "None":
            language_probs = self.model.detect_language(mel)[1]
            filtered_probs = {
                lang: prob
                for lang, prob in language_probs.items()
                if lang in ["ko", "en"]
            }
            language = max(filtered_probs, key=filtered_probs.get)

        options = whisper.DecodingOptions(language=language, without_timestamps=True)

        if inference_type == "transcribe":
            outputs = whisper.transcribe(
                self.model,
                input_audio.reshape(-1),
                language=language,
            )
            repetition = False
        elif inference_type == "streaming":
            outputs = self.model.decode(mel, options)
            repetition = outputs.compression_ratio >= 2.0

        return outputs, repetition, language

    def construct_response(self, inference_type, outputs, repetition, language):
        if inference_type == "streaming":
            return pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "transcription",
                        np.array([json.dumps(outputs.text)], dtype=np.string_),
                    ),
                    pb_utils.Tensor(
                        "repetition", np.array([repetition], dtype=np.bool_)
                    ),
                    pb_utils.Tensor(
                        "language",
                        np.array([json.dumps(language)], dtype=np.string_),
                    ),
                ]
            )

        elif inference_type == "transcribe":
            return pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "transcription",
                        np.array([json.dumps(outputs)], dtype=np.string_),
                    ),
                    pb_utils.Tensor(
                        "repetition", np.array([repetition], dtype=np.bool_)
                    ),
                    pb_utils.Tensor(
                        "language",
                        np.array([json.dumps(language)], dtype=np.string_),
                    ),
                ]
            )

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                inp = pb_utils.get_input_tensor_by_name(request, "audio")
                language = (
                    pb_utils.get_input_tensor_by_name(request, "language")
                    .as_numpy()[0]
                    .decode("utf-8")
                )
                inference_type = (
                    pb_utils.get_input_tensor_by_name(request, "type")
                    .as_numpy()[0]
                    .decode("utf-8")
                )

                input_audio = inp.as_numpy()
                outputs, repetition, language = self.execute_inference(
                    request, inference_type, language, input_audio
                )

                inference_response = self.construct_response(
                    inference_type, outputs, repetition, language
                )
                responses.append(inference_response)
            except Exception as e:
                logger.error(f"Failed to execute request: {e}")

        return responses
