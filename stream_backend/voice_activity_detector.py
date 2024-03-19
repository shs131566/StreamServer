import torch
from loguru import logger

from stream_backend.config import settings


class VADIterator:
    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError(
                "VADIterator does not support sampling rates other than [8000, 16000]"
            )
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False):
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = (
                self.current_sample - self.speech_pad_samples - window_size_samples
            )
            return {
                "start": (
                    int(speech_start)
                    if not return_seconds
                    else round(speech_start / self.sampling_rate, 1)
                )
            }

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = (
                    self.temp_end + self.speech_pad_samples - window_size_samples
                )
                self.temp_end = 0
                self.triggered = False
                return {
                    "end": (
                        int(speech_end)
                        if not return_seconds
                        else round(speech_end / self.sampling_rate, 1)
                    )
                }

        return None


class VoiceActivityDetect:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = self._init_model(settings.VAD_MODEL_PATH, self.device)
        self.iterator = VADIterator(self.model)
        logger.info("Voice Activity Detection model initialized.")

    def _init_model(self, model_path: str, device: str):
        torch.set_grad_enabled(False)
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model
