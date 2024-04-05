import argparse
import json

import librosa
import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput

# Use argparse to handle command line arguments
parser = argparse.ArgumentParser(
    description="Send an audio file to Triton for inference."
)
parser.add_argument("-f", "--file", required=True, help="Path to the audio file.")
parser.add_argument(
    "-u",
    "--url",
    default="localhost:8001",
    help="Triton server URL. Default is localhost:8001",
)
parser.add_argument(
    "-l",
    "--language",
    default="ko",
)
args = parser.parse_args()

audio_path = args.file  # Path to the audio file
url = args.url  # URL of the Triton server
language = args.language
model_name = "whisper"

triton_client = InferenceServerClient(url=url)

# Load the audio file using librosa
audio, sr = librosa.load(audio_path, sr=16000)
audio = audio.reshape(1, -1)

# Set up the inputs
audio_input = InferInput(name="audio", shape=audio.shape, datatype="FP32")
sr_input = InferInput(name="sample_rate", shape=[1], datatype="INT32")
language_input = InferInput(name="language", shape=[1], datatype="BYTES")
inference_type = InferInput(name="type", shape=[1], datatype="BYTES")

# Set the data for the inputs
audio_input.set_data_from_numpy(audio)
sr_input.set_data_from_numpy(np.array([sr], dtype=np.int32))
language_input.set_data_from_numpy(np.array([language.encode("utf-8")], dtype=object))
inference_type.set_data_from_numpy(
    np.array(["streaming".encode("utf-8")], dtype=object)
)
# Request inference from the Triton client
result = triton_client.infer(
    model_name=model_name,
    inputs=[audio_input, sr_input, language_input, inference_type],
    timeout=360000,
)

# Process the result
transcripts = result.as_numpy("transcription")
print(json.loads(transcripts[0]))
