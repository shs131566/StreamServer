import argparse
import json
import time

import librosa
import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    return audio.reshape(1, -1), sr


def setup_inputs(audio, sr, language, inference_type="transcribe"):
    audio_input = InferInput(name="audio", shape=audio.shape, datatype="FP32")
    sr_input = InferInput(name="sample_rate", shape=[1], datatype="INT32")
    language_input = InferInput(name="language", shape=[1], datatype="BYTES")
    inference_type_input = InferInput(name="type", shape=[1], datatype="BYTES")

    audio_input.set_data_from_numpy(audio)
    sr_input.set_data_from_numpy(np.array([sr], dtype=np.int32))
    language_input.set_data_from_numpy(
        np.array([language.encode("utf-8")], dtype=object)
    )
    inference_type_input.set_data_from_numpy(
        np.array([inference_type.encode("utf-8")], dtype=object)
    )

    return [audio_input, sr_input, language_input, inference_type_input]


def setup_transcription_input(transcripts):
    transcript_input = InferInput("transcription", [1], "BYTES")
    transcript_input.set_data_from_numpy(np.array(transcripts, dtype=object))
    return transcript_input


def perform_inference(client, model_name, inputs, timeout=36000):
    start_time = time.time()
    result = client.infer(model_name=model_name, inputs=inputs, timeout=timeout)
    end_time = time.time()
    print(f"{model_name} model inference time: {end_time - start_time} seconds")
    return result


def main(args):
    triton_client = InferenceServerClient(url=args.url)
    audio, sr = load_audio(args.file)
    inputs = setup_inputs(audio, sr, args.language)

    whisper_results = perform_inference(triton_client, "whisper", inputs)
    transcripts = whisper_results.as_numpy("transcription")

    transcript_input = InferInput("transcription", [1], "BYTES")
    transcript_input.set_data_from_numpy(transcripts)

    embedding_inputs = [inputs[0], inputs[1], setup_transcription_input(transcripts)]

    embedding_results = perform_inference(triton_client, "embedding", embedding_inputs)
    embeddings = embedding_results.as_numpy("embeddings")

    transcripts = embedding_results.as_numpy("transcription")

    transcripts_json = json.loads(transcripts[0])
    language = json.loads(whisper_results.as_numpy("language")[0])

    print(f"Language: {language}")
    for transcript in transcripts_json:
        print(f"Speaker {transcript['speaker']}: {transcript['text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-u", "--url", default="localhost:8001")
    parser.add_argument("-l", "--language", default="ko")
    args = parser.parse_args()

    main(args)
