import argparse
import json
import threading
import wave
from urllib.parse import urlencode

import pyaudio
import websocket

parser = argparse.ArgumentParser(description="Audio streaming and translation client.")
parser.add_argument(
    "-f", "--audio_file_path", type=str, required=True, help="Path to the audio file."
)
parser.add_argument(
    "-r",
    "--realtime_playback",
    type=bool,
    default=False,
    help="Whether to play the audio in real time.",
)
parser.add_argument(
    "-s", "--server_url", type=str, required=True, help="URL of the server."
)
parser.add_argument(
    "-t",
    "--translate",
    type=bool,
    default=False,
    help="Whether to translate the audio.",
)
parser.add_argument(
    "-sl",
    "--source_lang",
    type=str,
    default=None,
    help="Source language for translation.",
)
parser.add_argument(
    "-tl",
    "--target_lang",
    type=str,
    default=None,
    help="Target language for translation.",
)
args = parser.parse_args()


query_params = {"translate_flag": str(args.translate).lower()}
if args.source_lang is not None:
    query_params["src_lang"] = args.source_lang
if args.target_lang is not None:
    query_params["tgt_lang"] = args.target_lang


last_message_id = None
messages = {}


def send_audio(ws, audio_file_path, realtime_playback):
    if realtime_playback:
        p = pyaudio.PyAudio()

    with wave.open(audio_file_path, "rb") as wf:
        framerate = wf.getframerate()
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()

        if realtime_playback:
            stream = p.open(
                format=p.get_format_from_width(sampwidth),
                channels=nchannels,
                rate=framerate,
                output=realtime_playback,
                input=not realtime_playback,
            )

        chunk_size = int(framerate * 0.1)
        data = wf.readframes(chunk_size)

        while len(data) > 0:
            if realtime_playback:
                stream.write(data)  # play audio data
            ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)  # send audio data
            data = wf.readframes(chunk_size)

    ws.close()
    if realtime_playback:
        stream.stop_stream()
        stream.close()
        p.terminate()


def on_open(ws):
    threading.Thread(
        target=send_audio, args=(ws, args.audio_file_path, args.realtime_playback)
    ).start()


def on_message(ws, message):
    global last_message_id
    global messages

    message = json.loads(message)
    print(message)
    if last_message_id != message["message_id"]:
        last_message_id = message["message_id"]

        if message["transcript"]:
            messages[message["message_id"]] = {
                "transcript": message["transcript"],
                "translate": None,
            }
        if message["translate"]:
            messages[message["message_id"]]["translate"] = message["translate"]
    else:
        if message["transcript"]:
            messages[message["message_id"]]["transcript"] = message["transcript"]
        if message["translate"]:
            messages[message["message_id"]]["translate"] = message["translate"]

    print("\033[2J")
    for message_id in sorted(messages.keys()):
        message = messages[message_id]
        print(f"{message_id} : {message['transcript']} -> {message['translate']}")
        print()


def on_error(ws, error):
    print("Error: ", error)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


websocket_url = f"{args.server_url}/api/v1/stream/overlap?{urlencode(query_params)}"
print(websocket_url)
ws = websocket.WebSocketApp(
    websocket_url,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)

ws.run_forever()
