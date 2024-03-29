import json
import threading
import time
import wave

import pyaudio
import websocket


def on_open(ws):
    def send_audio(*args):
        with wave.open(audio_file_path, "rb") as wf:
            framerate = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()

            chunk_size = int(framerate * 0.1)
            data = wf.readframes(chunk_size)
            print(
                f"framerate {framerate}, nchannels {nchannels}, sampwidth {sampwidth}, chunk_size {chunk_size}, {len(data)}"
            )

            while len(data) > 0:
                ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
                data = wf.readframes(chunk_size)

                time.sleep(0.1)

        ws.close()

    thread = threading.Thread(target=send_audio)
    thread.start()


last_message_id = None
messages = {}


def on_message(ws, message):
    global last_message_id
    global messages

    message = json.loads(message)

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


websocket_url = "ws://localhost:8080/api/v1/stream/overlap?translate_flag=true"
audio_file_path = "news.wav"

ws = websocket.WebSocketApp(
    websocket_url,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)

ws.run_forever()
