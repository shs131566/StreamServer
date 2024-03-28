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

                # time.sleep(0.1)

        time.sleep(300)
        ws.close()

    thread = threading.Thread(target=send_audio)
    thread.start()


def on_message(ws, message):
    message = json.loads(message)
    if message["transcript"]:
        print(f"{message['message_id']}: {message['transcript']}")
    elif message["translate"]:
        print(f"{message['message_id']}: {message['translate']}")


def on_error(ws, error):
    print("Error: ", error)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


websocket_url = "ws://localhost:8080/api/v1/stream/transcribe?translate_flag=true"
audio_file_path = "news.wav"

ws = websocket.WebSocketApp(
    websocket_url,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)

ws.run_forever()
