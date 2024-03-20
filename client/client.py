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
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(sampwidth),
                channels=nchannels,
                rate=framerate,
                output=True,
            )

            chunk_size = int(framerate * 0.1)
            data = wf.readframes(chunk_size)
            print(
                f"framerate {framerate}, nchannels {nchannels}, sampwidth {sampwidth}, chunk_size {chunk_size}, {len(data)}"
            )

            while len(data) > 0:
                ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
                stream.write(data)  # 오디오 데이터 재생
                data = wf.readframes(chunk_size)

                time.sleep(0.1)

        stream.stop_stream()
        stream.close()
        p.terminate()
        ws.close()

    thread = threading.Thread(target=send_audio)
    thread.start()


def on_message(ws, message):
    print("Received message from server: ", message)


def on_error(ws, error):
    print("Error: ", error)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


websocket_url = "ws://localhost:8080/api/v1/stream/transcribe"
audio_file_path = "0320.wav"

ws = websocket.WebSocketApp(
    websocket_url,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)

ws.run_forever()
