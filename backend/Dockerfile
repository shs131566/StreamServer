FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

COPY ./backend/requirements.txt /app/requirements.txt
RUN apt update
RUN apt-get install -y ffmpeg
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

ENV PORT 8080
EXPOSE 8080

COPY ./backend/assets /app/assets
COPY ./backend/stream_backend /app/stream_backend
COPY ./backend/main.py /app/main.py
