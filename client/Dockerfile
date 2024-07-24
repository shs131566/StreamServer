FROM python:3.10-slim

WORKDIR /app

RUN apt-get update

COPY client.py .
COPY requirements.txt .
COPY EngKorNews_16K.wav .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "client.py", "-f", "EngKorNews_16K.wav"]
