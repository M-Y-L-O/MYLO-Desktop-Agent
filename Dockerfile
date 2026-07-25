FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY App.py .
COPY Core ./Core
COPY Processing ./Processing
COPY Types ./Types
COPY Utils ./Utils
COPY .env .

ENV SERVER_IP=0.0.0.0
ENV SERVER_PORT=8080

EXPOSE 8080

CMD ["python", "App.py"]

# docker build -t mylo-agent:local .