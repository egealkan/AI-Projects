FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./AI/config/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./AI/src/ ./src/

ENV GEMINI_API_KEY="AIzaSyBiQ_A-k3kjwVtkBGHqz3xTjiq3Qf6CcNw"
ENV BENCH_API_URL="https://api.victoriouscliff-d85d07e6.westeurope.azurecontainerapps.io/api/benches/67ac671f485bbebbe866d4a5"
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

EXPOSE 8080

CMD ["python", "src/api/endpoints.py"]