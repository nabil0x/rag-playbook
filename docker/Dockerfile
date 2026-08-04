# Project 32: RAG API service image.
# Build from the repo root:  docker build -f docker/Dockerfile -t rag-api .
FROM python:3.11-slim

WORKDIR /app

# curl is only for the compose healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn redis

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
