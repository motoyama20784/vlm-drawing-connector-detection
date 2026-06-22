# Stage 1: React ビルド
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY annotator/frontend/package*.json ./
RUN npm install
COPY annotator/frontend/ ./
RUN npm run build

# Stage 2: FastAPI サーバー
FROM python:3.11-slim
WORKDIR /app
COPY docker/annotator/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*
COPY annotator/ ./annotator/
COPY --from=frontend-builder /frontend/dist/ ./annotator/backend/static/
ENV PYTHONPATH=/app
CMD ["uvicorn", "annotator.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
