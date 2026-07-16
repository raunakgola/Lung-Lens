# ---- Stage 1: build the React frontend ----
FROM node:20-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python API (CPU-only torch keeps the image small) ----
FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY backend/api_requirements.txt .
RUN pip install --no-cache-dir -r api_requirements.txt

COPY backend/ ./backend/
COPY --from=frontend /frontend/dist ./frontend/dist

WORKDIR /app/backend
# HF Spaces runs as a non-root user — give matplotlib/torch a writable cache dir
ENV MPLCONFIGDIR=/tmp TORCH_HOME=/tmp
ENV PORT=7860
EXPOSE 7860

CMD ["python", "Server.py"]
