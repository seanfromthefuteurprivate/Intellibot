# =============================================================================
# Intellibot Dashboard - Multi-Stage Dockerfile
# =============================================================================
# Stage 1: Build React frontend
# Stage 2: Run FastAPI backend with built frontend
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build React Frontend
# -----------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files first for better caching
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm ci --silent

# Copy frontend source code
COPY frontend/ ./

# Build production React app
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Python FastAPI Backend
# -----------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy API requirements and install Python dependencies
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire dashboard API
COPY api/ ./api/

# Copy built React frontend from Stage 1 to static folder
COPY --from=frontend-builder /app/frontend/build ./build

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port 8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DASHBOARD_PORT=8080
ENV DASHBOARD_HOST=0.0.0.0

# Run uvicorn server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
