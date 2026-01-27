FROM python:3.11-slim

WORKDIR /app

# Install system deps (safe defaults)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install deps (fallback to pip if uv not available)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt || true

# Default command (adjust if needed)
CMD ["python", "run_snake.py"]
