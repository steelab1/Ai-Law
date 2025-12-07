# Backend Dockerfile for Vietnamese Legal Q&A Chatbot
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set working directory to src
WORKDIR /app/src

# Expose port
EXPOSE 8002

# Default command (can be overridden in docker-compose)
CMD ["python", "app.py"]
