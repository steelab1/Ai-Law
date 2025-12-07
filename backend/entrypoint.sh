#!/bin/bash

# Entrypoint script for backend
# This script starts both FastAPI and Celery worker

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to src directory
cd "$SCRIPT_DIR/src" 2>/dev/null || cd /app/src

echo "Starting Vietnamese Legal Q&A Chatbot Backend..."
echo "Working directory: $(pwd)"

# Start FastAPI in background
echo "Starting FastAPI server on port 8002..."
python app.py &
FASTAPI_PID=$!

# Wait a bit for FastAPI to start
sleep 5

# Start Celery worker
echo "Starting Celery worker..."
celery -A tasks.celery_app worker --loglevel=info &
CELERY_PID=$!

# Function to handle shutdown
cleanup() {
    echo "Shutting down..."
    kill $FASTAPI_PID 2>/dev/null
    kill $CELERY_PID 2>/dev/null
    exit 0
}

# Trap SIGTERM and SIGINT
trap cleanup SIGTERM SIGINT

# Wait for processes
wait
