#!/bin/bash

# Cho Docker - giả sử WORKDIR là /ai hoặc /app
# Chuyển vào thư mục src của backend
cd /ai/backend/src 2>/dev/null || cd /app/backend/src 2>/dev/null || cd backend/src

echo "Working directory: $(pwd)"

# Chạy app.py trong nền
python app.py &

# Chạy Celery worker
celery -A tasks.celery_app worker --loglevel=debug
