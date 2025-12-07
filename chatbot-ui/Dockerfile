# Frontend Dockerfile for Streamlit UI
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY chat_interface.py .

# Expose Streamlit port
EXPOSE 8051

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8051/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "chat_interface.py", "--server.port=8051", "--server.address=0.0.0.0"]
