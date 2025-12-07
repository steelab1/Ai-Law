#!/bin/bash

# Entrypoint script for Streamlit frontend

set -e

echo "Starting Streamlit Chatbot Interface..."
echo "Backend URL: ${BACKEND_URL:-http://localhost:8002}"

# Run Streamlit
exec streamlit run chat_interface.py \
    --server.port=8051 \
    --server.address=0.0.0.0 \
    --server.headless=true
