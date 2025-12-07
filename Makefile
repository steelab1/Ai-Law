# Makefile for Vietnamese Legal Q&A Chatbot

.PHONY: help build up down logs restart clean setup

# Default target
help:
	@echo "Vietnamese Legal Q&A Chatbot - Docker Commands"
	@echo ""
	@echo "Usage:"
	@echo "  make setup    - Copy .env.example to .env (first time setup)"
	@echo "  make build    - Build all Docker images"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make restart  - Restart all services"
	@echo "  make logs     - View logs from all services"
	@echo "  make logs-api - View logs from backend API"
	@echo "  make logs-celery - View logs from Celery worker"
	@echo "  make logs-frontend - View logs from frontend"
	@echo "  make clean    - Remove all containers, volumes, and images"
	@echo "  make status   - Show status of all services"
	@echo ""

# First time setup
setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please edit it with your API keys."; \
	else \
		echo ".env file already exists."; \
	fi

# Build all images
build:
	docker compose build

# Start all services
up:
	docker compose up -d

# Start with build
up-build:
	docker compose up -d --build

# Stop all services
down:
	docker compose down

# Restart all services
restart:
	docker compose restart

# View all logs
logs:
	docker compose logs -f

# View specific service logs
logs-api:
	docker compose logs -f backend-api

logs-celery:
	docker compose logs -f backend-celery

logs-frontend:
	docker compose logs -f frontend

logs-redis:
	docker compose logs -f redis

logs-mongodb:
	docker compose logs -f mongodb

logs-qdrant:
	docker compose logs -f qdrant

logs-elasticsearch:
	docker compose logs -f elasticsearch

# Show status
status:
	docker compose ps

# Clean everything
clean:
	docker compose down -v --rmi all
	@echo "All containers, volumes, and images removed."

# Clean volumes only
clean-volumes:
	docker compose down -v
	@echo "All containers and volumes removed."

# Shell into services
shell-api:
	docker compose exec backend-api /bin/bash

shell-celery:
	docker compose exec backend-celery /bin/bash

shell-frontend:
	docker compose exec frontend /bin/bash

# Health check
health:
	@echo "Checking services health..."
	@curl -s http://localhost:8002/ > /dev/null && echo "✓ Backend API: OK" || echo "✗ Backend API: FAILED"
	@curl -s http://localhost:8051/_stcore/health > /dev/null && echo "✓ Frontend: OK" || echo "✗ Frontend: FAILED"
	@curl -s http://localhost:6333/ > /dev/null && echo "✓ Qdrant: OK" || echo "✗ Qdrant: FAILED"
	@curl -s http://localhost:9200/ > /dev/null && echo "✓ Elasticsearch: OK" || echo "✗ Elasticsearch: FAILED"
	@docker compose exec redis redis-cli ping > /dev/null && echo "✓ Redis: OK" || echo "✗ Redis: FAILED"
	@docker compose exec mongodb mongosh --eval "db.runCommand('ping')" > /dev/null 2>&1 && echo "✓ MongoDB: OK" || echo "✗ MongoDB: FAILED"
