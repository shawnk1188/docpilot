.PHONY: up down rebuild logs test run stop clean

run:
	podman rm -f fastapi-app qdrant ollama 2>/dev/null || true
	podman-compose up

stop:
	podman-compose down
	podman rm -f fastapi-app qdrant ollama 2>/dev/null || true

up:
	podman-compose up -d

down:
	podman-compose down

rebuild:
	podman-compose down
	podman-compose build --no-cache
	podman-compose up

logs:
	podman-compose logs -f fastapi-app

test:
	cd backend && pip install pytest pytest-asyncio -q && pytest -v

clean:
	podman rm -f fastapi-app qdrant ollama 2>/dev/null || true
	podman network prune -f
	podman volume prune -f

health:
	curl -s http://localhost:8000/health | python3 -m json.tool

pull-model:
	podman exec ollama ollama pull llama3.2

status:
	@echo "=== Containers ==="
	@podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "=== FastAPI ==="
	@curl -s http://localhost:8000/health | python3 -m json.tool
	@echo ""
	@echo "=== Qdrant ==="
	@curl -s http://localhost:6333/ | python3 -m json.tool
	@echo ""
	@echo "=== Ollama models ==="
	@curl -s http://localhost:11434/api/tags | python3 -m json.tool