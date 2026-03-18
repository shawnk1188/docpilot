.PHONY: up down rebuild run stop clean status \
        run-local up-local setup-local \
        logs logs-ui logs-all \
        test test-watch \
        deps lock add remove \
        pull-model ingest ask \
        test-groq grafana help

# ── Default ───────────────────────────────────────────────────────────────────
# Groq mode — no Ollama container needed
run:
	podman rm -f fastapi-app qdrant streamlit-app \
	  prometheus grafana 2>/dev/null || true
	podman-compose up

up:
	podman-compose up -d

down:
	podman-compose down

stop:
	podman-compose down
	podman rm -f fastapi-app qdrant streamlit-app \
	  prometheus grafana 2>/dev/null || true

# ── Local mode (Ollama) ───────────────────────────────────────────────────────
run-local:
	podman rm -f fastapi-app qdrant ollama streamlit-app \
	  prometheus grafana 2>/dev/null || true
	podman-compose --profile local up

up-local:
	podman-compose --profile local up -d

setup-local: up-local
	@echo "Pulling tinyllama — this takes a few minutes..."
	podman exec ollama ollama pull tinyllama
	@echo "Done. Update .env: LLM_PROVIDER=ollama then make rebuild"

pull-model:
	podman exec ollama ollama pull tinyllama

# ── Build ─────────────────────────────────────────────────────────────────────
rebuild:
	podman-compose down
	podman-compose build --no-cache
	podman-compose up

# ── Logs ─────────────────────────────────────────────────────────────────────
logs:
	podman-compose logs -f fastapi-app

logs-ui:
	podman-compose logs -f streamlit-app

logs-all:
	podman-compose logs -f

# ── Dependencies (uv) ─────────────────────────────────────────────────────────
deps:
	cd backend && uv sync
	cd frontend && uv sync

lock:
	cd backend && uv lock
	cd frontend && uv lock
	@echo "Lockfiles updated — commit uv.lock files"

add:
	@read -p "Service (backend/frontend): " svc; \
	read -p "Package (e.g. httpx==0.28.0): " pkg; \
	cd $$svc && uv add $$pkg && uv lock
	@echo "Done — run make rebuild to apply in containers"

remove:
	@read -p "Service (backend/frontend): " svc; \
	read -p "Package to remove: " pkg; \
	cd $$svc && uv remove $$pkg && uv lock
	@echo "Done — run make rebuild to apply in containers"

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	cd backend && uv run pytest -v

test-watch:
	cd backend && uv run pytest -v --tb=short -f

test-groq:
	@curl -s -X POST https://api.groq.com/openai/v1/chat/completions \
	  -H "Authorization: Bearer $$(grep GROQ_API_KEY .env | cut -d= -f2)" \
	  -H "Content-Type: application/json" \
	  -d '{"model":"llama-3.1-8b-instant","messages":[{"role":"user","content":"say hello in one word"}],"temperature":0.1}' \
	  | python3 -m json.tool

# ── Status & health ───────────────────────────────────────────────────────────
status:
	@echo "=== Containers ==="
	@podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "=== FastAPI health ==="
	@curl -s http://localhost:8000/health | python3 -m json.tool
	@echo ""
	@echo "=== Qdrant ==="
	@curl -s http://localhost:6333/ | python3 -m json.tool
	@echo ""
	@echo "=== Collection stats ==="
	@curl -s http://localhost:8000/api/stats | python3 -m json.tool

# ── Convenience ───────────────────────────────────────────────────────────────
ingest:
	@read -p "Path to file: " path; \
	curl -s -X POST http://localhost:8000/api/ingest \
	  -F "file=@$$path" | python3 -m json.tool

ask:
	@read -p "Question: " q; \
	curl -s -X POST http://localhost:8000/api/query \
	  -H "Content-Type: application/json" \
	  -d "{\"question\": \"$$q\", \"top_k\": 5}" \
	  | python3 -m json.tool

grafana:
	@echo "Grafana → http://localhost:3000  (admin / docpilot)"
	open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	podman rm -f fastapi-app qdrant ollama streamlit-app \
	  prometheus grafana 2>/dev/null || true
	podman network prune -f
	podman volume prune -f
	uv cache clean

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  docpilot — RAG system Makefile"
	@echo ""
	@echo "  Setup"
	@echo "    make up              start all containers (Groq mode)"
	@echo "    make up-local        start with Ollama (local LLM)"
	@echo "    make setup-local     start + pull tinyllama model"
	@echo "    make rebuild         full no-cache rebuild"
	@echo ""
	@echo "  Daily use"
	@echo "    make ingest          upload and ingest a document"
	@echo "    make ask             ask a question via terminal"
	@echo "    make status          health check all services"
	@echo "    make test-groq       test Groq API connection"
	@echo ""
	@echo "  Logs"
	@echo "    make logs            tail fastapi logs"
	@echo "    make logs-ui         tail streamlit logs"
	@echo "    make logs-all        tail all container logs"
	@echo ""
	@echo "  Dependencies"
	@echo "    make deps            install deps locally via uv"
	@echo "    make lock            regenerate uv.lock files"
	@echo "    make add             add a package to a service"
	@echo "    make remove          remove a package from a service"
	@echo ""
	@echo "  Testing"
	@echo "    make test            run pytest suite"
	@echo "    make test-watch      run pytest in watch mode"
	@echo ""
	@echo "  Cleanup"
	@echo "    make clean           remove containers + prune"
	@echo "    make down            stop all containers"
	@echo ""