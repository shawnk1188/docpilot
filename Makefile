.PHONY: up down rebuild logs test notebook

up:
	podman-compose up --build -d

down:
	podman-compose down

rebuild:
	podman-compose down
	podman-compose build --no-cache
	podman-compose up -d

logs:
	podman-compose logs -f fastapi-app

test:
	cd backend && pip install pytest pytest-asyncio -q && pytest -v

notebook:
	podman-compose up jupyter -d
	@echo "Jupyter → http://localhost:8888"

health:
	curl -s http://localhost:8000/health | python3 -m json.tool
