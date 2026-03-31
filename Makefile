.PHONY: up down build rebuild restart logs status dev db-shell clean help

# ── Docker ────────────────────────────────────────────────────────────────────

up: ## Arranca todos los servicios (db + web + mcp)
	docker compose up -d
	@echo ""
	@echo "  OEE Planta arrancado:"
	@echo "    Web:  http://localhost:8000"
	@echo "    DB:   postgresql://oee:oee@localhost:5432/oee_planta"
	@echo "    Docs: http://localhost:8000/api/docs"
	@echo ""

down: ## Para todos los servicios
	docker compose down

build: ## Reconstruye las imagenes (arch nativa)
	docker compose build

rebuild: ## Reconstruye sin cache y arranca
	docker compose build --no-cache
	docker compose up -d

restart: ## Reinicia todos los servicios
	docker compose restart

logs: ## Muestra logs en tiempo real
	docker compose logs -f

logs-web: ## Logs solo de la web
	docker compose logs -f web

logs-db: ## Logs solo de la base de datos
	docker compose logs -f db

logs-mcp: ## Logs solo del MCP server
	docker compose logs -f mcp

status: ## Muestra el estado de los contenedores
	docker compose ps

db-shell: ## Abre una shell psql en la base de datos
	docker compose exec db psql -U oee oee_planta

# ── Desarrollo local (sin Docker) ────────────────────────────────────────────

dev: ## Arranca el servidor en modo desarrollo (sin Docker)
	OEE_DEBUG=true python3 -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

install: ## Instala dependencias Python (para desarrollo local)
	pip install -r requirements.txt

# ── Utilidades ────────────────────────────────────────────────────────────────

clean: ## Elimina contenedores, volumenes y caches
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f data/oee.db

health: ## Comprueba el health de los servicios
	@curl -s http://localhost:8000/api/health | python3 -m json.tool

# ── Ayuda ─────────────────────────────────────────────────────────────────────

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
