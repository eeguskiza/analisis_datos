.PHONY: up down build rebuild restart logs status dev db-shell clean help ip

# ── Docker ────────────────────────────────────────────────────────────────────

up: ## Arranca todos los servicios
	docker compose up -d
	@echo ""
	@echo "  OEE Planta arrancado:"
	@echo "    Local:  http://localhost:8000"
	@echo "    HTTPS:  https://localhost  (cert auto-firmado)"
	@IP=$$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $$1}'); \
	 if [ -n "$$IP" ]; then echo "    LAN:    https://$$IP  (desde otros equipos)"; fi
	@echo "    Docs:   http://localhost:8000/api/docs"
	@echo ""

down: ## Para todos los servicios
	docker compose down

build: ## Reconstruye las imagenes
	docker compose build

rebuild: ## Reconstruye sin cache y arranca
	docker compose build --no-cache
	docker compose up -d

restart: ## Reinicia todos los servicios
	docker compose restart

logs: ## Muestra logs en tiempo real (todos)
	docker compose logs -f

logs-web: ## Logs solo de la web
	docker compose logs -f web

logs-db: ## Logs solo de la base de datos
	docker compose logs -f db

logs-mcp: ## Logs solo del MCP server
	docker compose logs -f mcp

status: ## Muestra el estado de los contenedores
	docker compose ps

db-shell: ## Abre una shell psql
	docker compose exec db psql -U oee oee_planta

ip: ## Muestra la IP local para compartir el enlace
	@IP=$$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $$1}'); \
	 echo "  Tu IP: $$IP"; \
	 echo "  Enlace LAN: https://$$IP"

# ── Desarrollo local (sin Docker) ────────────────────────────────────────────

dev: ## Arranca en modo desarrollo (sin Docker, SQLite)
	@PORT=8000; \
	 while ss -tln 2>/dev/null | awk '{print $$4}' | grep -qE "[:.]$$PORT$$" || \
	       (command -v lsof >/dev/null 2>&1 && lsof -iTCP:$$PORT -sTCP:LISTEN >/dev/null 2>&1); do \
		echo "  Puerto $$PORT en uso, probando siguiente..."; \
		PORT=$$((PORT + 1)); \
		if [ $$PORT -gt 8100 ]; then echo "  No se encontro puerto libre entre 8000-8100"; exit 1; fi; \
	 done; \
	 echo "  Arrancando en http://127.0.0.1:$$PORT"; \
	 OEE_DEBUG=true python3 -m uvicorn api.main:app --reload --host 127.0.0.1 --port $$PORT

install: ## Instala dependencias Python
	pip install -r requirements.txt

# ── Utilidades ────────────────────────────────────────────────────────────────

clean: ## Elimina contenedores, volumenes y caches
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f data/oee.db

health: ## Health check de los servicios
	@curl -s http://localhost:8000/api/health | python3 -m json.tool

# ── Ayuda ─────────────────────────────────────────────────────────────────────

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
