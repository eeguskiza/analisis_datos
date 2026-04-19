.PHONY: up down build rebuild restart logs status dev db-shell clean help ip nexo-init nexo-owner nexo-verify nexo-smoke nexo-setup

# ── Docker ────────────────────────────────────────────────────────────────────

up: ## Arranca todos los servicios
	docker compose up -d
	@echo ""
	@echo "  Nexo arrancado:"
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

# Macro para correr psql dentro del container db usando las env vars que
# el propio container ya tiene inyectadas (POSTGRES_USER, POSTGRES_DB).
# Evita parsear el .env del host (que puede usar OEE_* via compat layer
# y no tener NEXO_* literalmente definido).
PSQL_IN_DB = docker compose exec -T db bash -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

db-shell: ## Abre una shell psql en el Postgres de Nexo
	@docker compose exec -it db bash -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

# ── Nexo: schema + bootstrap (Phase 2 / Plan 02-01) ─────────────────────────

nexo-init: ## Crea schema nexo + 8 tablas + seed (idempotente)
	docker compose exec web python scripts/init_nexo_schema.py

nexo-owner: ## Crea el primer usuario 'propietario' (interactivo)
	docker compose exec -it web python scripts/create_propietario.py

nexo-verify: ## Lista las tablas del schema nexo y los usuarios
	@echo "── Tablas del schema nexo ──"
	@docker compose exec -T db bash -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -c "\\dt nexo.*"'
	@echo ""
	@echo "── Usuarios ──"
	@docker compose exec -T db bash -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -c "SELECT id, email, role, active, must_change_password FROM nexo.users;"'
	@echo ""
	@echo "── Seed de catalogos ──"
	@docker compose exec -T db bash -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -c "SELECT code FROM nexo.roles ORDER BY code;"'
	@docker compose exec -T db bash -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -c "SELECT code FROM nexo.departments ORDER BY code;"'

nexo-smoke: ## Smoke test de argon2id (hash + verify)
	@docker compose exec web python -c "from nexo.services.auth import hash_password, verify_password; h = hash_password('test12345678'); print('hash:', h[:40], '...'); print('ok :', verify_password(h, 'test12345678')); print('bad:', verify_password(h, 'equivocado'))"

nexo-setup: nexo-init nexo-verify ## Init completo (schema + verify). Owner se crea despues con 'make nexo-owner'

nexo-app-role: ## Crea rol nexo_app con GRANTs limitados (audit_log append-only). Plan 02-04 gate IDENT-06. Lee NEXO_PG_APP_PASSWORD del .env.
	@APP_PWD=$$(grep -E '^[[:space:]]*NEXO_PG_APP_PASSWORD=' .env 2>/dev/null | tail -1 | sed -E 's/^[[:space:]]*NEXO_PG_APP_PASSWORD=//' | tr -d '\r'); \
	 if [ -z "$$APP_PWD" ]; then \
		echo "ERROR: NEXO_PG_APP_PASSWORD no esta en .env. Anade (sin espacios iniciales):"; \
		echo "  NEXO_PG_APP_USER=nexo_app"; \
		echo "  NEXO_PG_APP_PASSWORD=<password-generado>"; \
		exit 1; \
	 fi; \
	 echo "→ Aplicando rol nexo_app..."; \
	 docker compose exec -T db bash -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -v nexo_app_password='"'$$APP_PWD'"' -f -' < scripts/create_nexo_app_role.sql

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
