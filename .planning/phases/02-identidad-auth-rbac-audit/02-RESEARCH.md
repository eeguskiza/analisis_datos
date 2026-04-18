# Phase 2: Identidad (auth + RBAC + audit) — Research

**Researched:** 2026-04-18
**Domain:** FastAPI auth stack, PostgreSQL identity schema, Argon2id, session management, audit middleware
**Confidence:** HIGH (stack decisions and APIs verified via Context7 + official docs + PyPI)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Roles: `propietario` | `directivo` | `usuario`. Un rol por usuario, N departamentos.
- `propietario` ignora departamento (global). Unico que gestiona usuarios y ve audit completo.
- Algoritmo: Argon2id via `argon2-cffi`. Parametros: `time_cost=3, memory_cost=65536, parallelism=4`.
- Bloqueo: 5 intentos fallidos consecutivos → 15 min lock sobre tupla `(user, IP)`.
- Cookie HttpOnly + Secure + SameSite=Lax. 12h sliding expiration.
- `nexo.audit_log` append-only: `REVOKE UPDATE, DELETE` al rol app. Middleware registra cada request autenticada con body sanitizado.
- Todas las tablas bajo schema `nexo` en Postgres.
- `engine_nexo` (Postgres) es nuevo en Phase 2. `engine_app` (SQL Server) permanece intacto.
- No se toca `OEE/db/connector.py` (MES) en esta phase.
- Herramientas: scripts Python + SQLAlchemy + `docker compose exec` (no Alembic en Mark-III).
- Bootstrap: script interactivo `scripts/create_propietario.py`.

### Claude's Discretion

- Mecanismo concreto de sesion: cookie firmada con `itsdangerous` (stateful, requiere `nexo.sessions`) vs JWT stateless. CONTEXT.md sugiere cookie firmada por permitir revocacion inmediata.
- Orden de retrofit de permisos en los 14 routers.
- Estructura minima `nexo/` en Phase 2 vs esperar Phase 3 para repositorios completos.
- Diseno concreto de `/ajustes/usuarios` y `/ajustes/auditoria`.

### Deferred Ideas (OUT OF SCOPE)

- 2FA (TOTP, WebAuthn) — Mark-IV.
- LDAP / Active Directory — Mark-IV.
- Audit filtrado por departamento para directivo — Mark-IV.
- Flujo "olvide mi contrasena" por email — SMTP Out of Scope Mark-III.
- Migracion a Alembic — Mark-IV.
- MCP con write access — descartado.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| IDENT-01 | `/login` funcional con bloqueo 5 intentos → 15 min lock `(user, IP)` | Seccion "Rate limiting + bloqueo progresivo" + tabla `nexo.login_attempts` |
| IDENT-02 | `nexo.users` con hash argon2id, role, departments[], active, last_login, must_change_password | Seccion "Argon2id con argon2-cffi" + modelo de tablas |
| IDENT-03 | `nexo.roles`, `nexo.departments`, `nexo.permissions` | Seccion "Arquitectura RBAC" |
| IDENT-04 | Middleware FastAPI redirige HTML a `/login`, devuelve 401 para `/api/*` no autenticadas | Seccion "Middleware chain" |
| IDENT-05 | `Depends(require_permission(...))` aplicado a los 14 routers | Seccion "Dependency RBAC" |
| IDENT-06 | `nexo.audit_log` append-only + middleware con body sanitizado | Seccion "Audit middleware" + "Postgres REVOKE" |
| IDENT-07 | Panel `/ajustes/auditoria` con filtros + export CSV | Seccion "UI" |
| IDENT-08 | `/ajustes/usuarios` CRUD (solo propietario) | Seccion "UI" |
| IDENT-09 | Primer login obliga cambio de password (flag `must_change_password`) | Seccion "Argon2id" + middleware auth |
| IDENT-10 | `global_exception_handler` sin fuga de traceback (regression check) | Seccion "Gotchas" |
</phase_requirements>

---

## Summary

Phase 2 introduce toda la capa de identidad de Nexo sobre FastAPI 0.135.3 + SQLAlchemy 2.0 + Postgres 16. El patron central es un middleware de auth que corre antes del middleware de audit: auth establece `request.state.user` (o aborta con redirect/401), y audit consume ese estado para escribir en `nexo.audit_log`. Los 14 routers existentes reciben un `Depends(require_permission(...))` como retrofit mecanico; no requieren refactor de logica.

La decision de sesion recomendada es **cookie firmada con `itsdangerous` + tabla `nexo.sessions`**: permite revocacion inmediata (logout forzado, desactivar usuario) que JWT stateless no puede dar sin un deny-list adicional. En LAN-only de bajo volumen, el overhead de la tabla es insignificante.

El nuevo paquete `nexo/` se introduce en su forma minima: `nexo/db/models.py` (tablas SQLAlchemy), `nexo/db/engine.py` (`engine_nexo`), y `nexo/services/auth.py` (toda la logica de auth, RBAC y sesion). El refactor completo de repositorios llega en Phase 3.

**Recomendacion primaria:** cookie firmada con `itsdangerous.URLSafeTimedSerializer` + tabla `nexo.sessions` + parametros Argon2id RFC 9106 LOW_MEMORY (`time_cost=3, memory_cost=65536, parallelism=4`) + `slowapi` para rate limiting por IP en el endpoint `/login`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Verificacion de sesion / cookie | API/Backend (middleware) | — | La cookie llega al servidor; el middleware la verifica antes de que el router ejecute |
| Hash y verificacion de password | API/Backend (service) | — | Nunca en el cliente; Argon2id requiere CPU del servidor |
| Emision y revocacion de sesion | API/Backend (service) | Database/Storage | `auth.py` escribe/borra `nexo.sessions`; la tabla es la fuente de verdad |
| Control de acceso por rol/dept | API/Backend (dependency) | — | `require_permission()` corre en el worker antes del handler |
| Bloqueo progresivo (login) | API/Backend (service) | Database/Storage | Contador en `nexo.login_attempts`; lectura/escritura en cada intento |
| Audit log | API/Backend (middleware) | Database/Storage | Middleware audit escribe fila tras cada request autenticada |
| Login UI / change-password UI | Frontend (Jinja2 + HTMX) | API/Backend | Formularios HTML; submit a endpoints FastAPI |
| Gestion de usuarios (CRUD) | API/Backend (router) | Database/Storage | Solo `propietario`; operaciones sobre `nexo.users` |
| Rate limiting IP en `/login` | API/Backend (decorator) | — | `slowapi` decora el endpoint; contador en memoria (LAN ok) |
| Bootstrap `propietario` | Script Python (offline) | Database/Storage | `scripts/create_propietario.py`; corre fuera de la app |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `argon2-cffi` | `25.1.0` | Hash/verify passwords con Argon2id | Unica biblioteca Python con bindings cffi oficiales para Argon2; OWASP-recomendada [VERIFIED: PyPI] |
| `itsdangerous` | `2.2.0` | Firmar cookies de sesion con HMAC | Usado por Flask, Werkzeug; minimal y battle-tested; ya en el ecosistema Pallets [VERIFIED: PyPI] |
| `slowapi` | `0.1.9` | Rate limiting por IP en FastAPI | Port de flask-limiter para Starlette/FastAPI; decorator-based; no requiere Redis en LAN [VERIFIED: PyPI] |
| `psycopg2-binary` | `2.9.11` | Adaptador Postgres (ya pineado) | Ya en requirements.txt; no anadir nueva dependencia [VERIFIED: codebase] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `python-multipart` | `0.0.26` | Parsear `application/x-www-form-urlencoded` del login form | Ya en requirements.txt; necesario para `OAuth2PasswordRequestForm` o form manual [VERIFIED: codebase] |
| `secrets` (stdlib) | stdlib | Generar session tokens criptograficos | `secrets.token_urlsafe(32)` para session_id; sin dependencia extra [ASSUMED] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `itsdangerous` + DB sessions | JWT stateless (`python-jose` / `PyJWT`) | JWT no permite revocacion sin deny-list adicional; en LAN con logout forzado y desactivacion de usuario, la tabla `nexo.sessions` es obligatoria de todas formas |
| `slowapi` | `fastapi-limiter` (Redis) | `fastapi-limiter` requiere Redis; `slowapi` funciona en memoria para LAN de bajo volumen |
| `slowapi` | Custom counter en `nexo.login_attempts` | La tabla ya existe para el bloqueo `(user, IP)`; puede reusarse para el rate limit por IP puro sin anadir otra lib |

**Installation (paquetes nuevos a anadir a requirements.txt):**

```bash
pip install argon2-cffi==25.1.0 itsdangerous==2.2.0 slowapi==0.1.9
```

**Version verification:**
```
argon2-cffi 25.1.0 — PyPI 2025-06-03 [VERIFIED: pypi.org/project/argon2-cffi/]
itsdangerous 2.2.0 — PyPI 2024-04-16 [VERIFIED: PyPI search 2026-04-18]
slowapi 0.1.9      — PyPI 2024-2025  [VERIFIED: PyPI search 2026-04-18]
```

---

## Architecture Patterns

### System Architecture Diagram

```
Browser
  |
  | HTTP request (cookie: nexo_session=<signed_token>)
  v
Caddy (reverse proxy)
  |
  v
FastAPI ASGI app
  |
  +---> [1] AuthMiddleware
  |       - Extrae cookie → verifica firma HMAC (itsdangerous)
  |       - Consulta nexo.sessions en Postgres: sesion existe, no expirada
  |       - Establece request.state.user = UserContext(id, role, departments)
  |       - Si falla:
  |           - path HTML → redirect 302 /login
  |           - path /api/* → JSONResponse 401
  |       - Whitelist: /login, /api/health (sin verificar)
  |
  +---> [2] AuditMiddleware (solo si request.state.user existe)
  |       - Lee body bytes (pattern receive-replay, FastAPI >= 0.108)
  |       - Sanitiza campos sensibles segun whitelist de endpoints
  |       - Llama call_next(request) → obtiene response
  |       - Escribe INSERT en nexo.audit_log (ts, user_id, ip, method, path, status, details_json)
  |       - Retorna response
  |
  +---> [3] Router
          - Depends(require_permission("modulo:accion"))
              - Lee request.state.user
              - Verifica rol en PERMISSION_MAP dict
              - Para directivo/usuario: interseccion departamentos
              - Si falla: HTTPException 403
          - Logica del endpoint (sin cambios de Phase 1)
```

### Recommended Project Structure

```
nexo/                          # nuevo paquete Phase 2 (minimo)
├── __init__.py
├── db/
│   ├── __init__.py
│   ├── engine.py              # engine_nexo (Postgres) + SessionLocal
│   └── models.py              # ORM: User, Role, Dept, UserDept, Permission,
│                              #       LoginAttempt, Session, AuditLog
└── services/
    ├── __init__.py
    └── auth.py                # hash_password, verify_password, create_session,
                               #  get_session, revoke_session, check_login_attempts,
                               #  record_login_attempt, clear_login_attempts,
                               #  require_permission (dependency factory)

api/
├── middleware/
│   ├── __init__.py
│   ├── auth.py                # AuthMiddleware (BaseHTTPMiddleware)
│   └── audit.py               # AuditMiddleware (BaseHTTPMiddleware)
├── routers/
│   ├── auth.py                # POST /login, POST /logout, GET/POST /cambiar-password
│   ├── usuarios.py            # CRUD /ajustes/usuarios (propietario only)
│   └── auditoria.py           # GET /ajustes/auditoria, GET /ajustes/auditoria/export
│   └── ... (14 routers existentes sin cambios de logica)
├── deps.py                    # anadir get_current_user, current_user_context
└── config.py                  # anadir: secret_key, session_cookie_name, session_ttl_hours

scripts/
└── create_propietario.py      # bootstrap interactivo: email + password

templates/
├── login.html                 # nueva
├── cambiar_password.html      # nueva
├── ajustes_usuarios.html      # nueva
└── ajustes_auditoria.html     # nueva
```

### Pattern 1: Argon2id — Hash, Verify, Rehash

**Parametros LOCKED (de `docs/AUTH_MODEL.md`):** `time_cost=3, memory_cost=65536, parallelism=4`.
Estos coinciden con `argon2.profiles.RFC_9106_LOW_MEMORY` (el default de `PasswordHasher()` desde v21.2.0).

```python
# nexo/services/auth.py
# Source: https://github.com/hynek/argon2-cffi (Context7 + PyPI)
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Los parametros coinciden con RFC_9106_LOW_MEMORY (default de argon2-cffi >= 21.2.0)
# y con lo especificado en docs/AUTH_MODEL.md.
_ph = PasswordHasher(
    time_cost=3,
    memory_cost=65536,   # 64 MiB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)

def hash_password(plaintext: str) -> str:
    return _ph.hash(plaintext)

def verify_password(hashed: str, plaintext: str) -> bool:
    """Returns True si coincide; False si no coincide.
    Raises VerifyMismatchError internamente; aqui la capturamos."""
    try:
        return _ph.verify(hashed, plaintext)
    except VerifyMismatchError:
        return False

def needs_rehash(hashed: str) -> bool:
    """True si el hash fue generado con parametros distintos a los actuales."""
    return _ph.check_needs_rehash(hashed)
```

**Por que RFC_9106_LOW_MEMORY y no OWASP minimo:**
OWASP 2024 recomienda un minimo de `m=19456, t=2, p=1` (19 MiB).
RFC 9106 LOW_MEMORY recomienda `m=65536, t=3, p=4` (64 MiB) — mas fuerte.
AUTH_MODEL.md usa los parametros RFC 9106. Son mas seguros que OWASP minimo y
razonables para el hardware LAN de ECS (i5 7a gen, 16 GB).
[VERIFIED: argon2-cffi docs Context7 + OWASP Cheat Sheet 2026-04-18]

### Pattern 2: Cookie firmada con itsdangerous + tabla nexo.sessions

```python
# nexo/services/auth.py
# Source: https://github.com/pallets/itsdangerous (Context7)
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import secrets
from datetime import datetime, timedelta, timezone

SESSION_COOKIE = "nexo_session"
SESSION_TTL_HOURS = 12

def _serializer(secret_key: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret_key, salt="nexo-session")

def create_session(user_id: int, secret_key: str, db_session) -> str:
    """Crea fila en nexo.sessions; devuelve el token firmado para la cookie."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)
    s = NexoSession(user_id=user_id, token=token, expires_at=expires_at)
    db_session.add(s)
    db_session.commit()
    # Firmamos solo el token (no user_id) para que el servidor valide en DB
    return _serializer(secret_key).dumps(token)

def get_session(signed_cookie: str, secret_key: str, db_session):
    """Valida firma + TTL + DB. Retorna User o None."""
    try:
        # max_age en segundos; protege si la cookie se queda en el navegador
        token = _serializer(secret_key).loads(
            signed_cookie,
            max_age=SESSION_TTL_HOURS * 3600,
        )
    except (BadSignature, SignatureExpired):
        return None
    session = db_session.query(NexoSession).filter_by(token=token).first()
    if not session or session.expires_at < datetime.now(timezone.utc):
        return None
    # Sliding expiration: actualizar expires_at
    session.expires_at = datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)
    db_session.commit()
    return session.user  # relacion ORM

def revoke_session(signed_cookie: str, secret_key: str, db_session) -> None:
    """Logout: borra la fila. Inmediato, sin esperar TTL."""
    try:
        token = _serializer(secret_key).loads(signed_cookie, max_age=SESSION_TTL_HOURS * 3600)
    except Exception:
        return
    db_session.query(NexoSession).filter_by(token=token).delete()
    db_session.commit()
```

**Por que cookie firmada + DB frente a JWT:**
- Revocacion inmediata (logout forzado, desactivar usuario): con JWT puro no es posible sin deny-list.
- En LAN-only, bajo volumen: la tabla `nexo.sessions` no es un cuello de botella.
- `itsdangerous` ya es dependencia transitiva de Jinja2/Werkzeug — no es dependencia nueva en el ecosistema.
- JWT requiere `python-jose` o `PyJWT` y complica el manejo del sliding expiration.
[CITED: itsdangerous docs, itsdangerous.palletsprojects.com]

### Pattern 3: AuthMiddleware — Verificar cookie y establecer request.state.user

```python
# api/middleware/auth.py
# Source: FastAPI docs middleware + Starlette BaseHTTPMiddleware [Context7]
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, JSONResponse

# Rutas que NO requieren autenticacion
_PUBLIC_PATHS = {"/login", "/api/health"}
_PUBLIC_PREFIXES = ("/static/", "/favicon.ico")

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Bypass para rutas publicas
        if path in _PUBLIC_PATHS or path.startswith(_PUBLIC_PREFIXES):
            return await call_next(request)

        cookie = request.cookies.get("nexo_session")
        user = None
        if cookie:
            with SessionLocalNexo() as db:
                user = get_session(cookie, settings.secret_key, db)

        if user is None or not user.active:
            if path.startswith("/api/"):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)
            return RedirectResponse("/login", status_code=302)

        # must_change_password: redirigir si no es la pagina de cambio
        if user.must_change_password and path != "/cambiar-password":
            return RedirectResponse("/cambiar-password", status_code=302)

        request.state.user = user
        return await call_next(request)
```

**Orden de registro en api/main.py (CRITICO — ver Gotchas):**

```python
# api/main.py — registrar ANTES de include_router
from api.middleware.auth import AuthMiddleware
from api.middleware.audit import AuditMiddleware

app.add_middleware(AuditMiddleware)   # se registra segundo, ejecuta segundo
app.add_middleware(AuthMiddleware)    # se registra primero, ejecuta primero
```

Starlette ejecuta middlewares en orden LIFO de `add_middleware`: el ultimo en
registrarse es el primero en ejecutarse.
[VERIFIED: Starlette/FastAPI docs, Context7]

### Pattern 4: Dependency RBAC — require_permission factory

```python
# nexo/services/auth.py
# Source: FastAPI Security dependencies [Context7: /fastapi/fastapi]
from fastapi import HTTPException, Request, status
from functools import lru_cache

# Mapa LOCKED (propietario siempre pasa; otros filtran por departamento)
# Clave: "modulo:accion" — departamentos que tienen acceso
PERMISSION_MAP: dict[str, list[str]] = {
    "pipeline:run":        ["ingenieria", "produccion"],
    "pipeline:read":       ["ingenieria", "produccion", "gerencia"],
    "operarios:read":      ["rrhh"],
    "operarios:export":    ["rrhh"],
    "capacidad:read":      ["comercial", "ingenieria", "produccion", "gerencia"],
    "historial:read":      ["ingenieria", "produccion", "comercial", "gerencia", "rrhh"],
    "bbdd:read":           ["ingenieria"],
    "recursos:read":       ["ingenieria", "produccion"],
    "recursos:edit":       ["ingenieria"],
    "ciclos:read":         ["ingenieria"],
    "ciclos:edit":         ["ingenieria"],
    "centro_mando:read":   ["produccion", "ingenieria", "gerencia"],
    "informes:read":       ["ingenieria", "produccion", "comercial", "gerencia", "rrhh"],
    "luk4:read":           ["produccion", "ingenieria", "gerencia"],
    "datos:read":          ["ingenieria", "produccion"],
    "ajustes:manage":      [],   # solo propietario (gestionado en la funcion)
    "auditoria:read":      [],   # solo propietario en Mark-III
}

def require_permission(permission: str):
    """Factory que devuelve una dependency FastAPI."""
    async def _check(request: Request):
        user = getattr(request.state, "user", None)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        if user.role == "propietario":
            return user  # bypass total
        # directivo y usuario: verificar departamentos
        allowed_depts = PERMISSION_MAP.get(permission, [])
        user_depts = {d.code for d in user.departments}
        if not user_depts.intersection(allowed_depts):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail=f"Permiso requerido: {permission}")
        return user
    return _check

# Uso en routers existentes (retrofit mecanico):
# from nexo.services.auth import require_permission
# from fastapi import Depends
#
# router = APIRouter(dependencies=[Depends(require_permission("pipeline:read"))])
```

### Pattern 5: AuditMiddleware con body sanitization (receive-replay)

```python
# api/middleware/audit.py
# Source: https://sqlpey.com/python/fastapi-body-logging-strategies/
#         + FastAPI 0.108+ body caching behavior
import json
from starlette.middleware.base import BaseHTTPMiddleware

# Campos que NUNCA se graban, por endpoint o global
_REDACTED_ENDPOINTS = {"/api/conexion/config", "/api/conexion/test"}
_SENSITIVE_FIELDS = {"password", "pwd", "secret", "token", "clave", "contrasena"}

class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        user = getattr(request.state, "user", None)
        if user is None:
            return await call_next(request)  # no autenticado: auth lo maneja

        # Leer body SIN consumir el stream (FastAPI >= 0.108 cachea en request._body)
        # Despues de await request.body(), call_next puede seguir leyendo.
        body_bytes = await request.body()
        details = None

        if request.method in ("POST", "PUT", "PATCH"):
            path = request.url.path
            if path in _REDACTED_ENDPOINTS:
                details = None  # endpoint sensible: no grabar nada
            else:
                try:
                    body_dict = json.loads(body_bytes)
                    # Sanitizar campos sensibles
                    sanitized = {
                        k: "[REDACTED]" if k.lower() in _SENSITIVE_FIELDS else v
                        for k, v in body_dict.items()
                    }
                    details = json.dumps(sanitized)[:4096]  # limitar tamano
                except Exception:
                    details = None

        response = await call_next(request)

        # Escribir en nexo.audit_log (INSERT solo — rol app no tiene UPDATE/DELETE)
        try:
            with SessionLocalNexo() as db:
                db.add(AuditLog(
                    ts=datetime.now(timezone.utc),
                    user_id=user.id,
                    ip=request.client.host if request.client else "unknown",
                    method=request.method,
                    path=request.url.path,
                    status=response.status_code,
                    details_json=details,
                ))
                db.commit()
        except Exception:
            logger.exception("Error escribiendo audit_log")

        return response
```

**CRITICO — body en middleware:** En FastAPI >= 0.108 (el proyecto usa 0.135.3),
`await request.body()` cachea el contenido en `request._body`. La siguiente llamada
a `call_next(request)` puede releer el body normalmente. En versiones anteriores esto
consumia el stream y causaba errores en los handlers.
[VERIFIED: FastAPI changelog + GitHub discussions 8187 + sqlpey.com 2026-04-18]

### Pattern 6: engine_nexo (segundo engine, coexistencia con engine_app)

```python
# nexo/db/engine.py
# Source: SQLAlchemy 2.0 docs [Context7: /websites/sqlalchemy_en_20]
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.config import settings

engine_nexo = create_engine(
    f"postgresql+psycopg2://{settings.pg_user}:{settings.pg_password}"
    f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}",
    # Pool recomendado para low-concurrency LAN Postgres 16
    pool_size=5,           # conexiones persistentes
    max_overflow=5,        # overflow maximo (total: 10)
    pool_timeout=10,       # seg esperando conexion libre
    pool_recycle=1800,     # reciclar conexiones cada 30 min (evita timeout Postgres)
    pool_pre_ping=True,    # verificar conexion antes de usar (detecta drops)
    echo=settings.debug,   # SQL log solo en modo debug
)

SessionLocalNexo = sessionmaker(bind=engine_nexo, expire_on_commit=False)
```

SQLAlchemy 2.0 soporta multiples engines en la misma aplicacion sin conflicto.
`engine_app` (SQL Server, `api/database.py`) y `engine_nexo` (Postgres) son
completamente independientes: pools separados, conexiones separadas.
[VERIFIED: SQLAlchemy 2.0 docs Context7 + docs.sqlalchemy.org]

### Pattern 7: Schema nexo — modelos ORM con __table_args__ schema

```python
# nexo/db/models.py
# Source: SQLAlchemy 2.0 ORM [Context7: /websites/sqlalchemy_en_20]
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Table
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime, timezone

class NexoBase(DeclarativeBase):
    pass

# Tabla de asociacion N:M users <-> departments
user_departments = Table(
    "user_departments",
    NexoBase.metadata,
    Column("user_id", Integer, ForeignKey("nexo.users.id"), primary_key=True),
    Column("department_id", Integer, ForeignKey("nexo.departments.id"), primary_key=True),
    schema="nexo",
)

class NexoUser(NexoBase):
    __tablename__ = "users"
    __table_args__ = {"schema": "nexo"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(200), nullable=False, unique=True, index=True)
    password_hash = Column(String(200), nullable=False)
    role = Column(String(20), nullable=False)  # propietario | directivo | usuario
    active = Column(Boolean, default=True, nullable=False)
    must_change_password = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    departments = relationship("NexoDepartment", secondary=user_departments, back_populates="users")
    sessions = relationship("NexoSession", back_populates="user", cascade="all, delete-orphan")

class NexoDepartment(NexoBase):
    __tablename__ = "departments"
    __table_args__ = {"schema": "nexo"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(20), nullable=False, unique=True)  # rrhh | comercial | ...
    name = Column(String(100), nullable=False)
    users = relationship("NexoUser", secondary=user_departments, back_populates="departments")

class NexoSession(NexoBase):
    __tablename__ = "sessions"
    __table_args__ = {"schema": "nexo"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("nexo.users.id"), nullable=False)
    token = Column(String(64), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    user = relationship("NexoUser", back_populates="sessions")

class NexoLoginAttempt(NexoBase):
    __tablename__ = "login_attempts"
    __table_args__ = {"schema": "nexo"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(200), nullable=False)
    ip = Column(String(45), nullable=False)   # IPv4 o IPv6
    failed_at = Column(DateTime(timezone=True), nullable=False)

class NexoAuditLog(NexoBase):
    __tablename__ = "audit_log"
    __table_args__ = {"schema": "nexo"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("nexo.users.id"), nullable=True)  # nullable: logs pre-auth
    ip = Column(String(45))
    method = Column(String(10))
    path = Column(String(500))
    status = Column(Integer)
    details_json = Column(Text, nullable=True)
```

### Pattern 8: Init schema + REVOKE (append-only audit_log)

```python
# scripts/init_nexo_schema.py
# Source: PostgreSQL REVOKE docs + SQLAlchemy DDL [CITED: postgresql.org/docs/current/sql-revoke.html]
from sqlalchemy import text
from nexo.db.engine import engine_nexo
from nexo.db.models import NexoBase

def init_nexo_schema():
    with engine_nexo.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS nexo"))
        conn.commit()

    NexoBase.metadata.create_all(engine_nexo)

    # Hacer audit_log append-only para el rol app
    # El rol de la app (pg_user de settings) solo puede INSERT y SELECT.
    # Solo un rol DBA puede UPDATE o DELETE.
    with engine_nexo.connect() as conn:
        app_role = settings.pg_user  # e.g. "nexo_app"
        conn.execute(text(
            f"REVOKE UPDATE, DELETE ON nexo.audit_log FROM {app_role}"
        ))
        conn.commit()

    # Seed departamentos
    with SessionLocalNexo() as db:
        _seed_departments(db)
        _seed_roles(db)
```

**Nota:** `REVOKE UPDATE, DELETE` requiere que el rol `pg_user` tenga esos privilegios
previamente concedidos (por defecto en Postgres el owner tiene todos; los usuarios sin
ser owner necesitan GRANT explicito). Si el rol es el owner, el REVOKE surte efecto.
Si no, mejor otorgar solo INSERT + SELECT al rol app en vez de revocar.

Alternativa mas robusta (evita ambiguedad):
```sql
-- En el script de init como superuser:
GRANT SELECT, INSERT ON nexo.audit_log TO nexo_app;
-- Sin GRANT de UPDATE/DELETE, el rol no los tiene.
```

[CITED: postgresql.org/docs/current/sql-revoke.html]

### Pattern 9: Rate limiting por IP en /login

```python
# api/routers/auth.py
# Source: slowapi docs [Context7: /laurents/slowapi]
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

limiter = Limiter(key_func=get_remote_address)
# Registro en main.py:
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@router.post("/login")
@limiter.limit("20/minute")      # rate limit por IP (distribuido)
async def login(request: Request, form_data: LoginForm):
    # ... logica de bloqueo (user, IP) de nexo.login_attempts
    ...
```

**Bloqueo progresivo (IDENT-01) — implementacion con nexo.login_attempts:**

```python
# nexo/services/auth.py
LOCKOUT_THRESHOLD = 5
LOCKOUT_MINUTES = 15

def check_lockout(email: str, ip: str, db) -> bool:
    """True si la tupla (email, IP) esta bloqueada."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=LOCKOUT_MINUTES)
    count = db.query(NexoLoginAttempt).filter(
        NexoLoginAttempt.email == email,
        NexoLoginAttempt.ip == ip,
        NexoLoginAttempt.failed_at >= cutoff,
    ).count()
    return count >= LOCKOUT_THRESHOLD

def record_failed_attempt(email: str, ip: str, db) -> None:
    db.add(NexoLoginAttempt(email=email, ip=ip,
                            failed_at=datetime.now(timezone.utc)))
    db.commit()

def clear_attempts(email: str, ip: str, db) -> None:
    """Purgar al login exitoso."""
    db.query(NexoLoginAttempt).filter(
        NexoLoginAttempt.email == email,
        NexoLoginAttempt.ip == ip,
    ).delete()
    db.commit()
```

### Pattern 10: Inyectar current_user en templates Jinja2

```python
# api/routers/auth.py y cualquier router que renderice HTML
# Pattern: pasar current_user en el contexto de TemplateResponse
# Source: FastAPI docs templates [CITED: fastapi.tiangolo.com/advanced/templates/]
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

@router.get("/ajustes/usuarios", response_class=HTMLResponse)
async def ajustes_usuarios(request: Request):
    user = request.state.user   # ya establecido por AuthMiddleware
    return templates.TemplateResponse(
        "ajustes_usuarios.html",
        {"request": request, "current_user": user},
    )
```

**Para evitar repetir `current_user` en cada router**, el patron correcto en FastAPI es
usar `templates.env.globals` para variables estaticas (como ya hace `api/deps.py`),
pero `current_user` es dinamico por request. La solucion idiomatica en FastAPI es
pasarlo en el contexto de cada `TemplateResponse`. No existe un hook "context processor"
nativo en Jinja2Templates de FastAPI equivalente a Django.

**Alternativa (wrapper helper):**

```python
# api/deps.py — ampliar con helper
def render(template_name: str, request: Request, extra: dict = None) -> HTMLResponse:
    """Wrapper que siempre inyecta current_user en el contexto."""
    ctx = {"request": request, "current_user": getattr(request.state, "user", None)}
    if extra:
        ctx.update(extra)
    return templates.TemplateResponse(template_name, ctx)
```

Los routers lo llaman con `return render("ajustes.html", request, {"items": items})`.
[CITED: fastapi.tiangolo.com/advanced/templates/ + Jinja2 docs]

### Pattern 11: Bootstrap del primer propietario

```python
# scripts/create_propietario.py
# Pattern: getpass + idempotencia [ASSUMED — stdlib]
import getpass
import sys
sys.path.insert(0, ".")

from nexo.db.engine import engine_nexo, SessionLocalNexo
from nexo.db.models import NexoUser, NexoDepartment
from nexo.services.auth import hash_password

def main():
    with SessionLocalNexo() as db:
        if db.query(NexoUser).filter_by(role="propietario").count() > 0:
            print("Ya existe un usuario con rol propietario. Saliendo.")
            return

    email = input("Email del propietario: ").strip()
    pwd1 = getpass.getpass("Password (min 12 chars): ")
    pwd2 = getpass.getpass("Repetir password: ")

    if pwd1 != pwd2:
        print("Las contrasenas no coinciden.")
        sys.exit(1)
    if len(pwd1) < 12:
        print("La contrasena debe tener al menos 12 caracteres.")
        sys.exit(1)

    with SessionLocalNexo() as db:
        user = NexoUser(
            email=email,
            password_hash=hash_password(pwd1),
            role="propietario",
            active=True,
            must_change_password=False,   # propietario inicial no necesita cambio
        )
        db.add(user)
        db.commit()
        print(f"Propietario creado: {email}")

if __name__ == "__main__":
    main()
```

**Ejecucion:**

```bash
docker compose exec web python scripts/create_propietario.py
```

### Anti-Patterns to Avoid

- **JWT stateless sin deny-list:** No permite logout forzado ni desactivacion de usuario con efecto inmediato.
- **Registrar middlewares en orden incorrecto:** `add_middleware` usa LIFO; registrar audit antes de auth hace que audit se ejecute con `request.state.user = None`.
- **Leer `request.body()` en middleware con FastAPI < 0.108:** Consume el stream; los handlers reciben body vacio. El proyecto usa 0.135.3 — seguro.
- **Grabar `details_json` en endpoints de configuracion SQL:** `/api/conexion/config` acepta passwords de SQL Server. Whitelist explicita es obligatoria.
- **Usar `datetime.utcnow()` en `audit_log.ts`:** `utcnow()` genera datetimes naive. Usar `datetime.now(timezone.utc)` para datetimes timezone-aware en todas las columnas Postgres con `DateTime(timezone=True)`.
- **`propietario` en PERMISSION_MAP:** El propietario debe hacer bypass ANTES de consultar el mapa. Si se incluye en el mapa, un cambio posterior podria bloquearlo accidentalmente.
- **Parametros Argon2id en la request:** Los parametros (`time_cost`, etc.) van codificados en el hash en formato PHC. No hace falta almacenarlos por separado ni pasarlos al verificar.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hash de passwords | Funcion SHA256/bcrypt propia | `argon2-cffi` PasswordHasher | Argon2id tiene resistencia a GPU, side-channel protections; SHA256 para passwords es inseguro |
| Firma de cookies | HMAC manual | `itsdangerous.URLSafeTimedSerializer` | Gestiona padding, encoding, timing-safe comparison, rotacion de claves |
| Rate limiting por IP | Contador en memoria propio | `slowapi` | Gestiona ventanas de tiempo, TTL, excepciones, integracion con Starlette |
| Validacion de password minimo 12 chars | Regex custom | Validacion simple `len(pwd) >= 12` | No hay complejidad adicional (decision LOCKED en AUTH_MODEL.md) |
| Generacion de session tokens | `random.token_hex()` | `secrets.token_urlsafe(32)` | `secrets` usa OS random (CSPRNG); `random` no es criptografico |

**Key insight:** El unico componente sin libreria especializada es el bloqueo `(user, IP)`: se implementa directamente sobre `nexo.login_attempts` porque ya existe la tabla y la logica es simple (COUNT + timedelta). No hay libreria de "login throttling" que justifique la dependencia.

---

## Common Pitfalls

### Pitfall 1: Orden de middlewares (LIFO de add_middleware)

**What goes wrong:** Se registra `AuthMiddleware` despues de `AuditMiddleware`. Starlette ejecuta LIFO, asi que `AuditMiddleware` corre primero, cuando `request.state.user` aun no existe. El audit log escribe `user_id=None` para todas las requests.

**Why it happens:** `app.add_middleware()` usa una pila interna; el ultimo en anadirse es el primero en ejecutarse.

**How to avoid:** Registrar siempre en este orden exacto:
```python
app.add_middleware(AuditMiddleware)    # segundo en registrarse → segundo en ejecutar
app.add_middleware(AuthMiddleware)     # primero en registrarse → primero en ejecutar
```

**Warning signs:** Todas las filas de `audit_log` tienen `user_id=NULL`; el panel de auditoria muestra usuarios vacios.

### Pitfall 2: Body de request consumido en middleware

**What goes wrong:** El middleware llama `await request.body()` y despues `call_next(request)`. En versiones antiguas de Starlette, el stream se ha consumido y los handlers de FastAPI reciben un body vacio, causando errores de parsing.

**Why it happens:** HTTP bodies son streams de un solo uso.

**How to avoid:** El proyecto usa FastAPI 0.135.3 (>= 0.108): `await request.body()` cachea en `request._body`. `call_next` puede leer el body normalmente despues. No requiere workaround adicional.

**Warning signs:** Errores `422 Unprocessable Entity` en endpoints POST/PUT tras anadir el middleware; body llega vacio a los handlers.

### Pitfall 3: datetime.utcnow() vs datetime.now(timezone.utc)

**What goes wrong:** Usar `datetime.utcnow()` (deprecated en Python 3.12) genera objetos `datetime` naive (sin tzinfo). Postgres con columna `TIMESTAMP WITH TIME ZONE` convierte los valores, pero la comparacion en Python entre naive y aware lanza `TypeError`.

**Why it happens:** Python 3.12+ depreca `datetime.utcnow()`; SQLAlchemy `DateTime(timezone=True)` espera datetimes timezone-aware.

**How to avoid:** Usar siempre `datetime.now(timezone.utc)` en todo el codigo de `nexo/`. Nunca `datetime.utcnow()`.

**Warning signs:** `TypeError: can't compare offset-naive and offset-aware datetimes` al verificar sesiones expiradas o al comparar fechas en `login_attempts`.

### Pitfall 4: REVOKE sin contexto de privilegios previos

**What goes wrong:** El script ejecuta `REVOKE UPDATE, DELETE ON nexo.audit_log FROM nexo_app`. Si `nexo_app` es el owner de la tabla (creada con su cuenta), el REVOKE puede fallar o no tener el efecto esperado — Postgres permite al owner recuperar sus privilegios.

**Why it happens:** En Postgres, el owner tiene privilegios implicitos que REVOKE no puede eliminar permanentemente.

**How to avoid:** Usar un superuser para crear el schema y las tablas, y otorgar al rol de la app solo los privilegios necesarios (GRANT SELECT, INSERT ON nexo.audit_log TO nexo_app). Sin GRANT de UPDATE/DELETE, el rol no los tiene. Ver Pattern 8.

**Warning signs:** El `DELETE` desde el rol app tiene exito en lugar de fallar — el test IDENT-06 falla.

### Pitfall 5: engine_nexo accedido sin schema en Postgres

**What goes wrong:** Los modelos definen `__table_args__ = {"schema": "nexo"}` pero el script de init no crea el schema antes de `create_all()`. Postgres devuelve `ERROR: schema "nexo" does not exist`.

**Why it happens:** `NexoBase.metadata.create_all(engine_nexo)` no crea schemas automaticamente.

**How to avoid:** `CREATE SCHEMA IF NOT EXISTS nexo` debe ejecutarse ANTES de `create_all()`. Ver Pattern 8.

**Warning signs:** `sqlalchemy.exc.ProgrammingError: schema "nexo" does not exist` al correr el script de init.

### Pitfall 6: Cookie sin SameSite=Lax en HTMX

**What goes wrong:** Cookie con `SameSite=Strict` bloquea requests HTMX que vienen de navegacion directa (GET de primera carga), causando que el usuario aparezca como no autenticado en ciertos flows.

**Why it happens:** `SameSite=Strict` bloquea cookies en requests cross-site y en navigaciones top-level.

**How to avoid:** Usar `SameSite=Lax` (especificado en CONTEXT.md). `Lax` permite cookies en GET top-level y en requests HTMX del mismo origen.

**Warning signs:** Login exitoso pero la siguiente pagina muestra `/login` de nuevo.

### Pitfall 7: PERMISSION_MAP incompleto en retrofit de 14 routers

**What goes wrong:** Se olvida un router al hacer el retrofit de `require_permission`. El router queda sin proteccion. `propietario` accede sin problema (bypass), pero `usuario` o `directivo` tampoco son rechazados porque no hay `Depends`.

**Why it happens:** El retrofit es mecanico sobre 14 archivos; facil de omitir uno.

**How to avoid:** Al final del retrofit, verificar con un test de integracion que `/api/health` sin cookie devuelve 401 y cada endpoint protegido devuelve 403 a un usuario sin permiso.

**Warning signs:** Test IDENT-05 (IDENT-10 regression) falla; `curl /api/bbdd` sin cookie devuelve 200.

---

## Stack Decisions

### Decision 1: Cookie firmada (itsdangerous + nexo.sessions) — RECOMENDADO

**Contexto:** Claude's Discretion en CONTEXT.md.

**Recomendacion:** Cookie firmada con `itsdangerous.URLSafeTimedSerializer` + tabla `nexo.sessions`.

| Aspecto | Cookie firmada + DB | JWT stateless |
|---------|--------------------|--------------:|
| Revocacion inmediata | SI (borra fila) | NO (necesita deny-list) |
| Logout forzado (admin desactiva usuario) | SI | NO |
| Sliding expiration | Simple (UPDATE expires_at) | Requiere emitir nuevo token |
| Overhead DB | 1 query por request (SELECT session) | 0 queries (stateless) |
| Implementacion | itsdangerous + SQLAlchemy | PyJWT + logica denial |
| Adecuado para LAN low-volume | SI — el SELECT es trivial | SI, pero con limitaciones |

**Veredicto:** En LAN-only con <20 usuarios concurrentes, el overhead de 1 SELECT por request es completamente despreciable. La revocacion inmediata y el logout forzado son requisitos funcionales reales (propietario puede desactivar un usuario). JWT sin deny-list no los cumple.

### Decision 2: PERMISSION_MAP en codigo vs tabla nexo.permissions

**Contexto:** IDENT-03 menciona `nexo.permissions` pero CONTEXT.md deja abierta la implementacion.

**Recomendacion:** `PERMISSION_MAP` como dict en `nexo/services/auth.py` para Phase 2. Tabla `nexo.permissions` puede existir como catalogo de referencia, pero la logica de permisos efectiva esta en codigo.

**Justificacion:**
- Los 3 roles y 5 departamentos son LOCKED hasta Mark-IV. No hay UI para editar permisos.
- Una tabla de permisos sin UI de gestion solo anade una query extra sin valor.
- Phase 3 puede migrar a la tabla si el planner lo decide; el dict facilita ese refactor.

### Decision 3: Estructura nexo/ en Phase 2 (minima)

**Recomendacion:** Crear solo `nexo/db/` y `nexo/services/auth.py`. No crear `nexo/data/repositories/` — ese es el trabajo de Phase 3 (DATA-04).

**Justificacion:** Phase 3 introduce el patron completo de repositorios. Crearlo ahora parcialmente genera deuda de inconsistencia. El minimo necesario para Phase 2 son: modelos ORM, engine, y logica de auth.

---

## Out of Scope para Phase 2

- Repositorios completos (`nexo/data/repositories/`) — Phase 3.
- Filtros de auditoria por departamento para `directivo` — Mark-IV (IDENT-07 solo propietario en Mark-III).
- UI condicionada por rol en sidebar y nav — Phase 5 (UIROL-01/02).
- Split de `ajustes.html` en multiples sub-paginas — Phase 5 (UIROL-03).
- 2FA, LDAP, recuperacion de password por email — Mark-IV / Out of Scope.
- Alembic para migraciones — Mark-IV.
- `engine_mes` refactor — Phase 3.
- schema_guard en lifespan — Phase 3 (DATA-06).
- `nexo.query_log`, `nexo.query_thresholds` — Phase 4.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PostgreSQL 16 | engine_nexo, nexo.* tables | Via Docker | 16-alpine (docker-compose.yml) | — |
| psycopg2-binary | SQLAlchemy Postgres adapter | En requirements.txt | 2.9.11 | — |
| argon2-cffi | Hash passwords | NO (no esta en requirements.txt) | 25.1.0 a anadir | — |
| itsdangerous | Firma de cookies | NO (no esta en requirements.txt) | 2.2.0 a anadir | — |
| slowapi | Rate limiting IP | NO (no esta en requirements.txt) | 0.1.9 a anadir | Contador manual en DB |
| python-multipart | Parsear login form | En requirements.txt | 0.0.26 | — |
| secrets (stdlib) | Generar session tokens | stdlib Python 3.11 | stdlib | — |

**Missing dependencies con no fallback:**
- `argon2-cffi==25.1.0` — BLOQUEANTE para IDENT-02. Anadir a requirements.txt.
- `itsdangerous==2.2.0` — BLOQUEANTE para IDENT-04 (sesion). Anadir a requirements.txt.

**Missing dependencies con fallback:**
- `slowapi==0.1.9` — IDENT-01 rate limit por IP. Fallback: implementar contador por IP en `nexo.login_attempts` sin slowapi (mas codigo, menos garantias de sliding window). Recomendado anadir slowapi.

**Campos necesarios en api/config.py (nuevos):**
```python
secret_key: str = Field(..., validation_alias=AliasChoices("NEXO_SECRET_KEY"))
session_cookie_name: str = Field("nexo_session", ...)
session_ttl_hours: int = Field(12, ...)
pg_host: str = Field("db", validation_alias=AliasChoices("NEXO_PG_HOST"))
pg_port: int = Field(5432, validation_alias=AliasChoices("NEXO_PG_PORT"))
```

`NEXO_SECRET_KEY` debe estar en `.env` (y en `.env.example` sin valor). Si falta, FastAPI falla al arrancar con `ValidationError` — correcto, es una configuracion critica de seguridad.

---

## Validation Architecture

El proyecto no tiene `workflow.nyquist_validation` configurado en `.planning/config.json` (ausente = habilitado).

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (en requirements-dev.txt) |
| Config file | No detectado — crear `pytest.ini` o `pyproject.toml [tool.pytest]` en Wave 0 |
| Quick run command | `pytest tests/auth/ -x -q` |
| Full suite command | `pytest tests/ --cov=nexo --cov=api --cov-report=term-missing` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| IDENT-01 | Login bloquea tras 5 intentos fallidos | integration | `pytest tests/auth/test_login.py::test_lockout -x` | No — Wave 0 |
| IDENT-01 | Login exitoso limpia intentos | unit | `pytest tests/auth/test_auth_service.py::test_clear_attempts -x` | No — Wave 0 |
| IDENT-02 | hash_password retorna hash Argon2id | unit | `pytest tests/auth/test_auth_service.py::test_hash_password -x` | No — Wave 0 |
| IDENT-02 | verify_password correcto/incorrecto | unit | `pytest tests/auth/test_auth_service.py::test_verify_password -x` | No — Wave 0 |
| IDENT-04 | Request sin cookie a HTML devuelve 302 /login | integration | `pytest tests/auth/test_middleware.py::test_redirect_unauthenticated -x` | No — Wave 0 |
| IDENT-04 | Request sin cookie a /api/* devuelve 401 | integration | `pytest tests/auth/test_middleware.py::test_401_api -x` | No — Wave 0 |
| IDENT-05 | usuario sin departamento recibe 403 | integration | `pytest tests/auth/test_rbac.py::test_forbidden_wrong_dept -x` | No — Wave 0 |
| IDENT-05 | propietario pasa require_permission siempre | unit | `pytest tests/auth/test_rbac.py::test_owner_bypass -x` | No — Wave 0 |
| IDENT-06 | DELETE en audit_log falla con permission error | integration | `pytest tests/auth/test_audit.py::test_audit_append_only -x` | No — Wave 0 |
| IDENT-09 | must_change_password redirige a /cambiar-password | integration | `pytest tests/auth/test_middleware.py::test_must_change_password -x` | No — Wave 0 |
| IDENT-10 | 500 devuelve error_id sin traceback (regression) | integration | `pytest tests/test_exception_handler.py -x` | No — Wave 0 |

### Wave 0 Gaps

- [ ] `tests/auth/__init__.py`
- [ ] `tests/auth/test_auth_service.py` — unit tests para `nexo/services/auth.py`
- [ ] `tests/auth/test_login.py` — bloqueo progresivo, sliding expiration
- [ ] `tests/auth/test_middleware.py` — auth middleware, redirects, 401
- [ ] `tests/auth/test_rbac.py` — require_permission, departamentos
- [ ] `tests/auth/test_audit.py` — escritura en audit_log, append-only
- [ ] `tests/conftest.py` — fixtures: Postgres test DB, client FastAPI TestClient, usuario propietario/usuario de test
- [ ] Framework config: `pytest.ini` o `[tool.pytest.ini_options]` en `pyproject.toml`

---

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | YES | argon2-cffi PasswordHasher + bloqueo progresivo |
| V3 Session Management | YES | itsdangerous + nexo.sessions + HttpOnly/Secure/SameSite=Lax |
| V4 Access Control | YES | require_permission() dependency + RBAC PERMISSION_MAP |
| V5 Input Validation | YES | Pydantic models en endpoints login/usuarios; len(password) >= 12 |
| V6 Cryptography | YES — parcial | itsdangerous HMAC para cookies; argon2id para passwords; NO se implementa cifrado de campos en Phase 2 |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Credential stuffing / brute force | Elevation of Privilege | Bloqueo (user, IP) 5 → 15 min + slowapi rate limit por IP |
| Session hijacking | Spoofing | Cookie HttpOnly + Secure + SameSite=Lax; firma HMAC con itsdangerous |
| Session fixation | Elevation of Privilege | Generar nuevo token en cada login (secrets.token_urlsafe) |
| Privilege escalation horizontal | Elevation of Privilege | require_permission verifica departamentos del user en DB |
| Audit log tampering | Tampering | REVOKE UPDATE, DELETE en nexo.audit_log |
| Password exposure en audit log | Information Disclosure | Whitelist de endpoints sensibles; campos `_SENSITIVE_FIELDS` redactados |
| Traceback leakage via 500 | Information Disclosure | global_exception_handler ya cerrado (Sprint 0 NAMING-07); regression test IDENT-10 |
| CSRF en forms de login | Tampering | SameSite=Lax protege en LAN-only; sin CORS cross-origin; nivel de riesgo bajo en intranet |
| SQL injection en queries de auth | Tampering | SQLAlchemy ORM con parametros binding; nunca interpolacion de strings en queries |

---

## Open Questions for Planner

1. **health endpoint — requiere auth o es publico?**
   - CONTEXT.md: "excepto `/api/health`, decision a tomar en Phase 2".
   - Actualmente `/api/health` consulta SQL Server y Postgres. Si requiere auth, un check de CI/CD externo necesitaria credenciales.
   - Recomendacion: anadir a `_PUBLIC_PATHS` del middleware. El health check no expone datos sensibles.

2. **Que rol app de Postgres usar para `engine_nexo`?**
   - Si `pg_user` (`settings.pg_user`, actualmente `"oee"`) es el OWNER de las tablas `nexo.*`, el REVOKE de UPDATE/DELETE en `nexo.audit_log` no tiene efecto permanente (el owner puede recuperar sus privilegios).
   - Alternativa: crear un rol Postgres dedicado `nexo_app` con solo `SELECT, INSERT` en `nexo.audit_log`. Requiere un paso extra en el script de init como superuser.
   - Decision: el planner debe elegir entre (a) usar el owner y asumir que "nadie tiene acceso al psql en prod" o (b) separar roles Postgres ahora.

3. **Paginacion en /ajustes/auditoria?**
   - El audit log puede crecer rapido (una fila por request). Sin paginacion, la query devuelve todo.
   - Recomendacion: LIMIT + OFFSET con parametros de query (`?page=1&limit=100`). El planner debe decidir si paginar en servidor o en frontend (tabla Alpine).

4. **NEXO_SECRET_KEY: generacion y rotacion?**
   - Si la `secret_key` cambia, todas las cookies firmadas actuales se invalidan (logout masivo).
   - El script de init podria generar y escribir la clave en `.env` si no existe. El planner debe decidir el flujo de generacion inicial.

5. **Orden de retrofit de los 14 routers?**
   - Sugerencia (CONTEXT.md, Claude's Discretion): empezar por los mas simples (`health`, `centro_mando` readonly) para validar el Depends antes de aplicarlo a `pipeline`, `bbdd`, etc.
   - El planner puede definir el orden como parte del PLAN.md de tareas.

6. **`nexo.permissions` como tabla seed o como dict puro?**
   - Si existe como tabla con seed, Phase 5 (UIROL) puede leerla para construir el nav filtrado.
   - Si es solo un dict en codigo, Phase 5 necesita duplicar el mapeo o importar desde `nexo/services/auth.py`.
   - Recomendacion investigada: mantener el dict en codigo (Phase 2) pero seedear la tabla `nexo.permissions` como catalogo (para que Phase 5 pueda consultarla sin imports cruzados). Decision final al planner.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `slowapi 0.1.9` funciona con FastAPI 0.135.3 sin conflictos de dependencias | Standard Stack | Habria que implementar rate limit manual sobre `nexo.login_attempts` |
| A2 | `secrets.token_urlsafe` (stdlib) es el patron estandar para session tokens | Pattern 2 | Sin riesgo — stdlib no cambia |
| A3 | El rol Postgres del docker-compose actual (`oee`) es el owner de las tablas que crea | Pattern 8 | El REVOKE no surte efecto; el test IDENT-06 fallaria |
| A4 | FastAPI 0.135.3 hereda el comportamiento de caching de `request.body()` >= 0.108 | Pattern 5 | El body del audit middleware consumiria el stream; errores 422 en handlers |

**Si la tabla de Assumptions esta vacia:** No aplica — hay 4 asunciones identificadas; A3 y A4 son las de mayor riesgo.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `bcrypt` para passwords | `argon2id` via `argon2-cffi` | 2015+ (PHC ganador) | Argon2id resiste ataques GPU y side-channel mejor que bcrypt |
| JWT stateless para todos los casos | Cookie firmada + DB para casos con revocacion | 2020+ (dependiendo del caso) | JWT sigue siendo valido para APIs sin estado; para sesiones de usuario con logout forzado, DB es correcto |
| `@app.middleware("http")` decorador | `BaseHTTPMiddleware` (Starlette) | Desde Starlette 0.12 | `BaseHTTPMiddleware` es la clase recomendada; el decorador es un alias |
| `datetime.utcnow()` | `datetime.now(timezone.utc)` | Python 3.12 deprecation | `utcnow()` deprecated; usar aware datetimes siempre |

**Deprecated/outdated:**
- `datetime.utcnow()`: deprecated en Python 3.12; usar `datetime.now(timezone.utc)`.
- `OAuth2PasswordBearer` con JWT: valido para APIs puras, pero en Nexo (web app con sesiones, HTMX, logout) la cookie firmada es mas adecuada.

---

## Sources

### Primary (HIGH confidence)

- `Context7 /hynek/argon2-cffi` — PasswordHasher API, hash/verify/check_needs_rehash, RFC_9106_LOW_MEMORY profile values
- `Context7 /fastapi/fastapi` — Middleware, Security(), dependencies, cookie parameters, TemplateResponse
- `Context7 /pallets/itsdangerous` — URLSafeTimedSerializer dumps/loads/max_age
- `Context7 /laurents/slowapi` — Limiter setup, @limiter.limit decorator, FastAPI integration
- `Context7 /websites/sqlalchemy_en_20` — create_engine, pool settings, schema, DDL
- `https://github.com/hynek/argon2-cffi/blob/main/src/argon2/profiles.py` — RFC_9106_LOW_MEMORY = {time_cost=3, memory_cost=65536, parallelism=4}
- `https://pypi.org/project/argon2-cffi/` — version 25.1.0, released 2025-06-03
- `https://pypi.org/project/itsdangerous/` — version 2.2.0, released 2024-04-16

### Secondary (MEDIUM confidence)

- `https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html` — OWASP 2024 Argon2id minimum parameters (m=19456, t=2, p=1); el proyecto usa RFC 9106 LOW_MEMORY que es mas fuerte
- `https://sqlpey.com/python/fastapi-body-logging-strategies/` — body capture patterns en middleware, FastAPI >= 0.108 caching behavior
- `https://www.postgresql.org/docs/current/sql-revoke.html` — REVOKE syntax y semantica con owners

### Tertiary (LOW confidence)

- WebSearch "FastAPI middleware read request body 2024" — confirma que FastAPI >= 0.108 resuelve el problema del stream consumption; verificado con version real del proyecto (0.135.3)

---

## Metadata

**Confidence breakdown:**
- Standard stack (argon2-cffi, itsdangerous, slowapi): HIGH — versiones verificadas en PyPI; APIs verificadas en Context7
- Argon2id parametros: HIGH — RFC_9106_LOW_MEMORY verificado en codigo fuente argon2-cffi; coincide con AUTH_MODEL.md
- OWASP parametros: MEDIUM — OWASP recommends m=19456,t=2,p=1 minimum; AUTH_MODEL.md usa RFC 9106 que es mas fuerte; ambos son validos
- Middleware body pattern: HIGH — FastAPI 0.135.3 en requirements.txt; >= 0.108 confirmado
- PostgreSQL REVOKE semantica: MEDIUM — documentacion oficial verificada; gotcha del owner anotada como A3
- Arquitectura patterns: HIGH — derivados directamente del codigo existente y docs oficiales

**Research date:** 2026-04-18
**Valid until:** 2026-07-18 (stack estable; argon2-cffi y itsdangerous son librerias maduras de baja frecuencia de cambio)
