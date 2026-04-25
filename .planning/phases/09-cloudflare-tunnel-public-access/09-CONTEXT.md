# Phase 9: Cloudflare Tunnel + Public Access — Context

**Gathered:** 2026-04-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Nexo accesible desde **cualquier dispositivo (móvil, portátil, tablet) con un
link en el navegador**, sin VPN, sin app que instalar y sin tocar `/etc/hosts`.
Acceso limitado a una **allowlist explícita de emails** mantenida por
Erik Eguskiza (IT lead ECS Mobility) en el panel de Cloudflare Zero Trust.
La capa de auth de Nexo (argon2id + lockout 5/15min + audit append-only) se
preserva intacta como **segunda barrera** detrás de Cloudflare Access.

El servidor Ubuntu (LAN ECS, IP estática, mismo equipo donde corre Phase 6)
**no abre puertos a internet**: la conexión va por un **túnel saliente
(`cloudflared`) hacia el edge de Cloudflare**. Cloudflare termina TLS público
en su edge y reenvía al Caddy interno por el túnel. Nexo permanece invisible
desde internet salvo a través del nombre `nexo.app` gateado por Access.

El acceso LAN directo (`https://nexo.ecsmobility.local` con `tls internal` de
Phase 6) **se mantiene como fallback**: si Cloudflare cae, los usuarios en la
oficina siguen entrando vía LAN sin interrupción. Los cambios de esta phase
son aditivos (un servicio Docker más, headers de seguridad nuevos en Caddy,
docs nuevas) — nada existente se rompe.

**Estado de partida (commit `6c05487`):**

- Phase 6 dejó listo: `docker-compose.prod.yml`, `caddy/Caddyfile.prod`
  (`tls internal` + reverse_proxy a `web:8000`), `scripts/deploy.sh` (atómico
  con pre-deploy backup + smoke), `scripts/backup_nightly.sh`,
  `tests/infra/deploy_smoke.sh` (8 checks: cert/puertos/healthchecks/ufw),
  `docs/DEPLOY_LAN.md` runbook 740 líneas, Makefile prod-* targets.
- Phase 7 cerró DevEx hardening: pre-commit (ruff/format/mypy scoped a
  `^(api|nexo)/`), CI matrix 3.11+3.12, coverage gate 60%,
  `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md`, `docs/RELEASE.md`,
  `CHANGELOG.md` Keep-a-Changelog 1.1.0.
- Phase 8 al 5/11 plans (rediseño UI moderno; Centro de Mando refactor
  cerrado byte-for-byte respetando D-16 LOCKED). Plans 08-06..08-11
  pendientes (auth screens / data screens / config / ajustes / pa11y / release).
- CLAUDE.md decisión cerrada: "No comprar dominio público ni exponer Nexo a
  internet". **Esta phase revierte esa decisión explícitamente** con
  justificación documentada en ADR-001.
- Audit empírico de Phase 6 (sesión 2026-04-25, agente Explore): tres
  hallazgos accionables — (a) **falta de headers de seguridad** en
  `Caddyfile.prod` (sin HSTS, X-Frame-Options, X-Content-Type-Options,
  Referrer-Policy); (b) `deploy_smoke.sh` solo cubre 8 checks de los 11
  documentados — falta auth/login, MES caído, backup cron; (c) `deploy.sh`
  no tiene rollback automático si el smoke falla — manejable si supervisado,
  pero documentar como gotcha.

**Servidor target (Ubuntu LAN x300):**
- Existe físicamente, en la misma LAN que el SQL Server `dbizaro` (engine_mes
  read-only) y `ecs_mobility` (engine_app cfg/oee/luk4) — conectividad MES
  garantizada sin abrir reglas adicionales.
- Acceso administrativo: SSH desde el portátil del operador (WSL/Linux) o
  remoto desktop como fallback.
- OS asumido: Ubuntu Server 24.04 (consistente con `docs/DEPLOY_LAN.md`).
- Estado de instalación a confirmar el día del deploy: si Docker + repo
  clonado + `make prod-up` funcionando ya, fase 9 es aditiva pura. Si está
  virgen, primero se aplica `docs/DEPLOY_LAN.md` (Phase 6 runbook) y después
  `docs/CLOUDFLARE_DEPLOY.md` (esta phase).

**Cuenta Cloudflare:**
- La crea Erik Eguskiza con email a decidir (corporativo o personal — owner
  ship personal aceptable porque ECS no tiene cuenta CF gestionada y el
  dominio `ecsmobility.com` está externalizado a un tercero "que es un
  caos").
- Plan **Free**: Tunnel ilimitado, Access free hasta 50 usuarios. La empresa
  tiene <50 empleados con necesidad real de acceso → free es suficiente
  durante todo Mark-III y previsiblemente Mark-IV.
- Si en algún momento se cruza el umbral 50: replantear en su momento
  (~3 €/usuario/mes paid o migrar a SSO Workspace).

</domain>

<decisions>
## Implementation Decisions

### Dominio y registro

- **D-01:** Dominio = **`nexo.app`** como objetivo prioritario. TLD `.app`
  gestionado por Google con HSTS preload obligatorio (HTTPS forzado a nivel
  navegador, seguridad gratis). 4 letras + ".app" brandable y memorable.
  Coste ~14 €/año en Cloudflare Registrar (precio sin markup).
- **D-02:** Fallbacks si `nexo.app` está cogido (verificar al crear cuenta):
  primer fallback `nexo-app.com`, segundo `usenexo.com`, tercero
  `getnexo.com`, cuarto `nexoecs.com`. El operador escoge en el momento del
  alta. Lo decisivo es no quedarse bloqueado esperando: si los 5 están
  cogidos, escoger uno aceptable y avanzar.
- **D-03:** Registrar = **Cloudflare Registrar** directamente. Razón:
  cuenta Cloudflare ya necesaria para Tunnel + Access; tener dominio en la
  misma cuenta evita propagación DNS, configuración manual de nameservers y
  un proveedor más que mantener. Precio CF Registrar = wholesale, sin
  markup.
- **D-04:** URL final usuarios = `https://nexo.app` (raíz directa, sin
  subdominio). Limpia, fácil de dictar por teléfono. Subdominios reservados
  para futuro: `staging.nexo.app`, `api.nexo.app` si llega.

### Cuenta y ownership

- **D-05:** Owner de la cuenta Cloudflare = **Erik Eguskiza**, ownership
  personal. Justificación: ECS no tiene cuenta CF y tampoco la quiere
  gestionar; el dominio corporativo `ecsmobility.com` está externalizado a
  un tercero descrito como "caos". Erik es responsable de IT de Nexo end-to-
  end y prefiere control directo sobre infra que usa.
- **D-06:** Bus factor de la cuenta = **1**. Riesgo asumido. Mitigación:
  documentar credenciales en gestor de contraseñas corporativo de ECS
  Mobility con acceso de respaldo a un segundo administrador (a designar).
  Token recovery codes guardados en gestor + impreso en sobre cerrado.

### Cloudflare Tunnel

- **D-07:** Túnel ejecutado como servicio Docker en `docker-compose.prod.yml`,
  imagen oficial `cloudflare/cloudflared:latest`. NO instalación nativa en el
  host — mantiene el patrón "todo en compose" del proyecto.
- **D-08:** Modo del túnel = **token-based remote-managed**. Configuración
  vive en el dashboard de Cloudflare (no en YAML local). Una sola fuente de
  verdad. El servidor solo necesita el token.
- **D-09:** Token del túnel guardado en `/opt/nexo/.env` con perms `600`,
  variable de proyecto **`CLOUDFLARE_TUNNEL_TOKEN`**. Ejemplo (sin valor
  real) en `.env.example` con comentario de procedencia ("Get from Cloudflare
  Zero Trust → Networks → Tunnels → nexo-prod → Install connector → token").

  **REVISADO 2026-04-25 (research finding CORRECTION-01):** la imagen
  oficial `cloudflare/cloudflared:latest` lee la variable de entorno
  **`TUNNEL_TOKEN`** (no `CLOUDFLARE_TUNNEL_TOKEN`). En el bloque
  `environment:` del servicio compose hay que mapear explícitamente:
  ```yaml
  environment:
    - TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN}
  ```
  Esto preserva el naming del proyecto en `.env` (con prefijo
  `CLOUDFLARE_*` consistente con `NEXO_*`) sin pelearse con la convención
  upstream de Cloudflare. Sin este mapeo, el container falla silenciosamente
  con "no token provided".
- **D-10:** Target del túnel = **`https://caddy:443`** (apunta a Caddy
  interno, NO directo a `web:8000`). Justificación: (a) Caddy es la única
  capa que añade headers de seguridad (D-13), (b) mantiene arquitectura
  uniforme entre rama LAN y rama Cloudflare, (c) si en el futuro se añaden
  rate limits o middleware HTTP en Caddy, ambas ramas se benefician.
- **D-11:** TLS verify entre `cloudflared` y `caddy:443` = **off**
  (`noTLSVerify: true` en config del tunnel). Razón: el cert de Caddy es
  `tls internal` (CA local), no válido para `cloudflared`. El tráfico ya va
  cifrado E2E hasta el edge de Cloudflare; el tramo interno
  `cloudflared → caddy` corre por la red Docker (no expuesta) y aceptar el
  cert interno es irrelevante en términos de superficie. Documentar
  explícitamente en `docs/CLOUDFLARE_DEPLOY.md`.
- **D-12:** Routing alternativo descartado = `cloudflared → web:8000`
  saltándose Caddy. Pierde headers + asimetría LAN/CF + complica el routing.
  No.

### Headers de seguridad (PRE-REQUISITO crítico)

- **D-13:** Antes de exponer vía Cloudflare, `caddy/Caddyfile.prod` añade
  bloque `header` con:
  ```
  Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
  X-Frame-Options "DENY"
  X-Content-Type-Options "nosniff"
  Referrer-Policy "strict-origin-when-cross-origin"
  Permissions-Policy "geolocation=(), microphone=(), camera=()"
  ```
  Cabeceras aplicadas a **ambas ramas** (LAN y Cloudflare) porque ambas pasan
  por Caddy. Sin esto, exponer públicamente es vulnerabilidad real
  (clickjacking + MIME sniff + downgrade attack). Con esto, Mozilla
  Observatory baseline ≥ B+. CSP no se añade en esta phase (requiere
  inventario de assets inline — sub-plan futuro Mark-IV).

### Cloudflare Access (gating de acceso)

- **D-14:** Access App = una sola, nombre `nexo`, hostname `nexo.app`,
  cubre toda la app (`*`). NO se segmenta por path en esta phase (todo o
  nada). Segmentación por path (ej. `/api/health` público) descartada para
  simplicidad — fallback LAN ya cubre necesidad de health check sin auth.
- **D-15:** Access Policy = **email allowlist explícita**. Lista inicial:
  `e.eguskiza@ecsmobility.com` únicamente. Erik añade emails uno a uno
  desde el dashboard según va dando de alta usuarios reales.
- **D-16:** Identity Provider = **One-Time Passcode (OTP) por email**. Sin
  Google SSO ni Microsoft 365 (decisión: mantener simple, evitar
  dependencia de Workspace que ECS puede no tener configurado). El usuario
  recibe código de 6 dígitos en su email, lo mete, entra. Cookie session
  Cloudflare = **24 horas** (configurable hasta 30 días si fricción molesta;
  arrancar conservador).
- **D-17:** Denial page con branding Nexo + mensaje en español "Acceso
  restringido. Si crees que deberías tener acceso a Nexo, contacta con
  Erik Eguskiza – e.eguskiza@ecsmobility.com".

  **REVISADO 2026-04-25 (research finding BLOCKER-01):** la versión
  inicial planteaba subir un HTML custom al dashboard de Cloudflare Access
  → Settings → Custom Pages → Forbidden. **Esa feature requiere plan
  Pay-as-you-go ($7/usr/mes), NO está en Free tier**. Free tier solo
  ofrece (a) default page genérica de CF o (b) **Redirect URL** a una URL
  externa que tú controles. Decisión revisada en sesión 2026-04-25
  (Opción B):

  **Implementación**:
  1. HTML estático en `static/cloudflare-denial.html` con branding (logo +
     `tokens.css` colors) + mensaje "contacta con Erik".
  2. Bloque adicional en `Caddyfile.prod` que sirve la ruta
     `/access-denied` **públicamente** (sin gating de Cloudflare Access ni
     auth de Nexo — propósito explícito: ser accesible para usuarios
     rechazados que aún no tienen sesión).
  3. En el dashboard de Cloudflare Access App, configurar la opción
     "Redirect URL on identity denied" → `https://nexo.app/access-denied`.
  4. Flujo: usuario sin email autorizado abre `https://nexo.app` →
     Cloudflare Access intenta autenticar → email rechazado →
     **Cloudflare redirige el navegador** a `https://nexo.app/access-denied`
     → Caddy sirve la página estática con branding Nexo.

  **Coste**: 0 € (Free tier respetado). **UX**: indistinguible de la
  opción custom paid (mismo HTML, mismo dominio, mismo branding).
  **Trade-off**: la ruta `/access-denied` es pública en internet (no
  contiene datos sensibles, solo HTML estático con mensaje de contacto).
  El planner debe garantizar que **NO se loguea esta ruta en `audit_log`**
  (no requiere sesión, no es path autenticado, AuditMiddleware ya la
  excluye por estar en lista de paths sin auth — verificar).
- **D-18:** Logout de Access = endpoint `/cdn-cgi/access/logout` integrado
  como link en el dropdown de usuario de Nexo (sub-tarea Phase 8 auth
  refactor 08-06; en esta phase solo se documenta el endpoint, no se
  cablea).

### Coexistencia con LAN fallback

- **D-19:** Caddy sigue escuchando `:443` interno con `tls internal` para
  hostname `nexo.ecsmobility.local`. Cero cambios. Los usuarios en LAN con
  hosts-file + root CA instalada siguen entrando como hasta ahora. Si CF
  tiene incidente, oficina sigue trabajando.
- **D-20:** El túnel Cloudflare entra al **mismo Caddy** por la red Docker
  interna. Caddy no necesita config nueva por hostname (el `Caddyfile.prod`
  ya escucha `:443` para todo). El SNI será `caddy:443` interno, no
  `nexo.app` — y eso está OK porque `cloudflared` trae el header `Host:
  nexo.app` y Caddy lo respeta para reverse_proxy.

  **REVISADO 2026-04-25 (research finding CORRECTION-02):** la
  preocupación inicial sobre "421 Misdirected Request" está mitigada por
  default en Caddy v2.11.0+ (Dec 2024). La opción `strict_sni_host` está
  **deshabilitada por default** — Caddy acepta requests con SNI distinto
  del Host header sin devolver 421, siempre que el bloque catch-all `:443`
  esté presente (ya lo está en `Caddyfile.prod` actual de Phase 6). Riesgo
  R-05 baja de ALTO a MEDIO: confirmable en smoke `[CF-04]`. Mitigación si
  el smoke falla: añadir bloque explícito `nexo.app { ... }` con misma
  config o establecer `default_sni nexo.app` global. Plan de contingencia
  documentado pero improbablemente necesario.

### Smoke testing

- **D-21:** Mejora del smoke LAN existente: `tests/infra/deploy_smoke.sh`
  pasa de 8 a 11 checks. Añadir:
  - `[DEPLOY-09]` POST `/api/login` con creds dummy responde 401 (no 500)
  - `[DEPLOY-10]` GET `/api/health` con MES caído responde 200 con
    `{"ok": false, "checks": {...}}`
  - `[DEPLOY-11]` `cloudflared` container running + `tunnel info` exit 0
- **D-22:** Smoke externo nuevo: `tests/infra/cloudflare_smoke.sh`,
  ejecutable desde fuera de la LAN (4G del móvil con tethering, o cualquier
  red ajena). Checks:
  - `[CF-01]` `nexo.app` resuelve por DNS público (cualquier resolver)
  - `[CF-02]` `https://nexo.app` devuelve cert real (no `tls internal`),
    issuer = `Google Trust Services` o similar
  - `[CF-03]` `curl -H 'CF-Access-Jwt-Assertion: invalid' https://nexo.app/`
    es rechazado por Access (302 a login)
  - `[CF-04]` GET sin headers → redirige a `https://nexo.cloudflareaccess.com/`
    (Access login)
  - `[CF-05]` Headers de seguridad presentes en respuesta: HSTS,
    X-Frame-Options, X-Content-Type-Options
  - `[CF-06]` `tunnel info nexo-prod` desde el servidor reporta
    `connectorStatus: HEALTHY`
- **D-23:** El smoke externo se documenta como manual + scripted. Manual:
  abrir `https://nexo.app` desde móvil con 4G, debe pedir email, recibir
  OTP, entrar. Scripted: el `cloudflare_smoke.sh` cubre los puntos
  automatizables.

### Rollback y safety

- **D-24:** Rollback de la activación = **`docker compose stop cloudflared`**.
  Una sola línea. Tunnel cae, dominio público deja de responder, **LAN
  fallback sigue intacto**. Cero impacto en usuarios LAN. Sin pérdida de
  datos.
- **D-25:** Rollback del registro de dominio = no hay (es transacción de
  compra). Si la phase se aborta tras comprar, el dominio queda parqueado
  hasta renovación o transferencia a otro proyecto. Coste hundido ~14 €.
  Riesgo aceptado.
- **D-26:** No se añade rollback automático a `scripts/deploy.sh` en esta
  phase (el audit lo señaló como mejora — diferido a phase futura). Sigue
  documentado como gotcha en `docs/RUNBOOK.md`.

### ADR y override de decisiones cerradas

- **D-27:** ADR formal en `docs/decisions/ADR-001-cloudflare-tunnel.md`
  documenta:
  - Decisión cerrada que se revierte: "No comprar dominio público ni exponer
    Nexo a internet" (CLAUDE.md sección "Qué NO hacer")
  - Razones del cambio: necesidad real de acceso desde móvil + remoto, sin
    fricción de instalación de CA o VPN para usuarios no técnicos
  - Alternativas evaluadas y descartadas: (a) Tailscale — fricción de
    instalación por dispositivo; (b) port-forward 443 directo —
    inaceptable, peor postura de seguridad; (c) LAN-only con DNS-01 cert
    real — solo resuelve oficina, no remoto
  - Trade-offs aceptados: Cloudflare termina TLS en su edge y técnicamente
    puede ver tráfico descifrado; tolerable para datos OEE industriales
- **D-28:** `CLAUDE.md` se actualiza en plan 09-03 para reflejar el
  override:
  - Sección "Despliegue": cambiar de "LAN interna ECS" a "LAN interna ECS +
    público vía Cloudflare Tunnel con allowlist Access (ver ADR-001)"
  - Sección "Qué NO hacer": eliminar "No comprar dominio público ni exponer
    Nexo a internet"; añadir "No exponer puertos del servidor a internet
    directamente — todo tráfico público pasa obligatoriamente por
    Cloudflare Tunnel saliente"

### Mobile detection (out of scope, planteado)

- **D-29:** Detección de móvil para layouts diferenciados **NO entra en
  esta phase**. La detección se aborda en Phase 8 sub-plan "mobile-aware
  navigation" (futuro, post-09). Las opciones técnicas evaluadas y
  registradas en este context para referencia:
  - **(a) Tailwind responsive utilities** (`md:hidden`, `hidden md:block`,
    breakpoints `sm/md/lg/xl`) — para variaciones de tamaño/spacing.
    Stack actual ya las soporta vía CDN.
  - **(b) Alpine `$store.viewport`** — store global que detecta viewport
    al cargar y en `resize`, devuelve `'mobile' | 'tablet' | 'desktop'`.
    Templates usan `x-show="$store.viewport === 'mobile'"` para mostrar
    bloques estructurales distintos.
  - **(c) Server-side User-Agent parsing** en middleware FastAPI, render
    de templates totalmente distintos. Reservado para casos donde (a) y
    (b) no escalen.
  - **Recomendación**: (a)+(b) cubren 95% de los casos. Solo escalar a (c)
    si el peso de página es problema en 4G.
  - Lo que SÍ entra en Phase 9: dejar headers de seguridad correctos para
    que móviles renderizen bien (HSTS, sin redirect loops, etc.).

</decisions>

<open-questions>
## Open Questions (Resueltas en sesión 2026-04-25)

- **Q-01:** ¿Quién accede a Nexo? — Cualquiera con email autorizado por
  Erik, sin distinción dispositivo. Móvil + portátil obligatorio.
- **Q-02:** ¿Tailscale o Cloudflare? — Cloudflare. Razón: Tailscale exige
  instalar app en cada dispositivo, contradice el requisito "link y entrar".
- **Q-03:** ¿Es totalmente gratis? — No, dominio cuesta ~14 €/año. El resto
  (tunnel + Access ≤50 users + cert) sí es gratis.
- **Q-04:** ¿Es seguro? — Más seguro que abrir router. Menos privado que
  Tailscale (CF termina TLS). Para OEE industrial interno = aceptable.
- **Q-05:** ¿Domain a comprar? — `nexo.app` con 4 fallbacks documentados
  (D-02). Decisión final en el momento de la compra.
- **Q-06:** ¿Cuenta CF de quién? — Erik Eguskiza, ownership personal
  (ECS no la quiere gestionar).
- **Q-07:** ¿Servidor existe? — Sí, Ubuntu LAN x300, mismo equipo Phase 6,
  acceso SSH + remote.
- **Q-08:** ¿MES llega desde el servidor? — Sí, misma LAN que SQL Server
  `dbizaro`/`ecs_mobility`. Sin reglas adicionales.
- **Q-09:** ¿Fallback LAN se conserva? — Sí, decisión D-19. Coste cero
  (ya está hecho en Phase 6), valor alto (uptime durante incidentes CF).
- **Q-10:** ¿Custom denial page? — Sí, HTML con mensaje "contacta con
  Erik – e.eguskiza@ecsmobility.com". Subir manualmente al dashboard CF.
- **Q-11:** ¿Mobile-specific layouts? — No en Phase 9. Phase 8 futuro
  sub-plan. Phase 9 solo deja la base (headers correctos) para que móviles
  funcionen sin layout específico.

</open-questions>

<scope-out>
## Out of Scope (Phase 9 explicitamente NO cubre)

- **Layouts mobile-specific** (drawer reordenado, menús totalmente
  distintos en móvil). Phase 8 sub-plan futuro.
- **Rollback automático en `scripts/deploy.sh`**. Hallazgo del audit, pero
  diferido a phase futura — esta phase no degrada el comportamiento actual.
- **CSP (Content-Security-Policy) header**. Requiere inventario completo
  de assets inline (Alpine, scripts inline en templates). Sub-plan futuro.
- **SSO con Google Workspace o Azure AD**. OTP por email basta para Mark-III
  + Mark-IV. Si crece >50 users o hay demanda de auto-aprovisionamiento,
  reabrir.
- **Sync de backups a NAS externo o S3**. Diferido desde Phase 6, sigue
  diferido. Backups locales 7-day rotation en `/var/backups/nexo/` siguen
  válidos.
- **WAF rules avanzadas de Cloudflare** (rate limiting agresivo, bot
  fight, geofencing). Free tier WAF básico suficiente para Mark-III.
- **Custom Cloudflare Worker** que reescriba o filtre tráfico antes de
  Nexo. Mantener stack simple.
- **Tag de release v1.0.0**. Eso vive en plan 08-11 (cierre Mark-III) o en
  Mark-IV. Phase 9 se limita a infra de acceso.
- **Pruebas de carga / DDoS simulado**. Cloudflare protege gratis hasta
  niveles enterprise; preocupación válida solo si crece la base de usuarios
  o hay incidentes reales.

</scope-out>

<plan-sketch>
## Plan Sketch (orientativo — `/gsd-plan-phase 9` decide la granularidad real)

Estimación: **3 plans atómicos**, ~4-6 horas trabajo en repo + ~30 min
operación en oficina el día del deploy.

### 09-01 — Caddy hardening + smoke mejorado (PRE-REQUISITO)

Bloqueante para 09-02. Sin esto, exponer es vulnerabilidad real.

- Añadir bloque `header` a `caddy/Caddyfile.prod` con HSTS + X-Frame-Options
  + X-Content-Type-Options + Referrer-Policy + Permissions-Policy
- Mejorar `tests/infra/deploy_smoke.sh` de 8 a 11 checks (login + MES
  caído + headers presentes)
- Test unitario que parsea `Caddyfile.prod` y verifica los 5 headers
- Verificar que el setup LAN existente sigue verde tras los headers
- Coverage gate sigue ≥60%

### 09-02 — `cloudflared` service + bootstrap + Makefile + docs

El grueso de la phase. Aditivo puro.

- Servicio `cloudflared` en `docker-compose.prod.yml` (imagen oficial,
  `command: tunnel run --token ${CLOUDFLARE_TUNNEL_TOKEN}`,
  `depends_on: caddy`, `restart: unless-stopped`)
- Variable `CLOUDFLARE_TUNNEL_TOKEN` en `.env.example` con comentario de
  procedencia (panel CF)
- `scripts/cloudflare-bootstrap.sh` para validación de prerequisitos en el
  servidor (token presente, container arranca, túnel HEALTHY)
- Makefile targets: `cf-up`, `cf-down`, `cf-status`, `cf-logs`,
  `cf-smoke-external`
- `tests/infra/cloudflare_smoke.sh` con los 6 checks `[CF-01..CF-06]`
- Plantilla `static/cloudflare-denial.html` con branding Nexo
- `docs/CLOUDFLARE_DEPLOY.md` runbook completo: paso a paso desde "creo la
  cuenta" hasta "smoke desde 4G", incluye screenshots placeholders donde
  el operador adjuntará capturas reales tras el deploy
- `docs/decisions/ADR-001-cloudflare-tunnel.md` con el formato Nygard
  estándar (Context / Decision / Consequences)
- Update `CLAUDE.md` (override decisión "no internet" + nueva sección
  "Despliegue" que menciona ambas ramas)
- Update `CHANGELOG.md` con entry de la phase

### 09-03 — Activación en producción (runbook-driven, manual)

No es código nuevo en repo. Es ejecución del runbook 09-02.

- Operador: crea cuenta CF, compra `nexo.app`, crea Tunnel `nexo-prod`,
  copia token
- Operador: crea Access App `nexo`, policy con email allowlist, sube
  custom denial HTML
- Operador: SSH al servidor, `git pull`, pega token en `.env`, `make cf-up`
- Operador: ejecuta `make cf-smoke-external` desde 4G (tethering del móvil)
- Operador: verifica fallback LAN sigue funcionando (`https://nexo.ecsmobility.local`)
- Operador: añade emails reales a la allowlist según va onboarding usuarios
- VERIFICATION.md de la phase documenta: dominio comprado, túnel HEALTHY,
  Access policy con N emails, smoke 6/6 checks pasando, fallback LAN OK

</plan-sketch>

<requirements-mapping>
## Requirements (CLOUD-01..CLOUD-12, ver REQUIREMENTS.md sección CLOUD)

- **CLOUD-01**: Dominio `nexo.app` (o fallback) comprado en Cloudflare
  Registrar; nameservers de Cloudflare; DNS público resolviendo a edge CF
- **CLOUD-02**: Cuenta Cloudflare creada por Erik Eguskiza; 2FA activo;
  recovery codes guardados
- **CLOUD-03**: Caddyfile.prod con 5 headers de seguridad (HSTS,
  X-Frame-Options, X-Content-Type-Options, Referrer-Policy,
  Permissions-Policy); aplicados a ambas ramas LAN+CF
- **CLOUD-04**: Servicio `cloudflared` en docker-compose.prod.yml; modo
  token-based remote-managed; target `https://caddy:443` con TLS verify off
  documentado
- **CLOUD-05**: Variable `CLOUDFLARE_TUNNEL_TOKEN` en `.env.example` con
  procedencia documentada; perms 600 en `/opt/nexo/.env` real
- **CLOUD-06**: Cloudflare Access App `nexo` cubriendo `nexo.app/*`;
  policy email allowlist explícita; OTP por email como IdP; cookie 24h
- **CLOUD-07**: Custom denial page subida al dashboard CF con branding
  Nexo y mensaje "contacta con Erik – e.eguskiza@ecsmobility.com"
- **CLOUD-08**: `tests/infra/deploy_smoke.sh` mejorado a 11 checks (8
  existentes + login + MES caído + cloudflared running)
- **CLOUD-09**: `tests/infra/cloudflare_smoke.sh` ejecutable con 6 checks
  [CF-01..CF-06] desde fuera de la LAN
- **CLOUD-10**: Makefile con targets `cf-up`, `cf-down`, `cf-status`,
  `cf-logs`, `cf-smoke-external`
- **CLOUD-11**: `docs/CLOUDFLARE_DEPLOY.md` runbook + `docs/decisions/
  ADR-001-cloudflare-tunnel.md` ADR + `CLAUDE.md` actualizado +
  `CHANGELOG.md` entry
- **CLOUD-12**: Fallback LAN preservado: smoke LAN sigue verde tras
  activación CF; `https://nexo.ecsmobility.local` accesible desde la
  oficina sin cambios

</requirements-mapping>

<success-criteria>
## Success Criteria (lo que debe ser TRUE al cerrar Phase 9)

1. Un usuario en la lista de Erik abre `https://nexo.app` desde el móvil
   con 4G (sin VPN, sin app instalada), recibe OTP en su email, mete el
   código, ve el login de Nexo, mete user/pass argon2id, **entra**.
2. Un usuario fuera de la lista hace lo mismo: Cloudflare le rechaza con la
   página custom de denial que muestra "contacta con Erik –
   e.eguskiza@ecsmobility.com". **Nunca llega a ver el login de Nexo**.
3. El servidor Ubuntu **no tiene puerto 443 ni 80 abiertos a internet**
   (`ufw status` lo confirma — solo accesible desde subnet LAN). Cloudflare
   accede únicamente vía túnel saliente.
4. Headers de seguridad presentes en respuestas (verificable con
   `curl -I https://nexo.app/login` desde fuera): HSTS,
   X-Frame-Options, X-Content-Type-Options, Referrer-Policy,
   Permissions-Policy.
5. Mozilla Observatory contra `https://nexo.app` devuelve grade ≥ B+
   (objetivo conservador; A- es alcanzable cuando se añada CSP en futuro).
6. **LAN fallback intacto**: un usuario en la oficina con hosts-file +
   root CA accede a `https://nexo.ecsmobility.local` igual que antes,
   sin pasar por Cloudflare. Cuando se hace `docker compose stop
   cloudflared`, este flujo sigue funcionando.
7. `tests/infra/cloudflare_smoke.sh` desde fuera de la LAN devuelve 6/6
   checks OK. `tests/infra/deploy_smoke.sh` desde la LAN devuelve 11/11
   checks OK.
8. Runbook `docs/CLOUDFLARE_DEPLOY.md` permite a un admin de respaldo
   reinstalar el túnel desde cero sin contexto en <30 min (asumiendo
   acceso al panel CF).
9. ADR-001 publicado en `docs/decisions/`. CLAUDE.md actualizado.
   CHANGELOG.md con entry de la phase. PROJECT.md y REQUIREMENTS.md
   reflejan CLOUD-* como done al cierre.
10. `cloudflared` container reporta `Status: HEALTHY` en
    `tunnel info nexo-prod` durante 24h continuas tras la activación.

</success-criteria>

<dependencies>
## Dependencies

- **Phase 6 (Despliegue LAN HTTPS)** — el túnel apunta a Caddy ya existente.
  Si Phase 6 no está activa en el servidor target, primero ejecutar el
  runbook `docs/DEPLOY_LAN.md` y después esta phase.
- **Phase 7 (DevEx hardening)** — los nuevos checks de smoke pasan por
  CI con coverage gate 60%, los nuevos `.sh` pasan por shellcheck en
  pre-commit. Phase 7 ya cerrada.
- **No depende de Phase 8** — esta phase es ortogonal al rediseño UI.
  Phase 8 puede continuar en paralelo. Si Phase 8 introduce cambios en
  `Caddyfile.prod` o en endpoints relevantes para smoke, hay que
  reconciliar antes del cierre.

</dependencies>

<risks>
## Risks

- **R-01:** `nexo.app` está cogido. **Mitigación:** D-02 lista 4 fallbacks
  pre-aprobados. Decisión en el momento de la compra, no bloquea.
- **R-02:** Cuenta de Cloudflare se pierde (Erik olvida creds, dispositivo
  perdido). **Mitigación:** D-06 documenta gestor de contraseñas + recovery
  codes impresos + segundo administrador a designar.
- **R-03:** Cloudflare cambia free tier o sube precios. **Mitigación:** Plan
  free CF estable desde 2020; allowlist <50 users no presiona. Si cambian,
  ~3 €/user/mes paid es coste asumible o migración a Tailscale (proceso
  documentado en alternativas evaluadas, ADR-001).
- **R-04:** Cloudflare incidente global (cae el edge entero). **Mitigación:**
  D-19 + D-24, fallback LAN intacto. Usuarios oficina siguen entrando.
  Usuarios remotos esperan a que vuelva (incidentes históricos CF típicos
  ~30 min).
- **R-05:** El SNI mismatch entre `cloudflared → caddy:443` rompe el
  routing (Caddy responde 421). **Mitigación:** D-20 documenta plan de
  contingencia (bloque explícito `nexo.app` en Caddyfile). Verificar
  empíricamente en smoke externo.
- **R-06:** Token del túnel se filtra (commit accidental, etc.).
  **Mitigación:** rotación inmediata desde dashboard CF (1 click). El
  archivo `.env` está en `.gitignore` y `gitleaks` lo detectaría en CI.
  Documentar procedimiento de rotación en RUNBOOK.
- **R-07:** OTP por email no llega (filtros antispam, dominio del email
  bloqueado). **Mitigación:** documentar en RUNBOOK que el remitente es
  `noreply@notify.cloudflare.com` y debe whitelist-arse. Alternativa
  futura: cambiar a Google Workspace SSO.
- **R-08:** Latencia añadida por CF edge (~30-100ms vs LAN directa).
  **Mitigación:** aceptable para flujos no real-time (login + reportes
  OEE). Si Centro de Mando luk4 (timeline real-time) sufre, usuarios
  oficina pueden seguir usando LAN directa.

</risks>
</content>
</invoke>