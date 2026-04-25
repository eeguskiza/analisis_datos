# ADR-001: Cloudflare Tunnel + Access para acceso público a Nexo

**Status:** Proposed (será Accepted al cierre de Phase 9)
**Date:** 2026-04-25
**Author:** Erik Eguskiza (IT lead, ECS Mobility)
**Supersedes:** decisión cerrada en CLAUDE.md "No comprar dominio público ni exponer Nexo a internet"

---

## Context

Mark-III cerró Phase 6 con Nexo desplegado en LAN ECS sobre HTTPS interno
(Caddy `tls internal` + hostname `nexo.ecsmobility.local` + root CA distribuida
manualmente por equipo). La decisión cerrada en CLAUDE.md era explícitamente
"sin exposición a internet, LAN-only".

A partir de la sesión de discusión 2026-04-25, surge un requisito operativo
real que esa decisión no contempla:

- Acceso desde **móviles** dentro de la oficina. iOS no permite editar
  `/etc/hosts`, Android requiere root, y la root CA interna es fricción
  inaceptable para un usuario no técnico que solo quiere abrir Nexo en su
  teléfono durante un turno.
- Acceso **remoto** previsto: comerciales fuera de la oficina, gerencia desde
  casa, soporte de IT desde fuera, todos necesitan poder consultar OEE sin
  estar físicamente en la LAN.
- Distribuir manualmente la root CA a cada dispositivo de cada empleado nuevo
  no escala — es trabajo de IT que se acumula y bloquea onboarding.

El requisito explícito del operador, formulado en sesión: "necesito que sea
tan simple como recibe enlace con invitación y ya está".

## Decision

Adoptar **Cloudflare Tunnel + Cloudflare Access** como capa de exposición
pública para Nexo, con las siguientes restricciones:

1. **Túnel saliente** (`cloudflared`) ejecutado como servicio Docker en el
   mismo servidor Ubuntu de Phase 6. El servidor **no abre puertos a
   internet** — la conexión se inicia desde dentro hacia el edge de
   Cloudflare.
2. **Cloudflare Access** gateando todo el hostname (`nexo.app/*`) con
   **email allowlist explícita**, mantenida manualmente por Erik Eguskiza.
   Sin SSO inicialmente — One-Time Passcode por email como Identity Provider.
3. **Doble capa de auth**: Cloudflare Access (gate de red, "puede este email
   ver Nexo?") + auth interna de Nexo (gate de app, "puede este usuario
   loguearse y qué permisos tiene?"). Para entrar a cualquier funcionalidad
   hay que pasar las dos.
4. **Fallback LAN preservado**: el setup de Phase 6 (`tls internal` +
   `nexo.ecsmobility.local`) se conserva intacto. Si Cloudflare cae, los
   usuarios en oficina siguen entrando sin interrupción. Cero impacto.
5. **Plan Cloudflare Free**: Tunnel ilimitado, Access ≤50 users gratis.
   Cubre el tamaño actual de ECS Mobility con margen.
6. **Dominio**: `nexo.app` (con fallbacks documentados) comprado en
   Cloudflare Registrar para minimizar saltos. Coste ~14 €/año, asumido por
   Erik personalmente para evitar bloqueo administrativo de IT corporativo.
7. **Cuenta Cloudflare**: ownership personal de Erik Eguskiza. ECS
   Mobility no tiene cuenta CF y no la quiere gestionar. Bus factor 1
   mitigado con gestor de contraseñas + recovery codes + segundo
   administrador a designar.

## Alternatives Considered

### A. Tailscale (WireGuard managed)

**Pros:** TLS punta a punta sin tercero terminando, no rompe la decisión
"no internet" en CLAUDE.md, free hasta 100 dispositivos, ACLs por usuario.

**Contras:** **Requiere instalar app cliente en cada dispositivo** (iOS,
Android, Windows, Mac). Para un operario que abre Nexo en su móvil personal,
esto es fricción inaceptable. La sesión de discusión 2026-04-25 cerró
explícitamente este punto: "no necesito eso, necesito que sea tan simple
como recibe enlace con invitación y ya está".

**Veredicto:** Descartada por requisito de UX, no por mérito técnico.
Tailscale sigue siendo opción viable para acceso administrativo (SSH al
servidor, debug Postgres) — se puede coexistir con Cloudflare en el futuro
sin conflicto.

### B. LAN-only con cert real (LetsEncrypt DNS-01 challenge)

**Pros:** Soluciona el problema de "fricción de root CA en móviles" para
usuarios dentro de la oficina. Mantiene "no internet exposure" literal —
solo expone DNS público apuntando a IP privada.

**Contras:** No resuelve el caso de **acceso remoto**. Comerciales fuera
de la oficina, gerencia desde casa, IT desde fuera siguen sin acceso. Y
tarde o temprano ese acceso remoto lo necesitamos.

**Veredicto:** Resuelve la mitad del problema. Descartada por incompleta.

### C. Port-forward 443 directo en el router corporativo

**Pros:** Simple en concepto.

**Contras:** Postura de seguridad significativamente peor que Cloudflare
Tunnel. Servidor expuesto directamente a internet. Sin DDoS protection,
sin WAF, sin gating de email previo a la app. Cualquier escaneo automatizado
de internet llega al login de Nexo.

**Veredicto:** Inaceptable. No considerada seriamente.

### D. VPN corporativa (OpenVPN, WireGuard self-hosted)

**Pros:** Control total, sin dependencia de proveedor externo.

**Contras:** Mismo problema de UX que Tailscale (instalación cliente) +
overhead operativo de mantener VPN propia (renovación de certs, gestión
de usuarios, parches del servidor VPN). ECS no tiene equipo de seguridad
dedicado que pueda mantener esto.

**Veredicto:** Descartada por TCO operativo.

## Consequences

### Positivas

- **UX inmejorable** para el caso "link + entrar": OTP por email, cookie
  24h, navegador estándar, cero instalación.
- **Sin puertos abiertos** en el router corporativo. Cloudflare Tunnel
  inicia conexión saliente, mucho más seguro que port-forward.
- **DDoS protection y WAF básico** gratis incluido por Cloudflare.
- **Onboarding de usuarios nuevos** = añadir email a una lista en el
  dashboard. Cero distribución de CA, cero touchpoints en cada dispositivo.
- **Plan Free cubre el tamaño actual** de ECS Mobility con margen amplio.
  Coste real = 14 €/año (dominio).
- **Fallback LAN preservado** — uptime durante incidentes Cloudflare.

### Negativas (y cómo se mitigan)

- **Cloudflare termina TLS en su edge**: técnicamente puede ver tráfico
  descifrado entre su edge y nuestro servidor. Para datos OEE industriales
  internos, riesgo aceptado. Si en el futuro se manejan datos
  significativamente más sensibles (RRHH/nóminas/financiero), reevaluar
  con esta ADR como punto de partida.
- **Bus factor 1 en la cuenta Cloudflare**: mitigación documentada en
  D-06 del context (gestor de contraseñas + recovery codes + segundo
  administrador a designar dentro de los primeros 30 días post-deploy).
- **Dependencia de proveedor externo**: si Cloudflare cambia free tier o
  cae, hay impacto. Mitigación: fallback LAN sigue activo (cero downtime
  para oficina) + proceso de migración a Tailscale documentado en este
  ADR como plan B.
- **Latencia añadida** (~30-100ms vs LAN directa): aceptable para Nexo
  (login + reportes + dashboards). Si Centro de Mando luk4 (real-time
  timeline) sufre, los usuarios oficina siguen teniendo LAN directa
  disponible.

### Cambios en CLAUDE.md (aplicados en plan 09-02)

- Sección **"Despliegue"**: cambiar de "LAN interna ECS, sin exposición a
  internet" a "LAN interna ECS + público vía Cloudflare Tunnel saliente
  con allowlist Access (ver ADR-001)".
- Sección **"Qué NO hacer"**: eliminar la línea "No comprar dominio
  público ni exponer Nexo a internet". Añadir nueva línea: "No exponer
  puertos del servidor a internet directamente — todo tráfico público
  pasa obligatoriamente por Cloudflare Tunnel saliente, autenticado por
  Cloudflare Access antes de tocar Nexo".

## Validation

Al cierre de Phase 9 (criterios de éxito en `09-CONTEXT.md`):

1. Usuario en allowlist accede vía 4G en <60 segundos desde clic.
2. Usuario fuera de allowlist es bloqueado en página de denial custom.
3. `ufw status` confirma sin puertos 443/80 abiertos a internet.
4. Mozilla Observatory contra `nexo.app` ≥ B+.
5. Smoke externo `tests/infra/cloudflare_smoke.sh` 6/6 OK.
6. Smoke LAN sigue 11/11 OK con `cloudflared` corriendo.
7. Smoke LAN sigue 11/11 OK con `cloudflared` parado (verificación de
   independencia del fallback).

## Revision Notes

**2026-04-25 — research finding aplicado tras `/gsd-plan-phase 9` (sesión de research):**

- **Custom denial page**: la versión inicial asumía que era feature Free
  tier de Cloudflare Access. **No lo es** — requiere Pay-as-you-go
  ($7/usr/mes). Para no romper "free tier" como criterio del ADR, la
  denial page se sirve desde Caddy en una ruta pública
  `/access-denied`, y Cloudflare Access se configura con **Redirect URL**
  apuntando ahí. UX equivalente al custom paid, coste cero. Detalle en
  `09-CONTEXT.md` D-17 revisado.
- **Variable de entorno `cloudflared`**: la imagen oficial lee
  `TUNNEL_TOKEN`, no `CLOUDFLARE_TUNNEL_TOKEN`. El proyecto mantiene el
  naming `CLOUDFLARE_TUNNEL_TOKEN` en `.env` (consistencia con `NEXO_*`)
  y mapea en `environment:` del compose. Detalle en D-09 revisado.
- **Caddy v2.11 SNI behavior**: el riesgo R-05 (Misdirected Request 421)
  está mitigado por default en Caddy v2.11+ (`strict_sni_host: false`).
  Smoke `[CF-04]` lo confirma empíricamente; mitigación documentada por
  si acaso. Detalle en D-20 revisado.

## Notes

Esta ADR no documenta detalles de implementación — viven en
`.planning/phases/09-cloudflare-tunnel-public-access/09-CONTEXT.md` (29
decisiones D-01..D-29, con D-09/D-17/D-20 revisados 2026-04-25 tras
research) y `docs/CLOUDFLARE_DEPLOY.md` (runbook operativo generado por
plan 09-02).

Si en el futuro se decide migrar a Tailscale, esta ADR se marca como
`Superseded by ADR-XXX` y la nueva ADR documenta el motivo del cambio
(presumiblemente: cambio en sensibilidad de datos manejados, o usuarios
dispuestos a instalar cliente).
