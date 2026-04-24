# REPLAN — Nexo hacia Deploy Real
Fecha: 2026-04-24
Estado: BORRADOR — completar en sesión Q&A

---

## Objetivo

Deploy real hoy en servidor LAN nuevo.
Plataforma funcional, limpia, usable por usuarios reales.
Interfaz modernizada. Solo lo que está listo se muestra.

---

## Requisitos confirmados

### R1 — Ocultar funcionalidad no lista
- Solo **Centro de Mando** visible en nav para todos los roles.
- El resto de opciones actuales (pipeline, historial, bbdd, datos, informes, ajustes avanzados, etc.) → ocultas, no cargadas.
- No se eliminan del código, solo se esconden hasta que estén listas.

### R2 — Secciones nuevas (estructura de negocio)
- Crear secciones en el nav: **Fabricación, Logística, RRHH** + las que confirme el operador.
- De momento placeholders — página vacía con título y "Próximamente" o similar.
- Cada sección puede tener sub-secciones (a definir en Q&A).

### R3 — Configuración sesgada por rol
- **Propietario**: ve panel de config completo (usuarios, ajustes, audit, límites, todo).
- **Usuario normal**: ve solo lo relevante para su trabajo (a definir qué exactamente en Q&A).
- **Directivo**: por definir.

### R4 — Logging se mantiene
- Todo el logging de queries, audit_log, login_attempts se queda tal cual.
- Base para analytics del propietario.

### R5 — Usuarios con nombre obligatorio
- Campo `nombre` obligatorio en todos los usuarios (migración ya hecha en 08-03).
- Solo el propietario puede dar de alta usuarios.
- Flujo de alta: propietario crea usuario → usuario cambia password en primer login (ya funciona).

### R6 — Servidor nuevo desde cero
- IP: **PENDIENTE** (operador la da)
- OS asumido: Ubuntu Server 24.04 (confirmar)
- Setup completo: Docker + Compose + Caddy + clone + .env + seed usuarios
- Referencia base: `docs/DEPLOY_LAN.md`

### R7 — Rediseño de interfaz
- Metodología: Q&A ventana a ventana (operador responde, Claude implementa).
- Condición: se hace DESPUÉS de que R1–R6 estén en producción.
- Estado actual Phase 8: tokens + chrome + bienvenida + luk4 ya rediseñados.
- Pendiente: auth screens, data screens, config screens, ajustes.

---

## Preguntas abiertas (Q&A pendiente)

### Servidor
- [ ] IP del servidor
- [ ] OS ya instalado o desde cero
- [ ] Acceso SSH: usuario + método (clave / password)
- [ ] Hay Docker ya instalado?
- [ ] Hostname LAN que quieres usar (nexo.ecsmobility.local u otro)

### Secciones
- [ ] Lista completa de secciones: ¿solo Fabricación / Logística / RRHH o hay más?
- [ ] ¿Sub-secciones dentro de cada una?
- [ ] ¿Alguna sección tiene que mostrar datos reales ya o todas son placeholder?

### Usuarios de partida
- [ ] ¿Quién es el propietario inicial? (nombre, username, email)
- [ ] ¿Hay más usuarios a crear en el seed o los va creando el propietario desde la app?
- [ ] ¿Qué departamentos hay que tener configurados desde el día 1?

### Config sesgada
- [ ] ¿Qué ve exactamente un usuario normal en la config? ¿Solo su perfil?
- [ ] ¿El directivo tiene una vista intermedia o igual que usuario normal?
- [ ] ¿El audit log es solo para propietario?

### Centro de Mando
- [ ] ¿Sigue conectado al MES (SQL Server) en el servidor nuevo o de momento mock/desconectado?
- [ ] ¿Las credenciales MES se copian tal cual al nuevo servidor?

---

## Plan de fases propuesto (BORRADOR)

### Fase A — Servidor y deploy base
1. Provisionar Ubuntu + Docker + Caddy en IP nueva
2. Clone repo + .env con secretos reales
3. `make prod-up` + smoke check
4. Seed usuario propietario

### Fase B — Limitar nav (solo Centro de Mando)
1. Ocultar rutas/nav items no listos
2. Verificar que Centro de Mando carga limpio
3. Test regresión

### Fase C — Secciones nuevas
1. Crear rutas placeholder (Fabricación / Logística / RRHH / ...)
2. Nav actualizado con nueva estructura
3. Acceso por rol configurado

### Fase D — Config sesgada por rol
1. Vista propietario: config completa
2. Vista usuario normal: solo perfil + lo suyo
3. Vista directivo: TBD

### Fase E — Usuarios y alta solo por propietario
1. Verificar flujo de alta en prod
2. Confirmar nombre obligatorio funciona
3. Seed inicial de usuarios si los hay

### Fase F — Rediseño interfaz (Q&A ventana a ventana)
- Auth screens
- Data screens
- Config screens
- Ajustes

### Fase G — Release y cierre
1. CHANGELOG bump
2. Tag v1.0.0
3. Smoke final

---

## Decisiones pendientes de cerrar

- [ ] IP servidor
- [ ] Lista completa de secciones de negocio
- [ ] Usuarios seed iniciales
- [ ] Qué ve cada rol en config
- [ ] MES conectado o mock en nuevo servidor
