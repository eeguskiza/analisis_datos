# Nexo · Branding

Convenciones de marca y assets para Nexo + ECS Mobility. Actualizado
2026-04-18 al scaffold inicial (Paso 2 de la sesión de arranque Sprint 0).

---

## Marcas que conviven en el producto

- **Nexo** — nombre del producto interno (sucesor de "OEE Planta / analisis_datos").
- **ECS Mobility** — empresa propietaria. Aparece como marca corporativa.

Regla rápida:
- **Nexo = producto** → cabecera del sidebar, login, favicon.
- **ECS Mobility = empresa** → footer, documentación corporativa, emails automáticos.

---

## Ubicación de assets

```
static/img/brand/
  nexo/
    logo.png               Logo principal de Nexo (placeholder IA temporal)
    README.md              Nota sobre el placeholder y su regeneración en Mark-V
  ecs/
    logo.png               Principal (alias de horizontal-positivo)
    horizontal-negativo.png
    vertical-positivo.png
    vertical-negativo.png
    README.md              Nota corporativa + tabla de variantes
```

El archivo `data/ecs-logo.png` (consumido por `api/config.py` →
`logo_filename`) sigue existiendo para que los módulos OEE sigan estampando
el logo en los PDFs generados por matplotlib. La unificación con
`static/img/brand/ecs/logo.png` es trabajo de Sprint 2 (reorganización de
config y capa de datos). **No se toca en Sprint 0**.

---

## Variables de entorno

Añadir a `.env.example` (sección "Branding"):

```
# Branding
NEXO_LOGO_PATH=/static/img/brand/nexo/logo.png
NEXO_ECS_LOGO_PATH=/static/img/brand/ecs/logo.png
NEXO_APP_NAME=Nexo
NEXO_COMPANY_NAME=ECS Mobility
```

Semántica:

- `NEXO_LOGO_PATH` — logo del producto (Nexo). Se usa en la cabecera del sidebar y en el `<title>` cuando el template lo renderiza como `<img>`.
- `NEXO_ECS_LOGO_PATH` — logo de la empresa (ECS Mobility). Se usa en el footer, en mails automáticos (Mark-IV) y en la documentación corporativa.
- `NEXO_APP_NAME` — string del producto. Usado en `<title>`, `<h1>` de `base.html`, banner de splash.
- `NEXO_COMPANY_NAME` — string de la empresa. Usado en el footer y en subject de mails.

> **Este archivo (`.env.example`) no lo puedo editar yo** en esta sesión
> — está en denyList de permisos de Claude Code. Erik debe copiar el
> bloque de arriba al final del `.env.example` y al `.env` real en
> disco, durante o antes de Sprint 0 commit 7.

---

## Uso en templates

**NO** hardcodear rutas de logos en templates/código. Siempre leer del
contexto que inyecta FastAPI desde `settings`:

```python
# api/deps.py (o similar)
def template_context(request):
    return {
        "app_name": settings.app_name,            # NEXO_APP_NAME
        "company_name": settings.company_name,    # NEXO_COMPANY_NAME
        "logo_path": settings.logo_path,          # NEXO_LOGO_PATH
        "ecs_logo_path": settings.ecs_logo_path,  # NEXO_ECS_LOGO_PATH
    }
```

```jinja2
{# templates/base.html — cabecera del sidebar, logo Nexo #}
<img src="{{ logo_path }}" alt="{{ app_name }}" class="w-7 h-7 object-contain"
     onerror="this.style.display='none';this.parentElement.innerHTML='<span class=\'font-bold text-brand-600 text-sm\'>NEXO</span>'">

{# templates/base.html — footer, logo ECS #}
<footer class="text-xs text-gray-500 py-3">
  <img src="{{ ecs_logo_path }}" alt="{{ company_name }}" class="inline-block h-4 mr-1">
  © {{ company_name }}
</footer>
```

**Estado actual (post-Paso 2 / pre-Sprint-0-commit-7):** las 3 referencias
hardcoded (`templates/base.html`, `static/js/app.js`, `api/main.py`
favicon) apuntan provisionalmente a `/static/img/brand/ecs/logo.png` para
que la app siga arrancando tras el `git mv` de los assets. **Sprint 0
commit 7** ("refactor: update UI titles and metadata to Nexo") sustituye
esas rutas por las variables de contexto y pasa el logo principal del
sidebar a `NEXO_LOGO_PATH` (Nexo), añadiendo un footer con
`NEXO_ECS_LOGO_PATH` (ECS).

---

## Regeneración del placeholder de Nexo (Mark-V)

El logo actual de Nexo es un placeholder generado por IA (Gemini). Tiene
artefactos visuales conocidos. Sustitución prevista en Mark-V con:

1. Prompt limpio en Ideogram v3 o Recraft v3 (salida SVG).
2. Export a PNG 512×512 para fallback y a SVG para escalabilidad.
3. Sustitución de `static/img/brand/nexo/logo.png` (mantener nombre para
   no tocar variables ni templates).
4. Actualización del README.md de `static/img/brand/nexo/` para reflejar
   la versión definitiva.

Hasta Mark-V el placeholder es aceptable para uso interno en LAN.
