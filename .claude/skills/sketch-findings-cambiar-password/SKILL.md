---
screen: cambiar-password
template: templates/cambiar_password.html
route: /cambiar-password
variants_generated: 2
selected_variant: 1
selected_by: claude_auto
selected_on: 2026-04-22
reversible: true
revert_command: "git revert <commit-hash-of-08-06-impl>"
---

# Sketch findings: cambiar_password

## Chosen variant

**Variant 1 — Top-bar + centered card (480px).** Extends `base.html`
so the topbar + drawer (Plan 08-02 chrome) remain available. Centered
480px card with three stacked password inputs (actual / nuevo / repetir)
+ primary CTA `Guardar contraseña`. When `must_change=True`, a warn
banner above the form surfaces the forced-change copy.

## Rationale

1. Extending `base.html` is the right default: the user is
   authenticated at this point (the `AuthMiddleware` enforces session
   cookie), so the chrome is appropriate even during forced-change.
2. 480px is wider than login (400px) because there are 3 inputs + a
   hint line ("Mínimo 12 caracteres") under the new-password field.
3. Spanish copy matches the router-surfaced errors:
   - `La contrasena actual no es correcta.`
   - `Las contrasenas nuevas no coinciden.`
   - `La contrasena debe tener al menos 12 caracteres.`
4. `autocomplete="new-password"` on both new + repetir is the WHATWG
   recommendation for password-creation flows.

## Token inventory

- `bg-surface-base` / `border-subtle` / `--radius-lg` (via `.card`).
- `text-heading`, `text-body`, `text-muted`.
- `bg-warn-subtle` + `text-warn` — forced-change banner.
- `bg-error-subtle` + `text-error` — validation error banner.
- `.btn.btn-primary.btn-lg` — `Guardar contraseña`.

## Component inventory

- `.card` + `.card-body`.
- `.input-inline` × 3 (actual, nuevo, repetir).
- `.btn.btn-primary.btn-lg`.
- Inline hint paragraph `<p class="text-sm text-muted">` under the new
  password input.

## Autonomy override (D-12)

Operator AFK. Plan 08-06 explicitly authorizes Claude to pick
Variant 1. Reversible.

## Deviations

- **Plan 08-06 kept the "Cancelar y salir" legacy POST-to-`/logout`
  secondary action.** The original template had it; we preserve it
  but restyle it as a `.btn.btn-ghost` (Mark-III acceptable — a plain
  link would lose the logout-via-POST contract).

## Landmines

- **`must_change_password` is a server-side flag.** The client
  cannot spoof it; the middleware re-checks on every authenticated
  request and redirects here if the flag is still set.
- **Drawer visibility with `must_change=True`** — plan recommends
  keeping the drawer visible because the URL is session-protected
  anyway; the user can't navigate away and commit damage. Mark-III
  acceptable.
- **Forced-change copy** — Spanish, no accents (consistent with the
  rest of the auth copy in the router, which uses "contrasena" without
  the tilde).
- **Password validators are server-authoritative** — `minlength="12"`
  on the inputs is a client-side hint only; the server checks
  `len(password_nuevo) < MIN_PASSWORD_LEN`. Never trust the client.
