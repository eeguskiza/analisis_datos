---
screen: login
template: templates/login.html
route: /login
variants_generated: 4
selected_variant: 2
selected_by: claude_auto
selected_on: 2026-04-22
reversible: true
revert_command: "git revert <commit-hash-of-08-06-impl>"
---

# Sketch findings: login

## Chosen variant

**Variant 2 — Centered card on `bg-surface-app`.** 400px card centered
vertically + horizontally. Logo + `by {{ company_name }}` tagline above
the card. Inside the card: `Entrar` H1, stacked email + password
inputs, primary CTA `Entrar` full-width, error banner rendered above
the form when present (lockout 429 or invalid-credentials 401).

## Rationale

1. Matches UI-SPEC §"Per-Screen Adaptations" row `login.html`:
   "centered card, logo above, stacked fields".
2. Public route — must NOT extend `base.html` (no top bar or drawer).
   Uses the same stylesheet + Tailwind config as the rest of the app
   so tokens resolve identically.
3. Lockout copy matches UI-SPEC §"Copywriting Contract":
   `Cuenta temporalmente bloqueada. Reintenta en 15 minutos.` The
   router surfaces the message server-side (D-25 server-authoritative).
4. Autocomplete attrs (`username` on email, `current-password` on
   password) are retained from the Phase 2 login so password managers
   keep working.

## Token inventory

- `bg-surface-app` — outer page background.
- `bg-surface-base` (via `.card`) — card surface.
- `border-subtle` (via `.card`) — card border.
- `shadow-card` — N/A on this screen; we use `.card` which already has
  a border and no shadow to keep the auth page visually quiet.
- `text-heading`, `text-body`, `text-muted` — typography anchors.
- `bg-error-subtle` + `text-error` + `border-subtle` — error banner.
- `bg-primary` / `text-on-accent` (via `.btn-primary`) — CTA.
- `--radius-lg` (via `.card`) — 16px.

## Component inventory

- `.card` + `.card-body` — the login frame.
- `.input-inline` — email + password.
- `.btn.btn-primary.btn-lg` — `Entrar`.
- Inline `<div class="rounded-md bg-error-subtle text-error border border-subtle">`
  for the error banner (we do NOT use `.error-state` because that
  component is a larger "page-level" empty/error box; the inline rule
  is an error BANNER, not a whole-page error).

## Autonomy override (D-12)

Operator (`e.eguskiza@ecsmobility.com`) was AFK at execution time.
Per GSD auto-mode + Plan 08-06's explicit "Pick planner-recommended
variants" directive, Claude selected Variant 2 (the planner-recommended
default) without human sign-off. Reversible via `git revert` of the
08-06 template commit.

## Deviations

None.

## Landmines

- **Public route** — middleware whitelist MUST keep `/login` in the
  allow-list (it already does — `api/middleware/auth.py`).
- **No client-side validation** (D-25 server-authoritative). The
  `required` HTML attribute is kept for UX (browser form hints) but
  any server-side 400/401/429 is the only source of truth.
- **Lockout = HTTP 429** (not 401). Same error banner copy but
  different status code → the router chooses the message.
- **Password manager UX** — `autocomplete="username"` on email is
  correct per WHATWG spec; password managers use it to match on
  subsequent visits. Keep it.
