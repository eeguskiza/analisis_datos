/* ──────────────────────────────────────────────────────────────
   Nexo — Tailwind CDN config (Phase 8 / Plan 08-01, per D-02)
   Extracted from base.html. Loaded via
     <script src="/static/js/tailwind.config.js"></script>
   and MUST appear BEFORE the cdn.tailwindcss.com script tag
   (Pitfall 1 — JIT ordering). Values reference CSS vars declared
   in /static/css/tokens.css using Tailwind's
     rgb(var(--color-xxx) / <alpha-value>)
   pattern, which preserves `bg-primary/20` alpha utilities.
   ────────────────────────────────────────────────────────────── */

window.tailwind = window.tailwind || {};
window.tailwind.config = {
  theme: {
    extend: {
      colors: {
        /* Semantic surfaces (Layer 2) */
        surface: {
          DEFAULT: 'rgb(var(--color-surface-base) / <alpha-value>)',
          base:    'rgb(var(--color-surface-base) / <alpha-value>)',
          app:     'rgb(var(--color-surface-app) / <alpha-value>)',
          subtle:  'rgb(var(--color-surface-subtle) / <alpha-value>)',
          muted:   'rgb(var(--color-surface-muted) / <alpha-value>)',
          /* Raw scale preserved for legacy uses (mapa_pabellon, etc.) */
          0:   'rgb(var(--color-surface-0) / <alpha-value>)',
          50:  'rgb(var(--color-surface-50) / <alpha-value>)',
          100: 'rgb(var(--color-surface-100) / <alpha-value>)',
          200: 'rgb(var(--color-surface-200) / <alpha-value>)',
          300: 'rgb(var(--color-surface-300) / <alpha-value>)',
          400: 'rgb(var(--color-surface-400) / <alpha-value>)',
          500: 'rgb(var(--color-surface-500) / <alpha-value>)',
          600: 'rgb(var(--color-surface-600) / <alpha-value>)',
          700: 'rgb(var(--color-surface-700) / <alpha-value>)',
          800: 'rgb(var(--color-surface-800) / <alpha-value>)',
          900: 'rgb(var(--color-surface-900) / <alpha-value>)',
        },
        /* Brand scale preserved for legacy uses; new code should use `primary`. */
        brand: {
          50:  'rgb(var(--color-brand-50) / <alpha-value>)',
          100: 'rgb(var(--color-brand-100) / <alpha-value>)',
          200: 'rgb(var(--color-brand-200) / <alpha-value>)',
          300: 'rgb(var(--color-brand-300) / <alpha-value>)',
          400: 'rgb(var(--color-brand-400) / <alpha-value>)',
          500: 'rgb(var(--color-brand-500) / <alpha-value>)',
          600: 'rgb(var(--color-brand-600) / <alpha-value>)',
          700: 'rgb(var(--color-brand-700) / <alpha-value>)',
          800: 'rgb(var(--color-brand-800) / <alpha-value>)',
          900: 'rgb(var(--color-brand-900) / <alpha-value>)',
        },
        /* Primary (accent) */
        primary: {
          DEFAULT: 'rgb(var(--color-primary) / <alpha-value>)',
          hover:   'rgb(var(--color-primary-hover) / <alpha-value>)',
          active:  'rgb(var(--color-primary-active) / <alpha-value>)',
          subtle:  'rgb(var(--color-primary-subtle) / <alpha-value>)',
        },
        /* Semantic states */
        success: {
          DEFAULT: 'rgb(var(--color-success) / <alpha-value>)',
          subtle:  'rgb(var(--color-success-subtle) / <alpha-value>)',
        },
        warn: {
          DEFAULT: 'rgb(var(--color-warn) / <alpha-value>)',
          subtle:  'rgb(var(--color-warn-subtle) / <alpha-value>)',
        },
        error: {
          DEFAULT: 'rgb(var(--color-error) / <alpha-value>)',
          subtle:  'rgb(var(--color-error-subtle) / <alpha-value>)',
        },
        info: {
          DEFAULT: 'rgb(var(--color-info) / <alpha-value>)',
          subtle:  'rgb(var(--color-info-subtle) / <alpha-value>)',
        },
      },
      textColor: {
        body:      'rgb(var(--color-text-body) / <alpha-value>)',
        heading:   'rgb(var(--color-text-heading) / <alpha-value>)',
        muted:     'rgb(var(--color-text-muted) / <alpha-value>)',
        disabled:  'rgb(var(--color-text-disabled) / <alpha-value>)',
        'on-accent': 'rgb(var(--color-text-on-accent) / <alpha-value>)',
      },
      borderColor: {
        subtle: 'rgb(var(--color-border-subtle) / <alpha-value>)',
        strong: 'rgb(var(--color-border-strong) / <alpha-value>)',
      },
      ringColor: {
        primary: 'rgb(var(--color-focus-ring) / <alpha-value>)',
      },
      boxShadow: {
        card:    'var(--shadow-card)',
        popover: 'var(--shadow-popover)',
        modal:   'var(--shadow-modal)',
        drawer:  'var(--shadow-drawer)',
      },
      borderRadius: {
        sm:   'var(--radius-sm)',
        md:   'var(--radius-md)',
        lg:   'var(--radius-lg)',
        xl:   'var(--radius-xl)',
        pill: 'var(--radius-pill)',
      },
      transitionDuration: {
        fast: 'var(--duration-fast)',
        base: 'var(--duration-base)',
        slow: 'var(--duration-slow)',
      },
      transitionTimingFunction: {
        standard:    'var(--ease-standard)',
        emphasized:  'var(--ease-emphasized)',
        accelerate:  'var(--ease-accelerate)',
      },
      zIndex: {
        base:     'var(--z-base)',
        sticky:   'var(--z-sticky)',
        topbar:   'var(--z-topbar)',
        backdrop: 'var(--z-backdrop)',
        drawer:   'var(--z-drawer)',
        modal:    'var(--z-modal)',
        popover:  'var(--z-popover)',
        toast:    'var(--z-toast)',
      },
      fontFamily: {
        sans: 'var(--font-sans)',
        mono: 'var(--font-mono)',
      },
    },
  },
};
