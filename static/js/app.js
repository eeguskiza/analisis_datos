/* ── Nexo — JavaScript helpers ──────────────────────────────────────────────── */

// ── Toast notifications (UI-SPEC §Flash toast, D-30) ──────────────────────
// Canonical signature: showToast(type, title, msg)
//   type:  info | success | warn | error |
//          (legacy alias) producing | stopped | incidence | alarm | turno
//   title: short bold line (required)
//   msg:   optional secondary text (may be null/undefined/empty)
//
// Phase 8 Pitfall 3: Historically `base.html` declared a 3-arg form and
// `app.js` a 2-arg `(message, type)` form. Plan 08-02 locks the 3-arg
// contract and updates every caller. Legacy type names map to the new
// set so the Alpine pabellon telemetry keeps working without churn.

const _TOAST_VARIANT_ALIAS = Object.freeze({
  producing: 'success',
  stopped:   'warn',
  incidence: 'error',
  alarm:     'error',
  turno:     'info',
});

const _TOAST_ICON_PATH = Object.freeze({
  info:    'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
  success: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
  warn:    'M12 9v2m0 4h.01M12 3l9 16H3l9-16z',
  error:   'M6 18L18 6M6 6l12 12',
});

const _TOAST_BORDER_COLOR = Object.freeze({
  info:    'border-l-info',
  success: 'border-l-success',
  warn:    'border-l-warn',
  error:   'border-l-error',
});

const _TOAST_ICON_COLOR = Object.freeze({
  info:    'text-info',
  success: 'text-success',
  warn:    'text-warn',
  error:   'text-error',
});

const _TOAST_ROLE = Object.freeze({
  info:    'status',
  success: 'status',
  warn:    'alert',
  error:   'alert',
});

window.showToast = function (type, title, msg) {
  const root = document.getElementById('toast-root');
  if (!root) return;

  const variant = _TOAST_VARIANT_ALIAS[type] || type;
  const safeVariant = ['info', 'success', 'warn', 'error'].includes(variant) ? variant : 'info';

  const el = document.createElement('div');
  el.setAttribute('role', _TOAST_ROLE[safeVariant]);
  el.setAttribute('tabindex', '0');
  el.className = [
    'pointer-events-auto',
    'bg-surface-base',
    'border', 'border-subtle', 'border-l-4', _TOAST_BORDER_COLOR[safeVariant],
    'rounded-md', 'shadow-popover',
    'px-4', 'py-3',
    'flex', 'items-start', 'gap-3',
    'min-w-0',
    'transition-base', 'ease-standard',
  ].join(' ');
  el.style.transform = 'translateX(16px)';
  el.style.opacity = '0';

  el.innerHTML = `
    <svg class="w-5 h-5 shrink-0 mt-0.5 ${_TOAST_ICON_COLOR[safeVariant]}" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" aria-hidden="true">
      <path stroke-linecap="round" stroke-linejoin="round" d="${_TOAST_ICON_PATH[safeVariant]}"/>
    </svg>
    <div class="min-w-0 flex-1">
      <div class="text-sm font-semibold text-heading">${_escape(title || '')}</div>
      ${msg ? `<div class="text-sm text-body mt-0.5">${_escape(msg)}</div>` : ''}
    </div>
    <button type="button" aria-label="Cerrar aviso" class="btn-icon text-muted hover:text-body shrink-0">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.75" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
      </svg>
    </button>
  `;

  root.appendChild(el);
  // Trigger enter animation on next frame
  requestAnimationFrame(() => {
    el.style.transform = 'translateX(0)';
    el.style.opacity = '1';
  });

  let dismissTimer = null;
  const dismiss = () => {
    if (dismissTimer) { clearTimeout(dismissTimer); dismissTimer = null; }
    el.style.transform = 'translateX(16px)';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 200);
  };
  const arm = () => { dismissTimer = setTimeout(dismiss, 4000); };

  // Pause on hover (D-30)
  el.addEventListener('mouseenter', () => { if (dismissTimer) { clearTimeout(dismissTimer); dismissTimer = null; } });
  el.addEventListener('mouseleave', arm);
  el.querySelector('button').addEventListener('click', dismiss);
  arm();
};

function _escape(str) {
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

// ── nexoChrome() Alpine component (Phase 8 / Plan 08-02) ──────────────────
// Drives the drawer open/close state + the [ keyboard shortcut.
// Persists `drawerOpen` via @alpinejs/persist under key `nexo.ui.drawerOpen`.
// Pitfall 5: $persist on first load without a persisted value returns the
// default (false) — safe. No migration needed.
// Pitfall 7: the [ listener guards against firing while typing in inputs.

function nexoChrome() {
  return {
    drawerOpen: (window.Alpine && window.Alpine.$persist)
      ? window.Alpine.$persist(false).as('nexo.ui.drawerOpen')
      : false,  // Fallback if $persist not yet loaded — rare
    toggleDrawer() { this.drawerOpen = !this.drawerOpen; },
    openDrawer()   { this.drawerOpen = true; },
    closeDrawer()  { this.drawerOpen = false; },
    onKeydown(e) {
      // Esc closes drawer first (z-order priority handled by Alpine @keydown)
      if (e.key === 'Escape' && this.drawerOpen) {
        this.closeDrawer();
        return;
      }
      // [ toggles drawer, unless typing
      if (e.key === '[' && !_isTyping(e.target)) {
        e.preventDefault();
        this.toggleDrawer();
      }
    },
  };
}

function _isTyping(el) {
  if (!el) return false;
  const tag = (el.tagName || '').toUpperCase();
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (el.isContentEditable) return true;
  return false;
}

// Expose for Alpine.data() auto-pickup
window.nexoChrome = nexoChrome;

// ── serviceHealth() — LEDs de estado en topbar ───────────────────────────────
function serviceHealth() {
  return {
    svcs: [
      { key: 'db_local', label: 'DB Local',  ok: null },
      { key: 'db_ecs',   label: 'DB ECS',    ok: null },
      { key: 'db_izaro', label: 'DB Izaro',  ok: null },
    ],
    _timer: null,
    init() {
      this._fetch();
      this._timer = setInterval(() => this._fetch(), 30000);
    },
    destroy() {
      if (this._timer) { clearInterval(this._timer); this._timer = null; }
    },
    async _fetch() {
      try {
        const r = await fetch('/api/health');
        const data = await r.json();
        this.svcs = this.svcs.map(s => ({ ...s, ok: data.services[s.key]?.ok ?? null }));
      } catch {
        this.svcs = this.svcs.map(s => ({ ...s, ok: false }));
      }
    }
  };
}
window.serviceHealth = serviceHealth;


// ── Landing: reloj (Plan 08-04 / UI-SPEC §Landing) ─────────────────────────
// Alpine component que muestra HH:MM:SS con tick cada 1000ms. El interval
// se limpia en destroy() — sin esto el timer sobrevive a navegaciones HTMX
// (Pitfall 6 de 08-RESEARCH). Formato 24h sin animaciones (prefers-reduced-
// motion respetado naturalmente: no hay CSS transition, solo texto).

function bienvenidaPage() {
  return {
    clock: '',
    dateStr: '',
    services: [
      { key: 'web',      label: 'Web',       ok: null },
      { key: 'db_local', label: 'DB Local',  ok: null },
      { key: 'db_ecs',   label: 'DB ECS',    ok: null },
      { key: 'db_izaro', label: 'DB Izaro',  ok: null },
    ],
    _timer: null,
    _healthTimer: null,
    init() {
      this._tick();
      this._timer = setInterval(() => this._tick(), 1000);
      this._fetchHealth();
      this._healthTimer = setInterval(() => this._fetchHealth(), 30000);
    },
    destroy() {
      if (this._timer)       { clearInterval(this._timer);       this._timer = null; }
      if (this._healthTimer) { clearInterval(this._healthTimer); this._healthTimer = null; }
    },
    _tick() {
      const now = new Date();
      this.clock   = now.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
      this.dateStr = now.toLocaleDateString('es-ES', { weekday: 'long', day: 'numeric', month: 'long' });
    },
    async _fetchHealth() {
      try {
        const r = await fetch('/api/health');
        const data = await r.json();
        this.services = this.services.map(s => ({ ...s, ok: data.services[s.key]?.ok ?? null }));
      } catch {
        this.services = this.services.map(s => ({ ...s, ok: false }));
      }
    },
  };
}

window.bienvenidaPage = bienvenidaPage;


// ── Conexion badge (responde al hx-get /api/conexion/status) ────────────────
document.addEventListener('htmx:afterRequest', (evt) => {
  if (!evt.detail.pathInfo?.requestPath?.includes('/api/conexion/status')) return;
  const badge = document.getElementById('conn-badge');
  if (!badge) return;

  try {
    const data = JSON.parse(evt.detail.xhr.responseText);
    if (data.ok) {
      badge.className = 'text-xs px-3 py-1 rounded-full conn-ok';
      badge.innerHTML = `<span class="inline-block w-2 h-2 rounded-full bg-green-500 mr-1"></span> ${data.database} @ ${data.server}`;
    } else {
      badge.className = 'text-xs px-3 py-1 rounded-full conn-err';
      badge.innerHTML = `<span class="inline-block w-2 h-2 rounded-full bg-red-500 mr-1"></span> Sin conexion`;
    }
  } catch {
    badge.className = 'text-xs px-3 py-1 rounded-full conn-err';
    badge.innerHTML = `<span class="inline-block w-2 h-2 rounded-full bg-red-500 mr-1"></span> Error`;
  }

  evt.detail.shouldSwap = false;
});


// ── SSE Pipeline helper ─────────────────────────────────────────────────────

function runPipeline(formData) {
  const logBox = document.getElementById('log-console');
  const pdfsPanel = document.getElementById('pdfs-panel');
  const pdfsList = document.getElementById('pdfs-list');
  const btnRun = document.getElementById('btn-run');
  const statusEl = document.getElementById('run-status');

  // Reset UI
  logBox.textContent = '';
  pdfsPanel.classList.add('hidden');
  pdfsList.innerHTML = '';
  btnRun.disabled = true;
  statusEl.innerHTML = '<span class="spinner border-brand-600"></span> <span class="text-sm font-medium text-gray-600">Ejecutando pipeline...</span>';

  fetch('/api/pipeline/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData),
  }).then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    function read() {
      reader.read().then(({ done, value }) => {
        if (done) {
          btnRun.disabled = false;
          return;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const msg = line.slice(6);

          if (msg.startsWith('DONE:')) {
            const parts = msg.split(':');
            const ejecId = parseInt(parts[1]) || 0;
            const count = parseInt(parts[2]) || 0;
            const pdfs = parts.slice(3).join(':').split('|').filter(Boolean);

            statusEl.innerHTML = `<span class="flex items-center gap-2"><svg class="w-5 h-5 text-green-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg><span class="text-sm font-semibold text-green-700">${count} PDF(s) generados</span></span>`;
            btnRun.disabled = false;
            notifyBrowser(`Pipeline completado: ${count} PDF(s)`);

            if (pdfs.length > 0) {
              pdfsPanel.classList.remove('hidden');
              pdfsList.innerHTML = pdfs.map(p => {
                const name = p.split('/').pop();
                return `<a href="/api/informes/pdf/${p}" target="_blank" title="${name}"
                           class="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-surface-50 hover:bg-surface-100 text-[11px] text-brand-700">
                          <svg class="w-3 h-3 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clip-rule="evenodd"/>
                          </svg>${name}</a>`;
              }).join('');
            }

            // Render interactive dashboard — minimize log to make room
            if (ejecId > 0) {
              const dashEl = document.getElementById('oee-dashboard');
              const logWrap = document.getElementById('log-wrapper');
              if (logWrap) logWrap.style.maxHeight = '60px';
              if (dashEl) {
                dashEl.classList.remove('hidden');
                renderOeeDashboard(dashEl, ejecId, pdfs);
              }
            }
          } else if (msg.startsWith('ERROR')) {
            logBox.textContent += msg + '\n';
            statusEl.innerHTML = '<span class="flex items-center gap-2"><svg class="w-5 h-5 text-red-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg><span class="text-sm font-semibold text-red-700">Error</span></span>';
            btnRun.disabled = false;
          } else {
            logBox.textContent += msg + '\n';
          }

          logBox.scrollTop = logBox.scrollHeight;
        }

        read();
      });
    }

    read();
  }).catch(err => {
    logBox.textContent += 'Error de red: ' + err + '\n';
    statusEl.innerHTML = '<span class="text-red-600 font-semibold text-sm">Error de red</span>';
    btnRun.disabled = false;
  });
}


// ── SECTION_MAP (para auto-detectar seccion de un recurso) ──────────────────
const SECTION_MAP = {
  luk1: 'LINEAS', luk2: 'LINEAS', luk3: 'LINEAS', luk6: 'LINEAS',
  coroa: 'LINEAS', vw1: 'LINEAS', omr: 'LINEAS', t48: 'TALLADORAS',
};

function getSectionForResource(name) {
  return SECTION_MAP[(name || '').toLowerCase()] || 'GENERAL';
}


// ── Browser notifications ────────────────────────────────────────────────────

function notifyBrowser(message) {
  if (!('Notification' in window)) return;
  const cfg = window.NEXO_CONFIG || {};
  const title = cfg.appName || 'Nexo';
  const icon = cfg.logoPath || '/static/img/brand/nexo/logo.png';
  if (Notification.permission === 'granted') {
    new Notification(title, { body: message, icon });
  } else if (Notification.permission !== 'denied') {
    Notification.requestPermission().then(p => {
      if (p === 'granted') new Notification(title, { body: message, icon });
    });
  }
}

if ('Notification' in window && Notification.permission === 'default') {
  Notification.requestPermission();
}


// ── OEE Dashboard renderer ────────────────────────────────────────────────

let _dashboardCharts = [];

function _destroyCharts() {
  _dashboardCharts.forEach(c => c.destroy());
  _dashboardCharts = [];
}

function _pctColor(v) {
  if (v >= 85) return 'text-green-700';
  if (v >= 60) return 'text-amber-600';
  return 'text-red-600';
}

function _pctBg(v) {
  if (v >= 85) return 'bg-green-500';
  if (v >= 60) return 'bg-amber-500';
  return 'bg-red-500';
}

function _hhmm(hours) {
  if (!hours || hours === 0) return '0:00';
  const h = Math.floor(hours);
  const m = Math.round((hours - h) * 60);
  return `${h}:${m < 10 ? '0' : ''}${m}`;
}

function renderOeeDashboard(container, ejecIdOrData, pdfs) {
  _destroyCharts();
  container.innerHTML = '<div class="flex items-center justify-center h-full"><span class="spinner border-brand-600 w-5 h-5"></span></div>';

  const render = (data) => {
    if (!data || data.error || !data.secciones) {
      container.innerHTML = '<p class="text-xs text-gray-400 text-center py-8">Sin datos</p>';
      return;
    }
    container.innerHTML = '';
    const entries = Object.entries(data.secciones);
    const isMulti = entries.length > 1;
    container.classList.toggle('space-y-6', isMulti);
    entries.forEach(([secName, sec]) => _renderSection(container, secName, sec, pdfs || [], data, isMulti));
  };

  if (typeof ejecIdOrData === 'object') {
    render(ejecIdOrData);
  } else {
    fetch(`/api/historial/${ejecIdOrData}/metrics`)
      .then(r => r.json()).then(render)
      .catch(() => { container.innerHTML = '<p class="text-xs text-red-400 text-center py-8">Error</p>'; });
  }
}

function _renderSection(container, secName, sec, pdfs, data, isMulti = false) {
  const t = sec.totales;
  const wrapper = document.createElement('div');
  wrapper.className = isMulti ? 'flex flex-col gap-3' : 'h-full flex flex-col gap-3';
  if (isMulti) wrapper.style.minHeight = '560px';

  if (isMulti) {
    const secHeader = document.createElement('div');
    secHeader.className = 'flex items-center gap-2 shrink-0';
    const accent = secName === 'LINEAS' ? 'bg-blue-500' : secName === 'TALLADORAS' ? 'bg-amber-500' : 'bg-gray-400';
    secHeader.innerHTML = `
      <span class="inline-block w-1 h-4 rounded ${accent}"></span>
      <span class="text-sm font-bold text-gray-700 uppercase tracking-wider">${secName}</span>
      <span class="text-[10px] text-gray-400">${sec.maquinas.length} maquina(s) · OEE ${t.oee_pct.toFixed(1)}%</span>
      <span class="flex-1 border-t border-surface-200"></span>`;
    wrapper.appendChild(secHeader);
  }

  // ── ROW 1: KPI cards ──
  const kpiRow = document.createElement('div');
  kpiRow.className = 'grid grid-cols-4 gap-3 shrink-0';
  const kpis = [
    { l: 'OEE', v: t.oee_pct, sub: `${_hhmm(t.horas_operativo)}h operativas` },
    { l: 'Disponibilidad', v: t.disponibilidad_pct, sub: `${_hhmm(t.horas_brutas)}h brutas · ${_hhmm(t.horas_paros)}h paros` },
    { l: 'Rendimiento', v: t.rendimiento_pct, sub: `${t.piezas_totales.toLocaleString('es')} piezas producidas` },
    { l: 'Calidad', v: t.calidad_pct, sub: `${t.buenas_finales.toLocaleString('es')} buenas · ${(t.piezas_totales - t.buenas_finales).toLocaleString('es')} scrap` },
  ];
  kpiRow.innerHTML = kpis.map(k => {
    const barW = Math.min(Math.max(k.v, 0), 100);
    return `
    <div class="bg-white border border-surface-200 rounded-lg p-3 flex flex-col justify-between">
      <div class="flex items-center justify-between mb-1">
        <span class="text-[11px] font-semibold text-gray-500 uppercase tracking-wide">${k.l}</span>
        ${pdfs.length > 0 && k.l === 'OEE' ? `<div class="flex gap-0.5">${pdfs.slice(0, 4).map(p => `<a href="/api/informes/pdf/${p}" target="_blank" title="${p.split('/').pop()}" class="p-0.5 rounded hover:bg-surface-100"><svg class="w-3 h-3 text-red-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clip-rule="evenodd"/></svg></a>`).join('')}</div>` : ''}
      </div>
      <div class="text-2xl font-black ${_pctColor(k.v)} leading-none">${k.v.toFixed(1)}<span class="text-sm font-bold">%</span></div>
      <div class="mt-2">
        <div class="w-full h-1.5 bg-surface-100 rounded-full overflow-hidden">
          <div class="h-full rounded-full ${_pctBg(k.v)} transition-all" style="width:${barW}%"></div>
        </div>
        <div class="text-[10px] text-gray-400 mt-1">${k.sub}</div>
      </div>
    </div>`;
  }).join('');
  wrapper.appendChild(kpiRow);

  // ── ROW 2: Chart + Table ──
  const mainRow = document.createElement('div');
  mainRow.className = 'flex-1 min-h-0 flex gap-3';

  // LEFT: Chart
  const chartPanel = document.createElement('div');
  chartPanel.className = 'w-2/5 shrink-0 bg-white border border-surface-200 rounded-lg p-3 flex flex-col min-h-0';
  const multiDay = sec.resumen_diario && sec.resumen_diario.length > 1;
  chartPanel.innerHTML = `
    <div class="flex items-center justify-between mb-2 shrink-0">
      <span class="text-[11px] font-semibold text-gray-500 uppercase tracking-wide">${multiDay ? 'Tendencia diaria' : 'OEE por maquina'}</span>
      <span class="text-[10px] text-gray-400">${secName} · ${data.fecha_inicio}${data.fecha_inicio !== data.fecha_fin ? ' → ' + data.fecha_fin : ''}</span>
    </div>`;
  const canvasWrap = document.createElement('div');
  canvasWrap.className = 'flex-1 min-h-0 relative';
  const canvas = document.createElement('canvas');
  canvasWrap.appendChild(canvas);
  chartPanel.appendChild(canvasWrap);
  mainRow.appendChild(chartPanel);

  // RIGHT: Table
  const tablePanel = document.createElement('div');
  tablePanel.className = 'flex-1 min-w-0 bg-white border border-surface-200 rounded-lg flex flex-col min-h-0 overflow-hidden';

  const tableId = 'oee-tbl-' + Math.random().toString(36).slice(2, 8);
  tablePanel.innerHTML = `
    <div class="flex items-center justify-between px-3 py-2 border-b border-surface-100 shrink-0">
      <span class="text-[11px] font-semibold text-gray-500 uppercase tracking-wide">Desglose por maquina</span>
      <span class="text-[10px] text-gray-400">${sec.maquinas.length} maquina(s)</span>
    </div>
    <div class="flex-1 overflow-y-auto">
      <table id="${tableId}" class="w-full text-xs border-collapse">
        <thead class="sticky top-0 z-10 bg-surface-50">
          <tr class="text-[10px] text-gray-500 uppercase tracking-wider">
            <th class="text-left px-3 py-2 font-semibold">Maquina</th>
            <th class="text-right px-2 py-2 font-semibold">T.Disp</th>
            <th class="text-right px-2 py-2 font-semibold">T.Oper</th>
            <th class="text-right px-2 py-2 font-semibold">Paros</th>
            <th class="text-right px-2 py-2 font-semibold">Piezas</th>
            <th class="text-right px-2 py-2 font-semibold">Disp%</th>
            <th class="text-right px-2 py-2 font-semibold">Rend%</th>
            <th class="text-right px-2 py-2 font-semibold">Cal%</th>
            <th class="text-right px-2 py-2 font-semibold w-20">OEE%</th>
          </tr>
        </thead>
        <tbody>
          ${sec.maquinas.map((m, idx) => {
            const shifts = ['T1','T2','T3'].map(s => m.turnos?.[s]).filter(st => st && st.piezas_totales > 0);
            const hasShifts = shifts.length > 0;
            const rowId = tableId + '-r' + idx;
            return `
          <tr class="border-t border-surface-200 hover:bg-blue-50/30 cursor-pointer transition-colors" onclick="document.querySelectorAll('.${rowId}').forEach(r=>r.classList.toggle('hidden'))">
            <td class="px-3 py-2.5 font-semibold text-gray-800">
              <div class="flex items-center gap-2">
                ${hasShifts ? `<svg class="w-3 h-3 text-gray-300 shrink-0 shift-chevron" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/></svg>` : '<span class="w-3"></span>'}
                ${m.nombre}
              </div>
            </td>
            <td class="text-right px-2 py-2.5 font-mono text-gray-500">${_hhmm(m.horas_disponible)}</td>
            <td class="text-right px-2 py-2.5 font-mono text-gray-500">${_hhmm(m.horas_operativo)}</td>
            <td class="text-right px-2 py-2.5 font-mono text-gray-500">${_hhmm(m.horas_paros)}</td>
            <td class="text-right px-2 py-2.5 text-gray-700">${m.piezas_totales.toLocaleString('es')}</td>
            <td class="text-right px-2 py-2.5 font-medium ${_pctColor(m.disponibilidad_pct)}">${m.disponibilidad_pct.toFixed(1)}</td>
            <td class="text-right px-2 py-2.5 font-medium ${_pctColor(m.rendimiento_pct)}">${m.rendimiento_pct.toFixed(1)}</td>
            <td class="text-right px-2 py-2.5 font-medium ${_pctColor(m.calidad_pct)}">${m.calidad_pct.toFixed(1)}</td>
            <td class="text-right px-2 py-2.5">
              <div class="flex items-center justify-end gap-1.5">
                <div class="w-12 h-1.5 bg-surface-100 rounded-full overflow-hidden"><div class="h-full rounded-full ${_pctBg(m.oee_pct)}" style="width:${Math.min(m.oee_pct,100)}%"></div></div>
                <span class="font-bold ${_pctColor(m.oee_pct)} w-10 text-right">${m.oee_pct.toFixed(1)}</span>
              </div>
            </td>
          </tr>
          ${['T1','T2','T3'].map(s => {
            const st = m.turnos?.[s];
            if (!st || st.piezas_totales === 0) return '';
            return `
          <tr class="${rowId} hidden bg-surface-50/50 text-[11px] text-gray-500">
            <td class="px-3 py-1.5 pl-10">${s}</td>
            <td class="text-right px-2 py-1.5 font-mono">${_hhmm(st.horas_disponible)}</td>
            <td class="text-right px-2 py-1.5 font-mono">${_hhmm(st.horas_operativo)}</td>
            <td class="text-right px-2 py-1.5 font-mono">${_hhmm(st.horas_paros)}</td>
            <td class="text-right px-2 py-1.5">${st.piezas_totales.toLocaleString('es')}</td>
            <td class="text-right px-2 py-1.5">${st.disponibilidad_pct.toFixed(1)}</td>
            <td class="text-right px-2 py-1.5">${st.rendimiento_pct.toFixed(1)}</td>
            <td class="text-right px-2 py-1.5">${st.calidad_pct.toFixed(1)}</td>
            <td class="text-right px-2 py-1.5"><div class="flex items-center justify-end gap-1.5"><div class="w-12 h-1 bg-surface-100 rounded-full overflow-hidden"><div class="h-full rounded-full ${_pctBg(st.oee_pct)}" style="width:${Math.min(st.oee_pct,100)}%"></div></div><span class="w-10 text-right">${st.oee_pct.toFixed(1)}</span></div></td>
          </tr>`;
          }).join('')}`;
          }).join('')}
          <tr class="border-t-2 border-brand-300 bg-brand-50/60">
            <td class="px-3 py-2.5 font-bold text-gray-800"><span class="pl-5">TOTAL</span></td>
            <td class="text-right px-2 py-2.5 font-mono font-semibold">${_hhmm(t.horas_disponible)}</td>
            <td class="text-right px-2 py-2.5 font-mono font-semibold">${_hhmm(t.horas_operativo)}</td>
            <td class="text-right px-2 py-2.5 font-mono font-semibold">${_hhmm(t.horas_paros)}</td>
            <td class="text-right px-2 py-2.5 font-semibold">${t.piezas_totales.toLocaleString('es')}</td>
            <td class="text-right px-2 py-2.5 font-semibold ${_pctColor(t.disponibilidad_pct)}">${t.disponibilidad_pct.toFixed(1)}</td>
            <td class="text-right px-2 py-2.5 font-semibold ${_pctColor(t.rendimiento_pct)}">${t.rendimiento_pct.toFixed(1)}</td>
            <td class="text-right px-2 py-2.5 font-semibold ${_pctColor(t.calidad_pct)}">${t.calidad_pct.toFixed(1)}</td>
            <td class="text-right px-2 py-2.5"><div class="flex items-center justify-end gap-1.5"><div class="w-12 h-1.5 bg-surface-100 rounded-full overflow-hidden"><div class="h-full rounded-full ${_pctBg(t.oee_pct)}" style="width:${Math.min(t.oee_pct,100)}%"></div></div><span class="font-black ${_pctColor(t.oee_pct)} w-10 text-right">${t.oee_pct.toFixed(1)}</span></div></td>
          </tr>
        </tbody>
      </table>
    </div>`;
  mainRow.appendChild(tablePanel);
  wrapper.appendChild(mainRow);
  container.appendChild(wrapper);

  // ── Chart ──
  requestAnimationFrame(() => {
    const chartOpts = {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, padding: 10, font: { size: 10 } } } },
    };

    if (multiDay) {
      const days = sec.resumen_diario.map(d => d.fecha.slice(5));
      _dashboardCharts.push(new Chart(canvas, {
        type: 'line',
        data: {
          labels: days,
          datasets: [
            { label: 'OEE', data: sec.resumen_diario.map(d => d.oee_pct), borderColor: '#4f46e5', backgroundColor: '#4f46e515', fill: true, tension: 0.3, pointRadius: 3, borderWidth: 2.5 },
            { label: 'Disp.', data: sec.resumen_diario.map(d => d.disponibilidad_pct), borderColor: '#3b82f6', tension: 0.3, pointRadius: 2, borderWidth: 1.5, borderDash: [4, 2] },
            { label: 'Rend.', data: sec.resumen_diario.map(d => d.rendimiento_pct), borderColor: '#d97706', tension: 0.3, pointRadius: 2, borderWidth: 1.5, borderDash: [4, 2] },
            { label: 'Cal.', data: sec.resumen_diario.map(d => d.calidad_pct), borderColor: '#059669', tension: 0.3, pointRadius: 2, borderWidth: 1.5, borderDash: [4, 2] },
          ]
        },
        options: { ...chartOpts,
          scales: { y: { min: 50, max: 100, ticks: { callback: v => v + '%', font: { size: 10 } } }, x: { ticks: { font: { size: 10 } } } }
        }
      }));
    } else {
      // OEE per machine horizontal bar
      const names = sec.maquinas.map(m => m.nombre);
      _dashboardCharts.push(new Chart(canvas, {
        type: 'bar',
        data: {
          labels: names,
          datasets: [
            { label: 'Disp.', data: sec.maquinas.map(m => m.disponibilidad_pct), backgroundColor: '#3b82f6', borderRadius: 3, barPercentage: 0.7 },
            { label: 'Rend.', data: sec.maquinas.map(m => m.rendimiento_pct), backgroundColor: '#d97706', borderRadius: 3, barPercentage: 0.7 },
            { label: 'Cal.', data: sec.maquinas.map(m => m.calidad_pct), backgroundColor: '#059669', borderRadius: 3, barPercentage: 0.7 },
            { label: 'OEE', data: sec.maquinas.map(m => m.oee_pct), backgroundColor: '#4f46e5', borderRadius: 3, barPercentage: 0.7 },
          ]
        },
        options: { ...chartOpts, indexAxis: 'y',
          scales: { x: { min: 0, max: 100, ticks: { callback: v => v + '%', font: { size: 10 } } }, y: { ticks: { font: { size: 11, weight: 'bold' } } } }
        }
      }));
    }
  });
}


// ── Preflight modal (Phase 4 / Plan 04-02) ──────────────────────────────────
// Humanizes milliseconds into operator-friendly strings.
//
// Examples:
//   humanize_ms(500)    -> "~500ms"
//   humanize_ms(12000)  -> "~12s"
//   humanize_ms(125000) -> "~2 min 5s"
//   humanize_ms(4000000) -> "~1h 6 min"
function humanize_ms(ms) {
  if (ms == null || isNaN(ms)) return '~?';
  if (ms < 1000) return `~${Math.round(ms)}ms`;
  if (ms < 60000) return `~${Math.round(ms / 1000)}s`;
  if (ms < 3600000) {
    const mins = Math.floor(ms / 60000);
    const secs = Math.round((ms % 60000) / 1000);
    return `~${mins} min ${secs}s`;
  }
  const hours = Math.floor(ms / 3600000);
  const mins = Math.floor((ms % 3600000) / 60000);
  return `~${hours}h ${mins} min`;
}


// Alpine component factory — register on document ready so Alpine.data is available.
// Register preflightModal globally so any page can `<div x-data="preflightModal">`.
//
// Flow:
//   1. User clicks "Ejecutar" button; page calls attempt(endpoint, params, executeFn).
//   2. attempt POSTs /api/<endpoint>/preflight to obtain Estimation.
//   3a. Green: executes immediately (executeFn(false, null)).
//   3b. Amber/Red: stores state, opens corresponding modal.
//   4. Amber [Continuar] calls confirmRun → executeFn(true, null) (force=true).
//   5. Red [Solicitar aprobacion] POSTs /api/approvals → toast + modal close.
//
// URL re-dispatch (D-15 / PC-04-02): if URL has `?approval_id=<N>`, the first
// attempt() auto-executes with force=true + approval_id if preflight returns red,
// bypassing the modal. This is the happy path after user clicks "Ejecutar ahora"
// from /mis-solicitudes.
document.addEventListener('alpine:init', () => {
  if (!window.Alpine) return;
  Alpine.data('preflightModal', () => ({
    modalLevel: null,        // null | 'amber' | 'red'
    estimation: null,        // Estimation DTO from backend
    currentParams: null,     // payload that triggered preflight
    currentEndpoint: null,   // 'pipeline' | 'bbdd' | 'capacidad' | 'operarios'
    currentExecuteFn: null,  // callback(force, approvalId) -> Promise
    pendingApprovalId: null, // From URL ?approval_id=<N> (user re-dispatch)

    init() {
      const params = new URLSearchParams(window.location.search);
      const aid = params.get('approval_id');
      this.pendingApprovalId = aid ? parseInt(aid, 10) : null;
    },

    humanize(ms) {
      return humanize_ms(ms);
    },

    async attempt(endpoint, params, executeFn) {
      // Store for later use (modal callbacks + toast routing).
      this.currentParams = params;
      this.currentEndpoint = endpoint;
      this.currentExecuteFn = executeFn;

      let est;
      try {
        const res = await fetch(`/api/${endpoint}/preflight`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params),
        });
        if (!res.ok) {
          // Fallback: preflight endpoint not implemented (e.g. bbdd/query
          // keeps its gate in-route). Skip modal and execute directly — the
          // route itself will return 428/403 if needed.
          return await executeFn(false, null);
        }
        est = await res.json();
      } catch (err) {
        // Network error — fail open to execute (route will gate properly).
        return await executeFn(false, null);
      }

      this.estimation = est;

      // URL re-dispatch (user clicked "Ejecutar ahora" from /mis-solicitudes):
      // if URL carries approval_id and preflight says red, auto-execute with
      // force=true + that approval_id, skipping modal entirely.
      if (this.pendingApprovalId && est.level === 'red') {
        const aid = this.pendingApprovalId;
        this.pendingApprovalId = null;
        return await executeFn(true, aid);
      }

      if (est.level === 'green') {
        return await executeFn(false, null);
      }
      // amber or red → open modal; user decides.
      this.modalLevel = est.level;
    },

    async confirmRun() {
      // AMBER path: close modal, execute with force=true.
      this.modalLevel = null;
      await this.currentExecuteFn(true, null);
    },

    async requestApproval() {
      // RED path: POST /api/approvals (stub until Plan 04-03). Backend
      // returns { approval_id, status }. Show toast with link to /mis-solicitudes.
      try {
        const res = await fetch('/api/approvals', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            endpoint: this.currentEndpoint,
            params: this.currentParams,
            estimated_ms: this.estimation ? this.estimation.estimated_ms : null,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          const approvalId = data.approval_id != null ? data.approval_id : '?';
          window.showToast('info', 'Solicitud enviada', `Ver en /mis-solicitudes (#${approvalId}).`);
        } else {
          // Until Plan 04-03 lands, /api/approvals may 404 or 501.
          // Inform the user clearly instead of silently failing.
          window.showToast('error', 'Aprobaciones no disponibles', 'Esta función todavía no está habilitada.');
        }
      } catch (err) {
        window.showToast('error', 'Error', 'No se pudo enviar la solicitud.');
      }
      this.modalLevel = null;
    },

    cancel() {
      this.modalLevel = null;
      this.estimation = null;
      this.currentParams = null;
      this.currentExecuteFn = null;
    },
  }));
});
