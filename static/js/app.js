/* ── ECS Mobility — JavaScript helpers ──────────────────────────────────────── */

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
            const count = parseInt(parts[1]) || 0;
            const pdfs = parts.slice(2).join(':').split('|').filter(Boolean);

            statusEl.innerHTML = `<span class="flex items-center gap-2"><svg class="w-5 h-5 text-green-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg><span class="text-sm font-semibold text-green-700">${count} PDF(s) generados</span></span>`;
            btnRun.disabled = false;
            notifyBrowser(`Pipeline completado: ${count} PDF(s)`);

            if (pdfs.length > 0) {
              pdfsPanel.classList.remove('hidden');
              pdfsList.innerHTML = pdfs.map(p => {
                const name = p.split('/').pop();
                const section = p.split('/').slice(-2, -1)[0] || '';
                return `<a href="/api/informes/pdf/${p}" target="_blank"
                           class="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-surface-50 hover:bg-surface-100 transition-colors text-sm text-brand-700">
                          <svg class="w-4 h-4 text-red-400 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clip-rule="evenodd"/>
                          </svg>
                          <span class="truncate">${name}</span>
                          <span class="text-xs text-gray-400">${section}</span>
                        </a>`;
              }).join('');
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
  if (Notification.permission === 'granted') {
    new Notification('ECS Mobility', { body: message, icon: '/static/img/logo.png' });
  } else if (Notification.permission !== 'denied') {
    Notification.requestPermission().then(p => {
      if (p === 'granted') new Notification('ECS Mobility', { body: message });
    });
  }
}

if ('Notification' in window && Notification.permission === 'default') {
  Notification.requestPermission();
}


// ── Toast notifications ─────────────────────────────────────────────────────

function showToast(message, type = 'success') {
  const colors = {
    success: 'bg-green-600',
    error: 'bg-red-600',
    info: 'bg-brand-600',
  };
  const toast = document.createElement('div');
  toast.className = `fixed bottom-4 right-4 ${colors[type] || colors.info} text-white px-4 py-2.5 rounded-xl shadow-lg text-sm z-50 transition-opacity duration-300`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 2500);
}
