"""Informe OEE LUK4 — una hoja A4, logo, tabla de números + gráfico de disponibilidad."""
from __future__ import annotations

import io
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

CICLO_NOMINAL_S = 25.2
LOGO_PATH = Path(__file__).resolve().parents[2] / "static" / "img" / "logo.png"
BRAND  = "#1a3a5c"
GRAY   = "#6b7280"
LGRAY  = "#9ca3af"
XGRAY  = "#e5e7eb"
BG     = "#f9fafb"

C_PROD  = "#22c55e"   # produciendo
C_ALARM = "#f59e0b"   # incidencia (estado 3, sigue produciendo)
C_STOP  = "#ef4444"   # parada real


# ── Helpers ──────────────────────────────────────────────────────────────────

def _color(pct: float) -> str:
    if pct >= 75: return "#16a34a"
    if pct >= 50: return "#d97706"
    return "#dc2626"


def _sep(ax, y: float) -> None:
    ax.plot([0.06, 0.94], [y, y], transform=ax.transAxes,
            color=XGRAY, linewidth=0.9, solid_capstyle="round")


def _txt(ax, x, y, s, **kw):
    kw.setdefault("transform", ax.transAxes)
    kw.setdefault("va", "center")
    ax.text(x, y, s, **kw)


def _row_bg(ax, yc: float, color: str = BG, xpad: float = 0.06) -> None:
    ax.add_patch(mpatches.FancyBboxPatch(
        (xpad, yc - 0.018), 1 - 2 * xpad, 0.034,
        boxstyle="round,pad=0.004", facecolor=color,
        edgecolor="none", transform=ax.transAxes, clip_on=False,
    ))


def _kpi_box(ax, x, y, w, h, val, lbl, color):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        linewidth=1.5, edgecolor=color, facecolor=color + "1a",
        transform=ax.transAxes, clip_on=False,
    ))
    _txt(ax, x + w/2, y + h*0.62, val, ha="center",
         fontsize=23, fontweight="bold", color=color, fontfamily="monospace")
    _txt(ax, x + w/2, y + h*0.18, lbl, ha="center",
         fontsize=7, fontweight="bold", color=GRAY)


def _add_logo(ax, xc: float, yc: float, zoom: float = 0.072) -> None:
    if not LOGO_PATH.exists():
        return
    ib = OffsetImage(mpimg.imread(str(LOGO_PATH)), zoom=zoom)
    ax.add_artist(AnnotationBbox(ib, (xc, yc), xycoords="axes fraction",
                                 frameon=False, box_alignment=(0.5, 0.5)))


def _metricas(conn, ts_ini: datetime, ts_fin: datetime) -> dict:
    from sqlalchemy import text
    p = {"ts": ts_ini, "te": ts_fin}

    e_rows = conn.execute(text("""
        SELECT estado_global, COUNT(*) AS n
        FROM luk4.estado
        WHERE timestamp >= :ts AND timestamp < :te
        GROUP BY estado_global
    """), p).fetchall()

    pz = conn.execute(text("""
        SELECT
          COALESCE((SELECT TOP 1 contador_piezas_buenas FROM luk4.tiempos_ciclo
                    WHERE timestamp <  :ts AND contador_piezas_buenas IS NOT NULL
                    ORDER BY timestamp DESC),
                   (SELECT TOP 1 contador_piezas_buenas FROM luk4.tiempos_ciclo
                    WHERE timestamp >= :ts AND contador_piezas_buenas IS NOT NULL
                    ORDER BY timestamp ASC)),
          (SELECT TOP 1 contador_piezas_buenas FROM luk4.tiempos_ciclo
           WHERE timestamp < :te AND contador_piezas_buenas IS NOT NULL
           ORDER BY timestamp DESC),
          COALESCE((SELECT TOP 1 contador_piezas_malas FROM luk4.tiempos_ciclo
                    WHERE timestamp <  :ts AND contador_piezas_malas IS NOT NULL
                    ORDER BY timestamp DESC),
                   (SELECT TOP 1 contador_piezas_malas FROM luk4.tiempos_ciclo
                    WHERE timestamp >= :ts AND contador_piezas_malas IS NOT NULL
                    ORDER BY timestamp ASC)),
          (SELECT TOP 1 contador_piezas_malas FROM luk4.tiempos_ciclo
           WHERE timestamp < :te AND contador_piezas_malas IS NOT NULL
           ORDER BY timestamp DESC),
          COALESCE((SELECT TOP 1 contador_piezas_totales FROM luk4.tiempos_ciclo
                    WHERE timestamp <  :ts AND contador_piezas_totales IS NOT NULL
                    ORDER BY timestamp DESC),
                   (SELECT TOP 1 contador_piezas_totales FROM luk4.tiempos_ciclo
                    WHERE timestamp >= :ts AND contador_piezas_totales IS NOT NULL
                    ORDER BY timestamp ASC)),
          (SELECT TOP 1 contador_piezas_totales FROM luk4.tiempos_ciclo
           WHERE timestamp < :te AND contador_piezas_totales IS NOT NULL
           ORDER BY timestamp DESC)
    """), p).fetchone()

    span = conn.execute(text("""
        SELECT MIN(timestamp), MAX(timestamp)
        FROM luk4.tiempos_ciclo
        WHERE timestamp >= :ts AND timestamp < :te
    """), p).fetchone()

    ec      = {int(r[0]): int(r[1]) for r in e_rows}
    n_prod  = ec.get(1, 0)
    n_alarm = ec.get(3, 0)
    n_total = sum(ec.values())
    n_stop  = n_total - n_prod - n_alarm

    D = (n_prod + n_alarm) / n_total * 100 if n_total > 0 else 0

    pz_b = max(int((pz[1] or 0) - (pz[0] or 0)), 0) if pz else 0
    pz_m = max(int((pz[3] or 0) - (pz[2] or 0)), 0) if pz else 0
    # piezas inspeccionadas = buenas + malas (no usar contador_totales, que incluye
    # bandeja y piezas en proceso y da siempre ~100 % de calidad)
    pz_insp = pz_b + pz_m

    C = pz_b / pz_insp * 100 if pz_insp > 0 else 0

    span_s  = (span[1] - span[0]).total_seconds() if span and span[0] and span[1] else 0
    t_avail = (n_prod + n_alarm) / n_total * span_s if n_total > 0 and span_s > 0 else 0
    pz_teo  = t_avail / CICLO_NOMINAL_S if CICLO_NOMINAL_S > 0 else 1
    R       = min(pz_insp / pz_teo * 100, 100.0) if pz_teo > 0 else 0
    OEE     = D * R * C / 10000

    return {
        "D": round(D, 1), "R": round(R, 1), "C": round(C, 1), "OEE": round(OEE, 1),
        "pz_b": pz_b, "pz_m": pz_m, "pz_insp": pz_insp,
        "pz_teo": round(pz_teo),
        "h_avail": round(t_avail / 3600, 1),
        "n_prod": n_prod, "n_alarm": n_alarm, "n_stop": n_stop, "n_total": n_total,
        "has_data": n_total > 0,
    }


# ── Gráfico de disponibilidad por turno (barras apiladas horizontales) ────────

def _draw_turno_chart(fig, turnos: list, y_bottom: float, height: float) -> None:
    """3 barras horizontales apiladas T1/T2/T3: Produciendo (est=1+3) | Parada (est=0+2+4).

    Estado=3 = alarma activa pero la línea sigue produciendo → se suma al verde.
    """
    chart_ax = fig.add_axes([0.19, y_bottom, 0.72, height])
    chart_ax.set_facecolor("white")
    for spine in ["top", "right"]:
        chart_ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        chart_ax.spines[spine].set_color(XGRAY)

    SHIFT_H = 8.0   # duración nominal de cada turno en horas
    y_pos   = [2, 1, 0]   # T1 arriba, T3 abajo
    bar_h   = 0.52

    # estado=3 produce piezas a ritmo normal → cuenta como tiempo disponible
    h_disp = [(t["n_prod"] + t["n_alarm"]) / t["n_total"] * SHIFT_H
              if t["n_total"] > 0 else 0 for t in turnos]
    h_stop = [t["n_stop"] / t["n_total"] * SHIFT_H
              if t["n_total"] > 0 else 0 for t in turnos]

    chart_ax.barh(y_pos, h_disp, bar_h, color=C_PROD, label="Produciendo", zorder=3)
    chart_ax.barh(y_pos, h_stop, bar_h, left=h_disp, color=C_STOP, label="Parada", zorder=3)

    # Etiquetas dentro de cada segmento
    for i, yi in enumerate(y_pos):
        if h_disp[i] >= 0.5:
            chart_ax.text(h_disp[i] / 2, yi, f"{h_disp[i]:.1f}h",
                          ha="center", va="center", fontsize=8,
                          fontweight="bold", color="white", zorder=4)
        if h_stop[i] >= 0.4:
            chart_ax.text(h_disp[i] + h_stop[i] / 2, yi, f"{h_stop[i]:.1f}h",
                          ha="center", va="center", fontsize=8,
                          fontweight="bold", color="white", zorder=4)
        # % disponibilidad en el extremo derecho
        pct = turnos[i]["D"]
        chart_ax.text(SHIFT_H + 0.08, yi, f"{pct}%",
                      ha="left", va="center", fontsize=8.5,
                      fontweight="bold", color=_color(pct))

    chart_ax.set_yticks(y_pos)
    chart_ax.set_yticklabels(
        [t["nombre"] for t in turnos],
        fontsize=10, fontweight="bold", color=BRAND,
    )
    chart_ax.set_xlim(0, SHIFT_H + 0.7)
    chart_ax.tick_params(axis="x", labelsize=7, colors=GRAY)
    chart_ax.tick_params(axis="y", length=0)
    chart_ax.xaxis.set_ticks([0, 2, 4, 6, 8])
    chart_ax.set_xticklabels(["0h", "2h", "4h", "6h", "8h"])
    chart_ax.grid(axis="x", color=XGRAY, linewidth=0.5, zorder=0)


# ── Tabla de totales con columnas fijas ────────────────────────────────────────

# Cada par (label_x, value_x) define una columna; values siempre right-aligned.
_TOT_COLS = [(0.08, 0.335), (0.38, 0.625), (0.66, 0.94)]


def _tot_row(ax, y, row_data, row_bg=False):
    if row_bg:
        _row_bg(ax, y)
    for (lx, vx), (label, val, vc) in zip(_TOT_COLS, row_data):
        _txt(ax, lx, y, label, fontsize=8.5, color=GRAY)
        _txt(ax, vx, y, val, ha="right", fontsize=9, fontweight="bold",
             color=vc, fontfamily="monospace")


# ── Informe de un día con turnos ──────────────────────────────────────────────

def _informe_dia(fecha: date) -> bytes:
    from api.database import engine
    from sqlalchemy import text

    jornada_ini = datetime(fecha.year, fecha.month, fecha.day, 6, 0, 0)
    jornada_fin = jornada_ini + timedelta(hours=24)

    with engine.connect() as conn:
        global_m = _metricas(conn, jornada_ini, jornada_fin)

        turnos = []
        for nombre, h_ini, h_fin in [("T1", 6, 14), ("T2", 14, 22), ("T3", 22, 6)]:
            ts = datetime(fecha.year, fecha.month, fecha.day, h_ini)
            te = (datetime(fecha.year, fecha.month, fecha.day, h_fin)
                  if nombre != "T3"
                  else datetime(fecha.year, fecha.month, fecha.day, 6) + timedelta(days=1))
            horario = ("22:00–06:00" if nombre == "T3"
                       else f"{h_ini:02d}:00–{h_fin:02d}:00")
            m = _metricas(conn, ts, te)
            turnos.append({"nombre": nombre, "horario": horario, **m})

        alarmas = conn.execute(text("""
            SELECT TOP 5 e.codigo_error,
                   ISNULL(a.componente,'SISTEMA') AS comp,
                   ISNULL(a.mensaje,'Error '+CAST(e.codigo_error AS VARCHAR)) AS msg,
                   COUNT(*) AS n
            FROM luk4.estado e
            LEFT JOIN luk4.alarmas a ON a.codigo = e.codigo_error
            WHERE e.timestamp >= :ts AND e.timestamp < :te AND e.codigo_error > 0
            GROUP BY e.codigo_error, a.componente, a.mensaje
            ORDER BY n DESC
        """), {"ts": jornada_ini, "te": jornada_fin}).fetchall()

    D, R, C, O = global_m["D"], global_m["R"], global_m["C"], global_m["OEE"]
    pz_b, pz_m, pz_insp = global_m["pz_b"], global_m["pz_m"], global_m["pz_insp"]
    pz_teo  = global_m["pz_teo"]
    h_avail = global_m["h_avail"]
    h_stop  = round(24 - h_avail, 1)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        # ── Cabecera ──────────────────────────────────────────────────
        _add_logo(ax, xc=0.115, yc=0.958)
        _txt(ax, 0.27, 0.971, "LUK4 — Informe OEE",
             fontsize=16, fontweight="bold", color=BRAND)
        _txt(ax, 0.27, 0.948, f"Jornada  {fecha.strftime('%d/%m/%Y')}  ·  06:00 → 06:00  ·  3 turnos",
             fontsize=9, color=GRAY)
        _txt(ax, 0.94, 0.971, fecha.strftime("%d / %m / %Y"),
             ha="right", fontsize=12, fontweight="bold", color=BRAND, fontfamily="monospace")
        _txt(ax, 0.94, 0.948, "Línea general",
             ha="right", fontsize=8, color=LGRAY)

        _sep(ax, 0.934)

        # ── KPIs ──────────────────────────────────────────────────────
        for i, (val, lbl) in enumerate([
            (f"{D}%", "DISPONIBILIDAD"),
            (f"{R}%", "RENDIMIENTO"),
            (f"{C}%", "CALIDAD"),
            (f"{O}%", "OEE"),
        ]):
            _kpi_box(ax, 0.06 + i * 0.225, 0.820, 0.195, 0.096,
                     val, lbl, _color(float(val[:-1])))

        _sep(ax, 0.812)

        # ── Gráfico disponibilidad por turno ───────────────────────────
        _txt(ax, 0.06, 0.800, "DISPONIBILIDAD POR TURNO",
             fontsize=7.5, fontweight="bold", color=LGRAY)
        # leyenda inline (sin legend de matplotlib que solapa las barras)
        _txt(ax, 0.63, 0.800, "■", fontsize=10, color=C_PROD)
        _txt(ax, 0.650, 0.800, "Produciendo", fontsize=7, color=GRAY)
        _txt(ax, 0.775, 0.800, "■", fontsize=10, color=C_STOP)
        _txt(ax, 0.795, 0.800, "Parada", fontsize=7, color=GRAY)
        _draw_turno_chart(fig, turnos, y_bottom=0.686, height=0.104)
        _sep(ax, 0.662)

        # ── Tabla por turno ────────────────────────────────────────────
        _txt(ax, 0.06, 0.649, "DESGLOSE POR TURNO",
             fontsize=7.5, fontweight="bold", color=LGRAY)

        COL = [
            (0.06,  "left",  "TURNO"),
            (0.185, "left",  "HORARIO"),
            (0.325, "right", "DISP."),
            (0.405, "right", "REND."),
            (0.482, "right", "CAL."),
            (0.558, "right", "OEE"),
            (0.655, "right", "BUENAS"),
            (0.742, "right", "MALAS"),
            (0.832, "right", "INSP."),
            (0.940, "right", "MARCHA"),
        ]
        ch_y = 0.633
        for cx, ca, cl in COL:
            _txt(ax, cx, ch_y, cl, ha=ca, fontsize=6.8, fontweight="bold", color=LGRAY)

        row_h = 0.038
        for i, t in enumerate(turnos):
            ry = 0.607 - i * row_h
            if i % 2 == 0:
                _row_bg(ax, ry)
            vals = [
                (0.06,  "left",  t["nombre"],                                    BRAND,                                   True),
                (0.185, "left",  t["horario"],                                   GRAY,                                    False),
                (0.325, "right", f"{t['D']}%",                                   _color(t["D"]),                          True),
                (0.405, "right", f"{t['R']}%",                                   _color(t["R"]),                          True),
                (0.482, "right", f"{t['C']}%",                                   _color(t["C"]),                          True),
                (0.558, "right", f"{t['OEE']}%",                                 _color(t["OEE"]),                        True),
                (0.655, "right", f"{t['pz_b']:,}",                               "#374151",                               True),
                (0.742, "right", f"{t['pz_m']:,}",                               "#dc2626" if t["pz_m"] else "#374151",   True),
                (0.832, "right", f"{t['pz_insp']:,}",                             "#374151",                               True),
                (0.940, "right", f"{t['h_avail']} h",                            GRAY,                                    True),
            ]
            for cx, ca, cv, cc, bold in vals:
                _txt(ax, cx, ry, cv, ha=ca,
                     fontsize=8.5 if bold else 8,
                     fontweight="bold" if bold else "normal",
                     color=cc if t["has_data"] else LGRAY,
                     fontfamily="monospace" if bold else "sans-serif")

        sep3_y = 0.607 - len(turnos) * row_h - 0.008
        _sep(ax, sep3_y)

        # ── Totales jornada ────────────────────────────────────────────
        tot_y = sep3_y - 0.024
        _txt(ax, 0.06, tot_y, "TOTALES JORNADA",
             fontsize=7.5, fontweight="bold", color=LGRAY)

        r1y = tot_y - 0.030
        r2y = r1y - 0.038
        r3y = r2y - 0.038

        _tot_row(ax, r1y, [
            ("Inspeccionadas",      f"{pz_insp:,}",                   BRAND),
            ("Buenas",             f"{pz_b:,}",                      "#16a34a"),
            ("Malas",              f"{pz_m:,}",                      "#dc2626" if pz_m else BRAND),
        ], row_bg=True)
        ritmo_real = round(pz_insp / h_avail) if h_avail > 0 else 0
        ritmo_teo  = round(3600 / CICLO_NOMINAL_S)
        _tot_row(ax, r2y, [
            ("Horas en marcha",    f"{h_avail} h",                   BRAND),
            ("Horas parada",       f"{h_stop} h",                    "#d97706" if h_stop > 2 else BRAND),
            ("Oport. perdida",     f"{max(pz_teo - pz_insp, 0):,} pz", "#d97706"),
        ])
        _tot_row(ax, r3y, [
            ("Piezas teóricas",    f"{pz_teo:,}",                    GRAY),
            ("Teórico",            f"{ritmo_teo} pz/h",              GRAY),
            ("Real",               f"{ritmo_real} pz/h",             _color(ritmo_real / ritmo_teo * 100)),
        ], row_bg=True)

        sep4_y = r3y - 0.030
        _sep(ax, sep4_y)

        # ── Alarmas ────────────────────────────────────────────────────
        al_y = sep4_y - 0.024
        _txt(ax, 0.06, al_y, "ALARMAS DEL PERIODO",
             fontsize=7.5, fontweight="bold", color=LGRAY)

        if alarmas:
            for j, r in enumerate(alarmas):
                ry = al_y - 0.028 - j * 0.035
                if j % 2 == 0:
                    _row_bg(ax, ry, "#fff7ed")
                comp = (r[1] or "")[:15]
                msg  = (r[2] or "")[:52]
                _txt(ax, 0.08, ry, f"{comp}  ·  {msg}", fontsize=8, color="#374151")
                _txt(ax, 0.93, ry, f"×{r[3]}", ha="right",
                     fontsize=8, fontweight="bold", color="#dc2626", fontfamily="monospace")
        else:
            _txt(ax, 0.5, al_y - 0.028, "Sin alarmas en el periodo",
                 ha="center", fontsize=8.5, color=LGRAY)

        # ── Footer ────────────────────────────────────────────────────
        _sep(ax, 0.046)
        _txt(ax, 0.06, 0.030, f"Generado el {datetime.now().strftime('%d/%m/%Y  %H:%M')}",
             fontsize=7.5, color=XGRAY)
        _txt(ax, 0.94, 0.030, "ECS Mobility  ·  LUK4",
             ha="right", fontsize=7.5, color=XGRAY)

        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return buf.getvalue()


# ── Informe de rango ──────────────────────────────────────────────────────────

def _informe_rango(fecha_ini: date, fecha_fin: date) -> bytes:
    from api.database import engine
    from sqlalchemy import text

    ts_ini = datetime(fecha_ini.year, fecha_ini.month, fecha_ini.day, 0, 0, 0)
    ts_fin = datetime(fecha_fin.year,  fecha_fin.month,  fecha_fin.day,  23, 59, 59)
    n_dias = (fecha_fin - fecha_ini).days + 1
    fecha_label = (fecha_ini.strftime("%d/%m/%Y") if fecha_ini == fecha_fin
                   else f"{fecha_ini.strftime('%d/%m/%Y')}  —  {fecha_fin.strftime('%d/%m/%Y')}")

    with engine.connect() as conn:
        m = _metricas(conn, ts_ini, ts_fin)
        alarmas = conn.execute(text("""
            SELECT TOP 6 e.codigo_error,
                   ISNULL(a.componente,'SISTEMA') AS comp,
                   ISNULL(a.mensaje,'Error '+CAST(e.codigo_error AS VARCHAR)) AS msg,
                   COUNT(*) AS n
            FROM luk4.estado e
            LEFT JOIN luk4.alarmas a ON a.codigo = e.codigo_error
            WHERE e.timestamp >= :ts AND e.timestamp <= :te AND e.codigo_error > 0
            GROUP BY e.codigo_error, a.componente, a.mensaje
            ORDER BY n DESC
        """), {"ts": ts_ini, "te": ts_fin}).fetchall()

    D, R, C, O = m["D"], m["R"], m["C"], m["OEE"]
    pz_b, pz_m, pz_insp = m["pz_b"], m["pz_m"], m["pz_insp"]
    pz_teo  = m["pz_teo"]
    h_avail = m["h_avail"]
    h_stop  = round(n_dias * 24 - h_avail, 1)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        _add_logo(ax, xc=0.115, yc=0.958)
        _txt(ax, 0.27, 0.971, "LUK4 — Informe OEE",
             fontsize=16, fontweight="bold", color=BRAND)
        _txt(ax, 0.27, 0.948, f"Rango  {fecha_label}  ·  {n_dias} día{'s' if n_dias > 1 else ''}",
             fontsize=9, color=GRAY)
        _txt(ax, 0.94, 0.971, fecha_label,
             ha="right", fontsize=11, fontweight="bold", color=BRAND, fontfamily="monospace")
        _txt(ax, 0.94, 0.948, "Línea general",
             ha="right", fontsize=8, color=LGRAY)

        _sep(ax, 0.934)

        for i, (val, lbl) in enumerate([
            (f"{D}%", "DISPONIBILIDAD"),
            (f"{R}%", "RENDIMIENTO"),
            (f"{C}%", "CALIDAD"),
            (f"{O}%", "OEE"),
        ]):
            _kpi_box(ax, 0.06 + i * 0.225, 0.820, 0.195, 0.096,
                     val, lbl, _color(float(val[:-1])))

        _sep(ax, 0.812)
        _txt(ax, 0.06, 0.799, "PRODUCCIÓN",
             fontsize=7.5, fontweight="bold", color=LGRAY)

        # Tabla de producción (2 columnas fijas: label izq, valor der)
        rows_prod = [
            ("Inspeccionadas",      f"{pz_insp:,}",                           BRAND),
            ("Buenas",              f"{pz_b:,}",                              "#16a34a"),
            ("Malas",               f"{pz_m:,}",                              "#dc2626" if pz_m else BRAND),
            ("Calidad",             f"{C} %",                                  _color(C)),
            ("Horas en marcha",     f"{h_avail} h",                           BRAND),
            ("Horas parada",        f"{h_stop} h",                            "#d97706" if h_stop > 2 else BRAND),
            ("Piezas teóricas",     f"{pz_teo:,}",                            GRAY),
            ("Oportunidad perdida", f"{max(pz_teo - pz_insp, 0):,}",         "#d97706"),
            ("Ciclo nominal",       f"{CICLO_NOMINAL_S} s / {round(3600/CICLO_NOMINAL_S)} pz·h⁻¹", GRAY),
        ]
        rh = 0.042
        for j, (k, v, vc) in enumerate(rows_prod):
            ry = 0.775 - j * rh
            if j % 2 == 0:
                _row_bg(ax, ry)
            _txt(ax, 0.08, ry, k, fontsize=9.5, color="#374151")
            _txt(ax, 0.93, ry, v, ha="right", fontsize=9.5,
                 fontweight="bold", color=vc, fontfamily="monospace")

        sep2 = 0.775 - len(rows_prod) * rh - 0.010
        _sep(ax, sep2)

        al_y = sep2 - 0.024
        _txt(ax, 0.06, al_y, "ALARMAS DEL PERIODO",
             fontsize=7.5, fontweight="bold", color=LGRAY)
        if alarmas:
            for j, r in enumerate(alarmas):
                ry = al_y - 0.028 - j * 0.035
                if j % 2 == 0:
                    _row_bg(ax, ry, "#fff7ed")
                _txt(ax, 0.08, ry, f"{(r[1] or '')[:15]}  ·  {(r[2] or '')[:52]}",
                     fontsize=8, color="#374151")
                _txt(ax, 0.93, ry, f"×{r[3]}", ha="right",
                     fontsize=8, fontweight="bold", color="#dc2626", fontfamily="monospace")
        else:
            _txt(ax, 0.5, al_y - 0.028, "Sin alarmas en el periodo",
                 ha="center", fontsize=8.5, color=LGRAY)

        _sep(ax, 0.046)
        _txt(ax, 0.06, 0.030, f"Generado el {datetime.now().strftime('%d/%m/%Y  %H:%M')}",
             fontsize=7.5, color=XGRAY)
        _txt(ax, 0.94, 0.030, "ECS Mobility  ·  LUK4",
             ha="right", fontsize=7.5, color=XGRAY)

        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return buf.getvalue()


# ── Punto de entrada público ──────────────────────────────────────────────────

def generar_pdf_luk4_oee(fecha_ini: date, fecha_fin: date, modo: str = "rango") -> bytes:
    if modo == "dia":
        return _informe_dia(fecha_ini)
    return _informe_rango(fecha_ini, fecha_fin)
