"""Extrae datos de IZARO para luk3, luk6, coroa y t48 desde 2025."""
import csv
import sys
from collections import Counter
from datetime import date

sys.path.insert(0, "/home/eeguskiza/analisis_datos")
from OEE.db.connector import extraer_datos, load_config

cfg = load_config()
cfg["recursos"] = [
    {"centro_trabajo": 1003, "nombre": "luk3", "activo": True},
    {"centro_trabajo": 4001, "nombre": "luk6", "activo": True},
    {"centro_trabajo": 305,  "nombre": "coroa", "activo": True},
    {"centro_trabajo": 48,   "nombre": "t48", "activo": True},
]

print(f"Extrayendo 2025-01-01 → {date.today()} para luk3, luk6, coroa, t48...")
rows = extraer_datos(cfg, date(2025, 1, 1), date.today())
print(f"{len(rows)} registros")

if rows:
    out = "/home/eeguskiza/analisis_datos/data/export_2025_completo.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {out}")

    for rec, n in sorted(Counter(r["recurso"] for r in rows).items()):
        fechas = [r["fecha"] for r in rows if r["recurso"] == rec]
        print(f"  {rec}: {n} reg ({min(fechas)} → {max(fechas)})")
else:
    print("Sin datos")
