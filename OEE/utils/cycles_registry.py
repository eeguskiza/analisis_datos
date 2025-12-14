from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Tuple


def _normalize_machine(value: str) -> str:
    return (value or "").strip().lower()


def _normalize_reference(value: str) -> str:
    text = (value or "").strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def registrar_ciclos_faltantes(
    faltantes: Iterable[Tuple[str, str, str]], cycles_file: Path
) -> List[Tuple[str, str]]:
    """
    Garantiza que cada (máquina, referencia) tenga una fila en ciclos.csv.

    Retorna la lista de filas añadidas (máquina, referencia) para informar al usuario.
    """
    faltantes = list(faltantes)
    if not faltantes:
        return []

    cycles_file = Path(cycles_file)
    existing = set()
    if cycles_file.exists():
        with cycles_file.open(encoding="utf-8-sig", newline="") as handler:
            reader = csv.DictReader(handler)
            for row in reader:
                maquina = _normalize_machine(row.get("maquina", ""))
                referencia = _normalize_reference(row.get("referencia", ""))
                if maquina and referencia:
                    existing.add((maquina, referencia))
    else:
        cycles_file.parent.mkdir(parents=True, exist_ok=True)
        with cycles_file.open("w", encoding="utf-8", newline="") as handler:
            handler.write("maquina,referencia,tiempo_ciclo\n")

    nuevos: List[Tuple[str, str]] = []
    for maquina_raw, referencia_raw, referencia_norm in faltantes:
        maquina_norm = _normalize_machine(maquina_raw)
        ref_norm = _normalize_reference(referencia_norm or referencia_raw)
        if not maquina_norm or not ref_norm:
            continue
        key = (maquina_norm, ref_norm)
        if key in existing:
            continue
        existing.add(key)
        referencia_texto = referencia_raw or referencia_norm
        nuevos.append((maquina_norm, referencia_texto))

    if not nuevos:
        return []

    with cycles_file.open("a", encoding="utf-8", newline="") as handler:
        writer = csv.writer(handler)
        for maquina, referencia in nuevos:
            writer.writerow([maquina, referencia, "0"])

    return nuevos

