"""Aplica extracciones de entidades al JSON consolidado.

Recibe un dict {idx: {tipo, personas, naos, lugares, atributos}} y actualiza
los registros correspondientes en outputs/tomo_i_consolidado.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

JSON_PATH = Path("outputs/tomo_i_consolidado.json")


def aplicar(extracciones: dict[int, dict]) -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    for idx, ent in extracciones.items():
        if idx >= len(data):
            print(f"WARN idx {idx} fuera de rango")
            continue
        r = data[idx]
        r["tipo"] = ent.get("tipo", "")
        r["personas"] = "; ".join(ent.get("personas", []))
        r["naos"] = "; ".join(ent.get("naos", []))
        r["lugares"] = "; ".join(ent.get("lugares", []))
        atr = ent.get("atributos", {})
        r["atributos"] = json.dumps(atr, ensure_ascii=False) if atr else ""
    JSON_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aplicadas {len(extracciones)} extracciones")


def _procesado(r: dict) -> bool:
    """Un registro esta procesado si tiene tipo o personas."""
    return bool(
        (r.get("tipo") and str(r["tipo"]).strip())
        or (r.get("personas") and str(r.get("personas", "")).strip())
    )


def stats() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    total = len(data)
    con_asunto = sum(1 for r in data if r.get("asunto") and str(r["asunto"]).strip())
    proc = sum(1 for r in data if _procesado(r))
    con_ent = sum(1 for r in data if r.get("personas") and str(r.get("personas", "")).strip())
    pend = con_asunto - proc
    print(f"Total: {total} | Con asunto: {con_asunto} | Procesados: {proc} | Con entidades: {con_ent} | Pendientes: {pend}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats()
    else:
        chunk_path = Path("scripts/_chunk_extracciones.json")
        raw = json.loads(chunk_path.read_text(encoding="utf-8"))
        ext = {int(k): v for k, v in raw.items()}
        aplicar(ext)
        stats()
