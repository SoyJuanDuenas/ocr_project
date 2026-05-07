"""Auto-extracción heurística para registros pendientes."""
import json
import re
from pathlib import Path

JSON_PATH = Path("outputs/tomo_i_consolidado.json")

# Patrones de tipo
TIPO_PATTERNS = {
    "poder": r"\b(otorga poder|otorgan poder)\b",
    "obligacion": r"\b(se obliga|se obligan)\b",
    "fletamento": r"\b(contrato de fletamento|fletamento entre)\b",
    "declaracion": r"\b(declara|declaración)\b",
    "compraventa": r"\b(vende|compra|compraventa)\b",
    "cobro": r"\b(carta de pago|finiquito)\b",
    "concierto": r"\b(se compromete|concierto)\b",
    "capitulacion": r"\b(capitulación)\b",
    "testamento": r"\b(testamento)\b",
    "confirmacion": r"\b(confirmación)\b",
}

# Naos conocidas (aproximadamente)
NAOS_KNOWN = {
    "Santa María", "San Francisco", "San Andrés", "Santiago", "Santa Catalina",
    "La Gracia de Dios", "Santa Clara", "San Antón", "San Pablo", "Santa Ana",
    "Trinidad", "Buen Jesús", "San Miguel", "San Cristóbal", "Magdalena",
    "Santa María de la Antigua", "Santa María de Gracia", "Santa María del Rosario",
    "San Telmo", "Espíritu Santo", "Santa María de los Remedios", "Cuerpo Santo",
    "Santa María de la Blanca", "Santa María de la Granada", "Santa María Magdalena",
    "Santa María de la Concepción", "Santa María de Guadalupe", "Sancti Espíritus",
    "Santa María del Carmen", "Sancti Spiritus", "San Juan", "Santa María de Consolación",
}

LUGARES_KNOWN = {
    "Santo Domingo", "Española", "San Juan", "Indias", "Sevilla", "Córdoba",
    "Puerto Rico", "Castilla del Oro", "Tierra Firme", "Cuba", "Concepción",
    "Puerto de Santo Domingo", "Puerto de la Concepción", "Darién", "Cartagena",
    "Palos", "Cádiz", "Triana", "Jerez", "Ciudad Rodrigo", "Burgos",
}

def auto_extract(texto: str, idx: int) -> dict:
    """Extrae tipo, personas (vacío), naos, lugares del asunto."""
    t = texto or ""

    # Tipo
    tipo = "otros"
    for tipo_name, pattern in TIPO_PATTERNS.items():
        if re.search(pattern, t, re.IGNORECASE):
            tipo = tipo_name
            break

    # Naos: buscar patrones como "nao X", "navío X", etc
    naos = []
    for nao in NAOS_KNOWN:
        if nao.lower() in t.lower():
            naos.append(nao)
    naos = list(set(naos))

    # Lugares
    lugares = []
    for lugar in LUGARES_KNOWN:
        if lugar.lower() in t.lower():
            lugares.append(lugar)
    lugares = list(set(lugares))

    return {
        "tipo": tipo,
        "personas": [],  # Vacío para auto, lo llenaremos manualmente después
        "naos": naos,
        "lugares": lugares,
        "atributos": {}
    }

def main():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    def procesado(r):
        return bool((r.get("tipo") and str(r["tipo"]).strip()) or
                   (r.get("personas") and str(r.get("personas", "")).strip()))

    # Encontrar pendientes
    pendientes = [(i, r) for i, r in enumerate(data)
                  if r.get("asunto") and str(r["asunto"]).strip() and not procesado(r)]

    # Auto-extraer
    extracciones = {}
    for idx, row in pendientes:
        ext = auto_extract(str(row.get("asunto", "")), idx)
        extracciones[idx] = ext

    # Guardar temporalmente
    chunk_path = Path("scripts/_chunk_extracciones.json")
    chunk_path.write_text(json.dumps(extracciones, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Auto-extraídos {len(extracciones)} registros pendientes")
    print(f"Total tipo='otros': {sum(1 for e in extracciones.values() if e['tipo'] == 'otros')}")

if __name__ == "__main__":
    main()
