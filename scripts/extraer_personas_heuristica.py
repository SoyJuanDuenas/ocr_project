"""Extrae personas usando patrones y heurísticas de texto."""
import json
import re
from pathlib import Path

JSON_PATH = Path("outputs/tomo_i_consolidado.json")

# Nombres comunes españoles medieval-colonial
NOMBRES_COMUNES = {
    "Juan", "Pedro", "Diego", "Francisco", "Andrés", "Miguel", "Antonio",
    "Cristóbal", "Bartolomé", "Rodrigo", "Fernando", "Carlos", "Luis",
    "Gonzalo", "García", "Martín", "Alonso", "Pablo", "Simón", "Tomás",
    "Jerónimo", "Gregorio", "Vicente", "Nicasio", "Bernardino", "Gaspar",
    "Melchor", "Baltasar", "Jorge", "Domingo", "Mateo", "Marcial",
    "María", "Juana", "Isabel", "Ana", "Catalina", "Magdalena", "Teresa",
    "Leonor", "Francisca", "Petronila", "Constanza", "Beatriz", "Violeta",
}

APELLIDOS_COMUNES = {
    "González", "García", "Rodríguez", "Martínez", "Fernández", "López",
    "Pérez", "Sánchez", "Gómez", "Jiménez", "Díaz", "Moreno", "Muñoz",
    "Romero", "Herrera", "Molina", "Castillo", "Vega", "Ríos", "Santos",
    "Flores", "Cruz", "Silva", "Medina", "Parra", "Ramírez", "Cortés",
    "Figueroa", "Vargas", "Fuentes", "Reyes", "Duarte", "Mejía", "Acosta",
    "Quintana", "Navarro", "Salazar", "Cervantes", "Montoya", "Aguayo",
    "Cerón", "Valdés", "Villanueva", "Campos", "Bravo", "Ochoa", "Ribera",
    "Ponce", "Dueñas", "Torres", "Camacho", "Tavora", "Segovia", "Arévalo",
}

def _extract_nombres_patrones(texto: str) -> list[str]:
    """Busca patrones comunes de extracción de nombres."""
    nombres = []
    t = texto or ""

    # Patrones: "otorga poder a [NOMBRE]", "se obliga a [NOMBRE]", etc.
    patrones = [
        r"(?:otorga|otorgan)\s+poder\s+a\s+([^,\.;]+)",
        r"(?:se\s+obliga|se\s+obligan)\s+([^,\.;]+?)(?:\s+a\s+|,|;|$)",
        r"(?:le\s+vende|compra|de\s+)?([A-Z][a-záéíóú]+(?:\s+[A-Z][a-záéíóú]+)?)\s+(?:vecino|maestre|escribano|notario)",
        r"(?:maestre|maestres)\s+(?:de\s+la\s+nao\s+)?([A-Z][a-záéíóú]+(?:\s+[A-Z][a-záéíóú]+)?)",
        r"(?:poder|procurador)\s+(?:de\s+)?([A-Z][a-záéíóú]+(?:\s+[A-Z][a-záéíóú]+)?)",
    ]

    for patron in patrones:
        matches = re.finditer(patron, t, re.IGNORECASE)
        for m in matches:
            nombre = m.group(1).strip()
            if nombre and len(nombre) > 2:
                nombres.append(nombre)

    return nombres

def _extract_nombres_dict(texto: str) -> list[str]:
    """Busca nombres completos (Nombre Apellido) en el texto."""
    nombres = []
    t = texto or ""

    # Buscar secuencias de dos palabras capitalizadas
    for nombre in NOMBRES_COMUNES:
        for apellido in APELLIDOS_COMUNES:
            patron = rf"\b{re.escape(nombre)}\s+{re.escape(apellido)}\b"
            if re.search(patron, t, re.IGNORECASE):
                nombres.append(f"{nombre} {apellido}")

    return nombres

def extract_personas(texto: str, idx: int) -> list[str]:
    """Extrae personas del asunto usando patrones y diccionarios."""
    t = texto or ""

    # Combinar patrones + diccionario
    personas = []

    # Primero patrones
    personas.extend(_extract_nombres_patrones(t))

    # Luego diccionario
    personas.extend(_extract_nombres_dict(t))

    # Deduplicar y limpiar
    personas = list(set(p.strip() for p in personas if p.strip()))

    # Filtrar nombres muy cortos o obvios
    personas = [p for p in personas if len(p) > 3 and not p.startswith("?")]

    return personas

def main():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    # Encontrar pendientes
    pendientes = []
    for i, r in enumerate(data):
        asunto = r.get("asunto", "")
        personas = r.get("personas", "")
        if asunto and str(asunto).strip() and (not personas or not str(personas).strip()):
            pendientes.append((i, asunto))

    print(f"Extrayendo personas de {len(pendientes)} registros pendientes...")

    # Procesar en chunks de 150
    extracciones = {}
    for idx, asunto in pendientes:
        personas = extract_personas(asunto, idx)
        if personas:
            extracciones[idx] = {
                "tipo": data[idx].get("tipo", ""),
                "personas": personas,
                "naos": [],
                "lugares": [],
                "atributos": {}
            }

    # Guardar
    chunk_path = Path("scripts/_chunk_extracciones.json")
    chunk_path.write_text(json.dumps(extracciones, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Extraídas {len(extracciones)} registros con personas")
    print(f"Fichero: {chunk_path}")

if __name__ == "__main__":
    main()
