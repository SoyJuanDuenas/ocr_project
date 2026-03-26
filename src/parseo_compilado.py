"""
parseo_compilado.py
Parsea el texto OCR estructurado de compilado_tomo_I_XII.xlsx
y extrae campos individuales (año, oficio, escribanía, folio, fecha,
signatura, asunto, observaciones) en columnas separadas.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Regex y constantes
# ---------------------------------------------------------------------------

# Detecta "Libro del año:" con variantes OCR (Iibro, Jibro, Libri, etc.)
# Variantes vistas: "Libro del año:", "Libro del año;", "Libro del año.",
#   "Libro del año 1527" (sin puntuación), "Libro 1el año:", "Libro dle año:",
#   "Libro de año:", "Libro el año:", ": Libro .del . año:"
RE_INICIO = re.compile(
    r'[\w.]{0,5}br[oió]\s+[d1.]*[elé!.]+\s+\.?\s*a[ñn—]+[oó]\s*[:.;]?\s*', re.I
)
RE_INICIO_FALLBACK = re.compile(
    r'\b[d1.][elé!.]+\s+\.?\s*a[ñn—]+[oó]\s*[:.;]?\s*', re.I
)

# Separador entre campos: .— – - , ; y espacios
_SEP = r'[\s.—–\-,;]'
# Puntuación post-label: ":" o ";" (variantes OCR)
_PUNCT = r'[;:]'

# Divide header de body
RE_ASUNTO = re.compile(rf'{_SEP}+A[su][uú]n[t\']o\s*{_PUNCT}\s*', re.I)
RE_ASUNTO_FALLBACK = re.compile(rf'{_SEP}+Asunto\s+', re.I)
RE_OBSERVACIONES = re.compile(rf'{_SEP}+Observaciones\s*{_PUNCT}\s*', re.I)

# Labels del header (entre "Libro del año:" y "Asunto:")
# Acepta tanto ":" como ";" como puntuación post-label
HEADER_LABELS: List[Tuple[str, re.Pattern]] = [
    ('oficio',     re.compile(rf'{_SEP}+O[fl]icio\s*{_PUNCT}?\s*', re.I)),
    ('escribania', re.compile(rf'{_SEP}+Esc[ri]i?[rb]an[íifa]a\s*{_PUNCT}?\s*', re.I)),
    ('folio',      re.compile(rf'{_SEP}+Fol[ií—j]o\s*{_PUNCT}?\s*', re.I)),
    ('fecha',      re.compile(rf'{_SEP}+F[eé]cha\s*{_PUNCT}?\s*', re.I)),
    ('signatura',  re.compile(rf'{_SEP}+Signatura\s*{_PUNCT}?\s*', re.I)),
]

# Sub-libro dentro del valor de oficio (acepta "Libro I", "Libro: I", "Libro: único")
RE_SUB_LIBRO = re.compile(r'\bLibro\s*:?\s*([IVX]+|[Úú]nico|único)\b', re.I)

# Año numérico (4 dígitos)
RE_YEAR = re.compile(r'\b(\d{4})\b')

# Campos vacíos por defecto
_EMPTY: Dict[str, Any] = {
    'macrodatos': '',
    'año': '', 'año_num': pd.NA, 'oficio': '', 'libro': '',
    'escribania': '', 'folio': '', 'fecha': '', 'signatura': '',
    'asunto': '', 'observaciones': '', '_parse_ok': False,
}

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def limpiar_valor(s: str) -> str:
    """Strip separadores de borde y colapsar espacios internos."""
    s = s.strip()
    s = re.sub(r'^[\s.—–\-,;]+', '', s)
    s = re.sub(r'[\s.—–\-,;]+$', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def contar_subregistros(texto: str) -> int:
    """Cuenta ocurrencias de 'Libro del año:' (variantes OCR)."""
    n = len(RE_INICIO.findall(texto))
    if n == 0:
        n = len(RE_INICIO_FALLBACK.findall(texto))
    return max(n, 1)


def parsear_texto(texto: str) -> Dict[str, Any]:
    """
    Parsea el texto OCR de un registro notarial.
    Extrae: año, año_num, oficio, libro, escribania, folio, fecha,
            signatura, asunto, observaciones, _parse_ok.
    """
    if not isinstance(texto, str) or not texto.strip():
        return dict(_EMPTY)

    result = dict(_EMPTY)

    # 1. Buscar inicio: "Libro del año:"
    m_inicio = RE_INICIO.search(texto)
    if not m_inicio:
        m_inicio = RE_INICIO_FALLBACK.search(texto)
    if not m_inicio:
        # No se encontró patrón de inicio → parse fallido
        result['asunto'] = limpiar_valor(texto)
        return result

    after_inicio = texto[m_inicio.end():]

    # 2. Buscar "Asunto:" → divide header y body
    m_asunto = RE_ASUNTO.search(after_inicio)
    if not m_asunto:
        m_asunto = RE_ASUNTO_FALLBACK.search(after_inicio)
    if m_asunto:
        header = after_inicio[:m_asunto.start()]
        body = after_inicio[m_asunto.end():]
    else:
        # Sin "Asunto:" explícito: todo es header, body vacío
        header = after_inicio
        body = ''

    # Macrodatos: encabezado crudo desde "Libro del año:" hasta antes de "Asunto:"
    macro_end = m_asunto.start() if m_asunto else len(after_inicio)
    result['macrodatos'] = limpiar_valor(texto[m_inicio.start():m_inicio.end() + macro_end])

    # 3. En body: buscar "Observaciones:"
    if body:
        m_obs = RE_OBSERVACIONES.search(body)
        if m_obs:
            result['asunto'] = limpiar_valor(body[:m_obs.start()])
            result['observaciones'] = limpiar_valor(body[m_obs.end():])
        else:
            result['asunto'] = limpiar_valor(body)

    # 4. En header: buscar labels, ordenar por posición, extraer valores
    encontrados: List[Tuple[int, int, str]] = []  # (start, end, nombre)
    for nombre, patron in HEADER_LABELS:
        m = patron.search(header)
        if m:
            encontrados.append((m.start(), m.end(), nombre))

    # Ordenar por posición de aparición
    encontrados.sort(key=lambda x: x[0])

    # Año: texto antes del primer label (o todo el header si no hay labels)
    if encontrados:
        año_texto = header[:encontrados[0][0]]
    else:
        año_texto = header

    result['año'] = limpiar_valor(año_texto)

    # Extraer año numérico
    m_year = RE_YEAR.search(result['año'])
    if m_year:
        result['año_num'] = int(m_year.group(1))

    # Extraer valores entre labels consecutivos
    for i, (start, end, nombre) in enumerate(encontrados):
        if i + 1 < len(encontrados):
            valor = header[end:encontrados[i + 1][0]]
        else:
            valor = header[end:]
        result[nombre] = limpiar_valor(valor)

    # 5. Extraer sub-libro del valor de oficio
    if result['oficio']:
        m_libro = RE_SUB_LIBRO.search(result['oficio'])
        if m_libro:
            result['libro'] = 'Libro ' + m_libro.group(1)
            # Limpiar el "Libro X" del valor de oficio
            result['oficio'] = limpiar_valor(
                result['oficio'][:m_libro.start()] +
                result['oficio'][m_libro.end():]
            )

    # 6. Marcar como parseado correctamente si al menos tenemos año
    result['_parse_ok'] = bool(result['año'])

    return result


def parsear_lista(valor: Any) -> str:
    """
    Convierte un string representando una lista Python a string
    separado por '; ' para Excel.
    Ejemplo: "['Juan', 'Pedro']" → "Juan; Pedro"
    """
    if not isinstance(valor, str) or not valor.strip():
        return ''
    try:
        lst = ast.literal_eval(valor)
        if isinstance(lst, list):
            items = [str(x).strip() for x in lst if str(x).strip() and str(x).strip() != '...']
            return '; '.join(items)
    except (ValueError, SyntaxError):
        pass
    return ''


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def procesar_compilado(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Lee el Excel compilado, parsea cada fila, exporta resultado."""
    print(f"Leyendo {input_path} ...")
    df = pd.read_excel(input_path)
    print(f"  {len(df):,} filas cargadas")
    print(f"  Columnas: {list(df.columns)}")

    # Columna de texto a parsear
    col_texto = 'texto_ocr_fix'
    if col_texto not in df.columns:
        # Fallback si no existe
        for candidato in ['texto_original', 'texto_completo']:
            if candidato in df.columns:
                col_texto = candidato
                break
        else:
            raise ValueError(
                f"No se encontró columna de texto. "
                f"Columnas disponibles: {list(df.columns)}"
            )
    print(f"  Usando columna: '{col_texto}'")

    # Parsear cada fila
    registros: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        texto = row.get(col_texto, '')
        parsed = parsear_texto(texto)

        registro = {
            'tomo_id': row.get('tomo_id', ''),
            'id_num': row.get('id_num', ''),
            **parsed,
            'personas': parsear_lista(row.get('personas_lista', '')),
            'naos': parsear_lista(row.get('naos_lista', '')),
            'lugares': parsear_lista(row.get('lugares_lista', '')),
            'n_subregistros': contar_subregistros(str(texto) if isinstance(texto, str) else ''),
        }
        registros.append(registro)

    # Armar DataFrame de salida
    columnas = [
        'macrodatos', 'asunto',
        'tomo_id', 'id_num', 'año', 'año_num', 'oficio', 'libro',
        'escribania', 'folio', 'fecha', 'signatura',
        'observaciones',
        'personas', 'naos', 'lugares',
        'n_subregistros', '_parse_ok',
    ]
    df_out = pd.DataFrame(registros, columns=columnas)
    df_out['año_num'] = df_out['año_num'].astype('Int64')

    # Exportar
    df_out.to_excel(output_path, index=False)
    print(f"\nExportado: {output_path}")

    # Reporte de calidad
    _reporte_calidad(df_out)

    return df_out


def _reporte_calidad(df: pd.DataFrame) -> None:
    """Imprime reporte de calidad del parseo."""
    total = len(df)
    print(f"\n{'='*60}")
    print(f"REPORTE DE CALIDAD  ({total:,} registros)")
    print(f"{'='*60}")

    # Parse OK
    n_ok = df['_parse_ok'].sum()
    print(f"\n  _parse_ok:       {n_ok:>6,} / {total:,}  ({100*n_ok/total:.1f}%)")

    # Campos poblados
    campos = ['macrodatos', 'año', 'año_num', 'oficio', 'libro', 'escribania',
              'folio', 'fecha', 'signatura', 'asunto', 'observaciones',
              'personas', 'naos', 'lugares']
    print(f"\n  {'Campo':<16} {'Poblado':>8} {'%':>8}")
    print(f"  {'-'*16} {'-'*8} {'-'*8}")
    for campo in campos:
        if campo == 'año_num':
            n = df[campo].notna().sum()
        else:
            n = (df[campo].astype(str).str.strip() != '').sum()
        print(f"  {campo:<16} {n:>8,} {100*n/total:>7.1f}%")

    # Registros fusionados
    fusionados = (df['n_subregistros'] > 1).sum()
    print(f"\n  Fusionados (n_subregistros > 1): {fusionados:,} ({100*fusionados/total:.1f}%)")

    # Parse failures
    fallidos = df[~df['_parse_ok']]
    print(f"  Parse fallidos:                  {len(fallidos):,} ({100*len(fallidos)/total:.1f}%)")

    if len(fallidos) > 0:
        print(f"\n  Muestra de registros fallidos (primeros 100 chars):")
        for _, row in fallidos.head(10).iterrows():
            tomo = row.get('tomo_id', '?')
            id_n = row.get('id_num', '?')
            asunto = str(row.get('asunto', ''))[:100]
            print(f"    [{tomo} #{id_n}] {asunto}")

    print(f"\n{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Parsea texto OCR estructurado del compilado de tomos.'
    )
    parser.add_argument(
        '--input', '-i',
        default='compilado_tomo_I_XII.xlsx',
        help='Ruta al Excel de entrada (default: compilado_tomo_I_XII.xlsx)',
    )
    parser.add_argument(
        '--output', '-o',
        default='outputs/compilado_parseado.xlsx',
        help='Ruta al Excel de salida (default: outputs/compilado_parseado.xlsx)',
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: No se encontró el archivo de entrada: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    procesar_compilado(input_path, output_path)


if __name__ == '__main__':
    main()
