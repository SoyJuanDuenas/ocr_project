# -*- coding: utf-8 -*-
"""
Transforma compilado.xlsx en formato panel largo.

Una fila por mencion de entidad (persona/nao/lugar) por contrato.

Uso:
  python src/panelizar.py --compilado outputs/run_XXXXX/compilado.xlsx
  python src/panelizar.py  # usa el run mas reciente
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# Columnas del contrato que se conservan en el panel
COLS_CONTRATO = [
    "tomo_id", "id_num", "id_num_original", "id_status",
    "macrodatos", "año", "año_num", "oficio", "libro",
    "escribania", "folio", "fecha", "signatura",
    "asunto", "observaciones", "n_subregistros",
    "ocr_flag", "ocr_flagged",
]

MAPA_TIPOS = {"personas": "persona", "naos": "nao", "lugares": "lugar"}


def _split_entidades(valor: str) -> list[str]:
    """Separa string '; '-delimited en lista limpia."""
    if not isinstance(valor, str) or not valor.strip():
        return []
    return [e.strip() for e in valor.split(";") if e.strip()]


def _parse_atributos(valor: str) -> dict[str, str]:
    """Parsea string 'nombre::desc || nombre::desc' a dict."""
    if not isinstance(valor, str) or not valor.strip():
        return {}
    result = {}
    for par in valor.split("||"):
        par = par.strip()
        if "::" in par:
            nombre, desc = par.split("::", 1)
            result[nombre.strip()] = desc.strip()
    return result


def _format_id_num(valor: object) -> str:
    """Formatea id_num para IDs compuestos sin sufijos .0."""
    if pd.isna(valor):
        return ""
    try:
        return str(int(valor))
    except (TypeError, ValueError):
        return str(valor).strip()


def _roman_to_int(valor: str) -> int | None:
    """Convierte un romano simple a entero."""
    if not valor or not re.fullmatch(r"[IVXLCDM]+", valor.upper()):
        return None
    mapa = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(valor.upper()):
        num = mapa[ch]
        if num < prev:
            total -= num
        else:
            total += num
            prev = num
    return total


def _tomo_code(tomo_id: object) -> str:
    """Convierte Tomo_I / Tomo_XVI / Tomo_01 a codigo numerico de 2 digitos."""
    texto = str(tomo_id).strip()
    if not texto:
        return "00"

    m = re.search(r"(\d+)$", texto)
    if m:
        return f"{int(m.group(1)):02d}"

    ultimo = texto.split("_")[-1].strip()
    tomo_num = _roman_to_int(ultimo)
    if tomo_num is None:
        return "00"
    return f"{tomo_num:02d}"


def _contrato_code(row: pd.Series, orden_tomo: int) -> str:
    """Devuelve contrato en 4 digitos; si falta, usa orden tecnico dentro del tomo."""
    id_num = _format_id_num(row.get("id_num"))
    if id_num.isdigit():
        return f"{int(id_num):04d}"
    return f"{orden_tomo:04d}"


def _build_obs_base(row: pd.Series, orden_tomo: int) -> str:
    """Construye ID de observacion numerico: tomo(2) + contrato(4)."""
    return f"{_tomo_code(row.get('tomo_id'))}{_contrato_code(row, orden_tomo)}"


def _empty_ids() -> dict[str, str]:
    return {
        "IDTOMO": "",
        "IDCONT": "",
        "IDTOMOCONT": "",
        "IDPER": "",
        "IDNAO": "",
        "IDLUG": "",
        "IDTOMOCONTPER": "",
        "IDTOMOCONTNAO": "",
        "IDTOMOCONTLUG": "",
    }


def panelizar(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte compilado a panel largo: una fila por entidad por contrato."""

    cols_presentes = [c for c in COLS_CONTRATO if c in df.columns]
    filas: list[dict] = []
    orden_por_tomo: dict[str, int] = {}

    for idx, row in df.iterrows():
        base = {c: row[c] for c in cols_presentes}
        base["contrato_idx"] = idx
        tomo_key = str(row.get("tomo_id", ""))
        orden_por_tomo[tomo_key] = orden_por_tomo.get(tomo_key, 0) + 1
        id_tomo = _tomo_code(row.get("tomo_id"))
        id_cont = _contrato_code(row, orden_por_tomo[tomo_key])
        id_tomocont = f"{id_tomo}{id_cont}"

        # Parsear atributos por persona para este contrato
        atributos_dict = _parse_atributos(row.get("atributos", ""))

        tiene_entidad = False
        for tipo_fuente, tipo_panel in MAPA_TIPOS.items():
            entidades = _split_entidades(row.get(tipo_fuente, ""))
            for orden_tipo, entidad in enumerate(entidades, start=1):
                id_ent = f"{orden_tipo:02d}"
                fila = {
                    **base,
                    **_empty_ids(),
                    "IDTOMO": id_tomo,
                    "IDCONT": id_cont,
                    "IDTOMOCONT": id_tomocont,
                    "tipo_entidad": tipo_panel,
                    "entidad": entidad,
                    "atributo": "",
                }
                if tipo_panel == "persona":
                    fila["IDPER"] = id_ent
                    fila["IDTOMOCONTPER"] = f"{id_tomocont}{id_ent}"
                    fila["atributo"] = atributos_dict.get(entidad, "")
                elif tipo_panel == "nao":
                    fila["IDNAO"] = id_ent
                    fila["IDTOMOCONTNAO"] = f"{id_tomocont}{id_ent}"
                elif tipo_panel == "lugar":
                    fila["IDLUG"] = id_ent
                    fila["IDTOMOCONTLUG"] = f"{id_tomocont}{id_ent}"
                filas.append(fila)
                tiene_entidad = True

        if not tiene_entidad:
            filas.append({
                **base,
                **_empty_ids(),
                "IDTOMO": id_tomo,
                "IDCONT": id_cont,
                "IDTOMOCONT": id_tomocont,
                "tipo_entidad": "",
                "entidad": "",
                "atributo": "",
            })

    df_panel = pd.DataFrame(filas)

    cols_ids = [
        "IDTOMO", "IDCONT", "IDTOMOCONT",
        "IDPER", "IDNAO", "IDLUG",
        "IDTOMOCONTPER", "IDTOMOCONTNAO", "IDTOMOCONTLUG",
    ]
    cols_orden = ["contrato_idx", "tomo_id", "id_num", "tipo_entidad", "entidad", "atributo"] + cols_ids + [
        c for c in cols_presentes if c not in ("tomo_id", "id_num")
    ]
    cols_orden = [c for c in cols_orden if c in df_panel.columns]
    df_panel = df_panel[cols_orden]

    return df_panel


def exportar_panel_multihoja(df_panel: pd.DataFrame, out_path: Path) -> dict[str, int]:
    """Exporta panel a un xlsx con una hoja por tipo de entidad.

    Hojas: personas (con atributo), naos, lugares.
    Cada hoja solo incluye las columnas relevantes a su tipo de entidad.
    Retorna dict con cantidad de filas por hoja.
    """
    # Columnas base compartidas por todas las hojas
    cols_base = [
        "contrato_idx", "tomo_id", "id_num",
        "IDTOMO", "IDCONT", "IDTOMOCONT",
    ]
    # Columnas de contrato (las que existan)
    cols_contrato = [
        "id_num_original", "id_status", "macrodatos",
        "año", "año_num", "oficio", "libro", "escribania",
        "folio", "fecha", "signatura", "asunto", "observaciones",
        "n_subregistros", "ocr_flag", "ocr_flagged",
    ]

    def _cols_disponibles(cols: list[str], df: pd.DataFrame) -> list[str]:
        return [c for c in cols if c in df.columns]

    hojas = {
        "personas": {
            "filtro": df_panel["tipo_entidad"] == "persona",
            "cols_propias": ["entidad", "atributo", "IDPER", "IDTOMOCONTPER"],
        },
        "naos": {
            "filtro": df_panel["tipo_entidad"] == "nao",
            "cols_propias": ["entidad", "IDNAO", "IDTOMOCONTNAO"],
        },
        "lugares": {
            "filtro": df_panel["tipo_entidad"] == "lugar",
            "cols_propias": ["entidad", "IDLUG", "IDTOMOCONTLUG"],
        },
    }

    counts = {}
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for nombre, cfg in hojas.items():
            df_hoja = df_panel.loc[cfg["filtro"]].copy()
            cols = (
                _cols_disponibles(cols_base, df_hoja)
                + _cols_disponibles(cfg["cols_propias"], df_hoja)
                + _cols_disponibles(cols_contrato, df_hoja)
            )
            df_hoja = df_hoja[cols]
            # Renombrar entidad al tipo especifico
            if nombre == "personas":
                df_hoja = df_hoja.rename(columns={"entidad": "persona"})
            elif nombre == "naos":
                df_hoja = df_hoja.rename(columns={"entidad": "nao"})
            elif nombre == "lugares":
                df_hoja = df_hoja.rename(columns={"entidad": "lugar"})
            df_hoja.to_excel(writer, sheet_name=nombre, index=False)
            counts[nombre] = len(df_hoja)

    return counts


def separar_headers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa filas con id_num ausente para excluirlas del panel principal."""
    mask_headers = df["id_num"].isna() if "id_num" in df.columns else pd.Series(False, index=df.index)
    return df.loc[~mask_headers].copy(), df.loc[mask_headers].copy()


def main():
    parser = argparse.ArgumentParser(description="Panelizar compilado a formato largo.")
    parser.add_argument(
        "--compilado",
        default=None,
        help="Path al compilado.xlsx. Si no se indica, usa el run mas reciente.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path de salida. Default: mismo directorio que compilado, panel.xlsx",
    )
    parser.add_argument(
        "--output-headers",
        default=None,
        help="Path de salida para filas sin id_num. Default: mismo directorio, panel_headers.xlsx",
    )
    args = parser.parse_args()

    if args.compilado:
        compilado_path = Path(args.compilado)
    else:
        outputs = Path("outputs")
        runs = sorted(outputs.glob("run_*"), reverse=True)
        if not runs:
            print("ERROR: No se encontro ningun run en outputs/")
            return
        compilado_path = runs[0] / "compilado.xlsx"
        print(f"  Usando ultimo run: {runs[0].name}")

    if not compilado_path.exists():
        print(f"ERROR: No existe {compilado_path}")
        return

    print(f"  Cargando: {compilado_path}")
    df = pd.read_excel(compilado_path)
    print(f"  {len(df):,} contratos")

    df_panel_src, df_headers = separar_headers(df)
    print(f"  Headers sin id_num excluidos del panel: {len(df_headers):,}")

    print("  Panelizando...")
    df_panel = panelizar(df_panel_src)

    con_entidad = df_panel["entidad"].fillna("").str.len().gt(0)
    n_personas = (df_panel["tipo_entidad"] == "persona").sum()
    n_naos = (df_panel["tipo_entidad"] == "nao").sum()
    n_lugares = (df_panel["tipo_entidad"] == "lugar").sum()
    n_vacios = (~con_entidad).sum()

    print(f"  {len(df_panel):,} filas en panel:")
    print(f"    personas: {n_personas:,}")
    print(f"    naos:     {n_naos:,}")
    print(f"    lugares:  {n_lugares:,}")
    print(f"    sin entidad: {n_vacios:,}")

    ent_unicas = df_panel[con_entidad].groupby("tipo_entidad")["entidad"].nunique()
    print("  Entidades unicas:")
    for tipo, n in ent_unicas.items():
        print(f"    {tipo}: {n:,}")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = compilado_path.parent / "panel.xlsx"

    if args.output_headers:
        headers_path = Path(args.output_headers)
    else:
        headers_path = compilado_path.parent / "panel_headers.xlsx"

    df_panel.to_excel(out_path, index=False)
    df_headers.to_excel(headers_path, index=False)
    print(f"\n  Guardado panel:   {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Guardado headers: {headers_path} ({headers_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Exportar multi-hoja
    multihoja_path = out_path.parent / "panel_por_tipo.xlsx"
    counts = exportar_panel_multihoja(df_panel, multihoja_path)
    print(f"  Guardado multi-hoja: {multihoja_path} ({multihoja_path.stat().st_size / 1024 / 1024:.1f} MB)")
    for nombre, n in counts.items():
        print(f"    {nombre}: {n:,} filas")


if __name__ == "__main__":
    main()
