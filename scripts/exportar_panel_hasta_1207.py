# -*- coding: utf-8 -*-
"""Exporta paneles (panel.xlsx, panel_headers.xlsx, panel_por_tipo.xlsx)
desde outputs/tomo_i_consolidado.json filtrando hasta el contrato 1207.

Reutiliza las funciones de src/panelizar.py para mantener consistencia
con el pipeline principal (parseo de atributos, IDs compuestos, multi-hoja).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from panelizar import (  # noqa: E402
    exportar_panel_multihoja,
    panelizar,
    separar_headers,
)

JSON_PATH = ROOT / "outputs" / "tomo_i_consolidado.json"
OUT_DIR = ROOT / "outputs" / "panel_hasta_1207"
ID_NUM_MAX = 1207.0


def _buscar_corte(data: list[dict], id_num_max: float) -> int:
    """Devuelve idx (inclusive) del ultimo registro con id_num == id_num_max.

    Confiamos en el orden secuencial del JSON: registros posteriores al corte
    con id_num <= id_num_max son anomalias de OCR y se descartan.
    """
    ultimo = -1
    for i, r in enumerate(data):
        idn = r.get("id_num")
        if idn is not None and float(idn) == id_num_max:
            ultimo = i
    if ultimo < 0:
        raise ValueError(f"No se encontro id_num == {id_num_max} en el JSON")
    return ultimo


def cargar_df_filtrado(json_path: Path = JSON_PATH, id_num_max: float = ID_NUM_MAX) -> pd.DataFrame:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    corte = _buscar_corte(data, id_num_max)
    recortado = data[: corte + 1]
    df = pd.DataFrame(recortado)
    # id_num como Int64 nullable (consistente con el resto del pipeline)
    if "id_num" in df.columns:
        df["id_num"] = pd.to_numeric(df["id_num"], errors="coerce").astype("Int64")
    if "año_num" in df.columns:
        df["año_num"] = pd.to_numeric(df["año_num"], errors="coerce").astype("Int64")
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = cargar_df_filtrado()
    print(f"  Registros cargados: {len(df):,}")
    print(f"  Con id_num: {df['id_num'].notna().sum():,}")
    print(f"  Con tipo: {(df.get('tipo', pd.Series(dtype=str)).fillna('').str.strip() != '').sum():,}")
    print(f"  Con atributos: {(df.get('atributos', pd.Series(dtype=str)).fillna('').str.strip() != '').sum():,}")

    df_panel_src, df_headers = separar_headers(df)
    print(f"  Headers excluidos del panel (id_num NaN): {len(df_headers):,}")

    df_panel = panelizar(df_panel_src)

    n_personas = (df_panel["tipo_entidad"] == "persona").sum()
    n_naos = (df_panel["tipo_entidad"] == "nao").sum()
    n_lugares = (df_panel["tipo_entidad"] == "lugar").sum()
    print(f"\n  Filas panel: {len(df_panel):,}")
    print(f"    personas: {n_personas:,}")
    print(f"    naos:     {n_naos:,}")
    print(f"    lugares:  {n_lugares:,}")

    # Metricas de calidad de atributos en personas
    personas = df_panel[df_panel["tipo_entidad"] == "persona"]
    if len(personas):
        con_gent = (personas["gentilicio"].fillna("").str.strip() != "").sum()
        con_vec = (personas["vecindad"].fillna("").str.strip() != "").sum()
        con_ocu = (personas["ocupacion"].fillna("").str.strip() != "").sum()
        con_rel = (personas["relacion"].fillna("").str.strip() != "").sum()
        print("\n  Cobertura atributos en personas:")
        print(f"    gentilicio: {con_gent:,} / {len(personas):,} ({100*con_gent/len(personas):.1f}%)")
        print(f"    vecindad:   {con_vec:,} / {len(personas):,} ({100*con_vec/len(personas):.1f}%)")
        print(f"    ocupacion:  {con_ocu:,} / {len(personas):,} ({100*con_ocu/len(personas):.1f}%)")
        print(f"    relacion:   {con_rel:,} / {len(personas):,} ({100*con_rel/len(personas):.1f}%)")

    # Distribucion de tipo de contrato
    tipos = df_panel_src["tipo"].fillna("").value_counts()
    print("\n  Distribucion de 'tipo' (contratos):")
    for t, n in tipos.items():
        print(f"    {t or '(vacio)':15s}  {n:,}")

    panel_path = OUT_DIR / "panel.xlsx"
    headers_path = OUT_DIR / "panel_headers.xlsx"
    multihoja_path = OUT_DIR / "panel_por_tipo.xlsx"

    df_panel.to_excel(panel_path, index=False)
    df_headers.to_excel(headers_path, index=False)
    counts = exportar_panel_multihoja(df_panel, multihoja_path)

    print(f"\n  Exportado:")
    print(f"    {panel_path}       ({panel_path.stat().st_size/1024:.0f} KB)")
    print(f"    {headers_path}     ({headers_path.stat().st_size/1024:.0f} KB)")
    print(f"    {multihoja_path}   ({multihoja_path.stat().st_size/1024:.0f} KB)")
    for nombre, n in counts.items():
        print(f"      hoja {nombre}: {n:,} filas")


if __name__ == "__main__":
    main()
