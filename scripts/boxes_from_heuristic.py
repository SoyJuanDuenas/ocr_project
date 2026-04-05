"""
Genera bounding boxes 2D de contratos usando heurística de proyección.

- Y: perfil de proyección horizontal (gaps verticales entre contratos)
- X: perfil de proyección vertical dentro de cada segmento (dónde empieza/termina la tinta)

Uso:
  py scripts/boxes_from_heuristic.py --images-dir data/segmentation/images/train --out data/segmentation/prelabels
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from segmentar_visual import (
    perfil_horizontal,
    detectar_gaps,
    filtrar_gaps_contratos,
    fusionar_gaps_cercanos,
    clasificar_gaps_adaptativo,
    _rescatar_gap_final,
    _refinar_gaps_sobresegmentados,
)


def _limites_x(img_bin: np.ndarray, y0: int, y1: int, margen_px: int = 10) -> tuple[int, int]:
    """Calcula x_inicio y x_fin de la tinta en una franja vertical [y0, y1].

    Proyecta verticalmente (suma por columna) dentro de la franja
    y busca dónde la densidad supera un umbral.
    """
    franja = img_bin[y0:y1, :]
    # Invertir: tinta=1, fondo=0
    inv = (franja < 128).astype(np.uint8)
    perfil_v = inv.sum(axis=0).astype(np.float64)

    # Umbral: 2% de la altura de la franja (al menos unas pocas filas con tinta)
    umbral = max(2, (y1 - y0) * 0.02)

    ancho = len(perfil_v)
    x0 = 0
    x1 = ancho

    # Buscar primer columna con tinta
    for x in range(ancho):
        if perfil_v[x] > umbral:
            x0 = max(0, x - margen_px)
            break

    # Buscar última columna con tinta
    for x in range(ancho - 1, -1, -1):
        if perfil_v[x] > umbral:
            x1 = min(ancho, x + margen_px)
            break

    return x0, x1


def segmentar_pagina_2d(
    img_path: Path,
    *,
    umbral_frac: float = 0.05,
    min_gap_px: int = 10,
    margen_superior: float = 0.06,
    margen_inferior: float = 0.08,
    margen_x_px: int = 15,
) -> list[dict]:
    """Segmenta una página en contratos con bounding boxes 2D.

    Returns:
        Lista de dicts con: x0, y0, x1, y1, tipo
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {img_path}")

    altura, ancho = img.shape

    # --- Y: heurística de proyección horizontal ---
    perfil = perfil_horizontal(img)
    gaps_raw = detectar_gaps(perfil, umbral_frac=umbral_frac, min_gap_px=min_gap_px)
    gaps_zona = filtrar_gaps_contratos(
        gaps_raw, altura,
        margen_superior=margen_superior,
        margen_inferior=margen_inferior,
    )
    gaps_fusionados = fusionar_gaps_cercanos(gaps_zona, perfil)
    gaps = clasificar_gaps_adaptativo(gaps_fusionados)
    gaps = _rescatar_gap_final(
        gaps, gaps_raw,
        zona_fin=int(altura * (1 - margen_inferior)),
        min_gap_px=min_gap_px,
    )
    gaps, _ = _refinar_gaps_sobresegmentados(gaps)

    # Segmentos verticales
    zona_inicio = int(altura * margen_superior)
    zona_fin = int(altura * (1 - margen_inferior))

    segmentos_y = []
    prev = zona_inicio
    for gap_ini, gap_fin in gaps:
        if gap_ini > prev:
            segmentos_y.append((prev, gap_ini))
        prev = gap_fin
    if prev < zona_fin:
        segmentos_y.append((prev, zona_fin))

    # Filtrar segmentos muy pequeños
    min_seg_px = 60
    segmentos_y = [(a, b) for a, b in segmentos_y if b - a >= min_seg_px]

    # --- X: proyección vertical por segmento ---
    boxes = []
    for y0, y1 in segmentos_y:
        x0, x1 = _limites_x(img, y0, y1, margen_px=margen_x_px)
        boxes.append({
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "tipo": "contrato",
        })

    return boxes


def generar_boxes_batch(
    images_dir: Path,
    glob_pattern: str = "Tomo I_*_seg.png",
    out_dir: Path | None = None,
    **kwargs,
) -> list[dict]:
    """Genera cajas 2D para todas las imágenes."""
    paginas = sorted(images_dir.glob(glob_pattern))
    if not paginas:
        print(f"No se encontraron imágenes con '{glob_pattern}' en {images_dir}")
        return []

    print(f"Procesando {len(paginas)} páginas...")

    resultados = []
    total_boxes = 0

    for i, pag in enumerate(paginas):
        try:
            boxes = segmentar_pagina_2d(pag, **kwargs)
            total_boxes += len(boxes)
            img = cv2.imread(str(pag))
            img_h, img_w = img.shape[:2]

            for seg_id, box in enumerate(boxes, 1):
                resultados.append({
                    "pagina": pag.name,
                    "image_path": str(pag.resolve()),
                    "segment_id": seg_id,
                    "img_width": img_w,
                    "img_height": img_h,
                    "x0": box["x0"],
                    "y0": box["y0"],
                    "x1": box["x1"],
                    "y1": box["y1"],
                    "box_width": box["x1"] - box["x0"],
                    "box_height": box["y1"] - box["y0"],
                    "tipo": box["tipo"],
                })
        except Exception as e:
            print(f"  ERROR {pag.name}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(paginas)}] {total_boxes} cajas...")

    print(f"  Total: {len(paginas)} páginas, {total_boxes} cajas")

    if out_dir and resultados:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "contract_boxes_heuristic.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
            writer.writeheader()
            writer.writerows(resultados)
        print(f"  Exportado: {csv_path}")

    return resultados


def main():
    parser = argparse.ArgumentParser(
        description="Genera bounding boxes 2D con heurística de proyección"
    )
    parser.add_argument("--images-dir", default="data/segmentation/images/train")
    parser.add_argument("--glob", default="*_seg.png")
    parser.add_argument("--out", default="data/segmentation/prelabels")
    parser.add_argument("--margen-x", type=int, default=15,
                        help="Margen horizontal en px alrededor del texto (default: 15)")

    args = parser.parse_args()
    generar_boxes_batch(
        Path(args.images_dir), args.glob,
        out_dir=Path(args.out),
        margen_x_px=args.margen_x,
    )


if __name__ == "__main__":
    main()
