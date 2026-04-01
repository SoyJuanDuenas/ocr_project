"""
Segmentación visual de contratos en páginas escaneadas.

Usa perfil de proyección horizontal sobre imágenes binarizadas para detectar
los gaps verticales entre contratos. Cada gap indica un límite de contrato.

Uso:
    py src/segmentar_visual.py --image data/preprocess_v2/Tomo_I_p0008_prep.png
    py src/segmentar_visual.py --images-dir data/preprocess_v2 --glob "Tomo I*_prep.png" --out outputs/segmentacion
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Perfil de proyección horizontal
# ---------------------------------------------------------------------------

def perfil_horizontal(img_bin: np.ndarray, suavizado: int = 11) -> np.ndarray:
    """Cuenta píxeles negros (tinta) por fila, con suavizado.

    El suavizado (promedio móvil) elimina el ruido disperso del escáner
    en las zonas de gap, mientras que las líneas de texto con densidad
    alta se mantienen.

    Args:
        img_bin: imagen binarizada (0=negro/tinta, 255=blanco/fondo).
        suavizado: tamaño del kernel de promedio móvil (en filas).
                   0 o 1 para desactivar.

    Returns:
        Array 1-D con la densidad de tinta por fila (suavizada).
    """
    # Invertir: tinta=1, fondo=0
    inv = (img_bin < 128).astype(np.uint8)
    perfil = inv.sum(axis=1).astype(np.float64)

    if suavizado > 1:
        kernel = np.ones(suavizado) / suavizado
        perfil = np.convolve(perfil, kernel, mode="same")

    return perfil


def detectar_gaps(
    perfil: np.ndarray,
    *,
    umbral_frac: float = 0.05,
    min_gap_px: int = 10,
) -> list[tuple[int, int]]:
    """Detecta regiones vacías (gaps) en el perfil horizontal.

    El umbral se calcula sobre la **mediana de filas con contenido**, no
    sobre el máximo.  Esto es más robusto: una fila de texto normal tiene
    densidad entre 100-500 px; el ruido del escáner en gaps tiene < 30 px.
    Un umbral del 15% de la mediana de texto (típicamente ~50 px) separa
    limpiamente texto de gap sin depender del pico máximo.

    Args:
        perfil: densidad de tinta por fila (salida de perfil_horizontal).
        umbral_frac: fracción de la mediana de filas con contenido por
                     debajo de la cual se considera "vacío". Default 15%.
        min_gap_px: altura mínima del gap en píxeles para recoger como
                    candidato.

    Returns:
        Lista de (y_inicio, y_fin) de cada gap detectado.
    """
    umbral = perfil.max() * umbral_frac if umbral_frac < 1 else umbral_frac
    vacio = perfil <= umbral

    gaps = []
    en_gap = False
    inicio = 0

    for y, es_vacio in enumerate(vacio):
        if es_vacio and not en_gap:
            inicio = y
            en_gap = True
        elif not es_vacio and en_gap:
            if y - inicio >= min_gap_px:
                gaps.append((inicio, y))
            en_gap = False

    # Gap al final de la página
    if en_gap and len(perfil) - inicio >= min_gap_px:
        gaps.append((inicio, len(perfil)))

    return gaps


def fusionar_gaps_cercanos(
    gaps: list[tuple[int, int]],
    perfil: np.ndarray,
    *,
    max_puente_px: int = 30,
    umbral_puente_frac: float = 0.10,
) -> list[tuple[int, int]]:
    """Fusiona gaps separados por bandas estrechas de baja densidad.

    Cuando dos gaps están separados por una franja de pocos píxeles
    donde la densidad máxima no supera un umbral, se fusionan en uno
    solo.  Esto resuelve el caso de gaps inter-contrato que tienen
    ruido disperso del escáner que los fragmenta.

    Args:
        gaps: lista de (y_inicio, y_fin) ordenada por posición.
        perfil: perfil de densidad suavizado.
        max_puente_px: ancho máximo del puente entre gaps para fusionar.
        umbral_puente_frac: fracción del máximo del perfil; si la densidad
                            máxima del puente no la supera, se fusiona.
    """
    if len(gaps) <= 1:
        return gaps

    max_densidad = perfil.max()
    umbral_puente = max_densidad * umbral_puente_frac

    fusionados = [gaps[0]]
    for gap_ini, gap_fin in gaps[1:]:
        prev_ini, prev_fin = fusionados[-1]
        puente = gap_ini - prev_fin  # distancia entre fin del anterior e inicio del nuevo

        if puente <= max_puente_px and puente > 0:
            # Verificar densidad del puente
            densidad_puente = perfil[prev_fin:gap_ini].max()
            if densidad_puente <= umbral_puente:
                # Fusionar
                fusionados[-1] = (prev_ini, gap_fin)
                continue

        fusionados.append((gap_ini, gap_fin))

    return fusionados


def clasificar_gaps_adaptativo(
    gaps: list[tuple[int, int]],
    *,
    min_contratos: int = 2,
    salto_relativo: float = 0.5,
) -> list[tuple[int, int]]:
    """Filtra gaps para quedarse solo con los inter-contrato.

    Estrategia: ordena gaps por altura y busca el primer salto significativo
    entre alturas consecutivas.  Un salto es significativo si supera
    ``salto_relativo`` veces la altura del gap inferior (i.e., el gap
    siguiente es al menos 50 % más alto que el anterior).  Esto captura
    la transición interlineado → inter-contrato, que en las páginas del
    catálogo ocurre típicamente en el rango 30-55 px.

    Args:
        gaps: lista de (y_inicio, y_fin) candidatos.
        min_contratos: mínimo de gaps esperados.
        salto_relativo: proporción mínima de incremento relativo para
                        considerar un salto como significativo (default 0.5
                        = 50 % más alto).

    Returns:
        Gaps clasificados como inter-contrato.
    """
    if len(gaps) <= 1:
        return gaps

    alturas = sorted(b - a for a, b in gaps)

    # Buscar el primer salto relativo significativo
    umbral_altura = alturas[0]  # fallback: todos
    encontrado = False
    for i in range(len(alturas) - 1):
        h_actual = alturas[i]
        h_siguiente = alturas[i + 1]
        if h_actual > 0 and (h_siguiente - h_actual) / h_actual >= salto_relativo:
            umbral_altura = (h_actual + h_siguiente) / 2
            encontrado = True
            break

    if not encontrado:
        # Sin salto claro: usar percentil 60 como corte conservador
        umbral_altura = np.percentile(alturas, 60)

    resultado = [(a, b) for a, b in gaps if (b - a) >= umbral_altura]

    # Fallback: si quedaron muy pocos, bajar al percentil 40
    if len(resultado) < min_contratos and len(gaps) >= min_contratos:
        p40 = np.percentile(alturas, 40)
        resultado = [(a, b) for a, b in gaps if (b - a) >= p40]

    return resultado


def filtrar_gaps_contratos(
    gaps: list[tuple[int, int]],
    altura_img: int,
    *,
    margen_superior: float = 0.05,
    margen_inferior: float = 0.08,
) -> list[tuple[int, int]]:
    """Filtra gaps que están en márgenes de página (header/footer).

    Args:
        gaps: lista de (y_inicio, y_fin).
        altura_img: altura total de la imagen.
        margen_superior: fracción superior a ignorar (encabezado).
        margen_inferior: fracción inferior a ignorar (pie de página).

    Returns:
        Gaps filtrados que están en la zona de contenido.
    """
    y_min = int(altura_img * margen_superior)
    y_max = int(altura_img * (1 - margen_inferior))

    return [(a, b) for a, b in gaps if a >= y_min and b <= y_max]


# ---------------------------------------------------------------------------
# Segmentación de una página
# ---------------------------------------------------------------------------

def segmentar_pagina(
    img_path: str | Path,
    *,
    umbral_frac: float = 0.05,
    min_gap_px: int = 10,
    margen_superior: float = 0.05,
    margen_inferior: float = 0.08,
    debug: bool = False,
    debug_dir: Path | None = None,
) -> dict:
    """Segmenta una página en contratos usando gaps visuales.

    Returns:
        dict con claves:
            - pagina: nombre del archivo
            - n_gaps: número de gaps detectados
            - n_segmentos: número de segmentos (n_gaps + 1 si hay gaps)
            - gaps: lista de (y_inicio, y_fin)
            - segmentos: lista de (y_inicio, y_fin) de cada segmento de texto
    """
    img_path = Path(img_path)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {img_path}")

    altura, ancho = img.shape

    perfil = perfil_horizontal(img)
    gaps_raw = detectar_gaps(perfil, umbral_frac=umbral_frac, min_gap_px=min_gap_px)
    gaps_zona = filtrar_gaps_contratos(
        gaps_raw, altura,
        margen_superior=margen_superior,
        margen_inferior=margen_inferior,
    )
    gaps_fusionados = fusionar_gaps_cercanos(gaps_zona, perfil)
    gaps = clasificar_gaps_adaptativo(gaps_fusionados)

    # Calcular segmentos de contenido entre gaps
    zona_inicio = int(altura * margen_superior)
    zona_fin = int(altura * (1 - margen_inferior))

    segmentos_raw = []
    prev = zona_inicio
    for gap_ini, gap_fin in gaps:
        if gap_ini > prev:
            segmentos_raw.append((prev, gap_ini))
        prev = gap_fin
    if prev < zona_fin:
        segmentos_raw.append((prev, zona_fin))

    # Filtrar segmentos muy pequeños (ruido, no pueden ser un contrato)
    # Una línea de texto ocupa ~30-40px; un contrato mínimo tiene al menos 2 líneas
    min_seg_px = 60
    segmentos = [(a, b) for a, b in segmentos_raw if b - a >= min_seg_px]

    resultado = {
        "pagina": img_path.name,
        "n_gaps": len(gaps),
        "n_segmentos": len(segmentos),
        "gaps": gaps,
        "segmentos": segmentos,
        "altura": altura,
        "ancho": ancho,
    }

    # Debug: guardar imagen con gaps marcados y perfil
    if debug and debug_dir is not None:
        _guardar_debug(img, perfil, gaps, segmentos, img_path.stem, debug_dir)

    return resultado


def _guardar_debug(
    img: np.ndarray,
    perfil: np.ndarray,
    gaps: list[tuple[int, int]],
    segmentos: list[tuple[int, int]],
    nombre: str,
    debug_dir: Path,
) -> None:
    """Guarda imagen anotada con gaps y perfil de proyección."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Imagen con gaps marcados
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ancho = img.shape[1]

    for gap_ini, gap_fin in gaps:
        # Banda roja semi-transparente sobre el gap
        overlay = img_color.copy()
        cv2.rectangle(overlay, (0, gap_ini), (ancho, gap_fin), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, img_color, 0.7, 0, img_color)
        # Líneas de borde
        cv2.line(img_color, (0, gap_ini), (ancho, gap_ini), (0, 0, 255), 2)
        cv2.line(img_color, (0, gap_fin), (ancho, gap_fin), (0, 0, 255), 2)

    # Numerar segmentos
    for i, (seg_ini, seg_fin) in enumerate(segmentos):
        y_mid = (seg_ini + seg_fin) // 2
        cv2.putText(
            img_color, f"Seg {i+1}", (10, y_mid),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 0), 3,
        )

    cv2.imwrite(str(debug_dir / f"{nombre}_gaps.png"), img_color)

    # Perfil de proyección como gráfica
    perfil_img = np.full((img.shape[0], 300, 3), 255, dtype=np.uint8)
    max_val = perfil.max() if perfil.max() > 0 else 1
    for y, val in enumerate(perfil):
        x = int(val / max_val * 280)
        cv2.line(perfil_img, (0, y), (x, y), (0, 0, 0), 1)

    for gap_ini, gap_fin in gaps:
        cv2.rectangle(perfil_img, (0, gap_ini), (300, gap_fin), (0, 0, 255), 2)

    cv2.imwrite(str(debug_dir / f"{nombre}_perfil.png"), perfil_img)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def segmentar_batch(
    images_dir: str | Path,
    glob_pattern: str = "*_prep.png",
    *,
    out_dir: str | Path | None = None,
    debug: bool = False,
    **kwargs,
) -> list[dict]:
    """Segmenta todas las páginas que coinciden con el glob."""
    images_dir = Path(images_dir)
    paginas = sorted(images_dir.glob(glob_pattern))

    if not paginas:
        print(f"No se encontraron imagenes con {glob_pattern} en {images_dir}")
        return []

    out_path = Path(out_dir) if out_dir else None
    debug_dir = out_path / "debug" if out_path and debug else None

    resultados = []
    for i, pag in enumerate(paginas):
        print(f"  [{i+1}/{len(paginas)}] {pag.name}", end="", flush=True)
        try:
            r = segmentar_pagina(
                pag, debug=debug, debug_dir=debug_dir, **kwargs
            )
            resultados.append(r)
            print(f"  -> {r['n_gaps']} gaps, {r['n_segmentos']} segmentos")
        except Exception as e:
            print(f"  ERROR: {e}")
            resultados.append({"pagina": pag.name, "error": str(e)})

    # Exportar resumen CSV
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / "segmentacion_visual.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pagina", "n_gaps", "n_segmentos", "gaps_y"])
            for r in resultados:
                if "error" not in r:
                    gaps_str = "; ".join(f"{a}-{b}" for a, b in r["gaps"])
                    w.writerow([r["pagina"], r["n_gaps"], r["n_segmentos"], gaps_str])
        print(f"\nResumen exportado: {csv_path}")

    return resultados


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Segmentacion visual de contratos por gaps horizontales"
    )
    parser.add_argument("--image", type=str, help="Pagina individual a segmentar")
    parser.add_argument("--images-dir", type=str, help="Directorio con imagenes")
    parser.add_argument("--glob", type=str, default="*_prep.png",
                        help="Patron glob para filtrar imagenes (default: *_prep.png)")
    parser.add_argument("--out", type=str, help="Directorio de salida")
    parser.add_argument("--debug", action="store_true",
                        help="Guardar imagenes de debug con gaps marcados")
    parser.add_argument("--umbral-frac", type=float, default=0.05,
                        help="Fraccion del maximo de densidad para considerar vacio (default: 0.05)")
    parser.add_argument("--min-gap-px", type=int, default=10,
                        help="Altura minima de gap en pixeles (default: 10)")
    parser.add_argument("--margen-superior", type=float, default=0.05,
                        help="Fraccion superior de la pagina a ignorar (default: 0.05)")
    parser.add_argument("--margen-inferior", type=float, default=0.08,
                        help="Fraccion inferior de la pagina a ignorar (default: 0.05)")

    args = parser.parse_args()

    kwargs = dict(
        umbral_frac=args.umbral_frac,
        min_gap_px=args.min_gap_px,
        margen_superior=args.margen_superior,
        margen_inferior=args.margen_inferior,
    )

    if args.image:
        r = segmentar_pagina(
            args.image, debug=args.debug,
            debug_dir=Path(args.out) / "debug" if args.out else Path("debug_segmentacion"),
            **kwargs,
        )
        print(f"\n{r['pagina']}: {r['n_gaps']} gaps, {r['n_segmentos']} segmentos")
        for i, (a, b) in enumerate(r["gaps"]):
            print(f"  Gap {i+1}: y={a}-{b} ({b-a}px)")
        for i, (a, b) in enumerate(r["segmentos"]):
            print(f"  Segmento {i+1}: y={a}-{b} ({b-a}px)")

    elif args.images_dir:
        segmentar_batch(
            args.images_dir, args.glob,
            out_dir=args.out, debug=args.debug, **kwargs,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
