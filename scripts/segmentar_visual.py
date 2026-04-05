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


def _contar_componentes_pequenos(
    img_bin: np.ndarray,
    *,
    max_area: int = 20,
) -> int:
    """Cuenta componentes pequeños residuales de ruido."""
    inv = (img_bin < 128).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    count = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= max_area:
            count += 1
    return count


def _densidad_tinta(img_bin: np.ndarray, y0: int, y1: int) -> float:
    """Calcula densidad de tinta en una franja vertical."""
    y0 = max(0, int(y0))
    y1 = min(img_bin.shape[0], int(y1))
    if y1 <= y0:
        return 0.0
    band = img_bin[y0:y1, :]
    return float((band < 128).mean())


def _diagnosticar_segmentacion(
    img_bin: np.ndarray,
    *,
    gaps: list[tuple[int, int]],
    segmentos: list[tuple[int, int]],
    zona_fin: int,
    margen_inferior: float,
    refinement_applied: bool,
) -> dict:
    """Resume síntomas de páginas mal escaneadas o mal segmentadas."""
    altura = img_bin.shape[0]
    footer_start = zona_fin

    alturas_segmentos = [b - a for a, b in segmentos]
    mediana_seg = float(np.median(alturas_segmentos)) if alturas_segmentos else 0.0
    ultimo_seg = float(alturas_segmentos[-1]) if alturas_segmentos else 0.0
    ratio_ultimo = (ultimo_seg / mediana_seg) if mediana_seg > 0 else 0.0
    footer_density = _densidad_tinta(img_bin, footer_start, altura)
    small_components = _contar_componentes_pequenos(img_bin)

    flags: list[str] = []
    if len(segmentos) < 4 or len(segmentos) > 9:
        flags.append("n_segmentos_atipico")
    if ratio_ultimo >= 1.8:
        flags.append("ultimo_segmento_grande")
    if footer_density >= 0.025:
        flags.append("footer_con_tinta")
    if small_components >= 1200:
        flags.append("ruido_residual")
    if len(gaps) <= 2:
        flags.append("muy_pocos_gaps")
    if refinement_applied:
        flags.append("refinamiento_adaptativo")

    score = len(flags)
    if score >= 4:
        estado = "probable_mal_escaneo"
    elif score >= 2:
        estado = "revisar"
    else:
        estado = "ok"

    return {
        "mediana_altura_segmentos": round(mediana_seg, 1),
        "altura_ultimo_segmento": round(ultimo_seg, 1),
        "ratio_ultimo_segmento": round(ratio_ultimo, 2),
        "footer_density": round(footer_density, 4),
        "small_components": small_components,
        "diagnostico_flags": "; ".join(flags),
        "diagnostico_score": score,
        "diagnostico_estado": estado,
        "margen_inferior_pct": round(margen_inferior * 100, 2),
    }


def filtrar_gaps_contratos(
    gaps: list[tuple[int, int]],
    altura_img: int,
    *,
    margen_superior: float = 0.06,
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


def _rescatar_gap_final(
    gaps_finales: list[tuple[int, int]],
    gaps_raw: list[tuple[int, int]],
    *,
    zona_fin: int,
    min_gap_px: int,
    min_tail_segment_px: int = 350,
    min_footer_overlap_px: int = 120,
) -> list[tuple[int, int]]:
    """Recupera un gap final que cae parcialmente dentro del footer.

    Solo actúa cuando:
    - el último gap aceptado quedó lejos del final de la zona útil
    - existe un gap crudo que empieza dentro del contenido y se extiende al footer

    Esto evita fusionar el último contrato con el pie de página, sin abrir
    gaps extra en páginas donde la cola ya estaba bien segmentada.
    """
    if not gaps_raw:
        return gaps_finales

    last_end = gaps_finales[-1][1] if gaps_finales else 0
    tail_len = zona_fin - last_end
    if tail_len < min_tail_segment_px:
        return gaps_finales

    candidatos = []
    for a, b in gaps_raw:
        if a <= last_end:
            continue
        if a >= zona_fin:
            continue
        if b <= zona_fin:
            continue

        overlap = zona_fin - a
        if overlap < max(min_gap_px, min_footer_overlap_px):
            continue
        candidatos.append((a, zona_fin))

    if not candidatos:
        return gaps_finales

    # Elegimos el gap más cercano al footer para partir la cola larga.
    rescued = max(candidatos, key=lambda g: g[0])
    return sorted(gaps_finales + [rescued], key=lambda x: x[0])


def _refinar_gaps_sobresegmentados(
    gaps: list[tuple[int, int]],
    *,
    small_gap_px: int = 55,
    min_gaps_para_refinar: int = 10,
    min_ratio_small: float = 0.45,
) -> tuple[list[tuple[int, int]], bool]:
    """Endurece la selección cuando una página quedó hiperfragmentada.

    La idea es general: si la segmentación aceptó demasiados gaps y una
    fracción grande de ellos es muy pequeña, probablemente estamos
    capturando interlineado o separación de párrafos, no contratos.
    """
    if len(gaps) < min_gaps_para_refinar:
        return gaps, False

    alturas = np.array([b - a for a, b in gaps], dtype=np.float64)
    if len(alturas) == 0:
        return gaps, False

    ratio_small = float((alturas < small_gap_px).mean())
    if ratio_small < min_ratio_small:
        return gaps, False

    # Umbral más estricto, pero adaptativo a la distribución de la página.
    threshold = max(
        float(small_gap_px),
        float(np.percentile(alturas, 60)),
    )
    refined = [(a, b) for a, b in gaps if (b - a) >= threshold]

    # Evitar colapsar páginas buenas a muy pocos bloques.
    if len(refined) < 4:
        return gaps, False

    return refined, True


# ---------------------------------------------------------------------------
# Segmentación de una página
# ---------------------------------------------------------------------------

def segmentar_pagina(
    img_path: str | Path,
    *,
    umbral_frac: float = 0.05,
    min_gap_px: int = 10,
    margen_superior: float = 0.06,
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
    gaps = _rescatar_gap_final(
        gaps,
        gaps_raw,
        zona_fin=int(altura * (1 - margen_inferior)),
        min_gap_px=min_gap_px,
    )
    gaps, refinement_applied = _refinar_gaps_sobresegmentados(gaps)

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
        "margen_superior": margen_superior,
        "margen_inferior": margen_inferior,
        "zona_inicio": zona_inicio,
        "zona_fin": zona_fin,
        "refinement_applied": refinement_applied,
    }
    resultado.update(
        _diagnosticar_segmentacion(
            img,
            gaps=gaps,
            segmentos=segmentos,
            zona_fin=zona_fin,
            margen_inferior=margen_inferior,
            refinement_applied=refinement_applied,
        )
    )

    # Debug: guardar imagen con gaps marcados y perfil
    if debug and debug_dir is not None:
        _guardar_debug(
            img,
            perfil,
            gaps,
            segmentos,
            img_path.stem,
            debug_dir,
            margen_superior=margen_superior,
            margen_inferior=margen_inferior,
            zona_inicio=zona_inicio,
            zona_fin=zona_fin,
            diagnostico_estado=resultado["diagnostico_estado"],
            diagnostico_score=resultado["diagnostico_score"],
        )

    return resultado


def _guardar_debug(
    img: np.ndarray,
    perfil: np.ndarray,
    gaps: list[tuple[int, int]],
    segmentos: list[tuple[int, int]],
    nombre: str,
    debug_dir: Path,
    *,
    margen_superior: float,
    margen_inferior: float,
    zona_inicio: int,
    zona_fin: int,
    diagnostico_estado: str,
    diagnostico_score: int,
) -> None:
    """Guarda imagen anotada con gaps y perfil de proyección."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Imagen con gaps marcados
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ancho = img.shape[1]
    altura = img.shape[0]

    # Visualizar márgenes excluidos para que el recorte sea auditable.
    if zona_inicio > 0:
        overlay = img_color.copy()
        cv2.rectangle(overlay, (0, 0), (ancho, zona_inicio), (255, 220, 0), -1)
        cv2.addWeighted(overlay, 0.18, img_color, 0.82, 0, img_color)
    if zona_fin < altura:
        overlay = img_color.copy()
        cv2.rectangle(overlay, (0, zona_fin), (ancho, altura), (255, 220, 0), -1)
        cv2.addWeighted(overlay, 0.18, img_color, 0.82, 0, img_color)

    cv2.line(img_color, (0, zona_inicio), (ancho, zona_inicio), (255, 170, 0), 2)
    cv2.line(img_color, (0, zona_fin), (ancho, zona_fin), (255, 170, 0), 2)

    cv2.putText(
        img_color,
        f"top margin {margen_superior*100:.1f}% ({zona_inicio}px)",
        (20, max(35, zona_inicio - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 120, 0),
        2,
    )
    cv2.putText(
        img_color,
        f"bottom margin {margen_inferior*100:.1f}% ({altura - zona_fin}px)",
        (20, min(altura - 20, zona_fin + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 120, 0),
        2,
    )
    cv2.putText(
        img_color,
        f"diag: {diagnostico_estado} | score {diagnostico_score}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 140, 255),
        2,
    )

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
        boxes_path = out_path / "contract_boxes_proposed.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "pagina", "n_gaps", "n_segmentos", "gaps_y",
                "mediana_altura_segmentos", "altura_ultimo_segmento",
                "ratio_ultimo_segmento", "footer_density", "small_components",
                "diagnostico_score", "diagnostico_estado", "diagnostico_flags",
            ])
            for r in resultados:
                if "error" not in r:
                    gaps_str = "; ".join(f"{a}-{b}" for a, b in r["gaps"])
                    w.writerow([
                        r["pagina"],
                        r["n_gaps"],
                        r["n_segmentos"],
                        gaps_str,
                        r["mediana_altura_segmentos"],
                        r["altura_ultimo_segmento"],
                        r["ratio_ultimo_segmento"],
                        r["footer_density"],
                        r["small_components"],
                        r["diagnostico_score"],
                        r["diagnostico_estado"],
                        r["diagnostico_flags"],
                    ])
        with open(boxes_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "pagina", "image_path", "segment_id",
                "img_width", "img_height",
                "x0", "y0", "x1", "y1",
                "box_width", "box_height",
                "diagnostico_estado", "diagnostico_score", "diagnostico_flags",
            ])
            for r in resultados:
                if "error" in r:
                    continue
                img_width = r.get("ancho", 0)
                img_height = r.get("altura", 0)
                image_path = str((images_dir / r["pagina"]).resolve()) if out_path else r["pagina"]
                for seg_id, (y0, y1) in enumerate(r["segmentos"], start=1):
                    x0 = 0
                    x1 = img_width
                    w.writerow([
                        r["pagina"],
                        image_path,
                        seg_id,
                        img_width,
                        img_height,
                        x0,
                        y0,
                        x1,
                        y1,
                        x1 - x0,
                        y1 - y0,
                        r["diagnostico_estado"],
                        r["diagnostico_score"],
                        r["diagnostico_flags"],
                    ])
        print(f"\nResumen exportado: {csv_path}")
        print(f"Cajas propuestas exportadas: {boxes_path}")

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
    parser.add_argument("--margen-superior", type=float, default=0.06,
                        help="Fraccion superior de la pagina a ignorar (default: 0.06)")
    parser.add_argument("--margen-inferior", type=float, default=0.08,
                        help="Fraccion inferior de la pagina a ignorar (default: 0.08)")

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
