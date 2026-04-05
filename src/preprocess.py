"""
Preprocesa imágenes (PNG/JPG/TIF/WEBP/BMP) y PDFs (página a página) para OCR.

Uso CLI — preprocesar:
    py src/preprocess.py --in data/raw --out data/preprocess_v2 --save-debug
    py src/preprocess.py --in data/raw --out data/preprocess_v2 --bin sauvola --sauvola-w 31 --sauvola-k 0.45

Uso CLI — tuning de perfiles por tomo:
    py src/preprocess.py --tune --in data/raw --out data/preprocess_v2
    py src/preprocess.py --tune --in data/raw --out data/preprocess_v2 --tune-only
    py src/preprocess.py --tune --in data/raw --out data/preprocess_v2 --reference-tomo "Tomo I" --sample-pages 8 --max-candidates 8

El modo --tune:
  1. Lee *_prep.png existentes en --out para calcular métricas por tomo
  2. Compara cada tomo contra el de referencia (Tomo I por defecto)
  3. Grid search de parámetros en los tomos más distantes
  4. Actualiza PREPROCESS_PROFILE_OVERRIDES en este mismo archivo
  5. Ejecuta preprocesamiento con los nuevos perfiles (salvo --tune-only)

Uso desde otro script:
    from preprocess import preprocess, tune
    result = preprocess("data/raw", "data/preprocess_v2", save_debug=True)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# -------------------- Extensiones permitidas --------------------
VALID_IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
VALID_PDF_EXT = {".pdf"}

TOMO_PAGE_RE = re.compile(r"^(?P<tomo>.+?)_p\d{4}$")
PAGE_PREP_RE = re.compile(r"^(?P<tomo>.+?)_p(?P<page>\d+)_prep\.png$")
PDF_RE = re.compile(r"^(?P<tomo>.+)\.pdf$", re.IGNORECASE)

# --- Inicio PREPROCESS_PROFILE_OVERRIDES ---
PREPROCESS_PROFILE_OVERRIDES = {
    "Tomo IX": {
        "bg_ksize": 15,
        "sauvola_w": 51,
        "sauvola_k": 0.18,
        "close_ksize": 1,
        "open_ksize": 0,
        "denoise_ksize": 1,
    },
    "Tomo VI": {
        "bg_ksize": 11,
        "sauvola_w": 51,
        "sauvola_k": 0.18,
        "close_ksize": 1,
        "open_ksize": 0,
        "denoise_ksize": 1,
    },
    "Tomo VIII": {
        "bg_ksize": 11,
        "sauvola_w": 51,
        "sauvola_k": 0.18,
        "close_ksize": 1,
        "open_ksize": 0,
        "denoise_ksize": 1,
    },
    "Tomo X": {
        "bg_ksize": 11,
        "sauvola_w": 51,
        "sauvola_k": 0.18,
        "close_ksize": 1,
        "open_ksize": 0,
        "denoise_ksize": 1,
    },
    "Tomo XI": {
        "bg_ksize": 11,
        "sauvola_w": 51,
        "sauvola_k": 0.18,
        "close_ksize": 1,
        "open_ksize": 0,
        "denoise_ksize": 1,
    },
    "Tomo XIV": {
        "bg_ksize": 11,
        "sauvola_w": 31,
        "sauvola_k": 0.24,
        "close_ksize": 1,
        "open_ksize": 0,
        "denoise_ksize": 1,
    },
}
# --- Fin PREPROCESS_PROFILE_OVERRIDES ---

# -------------------- Lectura --------------------
def _read_image_bgr(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv.imdecode(data, cv.IMREAD_COLOR)

def _read_dpi(path: Path, default_dpi: int = 300) -> int:
    if Image is None:
        return default_dpi
    try:
        with Image.open(path) as im:
            dpi = im.info.get("dpi", None)
            if isinstance(dpi, tuple) and len(dpi) >= 1:
                xdpi = dpi[0] or default_dpi
                return int(xdpi)
    except Exception:
        pass
    return default_dpi

# -------------------- Transformaciones --------------------
def _to_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

def _resize_to_dpi(img, dpi_src=300, dpi_tgt=300):
    if dpi_src <= 0 or dpi_src == dpi_tgt:
        return img
    scale = dpi_tgt / float(dpi_src)
    return cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

def _zoom(gray, factor=1.15):
    if factor == 1.0:
        return gray
    h, w = gray.shape[:2]
    return cv.resize(gray, (int(w * factor), int(h * factor)), interpolation=cv.INTER_CUBIC)

def _normalize_background(gray, bg_ksize=31):
    if bg_ksize <= 0:
        return gray
    if bg_ksize % 2 == 0:
        bg_ksize += 1
    bg = cv.medianBlur(gray, bg_ksize)
    norm = cv.divide(gray, bg, scale=255)
    return norm

def _binarize_sauvola(gray, w=31, k=0.34):
    if w % 2 == 0:
        w += 1
    m = cv.boxFilter(gray, ddepth=-1, ksize=(w, w))
    g32 = gray.astype(np.float32)
    m2 = cv.boxFilter(g32 * g32, ddepth=-1, ksize=(w, w))
    s = np.sqrt(np.maximum(m2 - m.astype(np.float32) ** 2, 0))
    t = m.astype(np.float32) * (1 + k * ((s / 128.0) - 1))
    bin_img = (g32 > t).astype(np.uint8) * 255
    return bin_img

def _binarize(gray, method="auto", block_size=35, C=11, sauvola_w=31, sauvola_k=0.34):
    if method == "otsu":
        return cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    if method == "adaptive":
        if block_size % 2 == 0:
            block_size += 1
        return cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, C
        )
    if method == "sauvola":
        return _binarize_sauvola(gray, w=sauvola_w, k=sauvola_k)
    # auto = elige adaptativa u Otsu según dispersión
    adp = _binarize(gray, method="adaptive", block_size=block_size, C=C)
    ots = _binarize(gray, method="otsu")
    return adp if adp.std() > ots.std() else ots

def _fix_strokes(bin_img, close_ksize=2, open_ksize=0):
    """
    Reconecta trazos finos (cierre) y opcionalmente limpia punticos (apertura).
    Nota: Texto negro (0), fondo blanco (255). Invertimos para actuar sobre el texto.
    """
    inv = 255 - bin_img
    out = inv
    if close_ksize and close_ksize > 0:
        k = cv.getStructuringElement(cv.MORPH_RECT, (close_ksize, close_ksize))
        out = cv.morphologyEx(out, cv.MORPH_CLOSE, k)
    if open_ksize and open_ksize > 0:
        k2 = cv.getStructuringElement(cv.MORPH_RECT, (open_ksize, open_ksize))
        out = cv.morphologyEx(out, cv.MORPH_OPEN, k2)
    return 255 - out

def _denoise_after_bin(bin_img, ksize=3):
    k = int(ksize)
    if k <= 1:
        return bin_img
    if k % 2 == 0:
        k += 1
    return cv.medianBlur(bin_img, k)

def _trim_and_border(img_bin, pad=8):
    contours, _ = cv.findContours(255 - img_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv.copyMakeBorder(img_bin, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=255)
    x, y, w, h = cv.boundingRect(max(contours, key=cv.contourArea))
    cropped = img_bin[max(y - 3, 0): y + h + 3, max(x - 3, 0): x + w + 3]
    return cv.copyMakeBorder(cropped, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=255)

def _maybe_write(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imencode(".png", image)[1].tofile(str(path))


def _resolve_tomo_name(path: Path) -> str:
    stem = path.stem
    match = TOMO_PAGE_RE.match(stem)
    if match:
        return match.group("tomo")
    return stem


def _resolve_profile(tomo_name: str, base: dict) -> tuple[str, dict]:
    params = dict(base)
    overrides = PREPROCESS_PROFILE_OVERRIDES.get(tomo_name, {})
    params.update(overrides)
    profile_name = "baseline"
    if overrides:
        profile_name = f"tomo:{tomo_name}"
    return profile_name, params

# -------------------- Tuning --------------------
TUNE_METRIC_KEYS = ["ink_lt_128", "white_ge_240", "mean", "std", "p10", "p50", "p90"]

TUNE_PARAM_GRID = [
    {"name": "baseline"},
    {"name": "soft_bg15_k030", "bg_ksize": 15, "sauvola_w": 31, "sauvola_k": 0.30, "close_ksize": 2, "open_ksize": 1, "denoise_ksize": 1},
    {"name": "soft_bg15_k024", "bg_ksize": 15, "sauvola_w": 41, "sauvola_k": 0.24, "close_ksize": 1, "open_ksize": 1, "denoise_ksize": 1},
    {"name": "soft_bg11_k018", "bg_ksize": 11, "sauvola_w": 51, "sauvola_k": 0.18, "close_ksize": 1, "open_ksize": 0, "denoise_ksize": 1},
    {"name": "soft_bg21_k026", "bg_ksize": 21, "sauvola_w": 41, "sauvola_k": 0.26, "close_ksize": 1, "open_ksize": 1, "denoise_ksize": 1},
    {"name": "mid_bg21_k030", "bg_ksize": 21, "sauvola_w": 31, "sauvola_k": 0.30, "close_ksize": 2, "open_ksize": 1, "denoise_ksize": 1},
    {"name": "mid_bg31_k034", "bg_ksize": 31, "sauvola_w": 41, "sauvola_k": 0.34, "close_ksize": 2, "open_ksize": 1, "denoise_ksize": 3},
    {"name": "sharp_bg11_k024", "bg_ksize": 11, "sauvola_w": 31, "sauvola_k": 0.24, "close_ksize": 1, "open_ksize": 0, "denoise_ksize": 1},
    {"name": "sharp_bg15_k018", "bg_ksize": 15, "sauvola_w": 51, "sauvola_k": 0.18, "close_ksize": 1, "open_ksize": 0, "denoise_ksize": 1},
]

# Parámetros base para el grid (los que no se varían)
_TUNE_BASE = {
    "target_dpi": 300, "zoom": 1.15, "bin_method": "sauvola",
    "block_size": 35, "C": 11, "sauvola_w": 31, "sauvola_k": 0.45,
    "bg_ksize": 31, "close_ksize": 3, "open_ksize": 3, "denoise_ksize": 3,
}


def _tune_image_metrics(img: np.ndarray) -> dict:
    pixels = img.reshape(-1).astype(np.uint8)
    return {
        "ink_lt_128": float(np.mean(pixels < 128)),
        "white_ge_240": float(np.mean(pixels >= 240)),
        "mean": float(np.mean(pixels)),
        "std": float(np.std(pixels)),
        "p10": float(np.percentile(pixels, 10)),
        "p50": float(np.percentile(pixels, 50)),
        "p90": float(np.percentile(pixels, 90)),
    }


def _tune_summary(rows: list) -> dict:
    out = {}
    for key in TUNE_METRIC_KEYS:
        values = np.array([float(r[key]) for r in rows], dtype=np.float64)
        out[f"{key}_median"] = float(np.median(values))
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    return out


def _tune_distance(stats: dict, target: dict) -> float:
    total = 0.0
    for key in TUNE_METRIC_KEYS:
        center = target[f"{key}_median"]
        spread = max(target[f"{key}_std"], 1e-6)
        total += abs(stats[f"{key}_median"] - center) / spread
    return float(total / len(TUNE_METRIC_KEYS))


def _tune_collect_prep(prep_dir: Path) -> dict:
    """Colecta métricas de páginas *_prep.png agrupadas por tomo."""
    grouped: dict[str, list] = {}
    for path in sorted(prep_dir.glob("*_prep.png")):
        match = PAGE_PREP_RE.match(path.name)
        if not match:
            continue
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv.imdecode(data, cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        row = {
            "tomo": match.group("tomo"),
            "page_num": int(match.group("page")),
        }
        row.update(_tune_image_metrics(img))
        grouped.setdefault(row["tomo"], []).append(row)
    for tomo in grouped:
        grouped[tomo].sort(key=lambda r: r["page_num"])
    return grouped


def _tune_pick_pages(rows: list, n: int) -> list:
    rows_sorted = sorted(rows, key=lambda r: r["ink_lt_128"])
    if len(rows_sorted) <= n:
        return [r["page_num"] for r in rows_sorted]
    idxs = np.linspace(0, len(rows_sorted) - 1, num=n)
    pages, seen = [], set()
    for idx in idxs:
        page = rows_sorted[int(round(float(idx)))]["page_num"]
        if page not in seen:
            pages.append(page)
            seen.add(page)
    return pages


def _tune_render_page(pdf_path: Path, page_num: int, target_dpi: int = 300) -> np.ndarray:
    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(page_num - 1)
        zoom_pdf = target_dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom_pdf, zoom_pdf), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        return cv.cvtColor(img, cv.COLOR_RGB2BGR)
    finally:
        doc.close()


def _tune_process_array(bgr: np.ndarray, params: dict) -> np.ndarray:
    """Procesa una imagen con parámetros dados, devuelve imagen binarizada."""
    gray = _to_gray(bgr)
    gray = _zoom(gray, factor=params.get("zoom", 1.15))
    bg_k = params.get("bg_ksize", 31)
    gray = _normalize_background(gray, bg_ksize=bg_k) if bg_k > 0 else gray
    img_bin = _binarize(
        gray,
        method=params.get("bin_method", "sauvola"),
        block_size=params.get("block_size", 35),
        C=params.get("C", 11),
        sauvola_w=params.get("sauvola_w", 31),
        sauvola_k=params.get("sauvola_k", 0.34),
    )
    img_bin = _fix_strokes(img_bin, close_ksize=params.get("close_ksize", 2),
                           open_ksize=params.get("open_ksize", 0))
    img_bin = _denoise_after_bin(img_bin, ksize=params.get("denoise_ksize", 3))
    return img_bin


def _tune_update_script(new_overrides: dict) -> None:
    """Reescribe PREPROCESS_PROFILE_OVERRIDES en este mismo archivo."""
    script_path = Path(__file__).resolve()
    content = script_path.read_text(encoding="utf-8")

    marker_start = "# --- Inicio PREPROCESS_PROFILE_OVERRIDES ---"
    marker_end = "# --- Fin PREPROCESS_PROFILE_OVERRIDES ---"

    idx_start = content.index(marker_start)
    idx_end = content.index(marker_end) + len(marker_end)

    # Generar nuevo bloque
    lines = [marker_start, "PREPROCESS_PROFILE_OVERRIDES = {"]
    for tomo, params in sorted(new_overrides.items()):
        lines.append(f'    "{tomo}": {{')
        for k, v in params.items():
            if isinstance(v, float):
                lines.append(f'        "{k}": {v},')
            else:
                lines.append(f'        "{k}": {v},')
        lines.append("    },")
    lines.append("}")
    lines.append(marker_end)

    new_block = "\n".join(lines)
    content = content[:idx_start] + new_block + content[idx_end:]
    script_path.write_text(content, encoding="utf-8")


def tune(
    raw_dir: Path,
    prep_dir: Path,
    out_dir: Path,
    *,
    reference_tomo: str = "Tomo I",
    sample_pages: int = 6,
    max_candidates: int = 6,
) -> dict:
    """
    Grid search de parámetros por tomo, actualiza PREPROCESS_PROFILE_OVERRIDES.

    Retorna dict con los perfiles encontrados {tomo: {param: value, ...}}.
    """
    if fitz is None:
        raise SystemExit("PyMuPDF es requerido para tune: pip install PyMuPDF")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Colectar métricas del preprocesado actual
    print(f"Recolectando métricas de {prep_dir}...")
    grouped = _tune_collect_prep(prep_dir)
    if reference_tomo not in grouped:
        raise SystemExit(f"Tomo de referencia '{reference_tomo}' no encontrado en {prep_dir}")

    target_stats = _tune_summary(grouped[reference_tomo])
    print(f"Referencia: {reference_tomo} ({len(grouped[reference_tomo])} páginas)")

    # 2. Calcular distancia baseline de cada tomo
    baseline_distances = []
    for tomo, rows in grouped.items():
        stats = _tune_summary(rows)
        baseline_distances.append({
            "tomo": tomo,
            "n_paginas": len(rows),
            "distance": _tune_distance(stats, target_stats),
        })
    baseline_distances.sort(key=lambda r: r["distance"], reverse=True)

    print("\nDistancias al perfil de referencia (baseline):")
    for row in baseline_distances:
        print(f"  {row['tomo']}\tdist={row['distance']:.3f}\tpages={row['n_paginas']}")

    # 3. Seleccionar candidatos y mapear PDFs
    candidates = [r for r in baseline_distances if r["tomo"] != reference_tomo][:max_candidates]
    pdf_map = {}
    for path in raw_dir.glob("*.pdf"):
        match = PDF_RE.match(path.name)
        if match:
            key = re.sub(r"\s+", " ", match.group("tomo").strip()).casefold()
            pdf_map[key] = path

    # 4. Grid search por tomo candidato
    new_overrides = dict(PREPROCESS_PROFILE_OVERRIDES)
    results_rows = []

    for candidate in candidates:
        tomo = candidate["tomo"]
        tomo_key = re.sub(r"\s+", " ", tomo.strip()).casefold()
        pdf_path = pdf_map.get(tomo_key)
        if pdf_path is None:
            print(f"  WARN: no se encontró PDF para {tomo}, omitiendo")
            continue

        pages = _tune_pick_pages(grouped[tomo], sample_pages)
        print(f"\nTuneando {tomo} (dist={candidate['distance']:.3f}, páginas muestra: {pages})...")

        # Renderizar páginas de muestra
        rendered = {}
        for p in pages:
            rendered[p] = _tune_render_page(pdf_path, p)

        # Probar cada perfil del grid
        profile_results = []
        for grid_entry in TUNE_PARAM_GRID:
            params = dict(_TUNE_BASE)
            params.update({k: v for k, v in grid_entry.items() if k != "name"})
            profile_name = grid_entry["name"]

            page_metrics = []
            for page, bgr in rendered.items():
                img_bin = _tune_process_array(bgr, params)
                row = {"tomo": tomo, "page_num": page}
                row.update(_tune_image_metrics(img_bin))
                page_metrics.append(row)

            stats = _tune_summary(page_metrics)
            dist = _tune_distance(stats, target_stats)
            profile_results.append({
                "tomo": tomo,
                "profile": profile_name,
                "distance": dist,
                "params": {k: v for k, v in params.items() if k not in ("target_dpi", "zoom", "bin_method", "block_size", "C")},
            })
            results_rows.append({
                "tomo": tomo, "profile": profile_name, "distance": dist,
                **{k: v for k, v in params.items()},
            })

        profile_results.sort(key=lambda r: r["distance"])
        best = profile_results[0]
        baseline_match = next(r for r in profile_results if r["profile"] == "baseline")
        improvement = 100.0 * (baseline_match["distance"] - best["distance"]) / max(baseline_match["distance"], 1e-6)

        print(f"  Mejor: {best['profile']} (dist={best['distance']:.3f}, mejora={improvement:.1f}%)")

        # Solo actualizar si hay mejora real y no es baseline
        if best["profile"] != "baseline" and improvement > 5.0:
            new_overrides[tomo] = best["params"]
            print(f"  → Perfil actualizado: {best['params']}")
        elif tomo in new_overrides:
            print(f"  → Mantiene perfil actual")
        else:
            print(f"  → Sin mejora significativa, usa baseline")

    # 5. Exportar resultados CSV
    if results_rows:
        import csv
        csv_path = out_dir / "tuning_results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"\nResultados CSV: {csv_path}")

    # 6. Actualizar el script
    print(f"\nActualizando PREPROCESS_PROFILE_OVERRIDES en {Path(__file__).name}...")
    _tune_update_script(new_overrides)
    print("Perfiles actualizados:")
    for tomo, params in sorted(new_overrides.items()):
        print(f"  {tomo}: {params}")

    return new_overrides


# -------------------- Núcleo (array) --------------------
def _process_array(
    bgr_or_gray,
    out_dir: Path,
    stem: str,
    target_dpi: int = 300,
    zoom_factor: float = 1.15,
    bin_method: str = "auto",
    block_size: int = 35,
    C: int = 11,
    sauvola_w: int = 31,
    sauvola_k: float = 0.34,
    bg_ksize: int = 31,
    close_ksize: int = 2,
    open_ksize: int = 3,
    denoise_ksize: int = 3,
    save_debug: bool = False,
    src_dpi: int | None = None,
) -> Tuple[bool, str]:
    # 1) Gris
    gray = _to_gray(bgr_or_gray)

    # 2) Reescala a DPI
    if src_dpi is not None and src_dpi != target_dpi:
        gray = _resize_to_dpi(gray, dpi_src=src_dpi, dpi_tgt=target_dpi)

    # 3) Zoom
    gray_resized = _zoom(gray, factor=zoom_factor)

    # 4) Normalización de fondo
    gray_norm = _normalize_background(gray_resized, bg_ksize=bg_ksize) if bg_ksize > 0 else gray_resized

    # 5) Binarización
    img_bin = _binarize(
        gray_norm,
        method=bin_method,
        block_size=block_size,
        C=C,
        sauvola_w=sauvola_w,
        sauvola_k=sauvola_k,
    )

    # 6) Reconectar trazos
    img_bin = _fix_strokes(img_bin, close_ksize=close_ksize, open_ksize=open_ksize)

    # 7) Denoise
    img_bin_dn = _denoise_after_bin(img_bin, ksize=denoise_ksize)

    # 8) Salida final = versión denoised
    out_png = out_dir / f"{stem}_prep.png"
    _maybe_write(out_png, img_bin_dn)

    # 9) Guardar debug
    if save_debug:
        dbg_dir = out_dir / "_debug" / stem
        _maybe_write(dbg_dir / "01_gray.png", gray)
        _maybe_write(dbg_dir / "02_gray_resized.png", gray_resized)
        if bg_ksize > 0:
            _maybe_write(dbg_dir / "03_gray_norm.png", gray_norm)
        _maybe_write(dbg_dir / "04_bin.png", img_bin)
        _maybe_write(dbg_dir / "05_bin_dn.png", img_bin_dn)
        _maybe_write(dbg_dir / "06_final.png", _trim_and_border(img_bin_dn, pad=8))
    return True, str(out_png)

# -------------------- Wrappers de archivo --------------------
def _process_one_image(
    path: Path,
    out_dir: Path,
    *,
    target_dpi: int,
    zoom: float,
    bin_method: str,
    block_size: int,
    C: int,
    sauvola_w: int,
    sauvola_k: float,
    bg_ksize: int,
    close_ksize: int,
    open_ksize: int,
    denoise_ksize: int,
    save_debug: bool,
):
    bgr = _read_image_bgr(path)
    if bgr is None:
        return False, f"Error leyendo {path}"
    src_dpi = _read_dpi(path, default_dpi=target_dpi)
    stem = path.stem
    return _process_array(
        bgr,
        out_dir,
        stem,
        target_dpi=target_dpi,
        zoom_factor=zoom,
        bin_method=bin_method,
        block_size=block_size,
        C=C,
        sauvola_w=sauvola_w,
        sauvola_k=sauvola_k,
        bg_ksize=bg_ksize,
        close_ksize=close_ksize,
        open_ksize=open_ksize,
        denoise_ksize=denoise_ksize,
        save_debug=save_debug,
        src_dpi=src_dpi,
    )

def _process_pdf(
    path: Path,
    out_dir: Path,
    *,
    target_dpi: int,
    zoom: float,
    bin_method: str,
    block_size: int,
    C: int,
    sauvola_w: int,
    sauvola_k: float,
    bg_ksize: int,
    close_ksize: int,
    open_ksize: int,
    denoise_ksize: int,
    save_debug: bool,
):
    if fitz is None:
        return False, "PyMuPDF no instalado. pip install pymupdf"
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        return False, f"No se pudo abrir PDF: {path} ({e})"

    zoom_pdf = target_dpi / 72.0
    mat = fitz.Matrix(zoom_pdf, zoom_pdf)
    ok_all, msgs = True, []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img_rgb = cv.cvtColor(img_rgb, cv.COLOR_RGBA2RGB)
        bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
        stem = f"{path.stem}_p{i:04d}"
        ok, msg = _process_array(
            bgr,
            out_dir,
            stem,
            target_dpi=target_dpi,
            zoom_factor=zoom,
            bin_method=bin_method,
            block_size=block_size,
            C=C,
            sauvola_w=sauvola_w,
            sauvola_k=sauvola_k,
            bg_ksize=bg_ksize,
            close_ksize=close_ksize,
            open_ksize=open_ksize,
            denoise_ksize=denoise_ksize,
            save_debug=save_debug,
            src_dpi=None,
        )
        ok_all &= ok
        if not ok:
            msgs.append(msg)
    doc.close()
    return ok_all, "; ".join(msgs) if msgs else f"Procesado PDF {path.name}"

# -------------------- API pública --------------------
def preprocess(
    inp: str | Path,
    out: str | Path,
    *,
    target_dpi: int = 300,
    zoom: float = 1.15,
    bg_ksize: int = 31,
    bin_method: str = "auto",          # "auto" | "otsu" | "adaptive" | "sauvola"
    block_size: int = 35,
    C: int = 11,
    sauvola_w: int = 31,
    sauvola_k: float = 0.34,
    close_ksize: int = 2,
    open_ksize: int = 3,
    denoise_ksize: int = 3,
    save_debug: bool = False,
    recursive: bool = True,
) -> Dict[str, List[str]]:
    """
    Procesa todo archivo imagen/PDF dentro de 'inp' y guarda en 'out'.

    Returns:
        dict con:
            - 'processed': lista de rutas de salida generadas (str)
            - 'errors': lista de mensajes de error (str)
    """
    in_dir = Path(inp)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        return {"processed": [], "errors": [f"Ruta de entrada no existe: {in_dir}"]}

    walker = in_dir.rglob("*") if recursive else in_dir.glob("*")
    files = sorted([p for p in walker if p.suffix.lower() in (VALID_IMG_EXT | VALID_PDF_EXT)])

    processed: List[str] = []
    errors: List[str] = []
    base_params = {
        "target_dpi": target_dpi,
        "zoom": zoom,
        "bin_method": bin_method,
        "block_size": block_size,
        "C": C,
        "sauvola_w": sauvola_w,
        "sauvola_k": sauvola_k,
        "bg_ksize": bg_ksize,
        "close_ksize": close_ksize,
        "open_ksize": open_ksize,
        "denoise_ksize": denoise_ksize,
        "save_debug": save_debug,
    }
    used_profiles: set[tuple[str, str]] = set()

    for p in files:
        try:
            tomo_name = _resolve_tomo_name(p)
            profile_name, params = _resolve_profile(tomo_name, base_params)
            profile_key = (tomo_name, profile_name)
            if profile_key not in used_profiles:
                if profile_name == "baseline":
                    print(f"perfil aplicado a {tomo_name}: baseline")
                else:
                    print(f"perfil aplicado a {tomo_name}: {PREPROCESS_PROFILE_OVERRIDES[tomo_name]}")
                used_profiles.add(profile_key)
            if p.suffix.lower() in VALID_IMG_EXT:

                print(f"procesando {p}")

                ok, msg = _process_one_image(
                    p, out_dir,
                    target_dpi=params["target_dpi"],
                    zoom=params["zoom"],
                    bin_method=params["bin_method"],
                    block_size=params["block_size"],
                    C=params["C"],
                    sauvola_w=params["sauvola_w"],
                    sauvola_k=params["sauvola_k"],
                    bg_ksize=params["bg_ksize"],
                    close_ksize=params["close_ksize"],
                    open_ksize=params["open_ksize"],
                    denoise_ksize=params["denoise_ksize"],
                    save_debug=params["save_debug"],
                )
            else:
                ok, msg = _process_pdf(
                    p, out_dir,
                    target_dpi=params["target_dpi"],
                    zoom=params["zoom"],
                    bin_method=params["bin_method"],
                    block_size=params["block_size"],
                    C=params["C"],
                    sauvola_w=params["sauvola_w"],
                    sauvola_k=params["sauvola_k"],
                    bg_ksize=params["bg_ksize"],
                    close_ksize=params["close_ksize"],
                    open_ksize=params["open_ksize"],
                    denoise_ksize=params["denoise_ksize"],
                    save_debug=params["save_debug"],
                )

            if ok:
                # Cuando procesamos PDFs, el mensaje no es una ruta única.
                # Para imágenes, msg sí es la ruta del *_prep.png
                if isinstance(msg, str) and msg.lower().endswith("_prep.png"):
                    processed.append(msg)
            else:
                errors.append(msg if isinstance(msg, str) else str(msg))
        except Exception as e:
            errors.append(f"{p}: {e}")

    return {"processed": processed, "errors": errors}


# -------------------- CLI --------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocesa imágenes/PDFs para OCR"
    )
    parser.add_argument("--in", dest="inp",
                        help="Directorio de entrada con imágenes/PDFs")
    parser.add_argument("--out",
                        help="Directorio de salida")
    parser.add_argument("--target-dpi", type=int, default=300)
    parser.add_argument("--zoom", type=float, default=1.15)
    parser.add_argument("--bg-ksize", type=int, default=31,
                        help="Kernel para normalización de fondo (0=desactivar)")
    parser.add_argument("--bin", dest="bin_method", default="sauvola",
                        choices=["otsu", "adaptive", "sauvola", "auto"])
    parser.add_argument("--block-size", type=int, default=35)
    parser.add_argument("--C", type=int, default=11)
    parser.add_argument("--sauvola-w", type=int, default=31)
    parser.add_argument("--sauvola-k", type=float, default=0.34)
    parser.add_argument("--close", dest="close_ksize", type=int, default=2)
    parser.add_argument("--open", dest="open_ksize", type=int, default=3)
    parser.add_argument("--denoise-ksize", type=int, default=3)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--no-recursive", action="store_true",
                        help="No buscar recursivamente en subdirectorios")

    # Tuning
    parser.add_argument("--tune", action="store_true",
                        help="Grid search de parámetros por tomo. "
                             "Requiere --in (PDFs), --out (preprocessed). "
                             "Actualiza perfiles en el script y luego preprocesa.")
    parser.add_argument("--reference-tomo", default="Tomo I",
                        help="Tomo de referencia para tuning (default: Tomo I)")
    parser.add_argument("--sample-pages", type=int, default=6,
                        help="Páginas de muestra por tomo candidato (default: 6)")
    parser.add_argument("--max-candidates", type=int, default=6,
                        help="Máximo de tomos candidatos a tunear (default: 6)")
    parser.add_argument("--tune-only", action="store_true",
                        help="Solo tunear, no ejecutar preprocesamiento después")

    args = parser.parse_args()

    if args.tune:
        if not args.inp or not args.out:
            parser.error("--tune requiere --in (dir de PDFs crudos) y --out (dir de preprocesados)")
        new_profiles = tune(
            raw_dir=Path(args.inp),
            prep_dir=Path(args.out),
            out_dir=Path(args.out) / "_tuning",
            reference_tomo=args.reference_tomo,
            sample_pages=args.sample_pages,
            max_candidates=args.max_candidates,
        )
        if args.tune_only:
            print("\n--tune-only: preprocesamiento omitido.")
            return

        # Recargar perfiles actualizados en memoria
        PREPROCESS_PROFILE_OVERRIDES.clear()
        PREPROCESS_PROFILE_OVERRIDES.update(new_profiles)

        print(f"\nEjecutando preprocesamiento con perfiles actualizados...")
        result = preprocess(
            inp=args.inp,
            out=args.out,
            target_dpi=args.target_dpi,
            zoom=args.zoom,
            bg_ksize=args.bg_ksize,
            bin_method=args.bin_method,
            block_size=args.block_size,
            C=args.C,
            sauvola_w=args.sauvola_w,
            sauvola_k=args.sauvola_k,
            close_ksize=args.close_ksize,
            open_ksize=args.open_ksize,
            denoise_ksize=args.denoise_ksize,
            save_debug=args.save_debug,
            recursive=not args.no_recursive,
        )
    else:
        if not args.inp or not args.out:
            parser.error("Se requiere --in y --out")
        result = preprocess(
            inp=args.inp,
            out=args.out,
            target_dpi=args.target_dpi,
            zoom=args.zoom,
            bg_ksize=args.bg_ksize,
            bin_method=args.bin_method,
            block_size=args.block_size,
            C=args.C,
            sauvola_w=args.sauvola_w,
            sauvola_k=args.sauvola_k,
            close_ksize=args.close_ksize,
            open_ksize=args.open_ksize,
            denoise_ksize=args.denoise_ksize,
            save_debug=args.save_debug,
            recursive=not args.no_recursive,
        )

    print(f"\nResultado: {len(result['processed'])} procesados, {len(result['errors'])} errores")
    for e in result["errors"]:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()

