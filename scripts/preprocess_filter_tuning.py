"""
Tuning de filtros de preprocesamiento por tomo.

Flujo completo en 3 subcomandos:

  1. analizar  — Analiza distribuciones de intensidades en *_prep.png existentes
                  para detectar tomos con preprocesamiento deficiente.
  2. tunear    — Renderiza páginas de muestra desde PDFs, prueba una grilla de
                  parámetros y compara contra un tomo de referencia (Tomo I).
  3. aplicar   — Aplica los mejores perfiles encontrados, regenerando *_prep.png.

Los resultados del tuning se usan para actualizar PREPROCESS_PROFILE_OVERRIDES
en src/preprocess.py (perfiles hardcodeados por tomo).

Uso:
  py scripts/preprocess_filter_tuning.py analizar --images-dir data/preprocess_v2 --out outputs/tuning
  py scripts/preprocess_filter_tuning.py tunear --raw-dir data/raw --prep-dir data/preprocess_v2 --out outputs/tuning
  py scripts/preprocess_filter_tuning.py aplicar --raw-dir data/raw --out-dir data/preprocess_v2 --tuning-results outputs/tuning/tuning_results.csv

Requisitos:
  pip install opencv-python numpy PyMuPDF
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np

# Importar funciones internas de preprocess.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import _binarize, _denoise_after_bin, _fix_strokes, _normalize_background, _to_gray, _zoom

# ---------------------------------------------------------------------------
# Constantes compartidas
# ---------------------------------------------------------------------------

PAGE_RE = re.compile(r"^(?P<tomo>.+?)_p(?P<page>\d+)_prep\.png$")
PDF_RE = re.compile(r"^(?P<tomo>.+)\.pdf$", re.IGNORECASE)

METRIC_KEYS = ["ink_lt_128", "white_ge_240", "mean", "std", "p10", "p50", "p90"]

BIN_SPECS = [
    ("negro_0_31", 0, 31),
    ("oscuro_32_63", 32, 63),
    ("gris_64_127", 64, 127),
    ("gris_claro_128_191", 128, 191),
    ("casi_blanco_192_223", 192, 223),
    ("muy_claro_224_239", 224, 239),
    ("blanco_sucio_240_247", 240, 247),
    ("blanco_248_255", 248, 255),
]

QUANTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


# ---------------------------------------------------------------------------
# Dataclass de parámetros
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    name: str = "baseline"
    target_dpi: int = 300
    zoom: float = 1.15
    bg_ksize: int = 31
    bin_method: str = "sauvola"
    block_size: int = 35
    C: int = 11
    sauvola_w: int = 31
    sauvola_k: float = 0.45
    close_ksize: int = 3
    open_ksize: int = 3
    denoise_ksize: int = 3


# ---------------------------------------------------------------------------
# Utilidades compartidas
# ---------------------------------------------------------------------------

def _norm_tomo(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _read_gray(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv.imdecode(data, cv.IMREAD_GRAYSCALE)


def _image_metrics(img: np.ndarray) -> dict[str, float]:
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


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_by_tomo(rows: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["tomo"], []).append(row)
    for tomo in grouped:
        grouped[tomo].sort(key=lambda r: r["page_num"])
    return grouped


def _summary(rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        values = np.array([float(r[key]) for r in rows], dtype=np.float64)
        out[f"{key}_median"] = float(np.median(values))
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    return out


def _distance_to_target(stats: dict[str, float], target_stats: dict[str, float]) -> float:
    total = 0.0
    for key in METRIC_KEYS:
        center = target_stats[f"{key}_median"]
        spread = max(target_stats[f"{key}_std"], 1e-6)
        total += abs(stats[f"{key}_median"] - center) / spread
    return float(total / len(METRIC_KEYS))


def _map_raw_pdfs(raw_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in raw_dir.glob("*.pdf"):
        match = PDF_RE.match(path.name)
        if not match:
            continue
        mapping[_norm_tomo(match.group("tomo"))] = path
    return mapping


def _process_page_array(bgr: np.ndarray, params: Params) -> np.ndarray:
    gray = _to_gray(bgr)
    gray_resized = _zoom(gray, factor=params.zoom)
    gray_norm = _normalize_background(gray_resized, bg_ksize=params.bg_ksize) if params.bg_ksize > 0 else gray_resized
    img_bin = _binarize(
        gray_norm,
        method=params.bin_method,
        block_size=params.block_size,
        C=params.C,
        sauvola_w=params.sauvola_w,
        sauvola_k=params.sauvola_k,
    )
    img_bin = _fix_strokes(img_bin, close_ksize=params.close_ksize, open_ksize=params.open_ksize)
    img_bin = _denoise_after_bin(img_bin, ksize=params.denoise_ksize)
    return img_bin


def _require_fitz():
    try:
        import fitz
        return fitz
    except Exception as exc:
        raise SystemExit("PyMuPDF es requerido: pip install PyMuPDF") from exc


def _render_pdf_page(fitz, pdf_path: Path, page_num: int, target_dpi: int) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Subcomando: analizar
# ---------------------------------------------------------------------------

def _page_stats(path: Path) -> dict | None:
    match = PAGE_RE.match(path.name)
    if not match:
        return None
    img = _read_gray(path)
    if img is None:
        return None

    pixels = img.reshape(-1)
    total = int(pixels.size)
    stats = {
        "tomo": match.group("tomo"),
        "page_num": int(match.group("page")),
        "archivo": path.name,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "pixel_count": total,
        "mean": float(np.mean(pixels)),
        "std": float(np.std(pixels)),
        "min": int(np.min(pixels)),
        "max": int(np.max(pixels)),
        "dark_lt_64": float(np.mean(pixels < 64)),
        "ink_lt_128": float(np.mean(pixels < 128)),
        "white_ge_240": float(np.mean(pixels >= 240)),
        "paper_ge_248": float(np.mean(pixels >= 248)),
    }
    for q in QUANTILES:
        stats[f"p{q:02d}"] = float(np.percentile(pixels, q))
    for name, lo, hi in BIN_SPECS:
        stats[name] = float(np.mean((pixels >= lo) & (pixels <= hi)))
    return stats


def _summarize_by_tomo(page_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_tomo: dict[str, list[dict]] = defaultdict(list)
    for row in page_rows:
        by_tomo[row["tomo"]].append(row)

    tomo_rows: list[dict] = []
    hist_rows: list[dict] = []
    metric_keys = [
        "mean", "std", "dark_lt_64", "ink_lt_128", "white_ge_240", "paper_ge_248",
        *[f"p{q:02d}" for q in QUANTILES],
        *[name for name, _, _ in BIN_SPECS],
    ]

    for tomo, rows in sorted(by_tomo.items()):
        tomo_row = {"tomo": tomo, "n_paginas": len(rows)}
        for key in metric_keys:
            values = np.array([float(r[key]) for r in rows], dtype=np.float64)
            tomo_row[f"{key}_mean"] = float(np.mean(values))
            tomo_row[f"{key}_median"] = float(np.median(values))
            tomo_row[f"{key}_std"] = float(np.std(values))
            tomo_row[f"{key}_min"] = float(np.min(values))
            tomo_row[f"{key}_max"] = float(np.max(values))
        tomo_rows.append(tomo_row)

        for name, _, _ in BIN_SPECS:
            vals = np.array([float(r[name]) for r in rows], dtype=np.float64)
            hist_rows.append({
                "tomo": tomo,
                "bin": name,
                "page_frac_mean": float(np.mean(vals)),
                "page_frac_median": float(np.median(vals)),
                "page_frac_std": float(np.std(vals)),
            })

    return tomo_rows, hist_rows


def cmd_analizar(args: argparse.Namespace) -> None:
    """Analiza distribuciones de píxeles por tomo."""
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)

    page_rows: list[dict] = []
    for path in sorted(images_dir.glob("*_prep.png")):
        row = _page_stats(path)
        if row is not None:
            page_rows.append(row)

    if not page_rows:
        raise SystemExit(f"No se encontraron *_prep.png en {images_dir}")

    page_rows.sort(key=lambda r: (r["tomo"], r["page_num"]))
    tomo_rows, hist_rows = _summarize_by_tomo(page_rows)

    _write_csv(out_dir / "page_pixel_stats.csv", page_rows)
    _write_csv(out_dir / "tomo_pixel_summary.csv", tomo_rows)
    _write_csv(out_dir / "tomo_histogram_long.csv", hist_rows)

    tomo_rows_sorted = sorted(tomo_rows, key=lambda r: (r["ink_lt_128_median"], -r["white_ge_240_median"]))

    print(f"Páginas analizadas: {len(page_rows):,}")
    print("")
    print("Tomos ordenados por menor tinta (<128) mediana:")
    print("tomo\tn_pag\tink_med\tdark_med\twhite240_med\tp50_med\tp95_med")
    for row in tomo_rows_sorted:
        fmt = lambda v: f"{v:.4f}"
        print("\t".join([
            row["tomo"], str(row["n_paginas"]),
            fmt(row["ink_lt_128_median"]), fmt(row["dark_lt_64_median"]),
            fmt(row["white_ge_240_median"]), fmt(row["p50_median"]),
            fmt(row["p95_median"]),
        ]))
    print(f"\nCSVs guardados en {out_dir}")


# ---------------------------------------------------------------------------
# Subcomando: tunear
# ---------------------------------------------------------------------------

def _build_param_grid() -> list[Params]:
    return [
        Params(name="baseline"),
        Params(name="soft_bg15_k030", bg_ksize=15, sauvola_w=31, sauvola_k=0.30, close_ksize=2, open_ksize=1, denoise_ksize=1),
        Params(name="soft_bg15_k024", bg_ksize=15, sauvola_w=41, sauvola_k=0.24, close_ksize=1, open_ksize=1, denoise_ksize=1),
        Params(name="soft_bg11_k018", bg_ksize=11, sauvola_w=51, sauvola_k=0.18, close_ksize=1, open_ksize=0, denoise_ksize=1),
        Params(name="soft_bg21_k026", bg_ksize=21, sauvola_w=41, sauvola_k=0.26, close_ksize=1, open_ksize=1, denoise_ksize=1),
        Params(name="mid_bg21_k030", bg_ksize=21, sauvola_w=31, sauvola_k=0.30, close_ksize=2, open_ksize=1, denoise_ksize=1),
        Params(name="mid_bg31_k034", bg_ksize=31, sauvola_w=41, sauvola_k=0.34, close_ksize=2, open_ksize=1, denoise_ksize=3),
        Params(name="sharp_bg11_k024", bg_ksize=11, sauvola_w=31, sauvola_k=0.24, close_ksize=1, open_ksize=0, denoise_ksize=1),
        Params(name="sharp_bg15_k018", bg_ksize=15, sauvola_w=51, sauvola_k=0.18, close_ksize=1, open_ksize=0, denoise_ksize=1),
    ]


def _collect_prep_pages(prep_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(prep_dir.glob("*_prep.png")):
        match = PAGE_RE.match(path.name)
        if not match:
            continue
        img = _read_gray(path)
        if img is None:
            continue
        row = {
            "tomo": match.group("tomo"),
            "page_num": int(match.group("page")),
            "archivo": path.name,
        }
        row.update(_image_metrics(img))
        rows.append(row)
    return rows


def _pick_sample_pages(rows: list[dict], per_tomo: int) -> list[int]:
    rows_sorted = sorted(rows, key=lambda r: r["ink_lt_128"])
    if len(rows_sorted) <= per_tomo:
        return [r["page_num"] for r in rows_sorted]
    idxs = np.linspace(0, len(rows_sorted) - 1, num=per_tomo)
    pages: list[int] = []
    seen: set[int] = set()
    for idx in idxs:
        page = rows_sorted[int(round(float(idx)))]["page_num"]
        if page not in seen:
            pages.append(page)
            seen.add(page)
    return pages


def cmd_tunear(args: argparse.Namespace) -> None:
    """Busca hiperparámetros óptimos comparando contra tomo de referencia."""
    fitz = _require_fitz()

    raw_dir = Path(args.raw_dir)
    prep_dir = Path(args.prep_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep_rows = _collect_prep_pages(prep_dir)
    grouped = _group_by_tomo(prep_rows)
    if args.reference_tomo not in grouped:
        raise SystemExit(f"No existe tomo de referencia {args.reference_tomo!r} en {prep_dir}")

    target_stats = _summary(grouped[args.reference_tomo])

    baseline_rows: list[dict] = []
    for tomo, rows in grouped.items():
        stats = _summary(rows)
        baseline_rows.append({
            "tomo": tomo,
            "n_paginas": len(rows),
            "baseline_distance": _distance_to_target(stats, target_stats),
            **stats,
        })
    baseline_rows.sort(key=lambda r: r["baseline_distance"], reverse=True)
    _write_csv(out_dir / "baseline_distances.csv", baseline_rows)

    candidate_rows = [r for r in baseline_rows if r["tomo"] != args.reference_tomo][:args.max_candidate_tomos]
    pdf_map = _map_raw_pdfs(raw_dir)
    grid = _build_param_grid()

    tuning_rows: list[dict] = []
    improvement_rows: list[dict] = []

    for candidate in candidate_rows:
        tomo = candidate["tomo"]
        pdf_path = pdf_map.get(_norm_tomo(tomo))
        if pdf_path is None:
            continue

        selected_pages = _pick_sample_pages(grouped[tomo], args.sample_pages)
        rendered = {page: _render_pdf_page(fitz, pdf_path, page, target_dpi=grid[0].target_dpi) for page in selected_pages}
        tomo_profile_rows: list[dict] = []

        for params in grid:
            page_metrics: list[dict] = []
            for page, bgr in rendered.items():
                img_bin = _process_page_array(bgr, params)
                row = {"tomo": tomo, "page_num": page, "profile": params.name}
                row.update(_image_metrics(img_bin))
                page_metrics.append(row)

            stats = _summary(page_metrics)
            tomo_profile_rows.append({
                "tomo": tomo,
                "profile": params.name,
                "distance_to_reference": _distance_to_target(stats, target_stats),
                **asdict(params),
                **stats,
            })

        tomo_profile_rows.sort(key=lambda r: r["distance_to_reference"])
        tuning_rows.extend(tomo_profile_rows)
        best = tomo_profile_rows[0]
        baseline_match = next(r for r in tomo_profile_rows if r["profile"] == "baseline")
        improvement_rows.append({
            "tomo": tomo,
            "baseline_distance_sample": baseline_match["distance_to_reference"],
            "best_profile": best["profile"],
            "best_distance_sample": best["distance_to_reference"],
            "improvement_pct": 100.0
            * (baseline_match["distance_to_reference"] - best["distance_to_reference"])
            / max(baseline_match["distance_to_reference"], 1e-6),
            "selected_pages": "; ".join(str(p) for p in selected_pages),
        })

    improvement_rows.sort(key=lambda r: r["improvement_pct"], reverse=True)
    _write_csv(out_dir / "tuning_results.csv", tuning_rows)
    _write_csv(out_dir / "tuning_improvements.csv", improvement_rows)

    print(f"Referencia visual: {args.reference_tomo}")
    print("\nTomos más lejos del perfil de referencia (baseline actual):")
    for row in baseline_rows[:10]:
        print(f"  {row['tomo']}\tdistance={row['baseline_distance']:.3f}\tpages={row['n_paginas']}")

    if improvement_rows:
        print("\nMejores mejoras en muestra renderizada:")
        for row in improvement_rows[:10]:
            print(
                f"  {row['tomo']}\tbest={row['best_profile']}\t"
                f"baseline={row['baseline_distance_sample']:.3f}\t"
                f"best_dist={row['best_distance_sample']:.3f}\t"
                f"improve={row['improvement_pct']:.1f}%"
            )

    print(f"\nResultados guardados en {out_dir}")


# ---------------------------------------------------------------------------
# Subcomando: aplicar
# ---------------------------------------------------------------------------

def _load_best_profiles(tuning_results_path: Path) -> dict[str, tuple[str, Params]]:
    rows = list(csv.DictReader(tuning_results_path.open("r", encoding="utf-8", newline="")))
    best: dict[str, tuple[str, Params, float]] = {}
    for row in rows:
        tomo = row["tomo"]
        dist = float(row["distance_to_reference"])
        params = Params(
            target_dpi=int(float(row["target_dpi"])),
            zoom=float(row["zoom"]),
            bg_ksize=int(float(row["bg_ksize"])),
            bin_method=row["bin_method"],
            block_size=int(float(row["block_size"])),
            C=int(float(row["C"])),
            sauvola_w=int(float(row["sauvola_w"])),
            sauvola_k=float(row["sauvola_k"]),
            close_ksize=int(float(row["close_ksize"])),
            open_ksize=int(float(row["open_ksize"])),
            denoise_ksize=int(float(row["denoise_ksize"])),
        )
        current = best.get(tomo)
        if current is None or dist < current[2]:
            best[tomo] = (row["profile"], params, dist)
    return {tomo: (profile, params) for tomo, (profile, params, _) in best.items()}


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    cv.imencode(".png", image)[1].tofile(str(tmp))
    tmp.replace(path)


def cmd_aplicar(args: argparse.Namespace) -> None:
    """Aplica los mejores perfiles encontrados, regenerando *_prep.png."""
    fitz = _require_fitz()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    tuning_results_path = Path(args.tuning_results)

    best_profiles = _load_best_profiles(tuning_results_path)
    pdf_map = _map_raw_pdfs(raw_dir)

    total_pages = 0
    for tomo_name, (profile_name, params) in sorted(best_profiles.items()):
        pdf_path = pdf_map.get(_norm_tomo(tomo_name))
        if pdf_path is None:
            print(f"WARN: no encontré PDF para {tomo_name}, se omite")
            continue
        print(f"Aplicando {profile_name} a {tomo_name} desde {pdf_path.name}")

        doc = fitz.open(str(pdf_path))
        try:
            zoom_pdf = params.target_dpi / 72.0
            mat = fitz.Matrix(zoom_pdf, zoom_pdf)
            count = 0
            for idx, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
                bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                final = _process_page_array(bgr, params)
                out_path = out_dir / f"{tomo_name}_p{idx:04d}_prep.png"
                _write_png(out_path, final)
                count += 1
                if idx % 25 == 0:
                    print(f"  {tomo_name}: {idx} páginas...")
            total_pages += count
            print(f"  {tomo_name}: {count} páginas reescritas")
        finally:
            doc.close()

    print(f"\nTotal de páginas reescritas: {total_pages}")
    print(f"Salida actualizada en {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tuning de filtros de preprocesamiento por tomo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analizar
    p_analizar = subparsers.add_parser("analizar", help="Analiza distribuciones de píxeles por tomo")
    p_analizar.add_argument("--images-dir", default="data/preprocess_v2")
    p_analizar.add_argument("--out", required=True)

    # tunear
    p_tunear = subparsers.add_parser("tunear", help="Busca hiperparámetros óptimos por tomo")
    p_tunear.add_argument("--raw-dir", default="data/raw")
    p_tunear.add_argument("--prep-dir", default="data/preprocess_v2")
    p_tunear.add_argument("--out", required=True)
    p_tunear.add_argument("--reference-tomo", default="Tomo I")
    p_tunear.add_argument("--sample-pages", type=int, default=6)
    p_tunear.add_argument("--max-candidate-tomos", type=int, default=6)

    # aplicar
    p_aplicar = subparsers.add_parser("aplicar", help="Aplica mejores perfiles a *_prep.png")
    p_aplicar.add_argument("--raw-dir", default="data/raw")
    p_aplicar.add_argument("--out-dir", default="data/preprocess_v2")
    p_aplicar.add_argument("--tuning-results", required=True)

    args = parser.parse_args()

    if args.command == "analizar":
        cmd_analizar(args)
    elif args.command == "tunear":
        cmd_tunear(args)
    elif args.command == "aplicar":
        cmd_aplicar(args)


if __name__ == "__main__":
    main()
