"""
Pipeline OCR completo.

Por defecto:
  imagenes preprocesadas -> YOLO -> recortes -> OCR por recorte -> postproceso

Produce dentro de outputs/run_*/:
  - boxes_manifest.csv          manifest por box detectado
  - ocr_boxes.csv              OCR tabular por box
  - pages/*/result.txt         texto recompuesto por pagina desde boxes
  - calidad_ocr.csv            flaggeo de paginas problematicas
  - tomos_txt/                 texto consolidado por tomo
  - contratos_segmentados.xlsx segmentacion cruda por contratos
  - compilado.xlsx             campos parseados + entidades + flags OCR

Uso:
  python src/pipeline.py
  python src/pipeline.py --images-dir data/preprocess_v2
  python src/pipeline.py --pages-dir outputs/pages
  python src/pipeline.py --skip-entidades
"""
from __future__ import annotations

import os
import sys
import re
import json
import glob
import logging
import unicodedata
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests

from panelizar import panelizar, separar_headers, exportar_panel_multihoja
from red_personas import construir_red, exportar as exportar_red_personas

# Importar funciones de parseo desde modulo hermano
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parseo_compilado import parsear_texto, contar_subregistros
from ocr_model_deepseek import (
    _PROMPTS_RETRY as OCR_PROMPTS_RETRY,
    _infer_una_pagina as _ocr_infer_una_pagina,
    _load_tokenizer_and_model as _ocr_load_tokenizer_and_model,
    _select_device as _ocr_select_device,
    _select_dtype as _ocr_select_dtype,
    _validar_output as _ocr_validar_output,
)

# =====================================================================
# Logging
# =====================================================================

logger = logging.getLogger("pipeline")
logger.setLevel(logging.DEBUG)

# Handler consola (INFO+)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
))
logger.addHandler(_console_handler)


def _add_file_handler(log_path: Path) -> None:
    """Agrega handler de archivo al logger (DEBUG+, con timestamps completos)."""
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)


# =====================================================================
# Utilidades
# =====================================================================

TOMO_PAGE_RE = re.compile(r"^(?P<tomo>.+?)_p(?P<page>\d+)$")
IMAGE_PAGE_RE = re.compile(r"^(?P<tomo>.+?)_p(?P<page>\d+)_prep\.\w+$", re.I)


def _normalizar_tomo(nombre: str) -> str:
    """Normaliza casing de tomo: 'tomo XV' -> 'Tomo XV'."""
    if nombre and nombre[0].islower():
        return nombre[0].upper() + nombre[1:]
    return nombre


def _parse_tomo_page(path: Path) -> dict:
    name = path.parent.name
    m = TOMO_PAGE_RE.match(name)
    if not m:
        return {"tomo": None, "page_num": None}
    tomo = _normalizar_tomo(m.group("tomo").strip())
    return {"tomo": tomo, "page_num": int(m.group("page"))}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())


def _normalize_tomo_selector(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value).strip()
    m = re.match(r"^(?i:tomo)\s+(.*)$", value)
    if m:
        suffix = m.group(1).strip()
    else:
        suffix = value
    return f"Tomo {suffix.upper()}".strip()


def _parse_tomos_arg(raw: str | None) -> set[str]:
    if not raw:
        return set()
    tomos = {
        _normalize_tomo_selector(part)
        for part in raw.split(",")
        if _normalize_tomo_selector(part)
    }
    return tomos


def _levenshtein(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    prev = list(range(m + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * m
        for j, cb in enumerate(b, start=1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[m]


def _parse_image_page(path: Path) -> dict:
    m = IMAGE_PAGE_RE.match(path.name)
    if not m:
        return {"tomo": None, "page_num": None, "page_label": None}
    tomo = _normalizar_tomo(m.group("tomo").strip())
    page_num = int(m.group("page"))
    return {
        "tomo": tomo,
        "page_num": page_num,
        "page_label": f"{tomo}_p{page_num:04d}",
    }


def _listar_imagenes_preprocesadas(images_dir: Path, tomos_filter: set[str] | None = None) -> list[Path]:
    images = [
        p for p in glob.glob(str(images_dir / "*_prep.*"))
        if Path(p).is_file() and IMAGE_PAGE_RE.match(Path(p).name)
    ]
    image_paths = [Path(p) for p in images]
    if tomos_filter:
        image_paths = [p for p in image_paths if _parse_image_page(p)["tomo"] in tomos_filter]
    image_paths.sort(key=lambda p: (
        _parse_image_page(p)["tomo"] or "",
        _parse_image_page(p)["page_num"] or 10**9,
        p.name.lower(),
    ))
    return image_paths


def _clamp_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def _rows_from_yolo_result(result, image_path: Path) -> list[dict]:
    meta = _parse_image_page(image_path)
    names = result.names if isinstance(result.names, dict) else {}
    rows: list[dict] = []

    try:
        height, width = result.orig_shape[:2]
    except Exception:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"No se pudo leer imagen {image_path}")
        height, width = img.shape[:2]

    def _class_name(class_id: int) -> str:
        return str(names.get(class_id, class_id))

    if getattr(result, "obb", None) is not None and len(result.obb) > 0:
        polys = result.obb.xyxyxyxy.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()
        classes = result.obb.cls.cpu().numpy().astype(int)
        for poly, conf, cls_id in zip(polys, confs, classes):
            xs = poly[:, 0]
            ys = poly[:, 1]
            x1, y1, x2, y2 = _clamp_box(
                int(np.floor(xs.min())),
                int(np.floor(ys.min())),
                int(np.ceil(xs.max())),
                int(np.ceil(ys.max())),
                width,
                height,
            )
            rows.append({
                **meta,
                "image_path": str(image_path),
                "class_id": int(cls_id),
                "class_name": _class_name(int(cls_id)),
                "conf": float(conf),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "image_width": width,
                "image_height": height,
            })
    elif getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for coords, conf, cls_id in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = _clamp_box(
                int(np.floor(coords[0])),
                int(np.floor(coords[1])),
                int(np.ceil(coords[2])),
                int(np.ceil(coords[3])),
                width,
                height,
            )
            rows.append({
                **meta,
                "image_path": str(image_path),
                "class_id": int(cls_id),
                "class_name": _class_name(int(cls_id)),
                "conf": float(conf),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "image_width": width,
                "image_height": height,
            })

    if not rows:
        rows.append({
            **meta,
            "image_path": str(image_path),
            "class_id": -1,
            "class_name": "page_fallback",
            "conf": 1.0,
            "x1": 0,
            "y1": 0,
            "x2": width,
            "y2": height,
            "image_width": width,
            "image_height": height,
        })

    rows.sort(key=lambda r: (r["y1"], r["x1"], -(r["y2"] - r["y1"]), -(r["x2"] - r["x1"])))
    for idx, row in enumerate(rows, start=1):
        row["sort_idx"] = idx
        row["box_id"] = f"{idx:04d}"
    return rows


def detectar_boxes_yolo(
    images_dir: Path,
    yolo_model_path: Path,
    tomos_filter: set[str] | None = None,
    conf: float = 0.25,
    imgsz: int = 1536,
    batch: int = 8,
) -> pd.DataFrame:
    from ultralytics import YOLO

    image_paths = _listar_imagenes_preprocesadas(images_dir, tomos_filter=tomos_filter)
    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imagenes *_prep.* en {images_dir}")
    if not yolo_model_path.exists():
        raise FileNotFoundError(f"No existe modelo YOLO: {yolo_model_path}")

    model = YOLO(str(yolo_model_path))
    rows: list[dict] = []
    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=conf,
        imgsz=imgsz,
        batch=batch,
        stream=True,
        verbose=False,
    )

    total = len(image_paths)
    for idx, (image_path, result) in enumerate(zip(image_paths, results), start=1):
        rows.extend(_rows_from_yolo_result(result, image_path))
        if idx % 100 == 0 or idx == total:
            logger.info(f"YOLO: {idx:,}/{total:,} paginas")

    return pd.DataFrame(rows)


def _crop_rows_for_image(image_path: str, rows: list[dict], crops_dir: Path) -> list[dict]:
    import cv2

    img = cv2.imread(image_path)
    result_rows: list[dict] = []
    if img is None:
        for row in rows:
            result_rows.append({**row, "crop_path": "", "crop_width": 0, "crop_height": 0, "crop_area": 0, "status": "image_read_error"})
        return result_rows

    for row in rows:
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            result_rows.append({**row, "crop_path": "", "crop_width": 0, "crop_height": 0, "crop_area": 0, "status": "empty_crop"})
            continue
        page_dir = crops_dir / row["page_label"]
        page_dir.mkdir(parents=True, exist_ok=True)
        crop_path = page_dir / f"{row['box_id']}.png"
        ok = cv2.imwrite(str(crop_path), crop)
        h, w = crop.shape[:2]
        result_rows.append({
            **row,
            "crop_path": str(crop_path),
            "crop_width": int(w),
            "crop_height": int(h),
            "crop_area": int(w * h),
            "status": "cropped" if ok else "crop_write_error",
        })
    return result_rows


def guardar_crops_y_manifest(df_boxes: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in df_boxes.to_dict("records"):
        grouped[row["image_path"]].append(row)

    rows_out: list[dict] = []
    max_workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_crop_rows_for_image, image_path, rows, crops_dir)
            for image_path, rows in grouped.items()
        ]
        for fut in as_completed(futures):
            rows_out.extend(fut.result())

    df_manifest = pd.DataFrame(rows_out)
    df_manifest = df_manifest.sort_values(["tomo", "page_num", "sort_idx"]).reset_index(drop=True)
    manifest_path = run_dir / "boxes_manifest.csv"
    df_manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    logger.info(f"{len(df_manifest):,} boxes en manifest: {manifest_path}")
    return df_manifest


def _ocr_texto_crop(model, tokenizer, crop_path: str, prompt: str, tmp_dir: Path, max_retries: int) -> tuple[str, str, str]:
    prompts = [prompt] + OCR_PROMPTS_RETRY[:max(max_retries - 1, 0)]
    best_text = ""
    best_flags: list[str] = ["vacio"]

    for try_prompt in prompts:
        try:
            res_text = _ocr_infer_una_pagina(
                model,
                tokenizer,
                prompt=try_prompt,
                img_path=crop_path,
                out_dir=str(tmp_dir),
                save_per_page=False,
                capture_stdout_fallback=True,
            )
            is_valid, flags = _ocr_validar_output(res_text or "")
            if not best_text:
                best_text = res_text or ""
                best_flags = flags
            if is_valid:
                return (res_text or "").strip(), "OK", "; ".join(flags)
            best_text = res_text or best_text
            best_flags = flags
        except Exception as e:
            best_flags = [f"error: {e}"]
    return best_text.strip(), "DEGRADADO", "; ".join(best_flags)


def ocr_boxes_a_paginas(
    df_manifest: pd.DataFrame,
    run_dir: Path,
    model_name: str = "deepseek-ai/DeepSeek-OCR",
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    device: str | None = None,
    dtype: str = "bfloat16",
    max_new_tokens: int = 4096,
    max_retries: int = 3,
) -> Path:
    import torch

    pages_dir = run_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    tmp_dir = run_dir / "_tmp_ocr"
    tmp_dir.mkdir(exist_ok=True)

    df_ok = df_manifest[df_manifest["status"] == "cropped"].copy()
    if df_ok.empty:
        raise RuntimeError("No hay crops validos para OCR")

    # Mejor throughput: crops pequeños primero; mantenemos un solo modelo GPU cargado.
    df_ok = df_ok.sort_values(["crop_area", "crop_height", "crop_width", "page_label", "sort_idx"])

    ocr_device = _ocr_select_device(device)
    torch_dtype = _ocr_select_dtype(dtype, ocr_device)
    tokenizer, model = _ocr_load_tokenizer_and_model(
        model_name,
        attn_impl=None,
        device=ocr_device,
        dtype=torch_dtype,
    )
    if max_new_tokens and hasattr(model, "generation_config"):
        model.generation_config.max_new_tokens = max_new_tokens

    rows_out: list[dict] = []
    page_chunks: dict[str, list[tuple[int, str]]] = defaultdict(list)
    total = len(df_ok)
    t0 = time.time()

    for count, row in enumerate(df_ok.to_dict("records"), start=1):
        text, ocr_status, ocr_flags = _ocr_texto_crop(
            model,
            tokenizer,
            row["crop_path"],
            prompt,
            tmp_dir,
            max_retries,
        )
        rows_out.append({
            **row,
            "ocr_text": text,
            "ocr_chars": len(text),
            "ocr_status": ocr_status,
            "ocr_flags": ocr_flags,
        })
        if text:
            page_chunks[row["page_label"]].append((int(row["sort_idx"]), text))

        if count % 100 == 0 or count == total:
            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 0
            logger.info(f"OCR boxes: {count:,}/{total:,} | {rate:.1f} crops/s")

        if ocr_device == "cuda":
            torch.cuda.empty_cache()

    df_ocr = pd.DataFrame(rows_out)
    ocr_path = run_dir / "ocr_boxes.csv"
    df_ocr.to_csv(ocr_path, index=False, encoding="utf-8-sig")

    for page_label in df_manifest["page_label"].dropna().unique():
        page_dir = pages_dir / page_label
        page_dir.mkdir(exist_ok=True)
        chunks = sorted(page_chunks.get(page_label, []), key=lambda x: x[0])
        page_text = "\n".join(t for _, t in chunks if t.strip())
        (page_dir / "result.txt").write_text(page_text, encoding="utf-8")

    logger.info(f"OCR por box exportado: {ocr_path}")
    logger.info(f"Paginas recompuestas en: {pages_dir}")
    return pages_dir


def generar_pages_desde_yolo_ocr(
    images_dir: Path,
    run_dir: Path,
    yolo_model_path: Path,
    tomos_filter: set[str] | None = None,
    yolo_conf: float = 0.25,
    yolo_imgsz: int = 1024,
    yolo_batch: int = 4,
    ocr_model: str = "deepseek-ai/DeepSeek-OCR",
    ocr_prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    ocr_device: str | None = None,
    ocr_dtype: str = "bfloat16",
    ocr_max_new_tokens: int = 4096,
    ocr_max_retries: int = 3,
) -> Path:
    logger.info(f"Imagenes: {images_dir}")
    logger.info(f"YOLO: {yolo_model_path} | conf={yolo_conf} | imgsz={yolo_imgsz} | batch={yolo_batch}")
    logger.info(f"OCR: {ocr_model}")

    df_boxes = detectar_boxes_yolo(
        images_dir=images_dir,
        yolo_model_path=yolo_model_path,
        tomos_filter=tomos_filter,
        conf=yolo_conf,
        imgsz=yolo_imgsz,
        batch=yolo_batch,
    )
    df_manifest = guardar_crops_y_manifest(df_boxes, run_dir)
    return ocr_boxes_a_paginas(
        df_manifest=df_manifest,
        run_dir=run_dir,
        model_name=ocr_model,
        prompt=ocr_prompt,
        device=ocr_device,
        dtype=ocr_dtype,
        max_new_tokens=ocr_max_new_tokens,
        max_retries=ocr_max_retries,
    )


# =====================================================================
# 1. FLAGGEO DE CALIDAD OCR
# =====================================================================

RE_NO_LATINO = re.compile(
    r"[\u0900-\u097F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]"
)
RE_ARTEFACTO = re.compile(r"<\|(?:ref|det|/ref|/det)\|>")


def _evaluar_calidad(texto: str) -> dict:
    if not texto.strip():
        return {
            "largo": 0,
            "ratio_alfa": 0.0,
            "chars_no_latino": 0,
            "lineas_repetidas": 0,
            "ratio_artefactos": 0.0,
        }

    largo = len(texto)
    ratio_alfa = sum(c.isalpha() for c in texto) / largo if largo else 0
    chars_no_latino = len(RE_NO_LATINO.findall(texto))

    lineas = [l.strip() for l in texto.split("\n") if l.strip()]
    counter = Counter(lineas)
    lineas_repetidas = sum(v - 1 for v in counter.values() if v > 1)

    lineas_artefacto = sum(1 for l in lineas if RE_ARTEFACTO.search(l))
    ratio_artefactos = lineas_artefacto / len(lineas) if lineas else 0

    return {
        "largo": largo,
        "ratio_alfa": round(ratio_alfa, 3),
        "chars_no_latino": chars_no_latino,
        "lineas_repetidas": lineas_repetidas,
        "ratio_artefactos": round(ratio_artefactos, 3),
    }


def flaggear_ocr(pages_dir: Path, run_dir: Path) -> pd.DataFrame:
    page_files = sorted(pages_dir.rglob("result.txt"))

    rows: list[dict] = []
    largos_tomo: dict[str, list[int]] = defaultdict(list)

    for path in page_files:
        meta = _parse_tomo_page(path)
        texto = _read_text(path)
        metricas = _evaluar_calidad(texto)
        row = {
            "tomo": meta["tomo"],
            "page_num": meta["page_num"],
            "archivo": str(path.relative_to(pages_dir)),
            **metricas,
        }
        rows.append(row)
        if meta["tomo"]:
            largos_tomo[meta["tomo"]].append(metricas["largo"])

    df = pd.DataFrame(rows)
    medianas = {t: np.median(v) for t, v in largos_tomo.items()}

    def _flag(row):
        flags = []
        if row["ratio_alfa"] < 0.40:
            flags.append("bajo_alfa")
        if row["chars_no_latino"] > 0:
            flags.append("script_no_latino")
        if row["lineas_repetidas"] > 5:
            flags.append("texto_repetitivo")
        if row["ratio_artefactos"] > 0.3:
            flags.append("artefactos")
        med = medianas.get(row["tomo"], 1)
        if med > 0:
            r = row["largo"] / med
            if r < 0.1:
                flags.append("muy_corto")
            elif r > 5:
                flags.append("muy_largo")
        return "; ".join(flags)

    df["flag"] = df.apply(_flag, axis=1)
    df["flagged"] = df["flag"] != ""

    out_path = run_dir / "calidad_ocr.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    n_flag = int(df["flagged"].sum())
    logger.info(f"{len(df):,} paginas evaluadas, {n_flag} flaggeadas")
    if n_flag > 0:
        all_flags = [
            f
            for flags in df[df["flagged"]]["flag"]
            for f in flags.split("; ")
            if f
        ]
        for f, c in Counter(all_flags).most_common():
            logger.info(f"  {f}: {c}")

    return df


# =====================================================================
# 2. LIMPIEZA DE PAGINAS
# =====================================================================


def _drop_angle_brackets(text: str) -> str:
    return "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("<"))


def _norm_catalogo(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()


_CATALOGO_NORM = _norm_catalogo("CATALOGO DE LOS FONDOS AMERICANOS")


def _is_catalogo_line(line: str, max_dist: int = 4) -> bool:
    norm = _norm_catalogo(line)
    if not norm:
        return False
    if _CATALOGO_NORM in norm:
        return True
    L = len(_CATALOGO_NORM)
    if len(norm) < L:
        return _levenshtein(norm, _CATALOGO_NORM) <= max_dist
    for i in range(len(norm) - L + 1):
        if _levenshtein(norm[i : i + L], _CATALOGO_NORM) <= max_dist:
            return True
    return False


def _drop_catalogo(text: str) -> str:
    return "\n".join(l for l in text.split("\n") if not _is_catalogo_line(l))


def _dedup_lineas_consecutivas(texto: str) -> str:
    """Elimina lineas duplicadas consecutivas (ignorando vacias intermedias).
    Artefacto de OCR que detecta la misma region dos veces."""
    lines = texto.split("\n")
    result: list[str] = []
    last_content = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append(line)
            continue
        if stripped == last_content and len(stripped) > 20:
            continue
        result.append(line)
        last_content = stripped
    return "\n".join(result)


def _limpiar_pagina(texto: str) -> str:
    texto = _drop_angle_brackets(texto)
    texto = _drop_catalogo(texto)
    texto = _dedup_lineas_consecutivas(texto)
    return texto


# =====================================================================
# 3. UNION INTELIGENTE DE PAGINAS
# =====================================================================

_PUNCT_FUERTE = set(".?!:;)")


def _unir_paginas(paginas: list[str]) -> str:
    """
    Une paginas de texto inteligentemente:
    - Ultima linea termina en guion: concatena sin espacio (palabra cortada)
    - Ultima linea no termina en puntuacion fuerte: une con espacio (frase continua)
    - Ultima linea termina en puntuacion fuerte: separa con doble salto (bloque nuevo)
    """
    if not paginas:
        return ""
    if len(paginas) == 1:
        return paginas[0]

    # Trabajamos con lista de lineas para manipulacion precisa
    all_lines: list[str] = paginas[0].rstrip().split("\n")

    for pagina in paginas[1:]:
        pagina_s = pagina.strip()
        if not pagina_s:
            continue

        page_lines = pagina_s.split("\n")

        # Ultima linea no vacia del acumulado
        last_idx = None
        for j in range(len(all_lines) - 1, -1, -1):
            if all_lines[j].strip():
                last_idx = j
                break

        # Primera linea no vacia de la pagina nueva
        first_idx = None
        for j, l in enumerate(page_lines):
            if l.strip():
                first_idx = j
                break

        if last_idx is None or first_idx is None:
            all_lines.append("")
            all_lines.extend(page_lines)
            continue

        ultima = all_lines[last_idx].rstrip()
        primera = page_lines[first_idx].strip()
        rest = page_lines[first_idx + 1 :]

        if ultima.endswith("-"):
            # Palabra cortada: quitar guion y pegar
            all_lines[last_idx] = ultima[:-1] + primera
        elif ultima and ultima[-1] not in _PUNCT_FUERTE:
            # Frase continua: unir con espacio
            all_lines[last_idx] = ultima + " " + primera
        else:
            # Separacion limpia
            all_lines.append("")
            all_lines.append(primera)

        all_lines.extend(rest)

    return "\n".join(all_lines)


def _join_soft_linebreaks(text: str) -> str:
    """Une lineas dentro de bloques (parrafos). Lineas vacias = separador."""
    lines = text.split("\n")
    joined: list[str] = []
    buf: list[str] = []

    for line in lines:
        if line.strip():
            buf.append(line.strip())
        else:
            if buf:
                joined.append(" ".join(buf))
                buf = []
            joined.append("")

    if buf:
        joined.append(" ".join(buf))

    return "\n".join(joined)


# =====================================================================
# 4. CONSOLIDAR TOMOS
# =====================================================================


def consolidar_tomos(pages_dir: Path, run_dir: Path) -> Path:
    tomos_dir = run_dir / "tomos_txt"
    tomos_dir.mkdir(exist_ok=True)

    page_files = sorted(pages_dir.rglob("result.txt"))

    by_tomo: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in page_files:
        meta = _parse_tomo_page(path)
        if meta["tomo"] is None:
            continue
        by_tomo[meta["tomo"]].append((meta["page_num"], path))

    for tomo, pages in by_tomo.items():
        pages.sort(key=lambda x: x[0])

        paginas_limpias = []
        for _page_num, path in pages:
            texto = _read_text(path)
            texto = _limpiar_pagina(texto)
            if texto.strip():
                paginas_limpias.append(texto)

        # Union inteligente entre paginas
        tomo_text = _unir_paginas(paginas_limpias)

        # Unir lineas sueltas dentro de parrafos
        tomo_text = _join_soft_linebreaks(tomo_text)

        out_path = tomos_dir / f"{_sanitize_name(tomo)}.txt"
        out_path.write_text(tomo_text, encoding="utf-8")

    logger.info(f"{len(by_tomo)} tomos consolidados en {tomos_dir}")
    return tomos_dir


# =====================================================================
# 5. SEGMENTACION DE CONTRATOS
# =====================================================================


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _is_libro_del_ano(words: list[str], max_distance: int = 3) -> bool:
    tokens = [_normalize_token(w) for w in words if w]
    if not tokens:
        return False
    target = "librodelano"
    n = len(tokens)
    for window in range(2, 6):
        if window > n:
            break
        for i in range(n - window + 1):
            combo = "".join(tokens[i : i + window])
            if _levenshtein(combo, target) <= max_distance:
                return True
    return False


def _segmentar_tomo(texto: str) -> list[dict]:
    """Segmenta texto de un tomo en contratos individuales."""
    records: list[dict] = []
    current: dict | None = None

    for line in texto.splitlines():
        line_s = line.strip()
        if not line_s:
            continue

        # Lineas que empiezan con "Nota" nunca inician nuevo contrato
        if re.match(r"^\s*Nota\b", line_s, flags=re.I):
            if current is not None:
                current["partes"].append(line_s)
            continue

        words = line_s.split()
        if _is_libro_del_ano(words[:10]):
            if current is not None:
                records.append(current)

            id_num = None
            header = line_s
            m = re.match(r"^(\d+)\s*[\.\)]?-?\s+(.*)$", header)
            if m:
                id_num = int(m.group(1))
                header = m.group(2).strip()

            current = {"id_num": id_num, "partes": [header]}
        else:
            if current is None:
                continue
            current["partes"].append(line_s)

    if current is not None:
        records.append(current)

    return records


def segmentar_tomos(tomos_dir: Path, run_dir: Path) -> pd.DataFrame:
    """Segmenta todos los tomos y produce DataFrame."""
    tomo_files = sorted(tomos_dir.glob("*.txt"))

    all_main: list[dict] = []
    all_debug: list[dict] = []

    for tomo_file in tomo_files:
        texto = tomo_file.read_text(encoding="utf-8")
        records = _segmentar_tomo(texto)

        for rec in records:
            texto_completo = " ".join(rec["partes"])
            all_main.append({
                "tomo_id": tomo_file.stem,
                "id_num": rec["id_num"],
                "texto_completo": texto_completo,
            })

            # Version debug con columnas separadas
            row_debug = {"tomo_id": tomo_file.stem, "id_num": rec["id_num"]}
            for i, parte in enumerate(rec["partes"]):
                row_debug[f"col_{i}"] = parte
            all_debug.append(row_debug)

    df_main = pd.DataFrame(all_main)

    # Exportar version segmentada para debug/revision
    df_debug = pd.DataFrame(all_debug)
    seg_path = run_dir / "contratos_segmentados.xlsx"
    df_debug.to_excel(seg_path, index=False)

    logger.info(f"{len(df_main):,} contratos segmentados de {len(tomo_files)} tomos")
    return df_main


# =====================================================================
# 6. PARSEO DE CAMPOS ESTRUCTURADOS
# =====================================================================


def parsear_contratos(df: pd.DataFrame) -> pd.DataFrame:
    """Parsea campos (macrodatos, ano, oficio, etc.) de cada contrato."""
    registros: list[dict] = []

    for _, row in df.iterrows():
        texto = row.get("texto_completo", "")
        parsed = parsear_texto(texto)

        registros.append({
            "tomo_id": row["tomo_id"],
            "id_num": row["id_num"],
            **parsed,
            "texto_completo": texto,
            "n_subregistros": contar_subregistros(texto),
        })

    columnas = [
        "tomo_id", "id_num", "macrodatos", "asunto",
        "año", "año_num", "oficio", "libro",
        "escribania", "folio", "fecha", "signatura",
        "observaciones", "texto_completo",
        "n_subregistros", "_parse_ok",
    ]

    df_out = pd.DataFrame(registros)
    for col in columnas:
        if col not in df_out.columns:
            df_out[col] = ""
    df_out = df_out[columnas]
    df_out["año_num"] = df_out["año_num"].astype("Int64")

    n_ok = int(df_out["_parse_ok"].sum())
    logger.info(f"{n_ok:,} / {len(df_out):,} parseados correctamente ({100*n_ok/len(df_out):.1f}%)")
    return df_out


# =====================================================================
# 7. CRUZAR FLAGS OCR CON CONTRATOS
# =====================================================================


def _cruzar_flags_ocr(df_compilado: pd.DataFrame, df_calidad: pd.DataFrame) -> pd.DataFrame:
    """Agrega flags OCR al compilado a nivel de contrato.

    Detecta problemas directamente en el texto del contrato usando
    las mismas metricas que el flaggeo por pagina.
    """
    df = df_compilado.copy()

    def _flag_contrato(texto: str) -> str:
        if not isinstance(texto, str) or not texto.strip():
            return "vacio"
        flags = []
        largo = len(texto)
        ratio_alfa = sum(c.isalpha() for c in texto) / largo if largo else 0
        if ratio_alfa < 0.40:
            flags.append("bajo_alfa")
        if RE_NO_LATINO.search(texto):
            flags.append("script_no_latino")
        lineas = [l.strip() for l in texto.split("\n") if l.strip()]
        if lineas:
            counter = Counter(lineas)
            repetidas = sum(v - 1 for v in counter.values() if v > 1)
            if repetidas > 5:
                flags.append("texto_repetitivo")
        if largo < 80:
            flags.append("muy_corto")
        return "; ".join(flags)

    df["ocr_flag"] = df["texto_completo"].apply(_flag_contrato)
    df["ocr_flagged"] = df["ocr_flag"] != ""

    n_flag = int(df["ocr_flagged"].sum())
    logger.info(f"{n_flag:,} contratos con flags OCR directas")
    if n_flag > 0:
        all_flags = [
            f for flags in df[df["ocr_flagged"]]["ocr_flag"]
            for f in flags.split("; ") if f
        ]
        for f, c in Counter(all_flags).most_common():
            logger.info(f"  {f}: {c}")
    return df


# =====================================================================
# 8. CORRECCION DE SECUENCIA id_num
# =====================================================================


def corregir_secuencia_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige id_num usando la secuencia esperada (siempre consecutiva por tomo).

    Reglas:
    - [2, NULL, 3]: el NULL es header → obviar
    - [2, NULL, 4]: gap=1, 1 NULL → rellenar con 3
    - [2, NULL, 5]: gap=2, 1 NULL → rellenar 3, flaggear 4 como perdido
    - id_num que rompe secuencia (bajo/salto/duplicado) → tratar como rellenable
    """
    df = df.copy()
    df["id_num_original"] = df["id_num"].copy()
    df["id_status"] = ""

    stats = {"ancla": 0, "rellenado": 0, "corregido": 0, "header": 0, "perdido_qty": 0}

    for tomo in df["tomo_id"].unique():
        mask_tomo = df["tomo_id"] == tomo
        indices = df[mask_tomo].index.tolist()
        n = len(indices)

        # Paso 1: identificar anclas (id_nums que forman secuencia creciente valida)
        anclas: list[tuple[int, int, int]] = []  # (pos_en_tomo, df_index, id_value)
        sosp_positions: set[int] = set()

        for pos in range(n):
            idx = indices[pos]
            val = df.at[idx, "id_num"]
            if pd.notna(val):
                val_int = int(val)
                if not anclas:
                    anclas.append((pos, idx, val_int))
                else:
                    _, _, prev_val = anclas[-1]
                    if val_int > prev_val and (val_int - prev_val) < 100:
                        anclas.append((pos, idx, val_int))
                    else:
                        sosp_positions.add(pos)

        # Marcar anclas
        for pos, idx, val in anclas:
            df.at[idx, "id_status"] = "ancla"
            stats["ancla"] += 1

        # Paso 2: entre cada par de anclas, clasificar filas intermedias
        for k in range(len(anclas) - 1):
            pos_a, idx_a, id_a = anclas[k]
            pos_b, idx_b, id_b = anclas[k + 1]
            gap = id_b - id_a - 1  # contratos intermedios esperados

            # Posiciones rellenables entre anclas (NULLs + sospechosos)
            fillable = []
            for pos in range(pos_a + 1, pos_b):
                idx = indices[pos]
                if pd.isna(df.at[idx, "id_num_original"]) or pos in sosp_positions:
                    fillable.append((pos, idx))

            if gap == 0:
                # Consecutivas: NULLs son headers
                for pos, idx in fillable:
                    df.at[idx, "id_status"] = "header"
                    stats["header"] += 1
            elif gap <= len(fillable):
                # Podemos rellenar todos los esperados
                fill_id = id_a + 1
                for i, (pos, idx) in enumerate(fillable):
                    if i < gap:
                        was_null = pd.isna(df.at[idx, "id_num_original"])
                        df.at[idx, "id_num"] = fill_id
                        status = "rellenado" if was_null else "corregido"
                        df.at[idx, "id_status"] = status
                        stats[status] += 1
                        fill_id += 1
                    else:
                        df.at[idx, "id_status"] = "header"
                        stats["header"] += 1
            else:
                # gap > fillable: rellenar lo que se pueda
                fill_id = id_a + 1
                for pos, idx in fillable:
                    was_null = pd.isna(df.at[idx, "id_num_original"])
                    df.at[idx, "id_num"] = fill_id
                    status = "rellenado" if was_null else "corregido"
                    df.at[idx, "id_status"] = status
                    stats[status] += 1
                    fill_id += 1
                # Los ids restantes son genuinamente perdidos
                stats["perdido_qty"] += gap - len(fillable)

        # Sospechosos despues de la ultima ancla (sin clasificar)
        if anclas:
            last_pos = anclas[-1][0]
            for pos in range(last_pos + 1, n):
                idx = indices[pos]
                if df.at[idx, "id_status"] == "" and pos in sosp_positions:
                    df.at[idx, "id_status"] = "sospechoso"

    # Convertir id_num a Int64
    df["id_num"] = pd.to_numeric(df["id_num"], errors="coerce").astype("Int64")
    df["id_num_original"] = pd.to_numeric(df["id_num_original"], errors="coerce").astype("Int64")

    logger.info(f"Anclas: {stats['ancla']:,} | Rellenados: {stats['rellenado']:,} | "
               f"Corregidos: {stats['corregido']:,} | Headers: {stats['header']:,}")
    logger.info(f"Contratos perdidos por OCR: {stats['perdido_qty']:,}")
    return df


# =====================================================================
# 9. RE-SEGMENTACION DE CONTRATOS ENTERRADOS
# =====================================================================


def resegmentar_perdidos(df: pd.DataFrame) -> pd.DataFrame:
    """Busca contratos perdidos enterrados en el texto de contratos vecinos.

    Cuando la consolidacion de paginas une dos paginas sin salto de linea,
    el header del siguiente contrato queda al final del texto del anterior.
    El segmentador no lo ve porque solo evalua el inicio de cada linea.

    Este paso usa el ground truth (IDs secuenciales) para buscar headers
    enterrados: si falta el ID N, busca "N Libro del a" dentro del
    texto_completo del contrato con el ID inmediatamente anterior.
    Si lo encuentra, parte el texto y crea una nueva fila.
    """
    df = df.copy()
    nuevas_filas: list[dict] = []
    indices_modificados: set[int] = set()
    stats = {"gaps_analizados": 0, "recuperados": 0}

    for tomo in df["tomo_id"].unique():
        mask_tomo = df["tomo_id"] == tomo
        t = df[mask_tomo]
        indices = t.index.tolist()
        n = len(indices)

        # Reconstruir anclas (misma logica que corregir_secuencia_ids)
        anclas: list[tuple[int, int, int]] = []  # (pos, df_index, id_value)
        sosp_positions: set[int] = set()

        for pos in range(n):
            idx = indices[pos]
            val = t.at[idx, "id_num"]
            if pd.notna(val):
                val_int = int(val)
                if not anclas:
                    anclas.append((pos, idx, val_int))
                else:
                    _, _, prev_val = anclas[-1]
                    if val_int > prev_val and (val_int - prev_val) < 100:
                        anclas.append((pos, idx, val_int))
                    else:
                        sosp_positions.add(pos)

        # Para cada par de anclas con gap, buscar headers enterrados
        for k in range(len(anclas) - 1):
            pos_a, idx_a, id_a = anclas[k]
            pos_b, idx_b, id_b = anclas[k + 1]
            gap = id_b - id_a - 1

            # Contar fillables
            fillable_count = 0
            for pos in range(pos_a + 1, pos_b):
                idx = indices[pos]
                orig = t.at[idx, "id_num_original"] if "id_num_original" in t.columns else t.at[idx, "id_num"]
                if pd.isna(orig) or pos in sosp_positions:
                    fillable_count += 1

            perdidos = gap - fillable_count
            if perdidos <= 0:
                continue

            stats["gaps_analizados"] += 1

            # IDs perdidos en este gap
            ids_perdidos = list(range(id_a + fillable_count + 1, id_b))

            # Buscar en el texto_completo del contrato ancla anterior
            texto_ancla = str(df.at[idx_a, "texto_completo"]) if pd.notna(df.at[idx_a, "texto_completo"]) else ""
            if not texto_ancla:
                continue

            for id_perdido in ids_perdidos:
                # Regex flexible: "88 . Libro del a", "88. Libro del a", "88 Libro del a"
                patron = rf"(?<!\d){id_perdido}\s*\.?\s*-?\s*Libro\s+del\s+a"
                match = re.search(patron, texto_ancla, re.IGNORECASE)
                if not match:
                    continue

                # Partir el texto
                pos_corte = match.start()
                texto_antes = texto_ancla[:pos_corte].rstrip()
                texto_nuevo = texto_ancla[pos_corte:]

                # Actualizar el contrato original (truncar)
                df.at[idx_a, "texto_completo"] = texto_antes
                indices_modificados.add(idx_a)

                # Re-parsear el contrato original
                parsed_antes = parsear_texto(texto_antes)
                for campo, valor in parsed_antes.items():
                    if campo in df.columns:
                        df.at[idx_a, campo] = valor
                df.at[idx_a, "n_subregistros"] = contar_subregistros(texto_antes)

                # Crear nueva fila para el contrato recuperado
                parsed_nuevo = parsear_texto(texto_nuevo)
                nueva_fila = {
                    "tomo_id": tomo,
                    "id_num": id_perdido,
                    "id_num_original": pd.NA,
                    "id_status": "resegmentado",
                    **parsed_nuevo,
                    "texto_completo": texto_nuevo,
                    "n_subregistros": contar_subregistros(texto_nuevo),
                    "_parse_ok": parsed_nuevo.get("_parse_ok", False),
                }

                # Copiar columnas que puedan existir pero no esten en parsed
                for col in df.columns:
                    if col not in nueva_fila:
                        nueva_fila[col] = "" if df[col].dtype == object else pd.NA

                nuevas_filas.append({
                    "insertar_despues_de": idx_a,
                    "fila": nueva_fila,
                })
                stats["recuperados"] += 1

                # Actualizar texto_ancla para buscar mas IDs en el texto restante
                # (no aplica: el texto_nuevo ya fue separado, texto_antes no tiene mas)
                break  # Solo un ID por contrato ancla en esta pasada

    # Insertar nuevas filas en el DataFrame
    if nuevas_filas:
        # Construir DataFrame final intercalando filas nuevas
        # Mapa: despues de idx_ref -> lista de filas a insertar
        insertar_map: dict[int, list[dict]] = defaultdict(list)
        for item in nuevas_filas:
            insertar_map[item["insertar_despues_de"]].append(item["fila"])

        partes: list[pd.DataFrame] = []
        indices_procesados = set(insertar_map.keys())
        for idx in df.index:
            partes.append(df.loc[[idx]])
            if idx in indices_procesados:
                nuevas_df = pd.DataFrame(insertar_map[idx])
                partes.append(nuevas_df)

        df = pd.concat(partes, ignore_index=True)

    logger.info(f"Gaps analizados: {stats['gaps_analizados']:,}")
    logger.info(f"Contratos recuperados por re-segmentacion: {stats['recuperados']:,}")
    if indices_modificados:
        logger.info(f"Contratos originales modificados (texto truncado): {len(indices_modificados):,}")

    return df


# =====================================================================
# 10. DIAGNOSTICO DE PAGINAS PARA RE-OCR
# =====================================================================


def _normalizar_busqueda(texto: str) -> str:
    """Normaliza texto para busqueda fuzzy: minusculas, sin acentos, sin espacios extra."""
    import unicodedata as ud
    texto = ud.normalize("NFD", texto)
    texto = "".join(c for c in texto if not ud.combining(c))
    texto = texto.lower()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def diagnosticar_paginas_reocr(
    df: pd.DataFrame,
    pages_dir: Path,
    run_dir: Path,
) -> pd.DataFrame:
    """Identifica paginas que necesitan re-OCR por contratos perdidos.

    Mapea contratos-ancla a paginas fisicas, luego para cada gap
    identifica el rango de paginas candidatas para re-OCR.
    """
    # Cargar todas las paginas por tomo
    page_files = sorted(pages_dir.rglob("result.txt"))
    paginas_por_tomo: dict[str, list[dict]] = defaultdict(list)

    for path in page_files:
        meta = _parse_tomo_page(path)
        if meta["tomo"]:
            texto = _read_text(path)
            texto_limpio = _limpiar_pagina(texto)
            paginas_por_tomo[meta["tomo"]].append({
                "page_num": meta["page_num"],
                "texto_norm": _normalizar_busqueda(texto_limpio),
            })

    for tomo in paginas_por_tomo:
        paginas_por_tomo[tomo].sort(key=lambda x: x["page_num"])

    # Para cada tomo, encontrar gaps y mapear a paginas
    resultados: list[dict] = []

    for tomo in sorted(df["tomo_id"].unique()):
        # Convertir tomo_id ("Tomo_I") a nombre de pagina ("Tomo I")
        tomo_nombre = tomo.replace("_", " ")
        paginas = paginas_por_tomo.get(tomo_nombre, [])
        if not paginas:
            continue

        t = df[df["tomo_id"] == tomo].copy()
        indices = t.index.tolist()
        n = len(indices)

        # Reconstruir anclas (misma logica que corregir_secuencia_ids)
        anclas = []
        for pos in range(n):
            idx = indices[pos]
            val = t.at[idx, "id_num_original"] if "id_num_original" in t.columns else t.at[idx, "id_num"]
            if pd.notna(val):
                val_int = int(val)
                if not anclas:
                    anclas.append((pos, idx, val_int))
                else:
                    _, _, prev_val = anclas[-1]
                    if val_int > prev_val and (val_int - prev_val) < 100:
                        anclas.append((pos, idx, val_int))

        # Funcion para buscar en que pagina esta un contrato
        def _buscar_pagina(df_idx: int, hint_start: int = 0) -> int | None:
            asunto = str(t.at[df_idx, "asunto"]) if pd.notna(t.at[df_idx, "asunto"]) else ""
            if len(asunto) < 15:
                # Usar texto_completo si asunto es muy corto
                asunto = str(t.at[df_idx, "texto_completo"]) if pd.notna(t.at[df_idx, "texto_completo"]) else ""
            if len(asunto) < 15:
                return None

            # Tomar fragmento distintivo (50-80 chars del asunto)
            key = _normalizar_busqueda(asunto[:80])
            if len(key) < 10:
                return None

            # Buscar desde hint hacia adelante
            for pi in range(max(0, hint_start - 2), len(paginas)):
                if key[:40] in paginas[pi]["texto_norm"]:
                    return paginas[pi]["page_num"]
            # Buscar hacia atras si no se encontro
            for pi in range(min(hint_start, len(paginas)) - 1, -1, -1):
                if key[:40] in paginas[pi]["texto_norm"]:
                    return paginas[pi]["page_num"]
            return None

        # Para cada par de anclas con gap > nulls, mapear a paginas
        hint = 0
        for k in range(len(anclas) - 1):
            pos_a, idx_a, id_a = anclas[k]
            pos_b, idx_b, id_b = anclas[k + 1]
            gap = id_b - id_a - 1

            # Contar fillables
            fillable_count = 0
            for pos in range(pos_a + 1, pos_b):
                idx = indices[pos]
                orig = t.at[idx, "id_num_original"] if "id_num_original" in t.columns else t.at[idx, "id_num"]
                if pd.isna(orig):
                    fillable_count += 1
                else:
                    val_int = int(orig)
                    if not (val_int > id_a and val_int < id_b and (val_int - id_a) < 100):
                        fillable_count += 1  # sospechoso

            perdidos = gap - fillable_count
            if perdidos <= 0:
                continue

            # Buscar paginas de los contratos vecinos
            page_antes = _buscar_pagina(idx_a, hint)
            page_despues = _buscar_pagina(idx_b, hint)

            if page_antes is not None:
                hint = next(
                    (pi for pi, p in enumerate(paginas) if p["page_num"] == page_antes), hint
                )

            # Rango de paginas candidatas (incluyendo +-1 por contratos entre paginas)
            if page_antes is not None and page_despues is not None:
                p_min = max(1, page_antes - 1)
                p_max = page_despues + 1
                paginas_reocr = list(range(p_min, p_max + 1))
            elif page_antes is not None:
                paginas_reocr = list(range(max(1, page_antes - 1), page_antes + 3))
            elif page_despues is not None:
                paginas_reocr = list(range(max(1, page_despues - 2), page_despues + 1))
            else:
                paginas_reocr = []

            # IDs perdidos
            ids_perdidos = list(range(id_a + fillable_count + 1, id_b))

            for lost_id in ids_perdidos:
                resultados.append({
                    "tomo_id": tomo,
                    "id_perdido": lost_id,
                    "id_ancla_antes": id_a,
                    "id_ancla_despues": id_b,
                    "perdidos_en_gap": perdidos,
                    "pagina_ancla_antes": page_antes,
                    "pagina_ancla_despues": page_despues,
                    "paginas_reocr": "; ".join(str(p) for p in paginas_reocr),
                })

    df_diag = pd.DataFrame(resultados)

    if len(df_diag) > 0:
        # Resumen de paginas unicas a re-OCR por tomo
        resumen: list[dict] = []
        for tomo in df_diag["tomo_id"].unique():
            sub = df_diag[df_diag["tomo_id"] == tomo]
            all_pages = set()
            for pages_str in sub["paginas_reocr"]:
                for p in str(pages_str).split("; "):
                    if p.strip().isdigit():
                        all_pages.add(int(p.strip()))
            resumen.append({
                "tomo_id": tomo,
                "contratos_perdidos": len(sub),
                "paginas_a_reocr": len(all_pages),
                "lista_paginas": "; ".join(str(p) for p in sorted(all_pages)),
            })
        df_resumen = pd.DataFrame(resumen)

        # Exportar
        diag_path = run_dir / "diagnostico_reocr.xlsx"
        with pd.ExcelWriter(diag_path, engine="openpyxl") as writer:
            df_resumen.to_excel(writer, sheet_name="resumen_por_tomo", index=False)
            df_diag.to_excel(writer, sheet_name="contratos_perdidos", index=False)

        n_pages = sum(r["paginas_a_reocr"] for r in resumen)
        logger.info(f"{len(df_diag):,} contratos perdidos en {n_pages} paginas candidatas")
        logger.info(f"Diagnostico exportado: {diag_path}")
    else:
        logger.info("Sin contratos perdidos detectados")

    return df_diag


# =====================================================================
# 10. EXTRACCION DE ENTIDADES VIA OLLAMA
# =====================================================================

PROMPT_ENTIDADES = """Eres un experto en historia colonial española.

Extrae del siguiente texto de un contrato notarial del siglo XV-XVI:

- personas: SOLO las partes contratantes — quienes firman, otorgan, se obligan, reciben, apodera, actúan como fiadores o testigos. NO incluyas personas que solo se mencionan como referencia, contexto o relación (ej. "siervo de X", "hijo de X difunto", "cobrar a X").
- naos: lista de naves, naos o carabelas (nombres de barcos).
- lugares: lista de lugares geográficos (ciudades, puertos, regiones).
- atributos: objeto donde cada clave es un nombre de persona (exactamente igual al de "personas") y el valor es UN STRING con todas sus características explícitas separadas por "; ". Incluir: gentilicio, vecindad, oficio/cargo, nao asignada, y relaciones con otras personas mencionadas en el texto (ej. "siervo de Cristóbal Colón", "hijo de Francisco Fernández difunto", "para cobrar a Diego de Nécesa"). No inventar nada.

EJEMPLO:
Texto: "Pepito Pérez, siervo de Cristóbal Colón, se compromete con Libia a..."
Resultado:
  personas: ["Pepito Pérez", "Libia"]
  atributos: {{"Pepito Pérez": "siervo de Cristóbal Colón", "Libia": ""}}
(Cristóbal Colón NO va en personas porque no es parte del contrato)

REGLAS:
- Si un campo no aplica, devuelve [].
- Si una persona no tiene atributos explícitos, pon "".
- Cada persona debe ser un nombre completo, no fragmentos ("don Martín de Tolosa", no "don Martín" + "de Tolosa").
- Une palabras partidas por guion/salto de línea: "Gri- maldo" -> "Grimaldo", "tone- ladas" -> "toneladas".
- Responde SOLO un JSON válido, sin texto adicional.

TEXTO:
\"\"\"{texto}\"\"\""""


def _check_ollama(ollama_url: str) -> bool:
    """Verifica que Ollama este corriendo y responda."""
    base_url = ollama_url.replace("/api/generate", "/api/tags")
    try:
        resp = requests.get(base_url, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _call_ollama(texto: str, ollama_url: str, model_name: str) -> Dict[str, List[str]]:
    """Llama a Ollama para extraer entidades de un texto."""
    payload = {
        "model": model_name,
        "prompt": PROMPT_ENTIDADES.format(texto=texto),
        "stream": False,
        "format": "json",
        "options": {"num_ctx": 8192},
        "think": False,
    }

    try:
        resp = requests.post(ollama_url, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        result = json.loads(raw)
    except Exception as e:
        logger.warning(f"Error en Ollama: {e}")
        return {"personas": [], "naos": [], "lugares": [], "atributos": {}}

    def _norm(lst):
        if not isinstance(lst, list):
            return []
        return [x.strip() for x in lst if isinstance(x, str) and x.strip()]

    personas = _norm(result.get("personas", []))
    atributos_raw = result.get("atributos", {})
    if not isinstance(atributos_raw, dict):
        atributos_raw = {}
    # Normalizar: solo claves que estan en personas, valores como string
    atributos = {}
    for p in personas:
        val = atributos_raw.get(p, "")
        if isinstance(val, str):
            atributos[p] = val.strip()
        else:
            atributos[p] = ""

    return {
        "personas": personas,
        "naos": _norm(result.get("naos", [])),
        "lugares": _norm(result.get("lugares", [])),
        "atributos": atributos,
    }


def extraer_entidades(
    df: pd.DataFrame,
    run_dir: Path,
    ollama_url: str = "http://localhost:11434/api/generate",
    model_name: str = "qwen3.5:9b",
    max_entidades: int = 0,
) -> pd.DataFrame:
    """Extrae personas, naos y lugares de la columna asunto via Ollama."""

    # Health check
    if not _check_ollama(ollama_url):
        logger.error(f"Ollama no disponible en {ollama_url}. Inicia con: ollama serve")
        logger.warning("Saltando extraccion de entidades.")
        df["personas"] = ""
        df["naos"] = ""
        df["lugares"] = ""
        df["atributos"] = ""
        return df

    logger.info(f"Modelo: {model_name} | URL: {ollama_url}")

    # Filtrar contratos con asunto no vacio
    mask = df["asunto"].fillna("").str.strip().astype(bool)
    indices = df.index[mask].tolist()

    if max_entidades > 0:
        indices = indices[:max_entidades]
        logger.info(f"Limitado a {max_entidades} contratos (--max-entidades)")

    total = len(indices)
    logger.info(f"{total:,} contratos con asunto para procesar")

    personas_col = [""] * len(df)
    naos_col = [""] * len(df)
    lugares_col = [""] * len(df)
    atributos_col = [""] * len(df)

    errores = 0
    t0 = time.time()

    for count, idx in enumerate(indices, start=1):
        texto = df.at[idx, "asunto"]
        entidades = _call_ollama(texto, ollama_url, model_name)

        if all(v == [] for k, v in entidades.items() if k != "atributos"):
            errores += 1

        pos = df.index.get_loc(idx)
        personas_col[pos] = "; ".join(entidades["personas"])
        naos_col[pos] = "; ".join(entidades["naos"])
        lugares_col[pos] = "; ".join(entidades["lugares"])
        # Serializar atributos como "nombre::desc || nombre::desc"
        attr = entidades.get("atributos", {})
        if attr:
            atributos_col[pos] = " || ".join(
                f"{k}::{v}" for k, v in attr.items() if v
            )

        # Progreso cada 100
        if count % 100 == 0:
            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 0
            eta = (total - count) / rate / 60 if rate > 0 else 0
            logger.info(f"Entidades: {count:,}/{total:,} ({100*count/total:.1f}%) "
                       f"| {rate:.1f} reg/s | ETA: {eta:.0f} min | errores: {errores}")

        # Checkpoint cada 500
        if count % 500 == 0:
            df_tmp = df.copy()
            df_tmp["personas"] = personas_col
            df_tmp["naos"] = naos_col
            df_tmp["lugares"] = lugares_col
            df_tmp["atributos"] = atributos_col
            ckpt_path = run_dir / "compilado_parcial.xlsx"
            df_tmp.to_excel(ckpt_path, index=False)

    df["personas"] = personas_col
    df["naos"] = naos_col
    df["lugares"] = lugares_col
    df["atributos"] = atributos_col

    elapsed = time.time() - t0
    logger.info(f"Extraccion completada en {elapsed/60:.1f} min | errores Ollama: {errores}")
    return df


# =====================================================================
# 0. MERGE SELECTIVO CON RE-OCR
# =====================================================================


def merge_con_reocr(
    pages_dir: Path,
    reocr_dir: Path,
    run_dir: Path,
) -> Path:
    """Crea directorio merged dentro del run: usa re-OCR donde es mejor.

    Criterio: se usa la version re-OCR de una pagina cuando tiene mas
    lineas nuevas que perdidas respecto al original Y no es garbage loop.

    Para las demas paginas, se usa un symlink/copia del original.
    Retorna el path al directorio merged (que reemplaza a pages_dir
    para el resto del pipeline).
    """
    import shutil

    merged_dir = run_dir / "pages_merged"
    merged_dir.mkdir(exist_ok=True)

    reocr_pages_dir = reocr_dir / "pages"
    if not reocr_pages_dir.exists():
        logger.warning(f"No existe {reocr_pages_dir}, usando originales")
        return pages_dir

    # Comparar original vs re-OCR para cada pagina con re-OCR disponible
    reocr_results = sorted(reocr_pages_dir.rglob("result.txt"))
    reocr_labels = {r.parent.name: r for r in reocr_results}

    usadas_reocr = 0
    usadas_original = 0
    garbage = 0

    # Iterar todas las paginas originales
    for orig_page_dir in sorted(pages_dir.iterdir()):
        if not orig_page_dir.is_dir():
            continue
        label = orig_page_dir.name
        orig_result = orig_page_dir / "result.txt"
        if not orig_result.exists():
            continue

        dest_dir = merged_dir / label
        dest_dir.mkdir(exist_ok=True)
        dest_result = dest_dir / "result.txt"

        if label in reocr_labels:
            reocr_result = reocr_labels[label]
            orig_text = _read_text(orig_result)
            reocr_text = _read_text(reocr_result)

            # Detectar garbage (>30KB = loop repetitivo)
            if len(reocr_text) > 30000:
                shutil.copy2(orig_result, dest_result)
                usadas_original += 1
                garbage += 1
                continue

            # Comparar por lineas
            orig_lines = set(l.strip() for l in orig_text.split("\n") if l.strip())
            reocr_lines = set(l.strip() for l in reocr_text.split("\n") if l.strip())
            nuevas = len(reocr_lines - orig_lines)
            perdidas = len(orig_lines - reocr_lines)

            if nuevas > perdidas:
                shutil.copy2(reocr_result, dest_result)
                usadas_reocr += 1
            else:
                shutil.copy2(orig_result, dest_result)
                usadas_original += 1
        else:
            shutil.copy2(orig_result, dest_result)
            usadas_original += 1

    total = usadas_reocr + usadas_original
    logger.info(f"{total:,} paginas merged: {usadas_reocr} re-OCR, "
               f"{usadas_original} original ({garbage} garbage descartados)")
    return merged_dir


def filtrar_pages_por_tomo(pages_dir: Path, run_dir: Path, tomos_filter: set[str]) -> Path:
    import shutil

    filtered_dir = run_dir / "pages_filtered"
    filtered_dir.mkdir(exist_ok=True)

    copied = 0
    for page_dir in sorted(pages_dir.iterdir()):
        if not page_dir.is_dir():
            continue
        meta = _parse_tomo_page(page_dir / "result.txt")
        if meta["tomo"] not in tomos_filter:
            continue
        dest_dir = filtered_dir / page_dir.name
        dest_dir.mkdir(exist_ok=True)
        src_result = page_dir / "result.txt"
        if src_result.exists():
            shutil.copy2(src_result, dest_dir / "result.txt")
            copied += 1

    logger.info(f"{copied:,} paginas copiadas tras filtro de tomos")
    return filtered_dir


# =====================================================================
# ORQUESTADOR
# =====================================================================


def run_pipeline(
    output_base: Path,
    images_dir: Path | None = None,
    pages_dir: Path | None = None,
    tomos_filter: set[str] | None = None,
    yolo_model_path: Path = Path("yolov8s.pt"),
    yolo_conf: float = 0.25,
    yolo_imgsz: int = 1024,
    yolo_batch: int = 4,
    ocr_model: str = "deepseek-ai/DeepSeek-OCR",
    ocr_prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    ocr_device: str | None = None,
    ocr_dtype: str = "bfloat16",
    ocr_max_new_tokens: int = 4096,
    ocr_max_retries: int = 3,
    solo_ocr: bool = False,
    skip_entidades: bool = False,
    skip_red_personas: bool = False,
    ollama_url: str = "http://localhost:11434/api/generate",
    ollama_model: str = "qwen3.5:9b",
    max_entidades: int = 0,
    reocr_dir: Path | None = None,
) -> Path:
    # Crear directorio de corrida
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Activar log a archivo dentro del run
    _add_file_handler(run_dir / "pipeline.log")

    tiene_reocr = reocr_dir is not None and reocr_dir.exists()
    usa_pages_existentes = pages_dir is not None

    if solo_ocr:
        n_pasos = 0
        if not usa_pages_existentes:
            n_pasos += 1  # YOLO + OCR
        if tiene_reocr:
            n_pasos += 1
    else:
        n_pasos = 9
        if not usa_pages_existentes:
            n_pasos += 1
        if tiene_reocr:
            n_pasos += 1
        if not skip_entidades:
            n_pasos += 1
        if not skip_red_personas:
            n_pasos += 1
    paso = 0

    logger.info("=" * 60)
    logger.info("PIPELINE OCR COMPLETO")
    logger.info(f"Run: {run_dir.name}")
    logger.info(f"Log: {run_dir / 'pipeline.log'}")
    if usa_pages_existentes:
        logger.info(f"Entrada pages: {pages_dir}")
    else:
        logger.info(f"Entrada images: {images_dir}")
    if tomos_filter:
        logger.info(f"Tomos: {', '.join(sorted(tomos_filter))}")
    if tiene_reocr:
        logger.info(f"Re-OCR: {reocr_dir}")
    if solo_ocr:
        logger.info("Modo: --solo-ocr (YOLO + OCR, sin post-proceso)")
    else:
        if skip_entidades:
            logger.info("Entidades: DESACTIVADO (--skip-entidades)")
        else:
            logger.info(f"Entidades: {ollama_model} @ {ollama_url}")
        if skip_red_personas:
            logger.info("Red personas: DESACTIVADO (--skip-red-personas)")
    logger.info("=" * 60)
    t_start = time.time()

    effective_pages = pages_dir
    if effective_pages is None:
        if images_dir is None:
            raise ValueError("Debes indicar images_dir o pages_dir")
        paso += 1
        logger.info(f"[{paso}/{n_pasos}] Deteccion YOLO + OCR por recortes...")
        effective_pages = generar_pages_desde_yolo_ocr(
            images_dir=images_dir,
            run_dir=run_dir,
            yolo_model_path=yolo_model_path,
            tomos_filter=tomos_filter,
            yolo_conf=yolo_conf,
            yolo_imgsz=yolo_imgsz,
            yolo_batch=yolo_batch,
            ocr_model=ocr_model,
            ocr_prompt=ocr_prompt,
            ocr_device=ocr_device,
            ocr_dtype=ocr_dtype,
            ocr_max_new_tokens=ocr_max_new_tokens,
            ocr_max_retries=ocr_max_retries,
        )
    elif tomos_filter:
        paso += 1
        logger.info(f"[{paso}/{n_pasos}] Filtrando pages por tomo...")
        effective_pages = filtrar_pages_por_tomo(effective_pages, run_dir, tomos_filter)

    # Merge selectivo con re-OCR (si se proporciona)
    if tiene_reocr:
        paso += 1
        logger.info(f"[{paso}/{n_pasos}] Merge selectivo con re-OCR...")
        effective_pages = merge_con_reocr(effective_pages, reocr_dir, run_dir)

    # --solo-ocr: parar aqui, solo YOLO + OCR + guardado
    if solo_ocr:
        elapsed = time.time() - t_start
        _log_resumen_run(run_dir, elapsed)
        return run_dir

    # 1. Calidad OCR
    paso += 1
    logger.info(f"[{paso}/{n_pasos}] Evaluando calidad OCR...")
    df_calidad = flaggear_ocr(effective_pages, run_dir)

    # 2. Consolidar tomos
    paso += 1
    logger.info(f"[{paso}/{n_pasos}] Consolidando tomos (union inteligente)...")
    tomos_dir = consolidar_tomos(effective_pages, run_dir)

    # 3. Segmentar contratos
    paso += 1
    logger.info(f"[{paso}/{n_pasos}] Segmentando contratos...")
    df_seg = segmentar_tomos(tomos_dir, run_dir)

    # 4. Parsear campos
    paso += 1
    logger.info(f"[{paso}/{n_pasos}] Parseando campos estructurados...")
    df_parsed = parsear_contratos(df_seg)

    # 5. Cruzar flags OCR
    paso += 1
    logger.info(f"[{paso}/{n_pasos}] Cruzando flags OCR...")
    df_final = _cruzar_flags_ocr(df_parsed, df_calidad)

    # 6. Corregir secuencia id_num
    paso += 1
    logger.info(f"[{paso}/{n_pasos}] Corrigiendo secuencia id_num...")
    df_final = corregir_secuencia_ids(df_final)

    # 7. Re-segmentar contratos enterrados
    paso += 1
    logger.info(f"[{paso}/{n_pasos}] Re-segmentando contratos enterrados...")
    df_final = resegmentar_perdidos(df_final)

    # 8. Diagnosticar paginas para re-OCR
    paso += 1
    logger.info(f"[{paso}/{n_pasos}] Diagnosticando paginas con contratos perdidos...")
    diagnosticar_paginas_reocr(df_final, effective_pages, run_dir)

    # 9. Extraer entidades (opcional)
    if not skip_entidades:
        paso += 1
        logger.info(f"[{paso}/{n_pasos}] Extrayendo entidades (personas, naos, lugares, atributos)...")
        df_final = extraer_entidades(df_final, run_dir, ollama_url, ollama_model, max_entidades)

    out_path = run_dir / "compilado.xlsx"
    df_final.to_excel(out_path, index=False)

    # 10. Panelizar
    paso += 1
    logger.info(f"[{paso}/{n_pasos}] Panelizando compilado...")
    df_panel_src, df_headers = separar_headers(df_final)
    df_panel = panelizar(df_panel_src)
    panel_path = run_dir / "panel.xlsx"
    headers_path = run_dir / "panel_headers.xlsx"
    df_panel.to_excel(panel_path, index=False)
    df_headers.to_excel(headers_path, index=False)

    # Exportar multi-hoja (una hoja por tipo de entidad)
    multihoja_path = run_dir / "panel_por_tipo.xlsx"
    counts = exportar_panel_multihoja(df_panel, multihoja_path)

    n_personas = counts.get("personas", 0)
    n_naos = counts.get("naos", 0)
    n_lugares = counts.get("lugares", 0)
    logger.info(f"{len(df_panel):,} filas: {n_personas:,} personas, {n_naos:,} naos, {n_lugares:,} lugares")

    # 11. Red de personas
    if not skip_red_personas:
        paso += 1
        logger.info(f"[{paso}/{n_pasos}] Construyendo red de personas...")
        G, stats = construir_red(df_final)
        exportar_red_personas(G, stats, run_dir)
        logger.info(f"{stats['nodos']:,} nodos, {stats['aristas']:,} aristas")

    elapsed = time.time() - t_start
    _log_resumen_run(run_dir, elapsed)
    return run_dir


def _log_resumen_run(run_dir: Path, elapsed: float) -> None:
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETADO en {elapsed / 60:.1f} min")
    logger.info(f"Directorio: {run_dir}")
    logger.info("Archivos:")
    for f in sorted(run_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            if size < 1024 * 1024:
                label = f"{size / 1024:.0f} KB"
            else:
                label = f"{size / 1024 / 1024:.1f} MB"
            logger.info(f"  {f.relative_to(run_dir)} ({label})")
    logger.info("=" * 60)


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline OCR completo.")
    parser.add_argument(
        "--images-dir",
        default="data/preprocess_v2",
        help="Directorio con imagenes preprocesadas *_prep.* (default: data/preprocess_v2)",
    )
    parser.add_argument(
        "--pages-dir",
        default=None,
        help="Directorio con result.txt por pagina. Si se indica, salta YOLO+OCR y usa pages existentes.",
    )
    parser.add_argument(
        "--tomos",
        default=None,
        help='Lista de tomos separada por coma. Ej: "Tomo I,Tomo III,Tomo XVI"',
    )
    parser.add_argument(
        "--output-base",
        default="outputs",
        help="Directorio base para carpetas de corrida (default: outputs)",
    )
    parser.add_argument(
        "--solo-ocr",
        action="store_true",
        help="Solo ejecutar YOLO + OCR (sin post-proceso). Genera crops/, boxes_manifest.csv, ocr_boxes.csv y pages/.",
    )
    parser.add_argument(
        "--skip-entidades",
        action="store_true",
        help="Saltar extraccion de entidades via Ollama",
    )
    parser.add_argument(
        "--skip-red-personas",
        action="store_true",
        help="Saltar generacion de red_personas.* al final del pipeline",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8s.pt",
        help="Path al modelo YOLO para detectar boxes (default: yolov8s.pt)",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="Confidence threshold de YOLO (default: 0.25)",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=1024,
        help="Tamano de inferencia YOLO (default: 1024)",
    )
    parser.add_argument(
        "--yolo-batch",
        type=int,
        default=4,
        help="Batch de YOLO sobre paginas (default: 4)",
    )
    parser.add_argument(
        "--ocr-model",
        default="deepseek-ai/DeepSeek-OCR",
        help="Modelo DeepSeek OCR para recortes (default: deepseek-ai/DeepSeek-OCR)",
    )
    parser.add_argument(
        "--ocr-prompt",
        default="<image>\n<|grounding|>Convert the document to markdown.",
        help="Prompt OCR para los recortes",
    )
    parser.add_argument(
        "--ocr-device",
        default=None,
        help="cuda|cpu|mps (default: auto)",
    )
    parser.add_argument(
        "--ocr-dtype",
        default="bfloat16",
        help="bfloat16|float16|float32 (default: bfloat16)",
    )
    parser.add_argument(
        "--ocr-max-new-tokens",
        type=int,
        default=4096,
        help="Max tokens por recorte OCR (default: 4096)",
    )
    parser.add_argument(
        "--ocr-max-retries",
        type=int,
        default=3,
        help="Reintentos OCR por recorte (default: 3)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/generate",
        help="URL de la API de Ollama (default: http://localhost:11434/api/generate)",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen3.5:9b",
        help="Modelo de Ollama para extraccion (default: qwen3.5:9b)",
    )
    parser.add_argument(
        "--max-entidades",
        type=int,
        default=0,
        help="Limitar extraccion a los primeros N contratos (0 = todos)",
    )
    parser.add_argument(
        "--reocr-dir",
        default=None,
        help="Directorio con resultado de re-OCR (contiene pages/). "
             "Si se indica, el pipeline hace merge selectivo antes de procesar.",
    )
    args = parser.parse_args()
    tomos_filter = _parse_tomos_arg(args.tomos)

    run_pipeline(
        output_base=Path(args.output_base),
        images_dir=Path(args.images_dir) if args.images_dir else None,
        pages_dir=Path(args.pages_dir) if args.pages_dir else None,
        tomos_filter=tomos_filter,
        yolo_model_path=Path(args.yolo_model),
        yolo_conf=args.yolo_conf,
        yolo_imgsz=args.yolo_imgsz,
        yolo_batch=args.yolo_batch,
        ocr_model=args.ocr_model,
        ocr_prompt=args.ocr_prompt,
        ocr_device=args.ocr_device,
        ocr_dtype=args.ocr_dtype,
        ocr_max_new_tokens=args.ocr_max_new_tokens,
        ocr_max_retries=args.ocr_max_retries,
        solo_ocr=args.solo_ocr,
        skip_entidades=args.skip_entidades,
        skip_red_personas=args.skip_red_personas,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        max_entidades=args.max_entidades,
        reocr_dir=Path(args.reocr_dir) if args.reocr_dir else None,
    )
