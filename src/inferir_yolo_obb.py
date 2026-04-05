"""
Inferencia YOLO OBB sobre páginas preprocesadas.

Modo principal (segmentación para OCR):
    py src/inferir_yolo_obb.py --images-dir data/preprocess_v2 --out outputs/segmentacion_obb

    Genera:
      outputs/segmentacion_obb/
      ├── manifest.csv          (página, box_id, clase, confianza, coordenadas)
      └── crops/                (recortes individuales por contrato/continuación)
          ├── Tomo I_p0001_box01_contrato.png
          └── ...

Modo visualización (debug, muestra aleatoria):
    py src/inferir_yolo_obb.py --visualizar --images-dir data/preprocess_v2 --n 100 --out outputs/inferencia_obb
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Clases del modelo
NOMBRES = {0: "contrato", 1: "continuacion"}
COLORES = {0: (0, 200, 0), 1: (0, 140, 255)}


# -------------------- Extracción de boxes --------------------

def _extraer_boxes(resultado) -> list[dict]:
    """Extrae boxes OBB de un resultado de YOLO como lista de dicts."""
    if resultado.obb is None or len(resultado.obb) == 0:
        return []

    obb = resultado.obb
    boxes = []
    for i in range(len(obb)):
        puntos = obb.xyxyxyxy[i].cpu().numpy()  # (4, 2) float
        cls_id = int(obb.cls[i].cpu().item())
        conf = float(obb.conf[i].cpu().item())
        boxes.append({
            "cls_id": cls_id,
            "cls_nombre": NOMBRES.get(cls_id, str(cls_id)),
            "conf": conf,
            "puntos": puntos,  # 4 esquinas (x, y)
        })

    # Ordenar de arriba a abajo (por coordenada Y mínima)
    boxes.sort(key=lambda b: b["puntos"][:, 1].min())
    return boxes


def _recortar_box(img: np.ndarray, puntos: np.ndarray, padding: int = 5) -> np.ndarray:
    """Recorta la región del OBB con warp perspectivo real.

    Reordena las 4 esquinas, estima un rectángulo destino y aplica
    `warpPerspective` para extraer el contenido orientado de la caja.
    """
    pts = np.asarray(puntos, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Se esperaban 4 puntos (4,2), recibido {pts.shape}")

    # Orden estándar: top-left, top-right, bottom-right, bottom-left.
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    height_left = np.linalg.norm(ordered[3] - ordered[0])

    width = max(1, int(round(max(width_top, width_bottom))))
    height = max(1, int(round(max(height_left, height_right))))

    dst = np.array(
        [
            [padding, padding],
            [padding + width - 1, padding],
            [padding + width - 1, padding + height - 1],
            [padding, padding + height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    out_w = width + 2 * padding
    out_h = height + 2 * padding
    crop = cv2.warpPerspective(
        img,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return crop


# -------------------- Modo batch (segmentación → OCR) --------------------

def segmentar_batch(
    images_dir: str | Path,
    model_path: str | Path,
    out_dir: str | Path,
    *,
    glob_pattern: str = "*_prep.png",
    conf: float = 0.25,
    imgsz: int = 640,
    padding: int = 5,
) -> Path:
    """
    Ejecuta inferencia OBB sobre todas las páginas y exporta recortes + manifiesto.

    Args:
        images_dir: directorio con *_prep.png
        model_path: ruta al modelo .pt
        out_dir: directorio de salida (se crean crops/ y manifest.csv)
        glob_pattern: patrón de imágenes
        conf: umbral de confianza
        imgsz: tamaño de imagen para inferencia
        padding: píxeles de margen alrededor de cada recorte

    Returns:
        Path al manifest.csv generado.
    """
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    patron = str(images_dir / glob_pattern)
    todas = sorted(glob.glob(patron))
    if not todas:
        raise FileNotFoundError(f"No se encontraron imágenes con: {patron}")

    print(f"Páginas encontradas: {len(todas)}")
    print(f"Cargando modelo: {model_path}")
    modelo = YOLO(str(model_path))

    manifest_path = out_dir / "manifest.csv"
    if manifest_path.exists():
        manifest_path.unlink()
    fieldnames = [
        "pagina", "box_id", "clase", "confianza",
        "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
        "crop_path",
    ]

    total_crops = 0
    total_paginas_con_det = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, ruta in enumerate(todas):
            nombre = os.path.basename(ruta)
            stem = Path(ruta).stem.replace("_prep", "")
            img = cv2.imread(ruta)
            if img is None:
                print(f"  [{i+1}/{len(todas)}] ERROR leyendo {nombre}")
                continue

            resultados = modelo(img, imgsz=imgsz, conf=conf, verbose=False)
            boxes = _extraer_boxes(resultados[0])

            if boxes:
                total_paginas_con_det += 1

            for box_idx, box in enumerate(boxes, 1):
                crop = _recortar_box(img, box["puntos"], padding=padding)
                crop_name = f"{stem}_box{box_idx:02d}_{box['cls_nombre']}.png"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                puntos = box["puntos"].astype(int)
                writer.writerow({
                    "pagina": nombre,
                    "box_id": box_idx,
                    "clase": box["cls_nombre"],
                    "confianza": f"{box['conf']:.4f}",
                    "x1": puntos[0][0], "y1": puntos[0][1],
                    "x2": puntos[1][0], "y2": puntos[1][1],
                    "x3": puntos[2][0], "y3": puntos[2][1],
                    "x4": puntos[3][0], "y4": puntos[3][1],
                    "crop_path": str(crop_path.relative_to(out_dir)),
                })
                total_crops += 1

            if (i + 1) % 100 == 0 or (i + 1) == len(todas):
                print(f"  [{i+1}/{len(todas)}] {total_crops} recortes acumulados...")

    print(f"\nResumen:")
    print(f"  Páginas procesadas: {len(todas)}")
    print(f"  Páginas con detecciones: {total_paginas_con_det}")
    print(f"  Recortes exportados: {total_crops}")
    print(f"  Manifiesto: {manifest_path}")
    print(f"  Recortes en: {crops_dir}")
    return manifest_path


# -------------------- Modo visualización --------------------

def _dibujar_obb(img, resultado):
    """Dibuja oriented bounding boxes sobre la imagen."""
    vis = img.copy()
    boxes = _extraer_boxes(resultado)
    for box in boxes:
        puntos = box["puntos"].astype(int)
        color = COLORES.get(box["cls_id"], (255, 255, 255))
        cv2.polylines(vis, [puntos], isClosed=True, color=color, thickness=3)
        label = f"{box['cls_nombre']} {box['conf']:.2f}"
        x_min, y_min = puntos.min(axis=0)
        cv2.putText(vis, label, (x_min, max(y_min - 8, 20)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def visualizar(
    images_dir: str | Path,
    model_path: str | Path,
    out_dir: str | Path,
    *,
    n: int = 100,
    glob_pattern: str = "*_prep.png",
    conf: float = 0.25,
    imgsz: int = 640,
    seed: int = 42,
) -> None:
    """Modo visualización: muestra aleatoria con boxes dibujadas."""
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patron = str(images_dir / glob_pattern)
    todas = sorted(glob.glob(patron))
    if not todas:
        print(f"No se encontraron imágenes con: {patron}")
        return

    random.seed(seed)
    n = min(n, len(todas))
    seleccion = sorted(random.sample(todas, n))
    print(f"Seleccionadas {n}/{len(todas)} imágenes (seed={seed})")

    print(f"Cargando modelo: {model_path}")
    modelo = YOLO(str(model_path))

    total_det = 0
    for i, ruta in enumerate(seleccion):
        nombre = os.path.basename(ruta)
        img = cv2.imread(ruta)
        if img is None:
            print(f"  [{i+1}/{n}] ERROR leyendo {nombre}")
            continue

        resultados = modelo(img, imgsz=imgsz, conf=conf, verbose=False)
        n_det = len(resultados[0].obb) if resultados[0].obb is not None else 0
        total_det += n_det

        vis = _dibujar_obb(img, resultados[0])
        salida = out_dir / nombre.replace("_prep.png", "_obb.png")
        cv2.imwrite(str(salida), vis)
        print(f"  [{i+1}/{n}] {nombre}: {n_det} detecciones")

    print(f"\nResumen: {total_det} detecciones en {n} imágenes")
    print(f"Visualizaciones en: {out_dir}")


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inferencia YOLO OBB: segmentación de contratos para OCR"
    )
    parser.add_argument("--images-dir", default="data/preprocess_v2",
                        help="Directorio con imágenes *_prep.png")
    parser.add_argument("--model", default="models/yolo_obb_v1/weights/best.pt",
                        help="Ruta al modelo entrenado")
    parser.add_argument("--out", default="outputs/segmentacion_obb",
                        help="Directorio de salida")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Umbral de confianza (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tamaño de imagen para inferencia (default: 640)")
    parser.add_argument("--padding", type=int, default=5,
                        help="Píxeles de margen en recortes (default: 5)")
    parser.add_argument("--glob", default="*_prep.png",
                        help="Patrón glob de imágenes (default: *_prep.png)")

    # Modo visualización
    parser.add_argument("--visualizar", action="store_true",
                        help="Modo debug: dibuja boxes sobre muestra aleatoria")
    parser.add_argument("--n", type=int, default=100,
                        help="Imágenes aleatorias en modo --visualizar")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para modo --visualizar")

    args = parser.parse_args()

    if args.visualizar:
        visualizar(
            args.images_dir, args.model, args.out,
            n=args.n, glob_pattern=args.glob, conf=args.conf,
            imgsz=args.imgsz, seed=args.seed,
        )
    else:
        segmentar_batch(
            args.images_dir, args.model, args.out,
            glob_pattern=args.glob, conf=args.conf,
            imgsz=args.imgsz, padding=args.padding,
        )


if __name__ == "__main__":
    main()
