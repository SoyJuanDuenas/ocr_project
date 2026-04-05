"""
Pipeline completo: augmentación con rotaciones + entrenamiento YOLOv8s-OBB.

Uso:
    python train_obb.py --source-dataset outputs/yolo_dataset
    python train_obb.py --source-dataset outputs/yolo_dataset --angles 3 5 8 --skip-augment
    python train_obb.py --source-dataset outputs/yolo_dataset --epochs 100 --batch 4 --workers 2

Log en: logs/train_obb.log
"""
from __future__ import annotations

import argparse
import threading
import datetime
import os
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np

LOG_PATH = "logs/train_obb.log"
MONITOR_INTERVAL = 10

os.makedirs("logs", exist_ok=True)


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# -------------------- Augmentación --------------------

def _yolo_to_corners(cx, cy, w, h):
    """Convierte label YOLO (cx, cy, w, h) normalizado a 4 esquinas normalizadas."""
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy - h / 2
    x3, y3 = cx + w / 2, cy + h / 2
    x4, y4 = cx - w / 2, cy + h / 2
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]


def _rotate_image(img, angle_deg):
    """Rota imagen alrededor del centro, fondo blanco (255)."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255) if len(img.shape) == 3 else 255)
    return rotated, M


def _rotate_corners(corners, M, img_w, img_h):
    """Rota 4 esquinas normalizadas usando la matriz de rotación. Devuelve esquinas normalizadas."""
    result = []
    for (nx, ny) in corners:
        px, py = nx * img_w, ny * img_h
        rx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ry = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        result.append((rx / img_w, ry / img_h))
    return result


def _clip_corners(corners):
    """Recorta esquinas al rango [0, 1]."""
    return [(max(0.0, min(1.0, x)), max(0.0, min(1.0, y))) for (x, y) in corners]


def _box_visible(corners, min_area_frac=0.001):
    """Verifica que la caja rotada tenga area minima visible tras recorte."""
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return w * h > min_area_frac


def augment_dataset(source_dir: Path, output_dir: Path, angles: list[int]):
    """
    Genera dataset OBB augmentado a partir de dataset YOLO regular.

    Por cada imagen de train, genera la original (con labels en formato OBB)
    + rotaciones positivas y negativas para cada ángulo.
    Val se copia sin augmentar (labels convertidos a OBB).
    """
    src_imgs_train = source_dir / "images" / "train"
    src_lbls_train = source_dir / "labels" / "train"
    src_imgs_val = source_dir / "images" / "val"
    src_lbls_val = source_dir / "labels" / "val"

    out_imgs_train = output_dir / "images" / "train"
    out_lbls_train = output_dir / "labels" / "train"
    out_imgs_val = output_dir / "images" / "val"
    out_lbls_val = output_dir / "labels" / "val"

    for d in [out_imgs_train, out_lbls_train, out_imgs_val, out_lbls_val]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Val: copiar sin augmentar, convertir labels a OBB ---
    val_imgs = sorted(src_imgs_val.glob("*.png"))
    log(f"Val: copiando {len(val_imgs)} imagenes sin augmentar")
    for img_path in val_imgs:
        shutil.copy2(str(img_path), str(out_imgs_val / img_path.name))
        lbl_path = src_lbls_val / img_path.with_suffix(".txt").name
        if lbl_path.exists():
            lines = lbl_path.read_text().strip().split("\n")
            obb_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = parts[0]
                cx, cy, w, h = map(float, parts[1:5])
                corners = _yolo_to_corners(cx, cy, w, h)
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
                obb_lines.append(f"{cls} {coords}")
            (out_lbls_val / lbl_path.name).write_text("\n".join(obb_lines) + "\n")

    # --- Train: original + rotaciones ---
    train_imgs = sorted(src_imgs_train.glob("*.png"))
    rotation_angles = []
    for a in angles:
        rotation_angles.extend([a, -a])

    total_expected = len(train_imgs) * (1 + len(rotation_angles))
    log(f"Train: {len(train_imgs)} originales x {1 + len(rotation_angles)} "
        f"(original + {len(rotation_angles)} rotaciones) = {total_expected} imagenes")

    total_imgs = 0
    total_boxes = 0

    for i, img_path in enumerate(train_imgs):
        stem = img_path.stem
        lbl_path = src_lbls_train / f"{stem}.txt"

        # Leer imagen
        img = cv2.imread(str(img_path))
        if img is None:
            log(f"  WARN: no se pudo leer {img_path}")
            continue
        img_h, img_w = img.shape[:2]

        # Leer labels YOLO
        boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = parts[0]
                    cx, cy, w, h = map(float, parts[1:5])
                    boxes.append((cls, cx, cy, w, h))

        # 1) Original (convertir a OBB)
        shutil.copy2(str(img_path), str(out_imgs_train / img_path.name))
        obb_lines = []
        for cls, cx, cy, w, h in boxes:
            corners = _yolo_to_corners(cx, cy, w, h)
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
            obb_lines.append(f"{cls} {coords}")
        (out_lbls_train / f"{stem}.txt").write_text("\n".join(obb_lines) + "\n")
        total_imgs += 1
        total_boxes += len(obb_lines)

        # 2) Rotaciones
        for angle in rotation_angles:
            sign = "+" if angle > 0 else ""
            rot_stem = f"{stem}_rot{sign}{angle}"

            rotated_img, M = _rotate_image(img, angle)
            cv2.imwrite(str(out_imgs_train / f"{rot_stem}.png"), rotated_img)

            rot_lines = []
            for cls, cx, cy, w, h in boxes:
                corners = _yolo_to_corners(cx, cy, w, h)
                rot_corners = _rotate_corners(corners, M, img_w, img_h)
                rot_corners = _clip_corners(rot_corners)
                if _box_visible(rot_corners):
                    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in rot_corners)
                    rot_lines.append(f"{cls} {coords}")

            (out_lbls_train / f"{rot_stem}.txt").write_text("\n".join(rot_lines) + "\n")
            total_imgs += 1
            total_boxes += len(rot_lines)

        if (i + 1) % 50 == 0:
            log(f"  Augmentacion: {i+1}/{len(train_imgs)} imagenes procesadas")

    # data.yaml
    yaml_content = f"""path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: contrato
  1: continuacion
"""
    (output_dir / "data.yaml").write_text(yaml_content)

    log(f"Augmentacion completa: {total_imgs} imagenes, {total_boxes} boxes")
    return total_imgs, total_boxes


# -------------------- Monitor de recursos --------------------

def monitor_resources(stop_event):
    import torch
    import psutil

    peak_ram = 0
    peak_vram = 0

    while not stop_event.is_set():
        try:
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / 1e9
            ram_avail_gb = ram.available / 1e9
            vram_used_gb = torch.cuda.memory_allocated(0) / 1e9
            vram_reserved_gb = torch.cuda.memory_reserved(0) / 1e9

            peak_ram = max(peak_ram, ram_used_gb)
            peak_vram = max(peak_vram, vram_reserved_gb)

            log(f"MONITOR | RAM usada: {ram_used_gb:.1f}GB libre: {ram_avail_gb:.1f}GB | "
                f"VRAM alloc: {vram_used_gb:.1f}GB reservada: {vram_reserved_gb:.1f}GB | "
                f"PICOS RAM: {peak_ram:.1f}GB VRAM: {peak_vram:.1f}GB")
        except Exception as e:
            log(f"MONITOR ERROR: {e}")

        stop_event.wait(MONITOR_INTERVAL)

    log(f"MONITOR FINAL | Pico RAM: {peak_ram:.1f}GB | Pico VRAM: {peak_vram:.1f}GB")


# -------------------- Entrenamiento --------------------

def train(dataset_dir: Path, epochs: int, batch: int, workers: int,
          patience: int, imgsz: int, project: str, name: str):
    import torch
    import psutil

    ram = psutil.virtual_memory()
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    log(f"RAM total: {ram.total/1e9:.1f}GB usada: {ram.used/1e9:.1f}GB libre: {ram.available/1e9:.1f}GB")
    log(f"Params: batch={batch}, workers={workers}, imgsz={imgsz}, epochs={epochs}, patience={patience}")

    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event,), daemon=True)
    monitor_thread.start()
    log("Monitor de recursos iniciado")

    try:
        from ultralytics import YOLO

        log("Cargando modelo yolov8s-obb.pt...")
        model = YOLO("yolov8s-obb.pt")
        log("Modelo cargado. Iniciando entrenamiento...")

        results = model.train(
            data=str(dataset_dir / "data.yaml"),
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch,
            device=0,
            workers=workers,
            project=project,
            name=name,
            exist_ok=True,
            verbose=True,
        )

        log("=" * 60)
        log("ENTRENAMIENTO COMPLETADO")
        for k, v in results.results_dict.items():
            log(f"  {k}: {v}")
        log("=" * 60)

        # Copiar best.pt a models/yolo_obb_v1/weights/ para uso en inferencia
        best_src = Path(project) / name / "weights" / "best.pt"
        best_dst = Path("models/yolo_obb_v1/weights/best.pt")
        if best_src.exists():
            best_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_src, best_dst)
            log(f"best.pt copiado a {best_dst}")

    finally:
        stop_event.set()
        monitor_thread.join(timeout=5)


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Augmentacion con rotaciones + entrenamiento YOLOv8s-OBB"
    )
    # Augmentacion
    parser.add_argument("--source-dataset", type=Path, default=Path("outputs/yolo_dataset"),
                        help="Dataset YOLO regular de entrada (con images/ y labels/)")
    parser.add_argument("--obb-dataset", type=Path, default=Path("outputs/yolo_obb_dataset"),
                        help="Directorio de salida para dataset OBB augmentado")
    parser.add_argument("--angles", type=int, nargs="+", default=[3, 5, 8],
                        help="Angulos de rotacion en grados (se aplican +/-)")
    parser.add_argument("--skip-augment", action="store_true",
                        help="Saltar augmentacion y usar dataset OBB existente")

    # Entrenamiento
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs/obb")
    parser.add_argument("--name", default="v1_obb_augmented")
    parser.add_argument("--skip-train", action="store_true",
                        help="Solo augmentar, no entrenar")

    args = parser.parse_args()

    log("=" * 60)
    log("INICIO pipeline augmentacion + entrenamiento OBB")
    log(f"PID: {os.getpid()}")

    # Paso 1: Augmentacion
    if not args.skip_augment:
        log(f"Augmentando desde {args.source_dataset} -> {args.obb_dataset}")
        log(f"Angulos: +/-{args.angles}")
        augment_dataset(args.source_dataset, args.obb_dataset, args.angles)
    else:
        log(f"Augmentacion saltada. Usando dataset existente: {args.obb_dataset}")

    # Paso 2: Entrenamiento
    if not args.skip_train:
        train(args.obb_dataset, args.epochs, args.batch, args.workers,
              args.patience, args.imgsz, args.project, args.name)
    else:
        log("Entrenamiento saltado.")

    log("Fin del script.")


if __name__ == "__main__":
    main()
