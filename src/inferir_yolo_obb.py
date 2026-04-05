"""
Inferencia YOLO OBB sobre páginas preprocesadas aleatorias.
Guarda imágenes con las cajas OBB dibujadas.

Uso:
    py src/inferir_yolo_obb.py --images-dir data/preprocess_v2 --n 100 --out outputs/inferencia_obb
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np
from ultralytics import YOLO

# Colores por clase (BGR)
COLORES = {
    0: (0, 200, 0),    # contrato  → verde
    1: (0, 140, 255),  # continuacion → naranja
}
NOMBRES = {0: "contrato", 1: "continuacion"}


def dibujar_obb(img, resultado):
    """Dibuja oriented bounding boxes sobre la imagen."""
    if resultado.obb is None or len(resultado.obb) == 0:
        return img

    vis = img.copy()
    obb = resultado.obb

    for i in range(len(obb)):
        # xyxyxyxy: 4 esquinas (x1,y1,x2,y2,x3,y3,x4,y4)
        puntos = obb.xyxyxyxy[i].cpu().numpy().astype(int)  # (4, 2)
        cls_id = int(obb.cls[i].cpu().item())
        conf = float(obb.conf[i].cpu().item())
        color = COLORES.get(cls_id, (255, 255, 255))
        nombre = NOMBRES.get(cls_id, str(cls_id))

        # Dibujar polígono
        cv2.polylines(vis, [puntos], isClosed=True, color=color, thickness=3)

        # Etiqueta
        label = f"{nombre} {conf:.2f}"
        x_min, y_min = puntos.min(axis=0)
        cv2.putText(vis, label, (x_min, max(y_min - 8, 20)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Inferencia YOLO OBB sobre páginas _prep")
    parser.add_argument("--images-dir", default="data/preprocess_v2",
                        help="Directorio con imágenes *_prep.png")
    parser.add_argument("--model", default="models/yolo_obb_v1/weights/best.pt",
                        help="Ruta al modelo entrenado")
    parser.add_argument("--n", type=int, default=100,
                        help="Número de imágenes aleatorias")
    parser.add_argument("--out", default="outputs/inferencia_obb",
                        help="Directorio de salida")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Umbral de confianza")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tamaño de imagen para inferencia")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad")
    args = parser.parse_args()

    # Buscar imágenes
    patron = os.path.join(args.images_dir, "*_prep.png")
    todas = sorted(glob.glob(patron))
    print(f"Imágenes encontradas: {len(todas)}")

    if len(todas) == 0:
        print(f"No se encontraron imágenes con patrón: {patron}")
        return

    # Selección aleatoria
    random.seed(args.seed)
    n = min(args.n, len(todas))
    seleccion = random.sample(todas, n)
    seleccion.sort()
    print(f"Seleccionadas {n} imágenes al azar (seed={args.seed})")

    # Cargar modelo
    print(f"Cargando modelo: {args.model}")
    modelo = YOLO(args.model)

    # Crear directorio de salida
    os.makedirs(args.out, exist_ok=True)

    # Inferencia
    total_detecciones = 0
    for i, ruta in enumerate(seleccion):
        nombre = os.path.basename(ruta)
        img = cv2.imread(ruta)
        if img is None:
            print(f"  [{i+1}/{n}] ERROR leyendo {nombre}")
            continue

        resultados = modelo(img, imgsz=args.imgsz, conf=args.conf, verbose=False)
        resultado = resultados[0]

        n_det = len(resultado.obb) if resultado.obb is not None else 0
        total_detecciones += n_det

        vis = dibujar_obb(img, resultado)

        salida = os.path.join(args.out, nombre.replace("_prep.png", "_obb.png"))
        cv2.imwrite(salida, vis)
        print(f"  [{i+1}/{n}] {nombre}: {n_det} detecciones → {os.path.basename(salida)}")

    print(f"\nResumen: {total_detecciones} detecciones en {n} imágenes")
    print(f"Resultados en: {args.out}")


if __name__ == "__main__":
    main()
