# YOLO OBB v1 — Segmentación de contratos

Modelo YOLOv8s-OBB entrenado para detectar contratos y continuaciones en
páginas escaneadas del Catálogo de los Fondos Americanos.

## Clases

| ID | Nombre        | Descripción                              |
|----|---------------|------------------------------------------|
| 0  | contrato      | Inicio de un contrato nuevo              |
| 1  | continuacion  | Continuación de contrato de página previa|

## Estructura

```
yolo_obb_v1/
├── train_obb.py      # Script de entrenamiento (incluye augmentation por rotación)
├── data.yaml         # Config del dataset YOLO
├── args.yaml         # Hiperparámetros del run de entrenamiento
├── labels/           # Anotaciones OBB (trackeadas en git)
│   ├── train/        # 2380 labels (340 base × 7 variantes de rotación)
│   └── val/          # 60 labels (sin augmentation)
└── weights/
    └── best.pt       # Peso entrenado (~23 MB, NO en git)
```

## Reproducir entrenamiento

### Requisitos
- `ultralytics`, `torch`, `psutil`
- GPU con CUDA (entrenado en RTX 4070 SUPER, 12GB VRAM)
- Dataset de imágenes en `outputs/yolo_obb_dataset/images/`

### Generar dataset + entrenar
```bash
# Desde la raíz del proyecto:
python models/yolo_obb_v1/train_obb.py \
    --source-dataset outputs/yolo_obb_dataset \
    --angles 3 5 8
```

El script:
1. Convierte labels estándar a formato OBB (8 puntos)
2. Genera rotaciones ±3°, ±5°, ±8° con labels transformados (on-disk augmentation)
3. Entrena YOLOv8s-OBB (75 epochs, batch=4, imgsz=640, patience=15)
4. Ultralytics aplica augmentations adicionales on-the-fly: mosaic, hsv, translate, scale, fliplr, randaugment, erasing

### Inferencia
```bash
python src/inferir_yolo_obb.py --model models/yolo_obb_v1/weights/best.pt --n 100
```

## Métricas
Ver `args.yaml` para la configuración completa del entrenamiento.
Los resultados detallados (curvas PR, matrices de confusión) están en
`runs/obb/runs/obb/v1_obb_augmented/` (no trackeado en git).
