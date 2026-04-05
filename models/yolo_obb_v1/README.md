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
- Dataset de imágenes con labels estándar en `outputs/yolo_dataset/` (exportado desde Label Studio via `scripts/labelstudio_sync.py`)

### Generar dataset OBB + entrenar
```bash
# Desde la raíz del proyecto:
python models/yolo_obb_v1/train_obb.py \
    --source-dataset outputs/yolo_dataset \
    --angles 3 5 8
```

El script ejecuta dos pasos:

1. **Augmentation** (genera `outputs/yolo_obb_dataset/`):
   - Convierte labels estándar (cx, cy, w, h) a formato OBB (8 puntos)
   - Genera rotaciones ±3°, ±5°, ±8° con labels transformados
   - Train: 340 base × 7 variantes = 2380 imágenes; Val: 60 sin augmentation

2. **Entrenamiento** (genera `runs/obb/.../weights/best.pt`):
   - YOLOv8s-OBB, 75 epochs, batch=4, imgsz=640, patience=15
   - Augmentations on-the-fly de Ultralytics: mosaic, hsv, translate, scale, fliplr, randaugment, erasing
   - Al terminar, copia `best.pt` automáticamente a `models/yolo_obb_v1/weights/`

Flags útiles:
```bash
# Solo augmentación (sin entrenar)
python models/yolo_obb_v1/train_obb.py --source-dataset outputs/yolo_dataset --skip-train

# Solo entrenar (reusar dataset OBB existente)
python models/yolo_obb_v1/train_obb.py --skip-augment

# Cambiar hiperparámetros
python models/yolo_obb_v1/train_obb.py --source-dataset outputs/yolo_dataset --epochs 100 --batch 8
```

### Inferencia
```bash
# 100 páginas aleatorias con visualización de boxes
python src/inferir_yolo_obb.py --n 100 --out outputs/inferencia_obb

# Página específica
python src/inferir_yolo_obb.py --images-dir data/preprocess_v2 --n 1 --out outputs/inferencia_obb
```

El modelo se carga desde `models/yolo_obb_v1/weights/best.pt` por defecto.

## Métricas
Ver `args.yaml` para la configuración completa del entrenamiento.
Los resultados detallados (curvas PR, matrices de confusión) se generan en
`runs/obb/` durante el entrenamiento (no trackeado en git).
