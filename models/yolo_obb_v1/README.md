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
models/yolo_obb_v1/
├── train_obb.py      # Entrenamiento (augmentation por rotación + YOLO)
├── data.yaml         # Config del dataset YOLO OBB (generado por train_obb.py)
├── args.yaml         # Hiperparámetros del run de entrenamiento
├── labels/           # Anotaciones OBB (trackeadas en git, generadas por train_obb.py)
│   ├── train/        # 2380 labels (340 base × 7 variantes de rotación)
│   └── val/          # 60 labels (sin augmentation)
└── weights/
    └── best.pt       # Peso entrenado (~23 MB, NO en git)
```

## Datos de entrenamiento

Los datos fuente están en `data/segmentation/` (ground truth anotado en Label Studio):

```
data/segmentation/
├── images/           # Imágenes *_seg.png para segmentación (NO en git)
│   ├── train/        # 340 imágenes base
│   └── val/          # 60 imágenes
├── prelabels/        # Pre-labels heurísticas (input a Label Studio)
├── labels/           # Ground truth de Label Studio (SÍ en git)
│   ├── train/*.txt   # Formato: class cx cy w h
│   └── val/*.txt
└── data.yaml
```

## Flujo completo desde cero

```bash
# 1. Generar pre-labels heurísticas sobre imágenes de segmentación
python scripts/boxes_from_heuristic.py \
    --images-dir data/segmentation/images/train

# 2. Importar a Label Studio para revisión humana
python scripts/labelstudio_sync.py serve --images-dir data/segmentation/images/train
python scripts/labelstudio_sync.py import \
    --proposed data/segmentation/prelabels/contract_boxes_heuristic.csv \
    --images-dir data/segmentation/images/train

# 3. Revisar y corregir en Label Studio → exportar ground truth
python scripts/labelstudio_sync.py export \
    --project-id <id> --output data/segmentation/labels/

# 4. Entrenar YOLO OBB (augmentation + training)
python models/yolo_obb_v1/train_obb.py \
    --source-dataset data/segmentation --angles 3 5 8

# 5. Inferencia sobre páginas preprocesadas
python src/inferir_yolo_obb.py --n 100 --out outputs/inferencia_obb
```

## Entrenamiento

### Requisitos
- `ultralytics`, `torch`, `psutil`
- GPU con CUDA (entrenado en RTX 4070 SUPER, 12GB VRAM)
- Imágenes en `data/segmentation/images/`

### Generar dataset OBB + entrenar
```bash
python models/yolo_obb_v1/train_obb.py \
    --source-dataset data/segmentation --angles 3 5 8
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
python models/yolo_obb_v1/train_obb.py --source-dataset data/segmentation --skip-train

# Solo entrenar (reusar dataset OBB existente)
python models/yolo_obb_v1/train_obb.py --skip-augment

# Cambiar hiperparámetros
python models/yolo_obb_v1/train_obb.py --source-dataset data/segmentation --epochs 100 --batch 8
```

### Inferencia
```bash
# 100 páginas aleatorias con visualización de boxes
python src/inferir_yolo_obb.py --n 100 --out outputs/inferencia_obb
```

El modelo se carga desde `models/yolo_obb_v1/weights/best.pt` por defecto.

## Métricas
Ver `args.yaml` para la configuración completa del entrenamiento.
Los resultados detallados (curvas PR, matrices de confusión) se generan en
`runs/obb/` durante el entrenamiento (no trackeado en git).
