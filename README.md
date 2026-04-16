# OCR Pipeline — Catálogo de los Fondos Americanos

Pipeline para digitalizar, transcribir y extraer información estructurada de los 16 tomos del *Catálogo de los Fondos Americanos del Archivo de Protocolos de Sevilla* (siglos XV–XVI, ~26,000 contratos notariales).

## Requisitos

- Python 3.10+
- GPU NVIDIA con CUDA (probado en RTX 4070 SUPER, 12 GB VRAM) para OCR
- [Ollama](https://ollama.com/) con modelo `qwen2.5:7b` para extracción de entidades

```bash
pip install -r requirements.txt
```

## Pipeline

| Paso | Script | Descripción |
|------|--------|-------------|
| 1 | `src/preprocess.py` | Preprocesamiento OCR con perfiles por tomo |
| 1b | `src/inferir_yolo_obb.py` | Segmentación visual con YOLO OBB (contratos + continuaciones) |
| 2 | `src/ocr_model_deepseek.py` | OCR por lotes con DeepSeek-OCR |
| 3 | `src/pipeline.py` | Orquestador post-OCR: consolidación, segmentación, parseo, corrección de secuencia, diagnóstico y extracción de entidades |
| 4 | `src/panelizar.py` | Conversión a formato panel largo (una fila por entidad) |
| 5 | `src/red_personas.py` | Red de co-ocurrencia de personas (se ejecuta automáticamente al final de `src/pipeline.py`; también puede correrse aparte) |

Módulos auxiliares:
- `src/parseo_compilado.py` — parser de campos estructurados, importado por `pipeline.py`
- `models/yolo_obb_v1/` — modelo YOLO OBB entrenado, labels y script de entrenamiento
- `scripts/` — herramientas auxiliares (tuning de filtros, heurísticas, Label Studio)

## Uso rápido

```bash
# 1. Preprocesar imágenes (con perfiles por tomo hardcodeados)
py src/preprocess.py --in data/raw --out data/preprocessed --target-dpi 300 \
   --bg-ksize 31 --bin sauvola --sauvola-w 31 --sauvola-k 0.45 \
   --close 3 --denoise-ksize 3 --zoom 1.15

# 1b. Segmentación visual con YOLO OBB
py src/inferir_yolo_obb.py --visualizar \
   --model models/yolo_obb_v1/weights/best.pt \
   --n 100 --out outputs/inferencia_obb

# 2. OCR por lotes
py src/ocr_model_deepseek.py --images-dir data/preprocess_v2 --glob "Tomo*_prep.png" --out outputs

# 3. Pipeline post-OCR completo
py src/pipeline.py                                          # con entidades
py src/pipeline.py --skip-entidades                         # sin entidades
py src/pipeline.py --skip-red-personas                      # sin exportar red_personas.*

# 4. Panelizar
py src/panelizar.py --compilado outputs/run_XXX/compilado.xlsx

# 5. Red de personas (opcional; el pipeline principal ya la genera)
py src/red_personas.py --compilado outputs/run_XXX/compilado.xlsx
```

## Estructura de salidas

```
outputs/run_YYYYMMDD_HHMMSS/
├── calidad_ocr.csv
├── tomos_txt/*.txt          (16 tomos consolidados)
├── contratos_segmentados.xlsx
├── compilado.xlsx           (dataset final con entidades)
├── panel.xlsx               (formato largo)
├── red_personas.gexf        (red para Gephi)
├── red_personas_nodos.csv
├── red_personas_aristas.csv
└── red_personas_stats.txt
```

## Resultados

- **7,016** páginas procesadas en 16 tomos
- **26,037** contratos segmentados (99.98% parse rate)
- **111** contratos recuperados vía re-OCR focalizado
- **26,447** personas únicas, **1,876** naos, **4,015** lugares extraídos
- Red de co-ocurrencia: 26,447 nodos, 123,846 aristas, componente gigante 92.9%

## Herramientas auxiliares (scripts/)

- **`scripts/preprocess_filter_tuning.py`** — Tuning de filtros de preprocesamiento con 3 subcomandos: `analizar`, `tunear`, `aplicar`. Los resultados se hardcodean en `PREPROCESS_PROFILE_OVERRIDES` de `src/preprocess.py`.
- **`scripts/segmentar_visual.py`** — Heurística de segmentación por proyección horizontal (reemplazada por YOLO OBB).
- **`scripts/boxes_from_heuristic.py`** — Generador de pre-labels con heurística 2D.
- **`scripts/labelstudio_sync.py`** — Integración con Label Studio para revisión de anotaciones.
