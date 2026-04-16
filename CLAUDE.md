# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pipeline for digitizing scanned historical documents (notarial records from 15th-16th century Spain). The workflow is: **scan images/PDFs → preprocessing → visual segmentation (YOLO OBB) → OCR → text cleaning → structured extraction → entity extraction via LLM → network analysis**.

The source documents are volumes ("tomos") of the "Catálogo de los Fondos Americanos" archive (16 volumes, ~26,000 contracts, 15th-16th century).

## Setup

```bash
# Python 3.10+, virtual environment at .venv/
pip install -r requirements.txt
# For OCR inference: needs CUDA GPU (tested on RTX 4070 SUPER, 12GB VRAM)
# For entity extraction: needs Ollama running locally with qwen2.5:7b
```

## Key Commands

```bash
# 1. Preprocess images (con perfiles por tomo hardcodeados)
py src/preprocess.py --in data/raw --out data/preprocessed --target-dpi 300 --save-debug --bg-ksize 31 --bin sauvola --sauvola-w 31 --sauvola-k 0.45 --close 3 --denoise-ksize 3 --zoom 1.15

# 1b. Visual segmentation with YOLO OBB → recortes por contrato para OCR
py src/inferir_yolo_obb.py --images-dir data/preprocess_v2 --out outputs/segmentacion_obb
py src/inferir_yolo_obb.py --visualizar --n 100 --out outputs/inferencia_obb  # debug: boxes dibujadas

# 2. Run OCR batch (standalone, sin YOLO)
py src/ocr_model_deepseek.py --images-dir data/preprocess_v2 --glob "Tomo I*_prep.png" --out outputs

# 3. Run full pipeline (YOLO + OCR + post-processing + entities + panel + red)
py src/pipeline.py --images-dir data/preprocess_v2     # desde imagenes (YOLO + OCR + todo)
py src/pipeline.py --pages-dir outputs/pages            # desde pages existentes (salta YOLO+OCR)
py src/pipeline.py --skip-entidades                     # sin entidades
py src/pipeline.py --skip-red-personas                  # sin red de personas
py src/pipeline.py --tomos "Tomo I,Tomo III"            # filtrar tomos
py src/pipeline.py --yolo-model yolov8s-obb.pt --yolo-conf 0.3  # configurar YOLO

# 4. Panelizar compilado (standalone)
py src/panelizar.py --compilado outputs/run_XXX/compilado.xlsx

# 5. Red de co-ocurrencia de personas (standalone)
py src/red_personas.py --compilado outputs/run_XXX/compilado.xlsx
```

## Architecture

### Pipeline Stages (in order)

1. **`src/preprocess.py`** — Image preprocessing. Raw scans/PDFs → binarized PNGs (`*_prep.png`). Steps: grayscale → DPI normalization → zoom → background normalization → Sauvola binarization → morphological stroke repair → median denoising. CLI via argparse. Incorpora perfiles por tomo hardcodeados para los casos ya validados (`Tomo IX`, `VI`, `VIII`, `X`, `XI`, `XIV`).

1b. **`src/inferir_yolo_obb.py`** — Segmentación visual pre-OCR con modelo YOLOv8s-OBB. Dos modos: (1) **batch** (default): infiere sobre todas las páginas, exporta recortes individuales por contrato en `crops/` y un `manifest.csv` con coordenadas OBB, clase y confianza; (2) **visualización** (`--visualizar`): muestra aleatoria con boxes dibujadas. Clases: `contrato`, `continuacion`. Modelo en `models/yolo_obb_v1/weights/best.pt`. Función pública `segmentar_batch()` retorna path al manifiesto.

2. **`src/ocr_model_deepseek.py`** — Batch OCR using DeepSeek-OCR (HuggingFace Transformers). Produces per-page text in `outputs/pages/<tomo>_p<NNNN>/result.txt`. Auto-detects device, retry logic (3 retries with alternate prompts), resume support. Validates output with quality flags: `vacio`, `solo_tags`, `muy_corto`, `bajo_alfa`, `texto_repetitivo`.

3. **`src/pipeline.py`** — Main orchestrator (YOLO + OCR + post-processing, up to 10 steps):
   - **Step 0** (if `--images-dir`, no `--pages-dir`): YOLO detection (`ultralytics`) → crop boxes → DeepSeek-OCR per crop → recompose `result.txt` per page
   - **Step 1**: Calidad OCR (flags por pagina)
   - **Step 2**: Limpieza de paginas (drop angle brackets, headers "Catálogo de los Fondos Americanos" via Levenshtein, dedup lineas consecutivas) + consolidacion de tomos (union inteligente con heuristica de saltos de linea)
   - **Step 3**: Segmentacion de contratos (fuzzy "Libro del año" via Levenshtein ≤ 3)
   - **Step 4**: Parseo de campos estructurados (año, oficio, escribania, etc.)
   - **Step 5**: Cruce de flags OCR a nivel contrato
   - **Step 6**: Correccion de secuencia id_num (anclas + relleno + diagnostico)
   - **Step 7**: Re-segmentacion de contratos "enterrados" en el texto de otros
   - **Step 8**: Extraccion de entidades via Ollama (opcional, `--skip-entidades` para saltar). Incluye `atributos` por persona.
   - **Step 9**: Panelizar (integrado, genera panel.xlsx + panel_headers.xlsx + panel_por_tipo.xlsx)
   - **Step 10**: Red de personas (integrado, opcional `--skip-red-personas`)

4. **`src/parseo_compilado.py`** — Módulo de parseo importado por pipeline.py. Extrae campos: macrodatos, año, oficio, libro, escribania, folio, fecha, signatura, asunto, observaciones.

5. **`src/panelizar.py`** — Convierte compilado.xlsx a formato panel largo (una fila por mencion de entidad). Genera IDs compuestos: IDTOMO (2 dígitos) + IDCONT (4 dígitos) + IDPER/NAO/LUG (2 dígitos).

6. **`src/red_personas.py`** — Construye red de co-ocurrencia de personas. Exporta GEXF (Gephi), CSVs de nodos/aristas, y estadisticas.

### Data Flow

```
data/
├── raw/                    (scans, PDFs)
├── preprocess_v2/          (*_prep.png — imágenes preprocesadas para OCR)
└── segmentation/           (datos de segmentación visual)
    ├── images/             (*_seg.png — imágenes para entrenar/anotar, NO en git)
    ├── prelabels/          (pre-labels heurísticas → input a Label Studio)
    ├── labels/             (ground truth de Label Studio, SÍ en git)
    └── data.yaml

models/yolo_obb_v1/         (modelo YOLO OBB entrenado)
├── labels/                 (labels OBB augmentados, SÍ en git)
└── weights/best.pt         (peso entrenado, SÍ en git)

data/preprocess_v2/*_prep.png
  → outputs/run_YYYYMMDD_HHMMSS/
      ├── crops/                   (recortes YOLO por pagina)
      ├── boxes_manifest.csv       (manifest de boxes detectados)
      ├── ocr_boxes.csv            (OCR tabular por box)
      ├── pages/                   (result.txt recompuesto por pagina)
      ├── pages_filtered/          (si --tomos)
      ├── calidad_ocr.csv
      ├── tomos_txt/*.txt          (tomos consolidados)
      ├── contratos_segmentados.xlsx
      ├── compilado.xlsx           (dataset final con entidades + atributos)
      ├── panel.xlsx               (formato largo)
      ├── panel_headers.xlsx       (filas header separadas)
      ├── panel_por_tipo.xlsx      (multi-hoja por tipo entidad)
      ├── red_personas.gexf        (red para Gephi)
      ├── red_personas_nodos.csv
      ├── red_personas_aristas.csv
      └── red_personas_stats.txt
```

### Key Data Conventions

- **Entity columns** (`personas`, `naos`, `lugares`): semicolon-delimited strings, NOT JSON (e.g., `"Juan; Pedro; María"`)
- **Atributos column**: pipe-delimited persona attributes (e.g., `"Juan Pérez::vecino de Sevilla || María López::viuda"`)
- **Nullable integers**: `id_num`, `año_num` use pandas `Int64` dtype for Excel compatibility
- **Encoding**: UTF-8 preferred, silent fallback to latin-1 on decode errors
- **OCR flags**: semicolon-separated strings (e.g., `"bajo_alfa; script_no_latino"`)
- **Tomo ID normalization**: spaces → underscores in DataFrames (e.g., `Tomo I` → `Tomo_I`)
- **Checkpointing**: OCR skips pages with existing `result.txt` when `--resume`; entity extraction saves `compilado_parcial.xlsx` every 500 contracts
- **Preprocess profiles**: `src/preprocess.py` aplica overrides por tomo para los tomos ya tuneados

### Naming Conventions

- Preprocessed images: `<Tomo Name>_p<NNNN>_prep.png` (e.g., `Tomo XVI_p0001_prep.png`)
- Page directories: `outputs/pages/<Tomo Name>_p<NNNN>/` — parsed by regex `^(?P<tomo>.+?)_p(?P<page>\d+)$`
- Run directories: `outputs/run_YYYYMMDD_HHMMSS/` — timestamped, self-contained

### Models

- **`models/yolo_obb_v1/`** — YOLOv8s-OBB entrenado para segmentación de contratos. Contiene script de entrenamiento, configs (data.yaml, args.yaml) y pesos (`weights/best.pt`, 23 MB, trackeado en git). Datos de entrenamiento (ground truth) en `data/segmentation/labels/`.

### Scripts auxiliares (scripts/)

- **`scripts/preprocess_filter_tuning.py`** — Tuning de filtros de preprocesamiento. Subcomandos: `analizar` (distribución de píxeles), `tunear` (grid search de parámetros), `aplicar` (regenerar *_prep.png con mejores perfiles). Los resultados se hardcodean en `PREPROCESS_PROFILE_OVERRIDES` de `src/preprocess.py`.
- **`scripts/segmentar_visual.py`** — Heurística de segmentación por proyección horizontal (reemplazada por YOLO OBB, conservada como referencia).
- **`scripts/boxes_from_heuristic.py`** — Generador de pre-labels usando heurística de proyección 2D. Output en `data/segmentation/prelabels/`. Depende de `segmentar_visual.py`.
- **`scripts/labelstudio_sync.py`** — Integración con Label Studio para revisión de anotaciones. Import: sube imágenes + pre-labels. Export: descarga ground truth a `data/segmentation/labels/`.

### External Dependencies

- **YOLO (ultralytics)**: Object detection for text region boxes. Default model `yolov8s.pt` (also `yolov8s-obb.pt` for oriented bounding boxes). Supports OBB and standard boxes with fallback to full-page crop.
- **DeepSeek-OCR**: `deepseek-ai/DeepSeek-OCR` model via HuggingFace (loaded with `trust_remote_code=True`). Used for per-crop OCR with retry logic and quality validation.
- **Ollama**: Local inference server at `http://localhost:11434/api/generate` for entity extraction with `qwen3.5:9b` (default). Health check via GET `/api/tags`. 120s timeout per request.

## Language

All code comments, variable names, docstrings, and user-facing text are in **Spanish**. The documents being processed are historical Spanish texts.
