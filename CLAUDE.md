# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pipeline for digitizing scanned historical documents (notarial records from 15th-16th century Spain). The workflow is: **scan images/PDFs → preprocessing → visual segmentation (YOLO OBB) → OCR → text cleaning → structured extraction → entity extraction via LLM → network analysis**.

The source documents are volumes ("tomos") of the "Catálogo de los Fondos Americanos" archive (16 volumes, ~26,000 contracts, 15th-16th century).

## Setup

```bash
# Python 3.10+, virtual environment at .venv/
pip install -r requirements.txt
# Implicit deps not in requirements.txt: pandas, networkx, requests, torch, openpyxl
# For OCR inference: needs CUDA GPU (tested on RTX 4070 SUPER, 12GB VRAM)
# For entity extraction: needs Ollama running locally with qwen2.5:7b
```

## Key Commands

```bash
# 1. Preprocess images (con perfiles por tomo hardcodeados)
py src/preprocess.py --in data/raw --out data/preprocessed --target-dpi 300 --save-debug --bg-ksize 31 --bin sauvola --sauvola-w 31 --sauvola-k 0.45 --close 3 --denoise-ksize 3 --zoom 1.15

# 1b. Visual segmentation with YOLO OBB (detect contract boundaries before OCR)
py src/inferir_yolo_obb.py --model models/yolo_obb_v1/weights/best.pt --n 100 --out outputs/inferencia_obb

# 2. Run OCR batch
py src/ocr_model_deepseek.py --images-dir data/preprocess_v2 --glob "Tomo I*_prep.png" --out outputs

# 3. Run full post-OCR pipeline (consolidation + segmentation + parsing + sequence correction + diagnosis + entities)
py src/pipeline.py
py src/pipeline.py --skip-entidades                    # sin entidades
py src/pipeline.py --reocr-dir outputs/run_XXX/reocr   # con merge de re-OCR

# 4. Re-OCR focalizado (paginas con contratos perdidos)
py src/reocr_perdidos.py --diagnostico outputs/run_XXX/diagnostico_reocr.xlsx --images-dir data/preprocess_v2 --original-pages outputs/pages --output-dir outputs/run_XXX/reocr

# 5. Panelizar compilado (una fila por entidad por contrato)
py src/panelizar.py --compilado outputs/run_XXX/compilado.xlsx

# 6. Red de co-ocurrencia de personas
py src/red_personas.py --compilado outputs/run_XXX/compilado.xlsx
```

## Architecture

### Pipeline Stages (in order)

1. **`src/preprocess.py`** — Image preprocessing. Raw scans/PDFs → binarized PNGs (`*_prep.png`). Steps: grayscale → DPI normalization → zoom → background normalization → Sauvola binarization → morphological stroke repair → median denoising. CLI via argparse. Incorpora perfiles por tomo hardcodeados para los casos ya validados (`Tomo IX`, `VI`, `VIII`, `X`, `XI`, `XIV`).

1b. **`src/inferir_yolo_obb.py`** — Segmentación visual pre-OCR con modelo YOLOv8s-OBB. Detecta contratos y continuaciones en páginas preprocesadas usando oriented bounding boxes. Clases: `contrato` (inicio de contrato nuevo), `continuacion` (continuación de página previa). Modelo en `models/yolo_obb_v1/weights/best.pt`.

2. **`src/ocr_model_deepseek.py`** — Batch OCR using DeepSeek-OCR (HuggingFace Transformers). Produces per-page text in `outputs/pages/<tomo>_p<NNNN>/result.txt`. Auto-detects device, retry logic (3 retries with alternate prompts), resume support. Validates output with quality flags: `vacio`, `solo_tags`, `muy_corto`, `bajo_alfa`, `texto_repetitivo`.

3. **`src/pipeline.py`** — Main post-OCR orchestrator (9 steps):
   - **Step 0**: Merge selectivo con re-OCR (si `--reocr-dir`; usa re-OCR si new_lines > lost_lines AND < 30KB)
   - **Step 1**: Calidad OCR (flags por pagina)
   - **Step 2**: Consolidacion de tomos (union inteligente de paginas con heuristica de saltos de linea)
   - **Step 3**: Segmentacion de contratos (fuzzy "Libro del año" via Levenshtein ≤ 3)
   - **Step 4**: Parseo de campos estructurados (año, oficio, escribania, etc.)
   - **Step 5**: Cruce de flags OCR a nivel contrato
   - **Step 6**: Correccion de secuencia id_num (anclas + relleno + diagnostico)
   - **Step 7**: Re-segmentacion de contratos "enterrados" en el texto de otros
   - **Step 8**: Diagnostico de paginas para re-OCR
   - **Step 9**: Extraccion de entidades via Ollama (opcional, `--skip-entidades` para saltar)

4. **`src/parseo_compilado.py`** — Módulo de parseo importado por pipeline.py. Extrae campos: macrodatos, año, oficio, libro, escribania, folio, fecha, signatura, asunto, observaciones.

5. **`src/reocr_perdidos.py`** — Re-OCR focalizado. Lee diagnostico_reocr.xlsx, re-procesa paginas con `crop_mode=False`, compara y exporta resultados. Soporta resume.

6. **`src/panelizar.py`** — Convierte compilado.xlsx a formato panel largo (una fila por mencion de entidad). Genera IDs compuestos: IDTOMO (2 dígitos) + IDCONT (4 dígitos) + IDPER/NAO/LUG (2 dígitos).

7. **`src/red_personas.py`** — Construye red de co-ocurrencia de personas. Exporta GEXF (Gephi), CSVs de nodos/aristas, y estadisticas.

### Data Flow

```
data/raw/ (scans, PDFs)
  → data/preprocess_v2/ (*_prep.png)
    → outputs/inferencia_obb/ (*_obb.png, segmentación visual con YOLO OBB)
    → outputs/pages/<Tomo>_p<NNNN>/result.txt
      → outputs/run_YYYYMMDD_HHMMSS/
          ├── pages_merged/          (si --reocr-dir)
          ├── calidad_ocr.csv
          ├── tomos_txt/*.txt        (16 tomos consolidados)
          ├── contratos_segmentados.xlsx
          ├── compilado.xlsx         (dataset final con entidades)
          ├── diagnostico_reocr.xlsx
          ├── panel.xlsx             (formato largo)
          ├── red_personas.gexf      (red para Gephi)
          ├── red_personas_nodos.csv
          ├── red_personas_aristas.csv
          └── red_personas_stats.txt
```

### Key Data Conventions

- **Entity columns** (`personas`, `naos`, `lugares`): semicolon-delimited strings, NOT JSON (e.g., `"Juan; Pedro; María"`)
- **Nullable integers**: `id_num`, `año_num` use pandas `Int64` dtype for Excel compatibility
- **Encoding**: UTF-8 preferred, silent fallback to latin-1 on decode errors
- **OCR flags**: space-separated strings (e.g., `"bajo_alfa; script_no_latino"`)
- **Tomo ID normalization**: spaces → underscores in DataFrames (e.g., `Tomo I` → `Tomo_I`)
- **Checkpointing**: OCR skips pages with existing `result.txt` when `--resume`; entity extraction saves `compilado_parcial.xlsx` every 500 contracts
- **Preprocess profiles**: `src/preprocess.py` aplica overrides por tomo para los tomos ya tuneados

### Naming Conventions

- Preprocessed images: `<Tomo Name>_p<NNNN>_prep.png` (e.g., `Tomo XVI_p0001_prep.png`)
- Page directories: `outputs/pages/<Tomo Name>_p<NNNN>/` — parsed by regex `^(?P<tomo>.+?)_p(?P<page>\d+)$`
- Run directories: `outputs/run_YYYYMMDD_HHMMSS/` — timestamped, self-contained

### Models

- **`models/yolo_obb_v1/`** — YOLOv8s-OBB entrenado para segmentación de contratos. Contiene labels (trackeados en git), configuración de entrenamiento, y script de training con augmentation por rotación. Pesos (`weights/best.pt`, 23 MB) no trackeados en git.

### Scripts auxiliares (scripts/)

- **`scripts/preprocess_filter_tuning.py`** — Tuning de filtros de preprocesamiento. Subcomandos: `analizar` (distribución de píxeles), `tunear` (grid search de parámetros), `aplicar` (regenerar *_prep.png con mejores perfiles). Los resultados se hardcodean en `PREPROCESS_PROFILE_OVERRIDES` de `src/preprocess.py`.
- **`scripts/segmentar_visual.py`** — Heurística de segmentación por proyección horizontal (reemplazada por YOLO OBB, conservada como referencia).
- **`scripts/boxes_from_heuristic.py`** — Generador de pre-labels usando heurística de proyección 2D. Depende de `segmentar_visual.py`.
- **`scripts/labelstudio_sync.py`** — Integración con Label Studio para revisión colaborativa de cajas de segmentación.

### External Dependencies

- **DeepSeek-OCR**: `deepseek-ai/DeepSeek-OCR` model via HuggingFace (loaded with `trust_remote_code=True`).
- **Ollama**: Local inference server at `http://localhost:11434/api/generate` for entity extraction with `qwen2.5:7b`. Health check via GET `/api/tags`. 120s timeout per request.

## Language

All code comments, variable names, docstrings, and user-facing text are in **Spanish**. The documents being processed are historical Spanish texts.
