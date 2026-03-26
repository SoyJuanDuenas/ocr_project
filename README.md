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
| 1 | `src/preprocess.py` | Preprocesamiento de imágenes (binarización Sauvola, 300 DPI) |
| 2 | `src/ocr_model_deepseek.py` | OCR por lotes con DeepSeek-OCR |
| 3 | `src/pipeline.py` | Orquestador post-OCR: consolidación, segmentación, parseo, corrección de secuencia, diagnóstico y extracción de entidades |
| 4 | `src/reocr_perdidos.py` | Re-OCR focalizado de páginas con contratos perdidos |
| 5 | `src/panelizar.py` | Conversión a formato panel largo (una fila por entidad) |
| 6 | `src/red_personas.py` | Red de co-ocurrencia de personas (GEXF + CSVs) |

Módulo auxiliar: `src/parseo_compilado.py` (parser de campos estructurados, importado por `pipeline.py`).

## Uso rápido

```bash
# 1. Preprocesar imágenes
py src/preprocess.py --in data/raw --out data/preprocessed --target-dpi 300 \
   --bg-ksize 31 --bin sauvola --sauvola-w 31 --sauvola-k 0.45 \
   --close 3 --denoise-ksize 3 --zoom 1.15

# 2. OCR por lotes
py src/ocr_model_deepseek.py --images-dir data/preprocess_v2 --glob "Tomo*_prep.png" --out outputs

# 3. Pipeline post-OCR completo
py src/pipeline.py                                          # con entidades
py src/pipeline.py --skip-entidades                         # sin entidades
py src/pipeline.py --reocr-dir outputs/run_XXX/reocr        # con merge de re-OCR previo

# 4. Re-OCR focalizado
py src/reocr_perdidos.py --diagnostico outputs/run_XXX/diagnostico_reocr.xlsx \
   --images-dir data/preprocess_v2 --original-pages outputs/pages \
   --output-dir outputs/run_XXX/reocr

# 5. Panelizar
py src/panelizar.py --compilado outputs/run_XXX/compilado.xlsx

# 6. Red de personas
py src/red_personas.py --compilado outputs/run_XXX/compilado.xlsx
```

## Estructura de salidas

```
outputs/run_YYYYMMDD_HHMMSS/
├── pages_merged/            (si --reocr-dir)
├── calidad_ocr.csv
├── tomos_txt/*.txt          (16 tomos consolidados)
├── contratos_segmentados.xlsx
├── compilado.xlsx           (dataset final con entidades)
├── diagnostico_reocr.xlsx
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
