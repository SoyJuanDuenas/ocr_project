# -*- coding: utf-8 -*-
"""
Re-OCR focalizado de paginas con contratos perdidos.

Lee diagnostico_reocr.xlsx (producido por pipeline.py paso 7),
re-procesa las paginas identificadas con crop_mode=False,
y exporta los contratos recuperados.

Uso:
  python src/reocr_perdidos.py --diagnostico outputs/run_.../diagnostico_reocr.xlsx
  python src/reocr_perdidos.py --diagnostico outputs/run_.../diagnostico_reocr.xlsx --images-dir data/preprocess_v2
  python src/reocr_perdidos.py --diagnostico outputs/run_.../diagnostico_reocr.xlsx --solo-listar
"""
from __future__ import annotations

import sys
import re
import argparse
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


# =====================================================================
# 1. LEER DIAGNOSTICO Y EXTRAER PAGINAS A RE-OCR
# =====================================================================

def leer_paginas_reocr(diagnostico_path: Path) -> dict[str, set[int]]:
    """Lee diagnostico_reocr.xlsx y retorna {tomo_id: {page_nums}}."""
    df = pd.read_excel(diagnostico_path, sheet_name="resumen_por_tomo")
    paginas_por_tomo: dict[str, set[int]] = {}

    for _, row in df.iterrows():
        tomo = row["tomo_id"]
        pages_str = str(row.get("lista_paginas", ""))
        pages = set()
        for p in pages_str.split(";"):
            p = p.strip()
            if p.isdigit():
                pages.add(int(p))
        if pages:
            paginas_por_tomo[tomo] = pages

    return paginas_por_tomo


# =====================================================================
# 2. ENCONTRAR IMAGENES PREPROCESADAS
# =====================================================================

_PREP_RE = re.compile(r"^(.+?)_p(\d+)_prep\.\w+$")


def encontrar_imagenes(
    paginas_por_tomo: dict[str, set[int]],
    images_dir: Path,
) -> list[dict]:
    """Encuentra las imagenes preprocesadas que corresponden a las paginas."""
    todas = sorted(images_dir.glob("*_prep.*"))
    imagenes = []

    for img_path in todas:
        m = _PREP_RE.match(img_path.name)
        if not m:
            continue
        tomo_nombre = m.group(1)  # "Tomo I", "Tomo XVI"
        page_num = int(m.group(2))

        # Convertir nombre a tomo_id: "Tomo I" -> "Tomo_I"
        tomo_id = re.sub(r"[^A-Za-z0-9._-]+", "_", tomo_nombre.strip())

        if tomo_id in paginas_por_tomo and page_num in paginas_por_tomo[tomo_id]:
            imagenes.append({
                "tomo_id": tomo_id,
                "tomo_nombre": tomo_nombre,
                "page_num": page_num,
                "image_path": img_path,
            })

    return imagenes


# =====================================================================
# 3. RE-OCR CON CROP_MODE=FALSE
# =====================================================================

def reocr_paginas(
    imagenes: list[dict],
    output_dir: Path,
    model_name: str = "deepseek-ai/DeepSeek-OCR",
    device: str | None = None,
    dtype: str = "bfloat16",
    max_new_tokens: int = 4096,
) -> list[dict]:
    """Re-OCR de paginas especificas con crop_mode=False."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(exist_ok=True)

    # Detectar device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Cargar modelo
    print(f"  Cargando modelo {model_name} en {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, use_safetensors=True,
    ).eval()

    if device == "cuda":
        torch_dtype = torch.bfloat16
        model = model.to(torch_dtype).cuda()
    else:
        model = model.to(torch.float32)

    if hasattr(model, "generation_config"):
        model.generation_config.max_new_tokens = max_new_tokens

    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    total = len(imagenes)
    resultados = []

    # Filtrar paginas ya procesadas (resume)
    pendientes = []
    ya_procesadas = 0
    for img_info in imagenes:
        tomo = img_info["tomo_nombre"]
        page = img_info["page_num"]
        page_label = f"{tomo}_p{page:04d}"
        result_file = pages_dir / page_label / "result.txt"
        if result_file.exists() and result_file.stat().st_size > 0:
            ya_procesadas += 1
        else:
            pendientes.append(img_info)

    if ya_procesadas > 0:
        print(f"  {ya_procesadas} paginas ya procesadas (resume), {len(pendientes)} pendientes")

    total = len(pendientes)
    if total == 0:
        print("  Todas las paginas ya fueron procesadas.")
        return resultados

    print(f"  Re-OCR de {total} paginas con crop_mode=False...")
    t0 = time.time()

    for i, img_info in enumerate(pendientes):
        tomo = img_info["tomo_nombre"]
        page = img_info["page_num"]
        img_path = img_info["image_path"]

        page_label = f"{tomo}_p{page:04d}"
        out_subdir = pages_dir / page_label
        out_subdir.mkdir(parents=True, exist_ok=True)
        result_file = out_subdir / "result.txt"

        # Inferencia con crop_mode=False (clave: no recorta la imagen)
        try:
            res_text = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=str(img_path),
                output_path=str(out_subdir),
                crop_mode=False,  # <-- diferencia clave
                save_results=True,
                eval_mode=True,
            )

            if not isinstance(res_text, str) or not res_text.strip():
                # Fallback: leer result.txt si model.infer no retorno texto
                if result_file.exists():
                    res_text = result_file.read_text(encoding="utf-8")

            if isinstance(res_text, str) and res_text.strip():
                result_file.write_text(res_text, encoding="utf-8")
                status = "OK"
            else:
                status = "VACIO"

        except Exception as e:
            res_text = ""
            status = f"ERROR: {e}"

        resultados.append({
            "tomo_id": img_info["tomo_id"],
            "page_num": page,
            "image": str(img_path),
            "status": status,
            "chars": len(res_text) if res_text else 0,
        })

        print(f"  [{i+1}/{total}] {page_label} — {status} ({len(res_text) if res_text else 0} chars)")

        if device == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Re-OCR completado en {elapsed/60:.1f} min")

    # Exportar log
    df_log = pd.DataFrame(resultados)
    log_path = output_dir / "reocr_log.csv"
    df_log.to_csv(log_path, index=False, encoding="utf-8-sig")

    return resultados


# =====================================================================
# 4. COMPARAR OCR ORIGINAL VS RE-OCR
# =====================================================================

def _sanitizar_excel(text: str) -> str:
    """Elimina caracteres ilegales para openpyxl (control chars excepto tab/newline)."""
    import re as _re
    return _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def comparar_resultados(
    reocr_dir: Path,
    original_pages_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Compara resultado original vs re-OCR para encontrar texto nuevo."""

    reocr_pages = sorted((reocr_dir / "pages").rglob("result.txt"))
    comparaciones = []

    for reocr_path in reocr_pages:
        page_label = reocr_path.parent.name  # "Tomo I_p0045"

        # Buscar original
        original_path = original_pages_dir / page_label / "result.txt"
        original_text = ""
        if original_path.exists():
            try:
                original_text = original_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                original_text = original_path.read_text(encoding="latin-1")

        reocr_text = reocr_path.read_text(encoding="utf-8")

        # Detectar garbage loops (texto repetitivo > 30KB)
        es_garbage = len(reocr_text) > 30000

        # Calcular diferencia
        orig_lines = set(l.strip() for l in original_text.split("\n") if l.strip())
        reocr_lines = set(l.strip() for l in reocr_text.split("\n") if l.strip())
        nuevas = reocr_lines - orig_lines
        perdidas = orig_lines - reocr_lines

        texto_nuevo = "\n".join(sorted(nuevas)[:20]) if nuevas else ""

        comparaciones.append({
            "pagina": page_label,
            "chars_original": len(original_text),
            "chars_reocr": len(reocr_text),
            "lineas_original": len(orig_lines),
            "lineas_reocr": len(reocr_lines),
            "lineas_nuevas": len(nuevas),
            "lineas_perdidas": len(perdidas),
            "garbage_loop": es_garbage,
            "texto_nuevo": _sanitizar_excel(texto_nuevo[:5000]),
        })

    df_comp = pd.DataFrame(comparaciones)

    comp_path = output_dir / "comparacion_reocr.xlsx"
    df_comp.to_excel(comp_path, index=False)
    print(f"  Comparacion exportada: {comp_path}")

    n_con_nuevo = int((df_comp["lineas_nuevas"] > 0).sum())
    print(f"  {n_con_nuevo}/{len(df_comp)} paginas con texto nuevo detectado")

    return df_comp


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-OCR focalizado de paginas con contratos perdidos."
    )
    parser.add_argument(
        "--diagnostico",
        required=True,
        help="Ruta a diagnostico_reocr.xlsx (producido por pipeline.py)",
    )
    parser.add_argument(
        "--images-dir",
        default="data/preprocess_v2",
        help="Directorio con imagenes preprocesadas (default: data/preprocess_v2)",
    )
    parser.add_argument(
        "--original-pages",
        default="outputs/pages",
        help="Directorio con result.txt originales (default: outputs/pages)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio de salida (default: mismo dir que diagnostico + /reocr)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-OCR",
        help="Modelo OCR (default: deepseek-ai/DeepSeek-OCR)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda|cpu (default: auto)",
    )
    parser.add_argument(
        "--solo-listar",
        action="store_true",
        help="Solo listar paginas a re-OCR, sin ejecutar",
    )
    parser.add_argument(
        "--solo-comparar",
        action="store_true",
        help="Solo comparar resultados ya existentes (no re-OCR)",
    )
    args = parser.parse_args()

    diagnostico_path = Path(args.diagnostico)
    images_dir = Path(args.images_dir)
    original_pages = Path(args.original_pages)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = diagnostico_path.parent / "reocr"

    # Leer diagnostico
    print(f"{'=' * 60}")
    print(f"  RE-OCR FOCALIZADO")
    print(f"  Diagnostico: {diagnostico_path}")
    print(f"{'=' * 60}")

    paginas_por_tomo = leer_paginas_reocr(diagnostico_path)
    total_paginas = sum(len(p) for p in paginas_por_tomo.values())
    print(f"\n  {total_paginas} paginas a re-OCR en {len(paginas_por_tomo)} tomos:")
    for tomo in sorted(paginas_por_tomo):
        pages = sorted(paginas_por_tomo[tomo])
        print(f"    {tomo}: {len(pages)} paginas")

    if args.solo_listar:
        print("\n  (--solo-listar: no se ejecuta re-OCR)")
        sys.exit(0)

    if args.solo_comparar:
        print(f"\n  Comparando resultados existentes...")
        comparar_resultados(output_dir, original_pages, output_dir)
        sys.exit(0)

    # Encontrar imagenes
    imagenes = encontrar_imagenes(paginas_por_tomo, images_dir)
    print(f"\n  {len(imagenes)} imagenes encontradas de {total_paginas} esperadas")

    if not imagenes:
        print("  ERROR: No se encontraron imagenes. Verifica --images-dir")
        sys.exit(1)

    no_encontradas = total_paginas - len(imagenes)
    if no_encontradas > 0:
        print(f"  WARNING: {no_encontradas} paginas sin imagen encontrada")

    # Re-OCR
    print(f"\n  Iniciando re-OCR con crop_mode=False...")
    reocr_paginas(
        imagenes,
        output_dir,
        model_name=args.model,
        device=args.device,
    )

    # Comparar
    print(f"\n  Comparando con OCR original...")
    comparar_resultados(output_dir, original_pages, output_dir)

    print(f"\n{'=' * 60}")
    print(f"  RE-OCR COMPLETADO")
    print(f"  Resultados en: {output_dir}")
    print(f"  Siguiente paso: revisar comparacion_reocr.xlsx")
    print(f"  y mergear contratos nuevos al compilado")
    print(f"{'=' * 60}")
