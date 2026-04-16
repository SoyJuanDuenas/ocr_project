# -*- coding: utf-8 -*-
"""
Batch OCR con DeepSeek-OCR (Transformers) sobre imágenes preprocesadas.

API principal:
    run_ocr_batch(
        images_dir: str | Path,
        glob_pattern: str = "Tomo *_prep.png",
        output_dir: str | Path = "outputs",
        model_name: str = "deepseek-ai/DeepSeek-OCR",
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        device: str | None = None,           # "cuda", "cpu", "mps"; None = auto
        dtype: str = "bfloat16",             # "bfloat16"|"float16"|"float32"
        attn_impl: str | None = None,        # None | "sdpa" | "eager"
        save_per_page: bool = True,
        capture_stdout_fallback: bool = True,
        tokenizer=None,
        model=None
    ) -> dict

Salida (dict):
    {
      "processed": int,
      "skipped": int,
      "combined_md_path": str,
      "log_csv_path": str,
      "pages_dir": str,
      "rows": List[dict]  # por página
    }

Uso desde otro .py:
    from ocr_batch_deepseek import run_ocr_batch
    res = run_ocr_batch("../data/preprocess_v2", glob_pattern="Tomo *_prep.png")

CLI opcional:
    python ocr_batch_deepseek.py --images-dir ../data/preprocess_v2 --glob "Tomo *_prep.png" --out outputs
"""
from __future__ import annotations

import os, io, re, csv, glob, contextlib, argparse

from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoModel, AutoTokenizer

# ---------- utilidades ----------
_NUM_PAT = re.compile(r"_p(\d+)_", re.IGNORECASE)

def _natural_key(p: Path):
    m = _NUM_PAT.search(p.name)
    return (int(m.group(1)) if m else 10**9, p.name.lower())

def _select_device(device: Optional[str]) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _select_dtype(dtype: str, device: str) -> torch.dtype:
    dtype = (dtype or "bfloat16").lower()
    if device == "cpu":
        # en CPU lo seguro es float32
        return torch.float32
    if dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype in ("float16", "fp16", "half"):
        return torch.float16
    return torch.float32

def _ensure_dirs(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    return pages_dir

# ---------- validación de output OCR ----------
_RE_NO_LATINO = re.compile(
    r"[\u0900-\u097F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]"
)


def _validar_output(text: str) -> Tuple[bool, List[str]]:
    """Valida calidad del output OCR.

    Retorna (es_valido, lista_de_flags).
    Un output inválido debería reintentarse con otro prompt.
    """
    flags: List[str] = []
    if not text or not text.strip():
        return False, ["vacio"]

    stripped = text.strip()

    # Muy corto
    if len(stripped) < 50:
        flags.append("muy_corto")

    # Ratio alfabético muy bajo (basura numérica / loops)
    if len(stripped) > 20:
        ratio = sum(c.isalpha() for c in stripped) / len(stripped)
        if ratio < 0.3:
            flags.append("bajo_alfa")

    # Detección de loop de decodificación: líneas repetidas
    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    if len(lines) > 5:
        counter = Counter(lines)
        repeated = sum(v - 1 for v in counter.values() if v > 1)
        if repeated / len(lines) > 0.5:
            flags.append("texto_repetitivo")

    # Caracteres no-latinos alucinados
    if _RE_NO_LATINO.search(stripped):
        flags.append("chars_no_latino")

    # Flags que invalidan el output (chars_no_latino es warning, no invalida)
    invalida = {"vacio", "bajo_alfa", "texto_repetitivo"}
    es_valido = not bool(invalida & set(flags))
    return es_valido, flags


# ---------- prompts de retry ----------
_PROMPTS_RETRY = [
    "<image>\nOCR this document to plain text.",
    "<image>\nThis is a section from a 16th century Spanish notarial catalog. "
    "Convert all text to markdown.",
]


# ---------- carga de modelo/tokenizer ----------
def _load_tokenizer_and_model(
    model_name: str,
    *,
    attn_impl: Optional[str],
    device: str,
    dtype: torch.dtype,
    tokenizer=None,
    model=None,
):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if model is None:
        model_kwargs = dict(trust_remote_code=True, use_safetensors=True)
        if attn_impl is not None:
            # solo si tu versión de transformers lo soporta
            model_kwargs["attn_implementation"] = attn_impl
        model = AutoModel.from_pretrained(model_name, **model_kwargs).eval()

    # mover a dispositivo y dtype
    if device == "cuda":
        model = model.to(dtype).cuda()
    elif device == "mps":
        # en MPS se recomienda float16/32; bfloat16 no siempre está disponible
        model = model.to(torch.float16 if dtype == torch.float16 else torch.float32).to("mps")
    else:  # cpu
        model = model.to(torch.float32)

    return tokenizer, model

# ---------- inferencia de una página ----------
def _infer_una_pagina(
    model,
    tokenizer,
    *,
    prompt: str,
    img_path: str,
    out_dir: str,
    save_per_page: bool,
    capture_stdout_fallback: bool,
) -> Optional[str]:
    """Ejecuta inferencia con un prompt dado. Retorna texto o None si falla."""
    res_text = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=img_path,
        output_path=out_dir,
        save_results=bool(save_per_page),
        eval_mode=True,
    )

    # fallback: capturar stdout si la versión no devuelve texto
    if capture_stdout_fallback and (not isinstance(res_text, str) or not res_text.strip()):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model.infer(
                tokenizer,
                prompt=prompt,
                image_file=img_path,
                output_path=out_dir,
                save_results=bool(save_per_page),
            )
        res_text = buf.getvalue()

    return res_text


# ---------- función pública ----------
def run_ocr_batch(
    images_dir: str | Path,
    glob_pattern: str = "Tomo *_prep.png",
    output_dir: str | Path = "outputs",
    model_name: str = "deepseek-ai/DeepSeek-OCR",
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    device: str | None = None,
    dtype: str = "bfloat16",
    attn_impl: str | None = None,
    save_per_page: bool = True,
    capture_stdout_fallback: bool = True,
    max_new_tokens: int = 4096,
    max_retries: int = 3,
    resume: bool = False,
    tokenizer=None,
    model=None,
) -> Dict[str, object]:
    """
    Ejecuta OCR por lotes sobre imágenes que cumplan el patrón indicado.

    Mejoras sobre la versión base:
    - max_new_tokens: limita tokens generados para cortar loops de decodificación.
    - max_retries: si la validación del output falla, reintenta con prompts alternativos.
    - resume: salta páginas que ya tienen un result.txt válido (checkpointing).
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    pages_dir = _ensure_dirs(output_dir)
    log_path = output_dir / "batch_log.csv"
    combined_md_path = output_dir / "combined.md"

    # listar imágenes
    image_paths = sorted(
        [Path(p) for p in glob.glob(str(images_dir / glob_pattern))],
        key=_natural_key,
    )
    if not image_paths:
        return {
            "processed": 0, "skipped": 0, "cached": 0,
            "combined_md_path": str(combined_md_path),
            "log_csv_path": str(log_path), "pages_dir": str(pages_dir),
            "rows": [], "note": f"No se encontraron imágenes con patrón {glob_pattern} en {images_dir.resolve()}"
        }

    # preparar modelo
    device = _select_device(device)
    torch_dtype = _select_dtype(dtype, device)
    tokenizer, model = _load_tokenizer_and_model(
        model_name, attn_impl=attn_impl, device=device, dtype=torch_dtype,
        tokenizer=tokenizer, model=model
    )

    # Intentar limitar max_new_tokens via generation_config
    if max_new_tokens and hasattr(model, "generation_config"):
        model.generation_config.max_new_tokens = max_new_tokens

    # Armar lista de prompts para reintentos
    prompts_a_probar = [prompt] + _PROMPTS_RETRY[:max(max_retries - 1, 0)]

    processed = 0
    skipped = 0
    cached = 0
    rows: List[Dict[str, str]] = []
    combined_md_parts: List[str] = []
    total = len(image_paths)

    for i, img_path in enumerate(image_paths):
        m = _NUM_PAT.search(img_path.name)
        page_id = int(m.group(1)) if m else None
        page_label = img_path.stem.replace("_prep", "")
        out_dir = pages_dir / page_label
        if save_per_page:
            out_dir.mkdir(parents=True, exist_ok=True)

        result_file = out_dir / "result.txt"

        # --- Checkpointing: saltar si ya existe un result.txt válido ---
        if resume and result_file.exists():
            existing = result_file.read_text(encoding="utf-8")
            is_valid, _ = _validar_output(existing)
            if is_valid:
                cached += 1
                combined_md_parts.append(
                    f"\n\n---\n\n# Página {page_id if page_id is not None else img_path.name}\n\n"
                )
                combined_md_parts.append(existing)
                rows.append({
                    "page": page_label, "image": str(img_path),
                    "status": "CACHED", "note": "",
                })
                print(f"  [{i+1}/{total}] {page_label} — CACHED (ya válido)")
                continue

        # --- Inferencia con reintentos ---
        best_text: Optional[str] = None
        best_flags: List[str] = ["no_intentado"]
        status = "SKIPPED"

        for attempt, try_prompt in enumerate(prompts_a_probar):
            try:
                res_text = _infer_una_pagina(
                    model, tokenizer,
                    prompt=try_prompt,
                    img_path=str(img_path),
                    out_dir=str(out_dir),
                    save_per_page=False,  # guardamos al final, no en cada intento
                    capture_stdout_fallback=capture_stdout_fallback,
                )

                is_valid, flags = _validar_output(res_text)

                # Guardar el mejor resultado (primera vez, o si es válido)
                if best_text is None or is_valid:
                    best_text = res_text
                    best_flags = flags

                if is_valid:
                    status = "OK"
                    break
                else:
                    label = f"intento {attempt+1}/{len(prompts_a_probar)}"
                    print(f"  [{i+1}/{total}] {page_label} — {label} inválido: {flags}")

            except Exception as e:
                if best_text is None:
                    best_flags = [f"error: {e}"]
                print(f"  [{i+1}/{total}] {page_label} — intento {attempt+1} excepción: {e}")

            # Limpiar VRAM entre reintentos
            if device == "cuda":
                torch.cuda.empty_cache()

        # --- Guardar mejor resultado ---
        final_text = best_text or ""
        if save_per_page:
            result_file.write_text(final_text, encoding="utf-8")

        combined_md_parts.append(
            f"\n\n---\n\n# Página {page_id if page_id is not None else img_path.name}\n\n"
        )
        combined_md_parts.append(final_text)

        if status == "OK":
            processed += 1
            note = ""
            if best_flags:
                note = f"flags: {'; '.join(best_flags)}" if best_flags != [] else ""
            print(f"  [{i+1}/{total}] {page_label} — OK")
        else:
            skipped += 1
            note = f"flags: {'; '.join(best_flags)}"
            print(f"  [{i+1}/{total}] {page_label} — DEGRADADO ({note})")

        rows.append({
            "page": page_label, "image": str(img_path),
            "status": status, "note": "; ".join(best_flags),
        })

        if device == "cuda":
            torch.cuda.empty_cache()

    # exports
    combined_md_path.write_text("".join(combined_md_parts), encoding="utf-8")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "image", "status", "note"])
        writer.writeheader()
        writer.writerows(rows)

    # Resumen
    print(f"\n  Resumen: {processed} OK, {cached} cached, {skipped} degradados de {total} total")

    return {
        "processed": processed,
        "skipped": skipped,
        "cached": cached,
        "combined_md_path": str(combined_md_path),
        "log_csv_path": str(log_path),
        "pages_dir": str(pages_dir),
        "rows": rows,
    }

# ---------- CLI opcional ----------
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=str, required=True, help="Carpeta con PNGs/TIF/etc.")
    ap.add_argument("--glob", dest="glob_pattern", type=str, default="Tomo I*_prep.png")
    ap.add_argument("--out", dest="output_dir", type=str, default="outputs")
    ap.add_argument("--model", dest="model_name", type=str, default="deepseek-ai/DeepSeek-OCR")
    ap.add_argument("--prompt", type=str, default="<image>\n<|grounding|>Convert the document to markdown.")
    ap.add_argument("--device", type=str, default=None, help="cuda|cpu|mps (None=auto)")
    ap.add_argument("--dtype", type=str, default="bfloat16", help="bfloat16|float16|float32")
    ap.add_argument("--attn", dest="attn_impl", type=str, default=None, help="sdpa|eager|None")
    ap.add_argument("--no-per-page", action="store_true", help="No guardar carpeta por página")
    ap.add_argument("--no-fallback", action="store_true", help="No capturar stdout como fallback")
    ap.add_argument("--max-new-tokens", type=int, default=4096, help="Max tokens generados por página (default: 4096)")
    ap.add_argument("--max-retries", type=int, default=3, help="Intentos con prompts alternativos si falla validación (default: 3)")
    ap.add_argument("--resume", action="store_true", help="Saltar páginas que ya tienen result.txt válido")
    args = ap.parse_args()

    res = run_ocr_batch(
        images_dir=args.images_dir,
        glob_pattern=args.glob_pattern,
        output_dir=args.output_dir,
        model_name=args.model_name,
        prompt=args.prompt,
        device=args.device,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        save_per_page=not args.no_per_page,
        capture_stdout_fallback=not args.no_fallback,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        resume=args.resume,
    )

    print("Listo")
    print(f"Procesadas: {res['processed']} | Cached: {res['cached']} | Degradadas: {res['skipped']}")
    print(f"- Markdown combinado: {res['combined_md_path']}")
    print(f"- Log: {res['log_csv_path']}")
    print(f"- Carpeta por página: {res['pages_dir']}")
