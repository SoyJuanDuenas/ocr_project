# -*- coding: utf-8 -*-
"""
Benchmark DeepSeek-OCR V1 vs V2 sobre un subset de paginas.

Selecciona automaticamente paginas representativas:
  - 2 paginas normales por tomo (primera + media)
  - Paginas problematicas del diagnostico de re-OCR

Compara: tiempo por pagina, longitud, ratio alfabetico, flags, y
diff contra el result.txt existente (V1).

Uso:
  python src/benchmark_ocr_v2.py
  python src/benchmark_ocr_v2.py --n-por-tomo 3
  python src/benchmark_ocr_v2.py --solo-problematicas
  python src/benchmark_ocr_v2.py --paginas "Tomo I_p0049,Tomo III_p0100"
"""
from __future__ import annotations

import argparse
import difflib
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# =====================================================================
# Constantes
# =====================================================================

MODEL_V1 = "deepseek-ai/DeepSeek-OCR"
MODEL_V2 = "deepseek-ai/DeepSeek-OCR-2"

PROMPT_V1 = "<image>\n<|grounding|>Convert the document to markdown."
# V2 puede usar un prompt mas limpio (verificar en model card)
PROMPT_V2 = "<image>\nConvert this scanned document page to markdown text."

IMAGES_DIR = Path("data/preprocess_v2")
PAGES_DIR = Path("outputs/pages")
DIAGNOSTICO = Path("outputs/run_20260301_220103/diagnostico_reocr.xlsx")
OUTPUT_DIR = Path("outputs/benchmark_v1_vs_v2")

# Validacion (copiado de ocr_model_deepseek.py)
_RE_TAG = re.compile(r"<\|[^|]*\|>")
_RE_COORDS = re.compile(r"\[\[\d[\d,\s]*\]\]")
_RE_NO_LATINO = re.compile(
    r"[\u0900-\u097F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]"
)


def _validar_output(text: str) -> tuple[bool, list[str]]:
    flags = []
    if not text or not text.strip():
        return False, ["vacio"]
    stripped = text.strip()
    clean = _RE_TAG.sub("", stripped)
    clean = _RE_COORDS.sub("", clean).strip()
    if len(clean) < 20:
        flags.append("solo_tags")
    if len(stripped) < 50:
        flags.append("muy_corto")
    if len(stripped) > 20:
        ratio = sum(c.isalpha() for c in stripped) / len(stripped)
        if ratio < 0.3:
            flags.append("bajo_alfa")
    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    if len(lines) > 5:
        counter = Counter(lines)
        repeated = sum(v - 1 for v in counter.values() if v > 1)
        if repeated / len(lines) > 0.5:
            flags.append("texto_repetitivo")
    if _RE_NO_LATINO.search(stripped):
        flags.append("chars_no_latino")
    invalida = {"vacio", "solo_tags", "bajo_alfa", "texto_repetitivo"}
    return not bool(invalida & set(flags)), flags


# =====================================================================
# Seleccion de paginas
# =====================================================================

def seleccionar_paginas(
    n_por_tomo: int = 2,
    solo_problematicas: bool = False,
    paginas_manual: list[str] | None = None,
) -> list[Path]:
    """Selecciona subset de imagenes para el benchmark."""

    if paginas_manual:
        result = []
        for label in paginas_manual:
            label = label.strip()
            matches = list(IMAGES_DIR.glob(f"{label}_prep.png"))
            if matches:
                result.append(matches[0])
            else:
                print(f"  WARN: No encontrada imagen para '{label}'")
        return result

    todas = sorted(IMAGES_DIR.glob("*_prep.png"))
    por_tomo: dict[str, list[Path]] = {}
    for p in todas:
        m = re.match(r"^(.+?)_p\d+", p.stem)
        if m:
            por_tomo.setdefault(m.group(1), []).append(p)

    seleccion = []

    # Paginas normales (primera + equiespaciadas)
    if not solo_problematicas:
        for tomo, imgs in sorted(por_tomo.items()):
            indices = [0]
            if n_por_tomo >= 2 and len(imgs) > 1:
                step = len(imgs) // n_por_tomo
                indices = [i * step for i in range(n_por_tomo)]
            for idx in indices:
                if idx < len(imgs):
                    seleccion.append(imgs[idx])

    # Paginas problematicas del diagnostico
    if DIAGNOSTICO.exists():
        try:
            df_diag = pd.read_excel(DIAGNOSTICO, sheet_name="contratos_perdidos")
            paginas_prob = set()
            for raw in df_diag["paginas_reocr"].dropna():
                for p in str(raw).split(";"):
                    p = p.strip()
                    if p.isdigit():
                        paginas_prob.add(int(p))

            # Tomar hasta 10 paginas problematicas distribuidas entre tomos
            prob_imgs = []
            for tomo, imgs in sorted(por_tomo.items()):
                for img in imgs:
                    m2 = re.search(r"_p(\d+)_", img.name)
                    if m2 and int(m2.group(1)) in paginas_prob:
                        prob_imgs.append(img)

            # Limitar a 10 distribuidas
            if len(prob_imgs) > 10:
                step = len(prob_imgs) // 10
                prob_imgs = prob_imgs[::step][:10]

            seleccion.extend(prob_imgs)
        except Exception as e:
            print(f"  WARN: No se pudo leer diagnostico: {e}")

    # Deduplicar manteniendo orden
    seen = set()
    result = []
    for p in seleccion:
        if p not in seen:
            seen.add(p)
            result.append(p)

    return result


# =====================================================================
# Carga de modelos
# =====================================================================

def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cargar_modelo(model_name: str, device: str, use_4bit: bool = False):
    """Carga modelo y tokenizer. Soporta cuantizacion 4-bit para V2."""
    print(f"  Cargando {model_name} en {device}" +
          (" (4-bit)" if use_4bit else "") + "...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs = dict(trust_remote_code=True, use_safetensors=True)

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        except ImportError:
            print("  WARN: bitsandbytes no disponible, cargando en float16")
            use_4bit = False

    model = AutoModel.from_pretrained(model_name, **model_kwargs).eval()

    if not use_4bit:
        if device == "cuda":
            # Intentar bfloat16; si no soporta, float16
            try:
                model = model.to(torch.bfloat16).cuda()
            except Exception:
                model = model.to(torch.float16).cuda()
        elif device == "mps":
            model = model.to(torch.float16).to("mps")
        else:
            model = model.to(torch.float32)

    elapsed = time.time() - t0
    # VRAM
    vram = ""
    if device == "cuda":
        mem = torch.cuda.memory_allocated() / 1024**3
        vram = f" | VRAM: {mem:.1f} GB"
    print(f"  Modelo cargado en {elapsed:.0f}s{vram}")

    return tokenizer, model


def inferir(model, tokenizer, prompt: str, img_path: str, tmp_dir: str = "") -> str:
    """Ejecuta inferencia en una pagina."""
    # V2 requiere output_path valido aunque save_results=False
    if not tmp_dir:
        tmp_dir = str(OUTPUT_DIR / "_tmp_infer")
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    try:
        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=img_path,
            output_path=tmp_dir,
            save_results=False,
            eval_mode=True,
        )
        if isinstance(res, str):
            return res
    except Exception as e:
        print(f"\n    ERROR infer: {e}")
    return ""


# =====================================================================
# Metricas de comparacion
# =====================================================================

def calcular_metricas(texto: str) -> dict:
    """Metricas basicas de un texto OCR."""
    if not texto or not texto.strip():
        return {"chars": 0, "lineas": 0, "ratio_alfa": 0.0, "valido": False, "flags": "vacio"}

    stripped = texto.strip()
    es_valido, flags = _validar_output(stripped)
    lineas = [l for l in stripped.split("\n") if l.strip()]
    ratio_alfa = sum(c.isalpha() for c in stripped) / len(stripped) if stripped else 0

    return {
        "chars": len(stripped),
        "lineas": len(lineas),
        "ratio_alfa": round(ratio_alfa, 3),
        "valido": es_valido,
        "flags": "; ".join(flags) if flags else "",
    }


def similarity_ratio(a: str, b: str) -> float:
    """Ratio de similitud entre dos textos (0-1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# =====================================================================
# Benchmark principal
# =====================================================================

def run_benchmark(
    paginas: list[Path],
    skip_v1: bool = False,
    use_4bit_v2: bool = True,
):
    """Ejecuta benchmark V1 vs V2 sobre las paginas seleccionadas."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = _select_device()
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK DeepSeek-OCR V1 vs V2")
    print(f"  Device: {device}")
    print(f"  Paginas: {len(paginas)}")
    print(f"  V2 4-bit: {use_4bit_v2}")
    print(f"{'=' * 60}\n")

    rows = []

    # --- V1 ---
    if not skip_v1:
        print("\n--- MODELO V1 ---")
        tok_v1, mod_v1 = cargar_modelo(MODEL_V1, device, use_4bit=False)

        if hasattr(mod_v1, "generation_config"):
            mod_v1.generation_config.max_new_tokens = 4096

        for i, img in enumerate(paginas):
            label = img.stem.replace("_prep", "")
            print(f"  V1 [{i+1}/{len(paginas)}] {label}...", end=" ", flush=True)

            t0 = time.time()
            texto = inferir(mod_v1, tok_v1, PROMPT_V1, str(img))
            elapsed = time.time() - t0

            metricas = calcular_metricas(texto)
            print(f"{elapsed:.1f}s | {metricas['chars']} chars | valido={metricas['valido']}")

            # Guardar texto
            out_file = OUTPUT_DIR / f"{label}_v1.txt"
            out_file.write_text(texto, encoding="utf-8")

            rows.append({
                "pagina": label,
                "modelo": "V1",
                "tiempo_s": round(elapsed, 2),
                **metricas,
            })

            if device == "cuda":
                torch.cuda.empty_cache()

        # Liberar V1
        del mod_v1, tok_v1
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        # Usar result.txt existentes como V1
        print("\n--- V1: usando result.txt existentes ---")
        for i, img in enumerate(paginas):
            label = img.stem.replace("_prep", "")
            result_path = PAGES_DIR / label / "result.txt"

            if result_path.exists():
                texto = result_path.read_text(encoding="utf-8")
            else:
                texto = ""
                print(f"  WARN: No existe {result_path}")

            metricas = calcular_metricas(texto)
            out_file = OUTPUT_DIR / f"{label}_v1.txt"
            out_file.write_text(texto, encoding="utf-8")

            rows.append({
                "pagina": label,
                "modelo": "V1",
                "tiempo_s": 0,
                **metricas,
            })

    # --- V2 ---
    print("\n--- MODELO V2 ---")
    tok_v2, mod_v2 = cargar_modelo(MODEL_V2, device, use_4bit=use_4bit_v2)

    if hasattr(mod_v2, "generation_config"):
        mod_v2.generation_config.max_new_tokens = 4096

    for i, img in enumerate(paginas):
        label = img.stem.replace("_prep", "")
        print(f"  V2 [{i+1}/{len(paginas)}] {label}...", end=" ", flush=True)

        t0 = time.time()
        texto = inferir(mod_v2, tok_v2, PROMPT_V2, str(img))
        elapsed = time.time() - t0

        metricas = calcular_metricas(texto)
        print(f"{elapsed:.1f}s | {metricas['chars']} chars | valido={metricas['valido']}")

        out_file = OUTPUT_DIR / f"{label}_v2.txt"
        out_file.write_text(texto, encoding="utf-8")

        rows.append({
            "pagina": label,
            "modelo": "V2",
            "tiempo_s": round(elapsed, 2),
            **metricas,
        })

        if device == "cuda":
            torch.cuda.empty_cache()

    del mod_v2, tok_v2
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Compilar resultados ---
    df = pd.DataFrame(rows)

    # Agregar similitud V1 vs V2
    similitudes = []
    for pagina in df["pagina"].unique():
        v1_file = OUTPUT_DIR / f"{pagina}_v1.txt"
        v2_file = OUTPUT_DIR / f"{pagina}_v2.txt"
        t1 = v1_file.read_text(encoding="utf-8") if v1_file.exists() else ""
        t2 = v2_file.read_text(encoding="utf-8") if v2_file.exists() else ""
        sim = similarity_ratio(t1.strip(), t2.strip())
        similitudes.append({"pagina": pagina, "similitud_v1_v2": round(sim, 3)})

    df_sim = pd.DataFrame(similitudes)

    # Pivot para comparacion lado a lado
    df_v1 = df[df["modelo"] == "V1"].set_index("pagina").add_suffix("_v1")
    df_v2 = df[df["modelo"] == "V2"].set_index("pagina").add_suffix("_v2")
    df_comp = df_v1.join(df_v2).join(df_sim.set_index("pagina"))

    # Columnas delta
    df_comp["delta_chars"] = df_comp["chars_v2"] - df_comp["chars_v1"]
    df_comp["delta_tiempo"] = df_comp["tiempo_s_v2"] - df_comp["tiempo_s_v1"]
    df_comp["delta_ratio_alfa"] = df_comp["ratio_alfa_v2"] - df_comp["ratio_alfa_v1"]

    # Guardar
    out_xlsx = OUTPUT_DIR / "benchmark_v1_vs_v2.xlsx"
    df_comp.to_excel(out_xlsx)

    out_raw = OUTPUT_DIR / "benchmark_raw.xlsx"
    df.to_excel(out_raw, index=False)

    # --- Resumen ---
    print(f"\n{'=' * 60}")
    print(f"  RESUMEN BENCHMARK")
    print(f"{'=' * 60}")
    print(f"  Paginas evaluadas: {len(paginas)}")
    print()

    for modelo in ["V1", "V2"]:
        sub = df[df["modelo"] == modelo]
        print(f"  {modelo}:")
        print(f"    Tiempo promedio: {sub['tiempo_s'].mean():.1f}s/pag")
        print(f"    Chars promedio:  {sub['chars'].mean():.0f}")
        print(f"    Ratio alfa prom: {sub['ratio_alfa'].mean():.3f}")
        print(f"    Validos:         {sub['valido'].sum()}/{len(sub)}")
        print()

    print(f"  Similitud V1 vs V2 promedio: {df_sim['similitud_v1_v2'].mean():.3f}")
    print(f"\n  Resultados: {out_xlsx}")
    print(f"  Textos:     {OUTPUT_DIR}/")
    print(f"{'=' * 60}")

    return df_comp


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-OCR V1 vs V2")
    parser.add_argument(
        "--n-por-tomo", type=int, default=2,
        help="Paginas normales por tomo (default: 2)",
    )
    parser.add_argument(
        "--solo-problematicas", action="store_true",
        help="Solo usar paginas del diagnostico de re-OCR",
    )
    parser.add_argument(
        "--paginas", type=str, default=None,
        help="Lista manual de paginas separadas por coma (e.g. 'Tomo I_p0049,Tomo III_p0100')",
    )
    parser.add_argument(
        "--skip-v1", action="store_true",
        help="Usar result.txt existentes como V1 en vez de re-ejecutar",
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="No usar cuantizacion 4-bit para V2 (requiere >=16GB VRAM)",
    )
    parser.add_argument(
        "--images-dir", type=str, default=None,
        help="Directorio de imagenes preprocesadas (default: data/preprocess_v2)",
    )
    parser.add_argument(
        "--diagnostico", type=str, default=None,
        help="Path al diagnostico_reocr.xlsx (default: ultimo run)",
    )
    args = parser.parse_args()

    if args.images_dir:
        IMAGES_DIR = Path(args.images_dir)
    if args.diagnostico:
        DIAGNOSTICO = Path(args.diagnostico)

    paginas_manual = None
    if args.paginas:
        paginas_manual = args.paginas.split(",")

    paginas = seleccionar_paginas(
        n_por_tomo=args.n_por_tomo,
        solo_problematicas=args.solo_problematicas,
        paginas_manual=paginas_manual,
    )

    if not paginas:
        print("ERROR: No se seleccionaron paginas para el benchmark.")
        exit(1)

    print(f"\nPaginas seleccionadas ({len(paginas)}):")
    for p in paginas:
        print(f"  {p.name}")

    run_benchmark(
        paginas=paginas,
        skip_v1=args.skip_v1,
        use_4bit_v2=not args.no_4bit,
    )
