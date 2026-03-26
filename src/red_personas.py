# -*- coding: utf-8 -*-
"""
Genera red de co-ocurrencia de personas a partir del compilado.

Dos personas estan conectadas si aparecen mencionadas en el mismo contrato.
El peso de la arista = numero de contratos compartidos.

Salidas:
  - red_personas.gexf       (para Gephi)
  - red_personas_nodos.csv   (nodo, frecuencia, tomos, grado)
  - red_personas_aristas.csv (persona_a, persona_b, peso, contratos)
  - red_personas_stats.txt   (resumen de la red)

Uso:
  python src/red_personas.py --compilado outputs/run_XXXXX/compilado.xlsx
  python src/red_personas.py  # usa el run mas reciente con compilado.xlsx
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
import networkx as nx


def _split(valor: str) -> list[str]:
    if not isinstance(valor, str) or not valor.strip():
        return []
    return [e.strip() for e in valor.split(";") if e.strip()]


def construir_red(df: pd.DataFrame) -> tuple[nx.Graph, dict]:
    """Construye grafo de co-ocurrencia de personas.

    Returns:
        G: grafo no dirigido con pesos
        stats: diccionario con estadisticas
    """
    # Extraer personas por contrato
    contratos_personas: list[tuple[dict, list[str]]] = []
    for _, row in df.iterrows():
        personas = _split(row.get("personas", ""))
        if len(personas) >= 1:
            meta = {
                "tomo_id": row.get("tomo_id", ""),
                "id_num": row.get("id_num"),
                "año_num": row.get("año_num"),
            }
            contratos_personas.append((meta, personas))

    # Frecuencia global de cada persona y tomos donde aparece
    freq: Counter = Counter()
    tomos_por_persona: dict[str, set] = defaultdict(set)
    años_por_persona: dict[str, list] = defaultdict(list)

    for meta, personas in contratos_personas:
        for p in personas:
            freq[p] += 1
            tomos_por_persona[p].add(meta["tomo_id"])
            if pd.notna(meta["año_num"]):
                años_por_persona[p].append(int(meta["año_num"]))

    # Construir aristas (pares co-ocurrentes)
    edge_counter: Counter = Counter()
    edge_contratos: dict[tuple, list] = defaultdict(list)

    for meta, personas in contratos_personas:
        unicos = sorted(set(personas))
        for a, b in combinations(unicos, 2):
            par = (a, b) if a < b else (b, a)
            edge_counter[par] += 1
            edge_contratos[par].append(f"{meta['tomo_id']}_{meta['id_num']}")

    # Crear grafo
    G = nx.Graph()

    # Agregar nodos
    for persona, f in freq.items():
        tomos = sorted(tomos_por_persona[persona])
        años = años_por_persona[persona]
        G.add_node(persona,
                    frecuencia=f,
                    n_tomos=len(tomos),
                    tomos="; ".join(tomos),
                    año_min=min(años) if años else 0,
                    año_max=max(años) if años else 0)

    # Agregar aristas
    for (a, b), peso in edge_counter.items():
        G.add_edge(a, b, weight=peso)

    # Stats
    stats = {
        "contratos_con_personas": len(contratos_personas),
        "contratos_con_2_o_mas": sum(1 for _, p in contratos_personas if len(set(p)) >= 2),
        "nodos": G.number_of_nodes(),
        "aristas": G.number_of_edges(),
        "densidad": nx.density(G),
        "componentes": nx.number_connected_components(G),
        "top_10_frecuencia": freq.most_common(10),
        "top_10_grado": sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10],
        "top_10_peso": edge_counter.most_common(10),
    }

    # Componente gigante
    if G.number_of_nodes() > 0:
        ccs = sorted(nx.connected_components(G), key=len, reverse=True)
        stats["componente_gigante"] = len(ccs[0])
        stats["pct_gigante"] = len(ccs[0]) / G.number_of_nodes() * 100
    else:
        stats["componente_gigante"] = 0
        stats["pct_gigante"] = 0

    return G, stats


def exportar(G: nx.Graph, stats: dict, output_dir: Path):
    """Exporta red en multiples formatos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. GEXF (Gephi)
    gexf_path = output_dir / "red_personas.gexf"
    nx.write_gexf(G, str(gexf_path))
    print(f"  GEXF: {gexf_path}")

    # 2. Nodos CSV
    nodos = []
    for nodo, data in G.nodes(data=True):
        nodos.append({
            "persona": nodo,
            "frecuencia": data.get("frecuencia", 0),
            "grado": G.degree(nodo),
            "grado_ponderado": sum(d["weight"] for _, _, d in G.edges(nodo, data=True)),
            "n_tomos": data.get("n_tomos", 0),
            "tomos": data.get("tomos", ""),
            "año_min": data.get("año_min", 0),
            "año_max": data.get("año_max", 0),
        })
    df_nodos = pd.DataFrame(nodos).sort_values("frecuencia", ascending=False)
    nodos_path = output_dir / "red_personas_nodos.csv"
    df_nodos.to_csv(nodos_path, index=False, encoding="utf-8-sig")
    print(f"  Nodos: {nodos_path}")

    # 3. Aristas CSV
    aristas = []
    for a, b, data in G.edges(data=True):
        aristas.append({
            "persona_a": a,
            "persona_b": b,
            "peso": data["weight"],
        })
    df_aristas = pd.DataFrame(aristas).sort_values("peso", ascending=False)
    aristas_path = output_dir / "red_personas_aristas.csv"
    df_aristas.to_csv(aristas_path, index=False, encoding="utf-8-sig")
    print(f"  Aristas: {aristas_path}")

    # 4. Stats
    stats_path = output_dir / "red_personas_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("RED DE CO-OCURRENCIA DE PERSONAS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Contratos con personas:     {stats['contratos_con_personas']:,}\n")
        f.write(f"Contratos con 2+ personas:  {stats['contratos_con_2_o_mas']:,}\n\n")
        f.write(f"Nodos (personas unicas):    {stats['nodos']:,}\n")
        f.write(f"Aristas (co-ocurrencias):   {stats['aristas']:,}\n")
        f.write(f"Densidad:                   {stats['densidad']:.6f}\n")
        f.write(f"Componentes conexos:        {stats['componentes']:,}\n")
        f.write(f"Componente gigante:         {stats['componente_gigante']:,} ({stats['pct_gigante']:.1f}%)\n\n")

        f.write("TOP 10 PERSONAS MAS FRECUENTES\n")
        f.write("-" * 50 + "\n")
        for p, c in stats["top_10_frecuencia"]:
            f.write(f"  {c:>5}x  {p}\n")

        f.write("\nTOP 10 PERSONAS MAS CONECTADAS (grado)\n")
        f.write("-" * 50 + "\n")
        for p, g in stats["top_10_grado"]:
            f.write(f"  {g:>5} conexiones  {p}\n")

        f.write("\nTOP 10 PARES MAS FRECUENTES\n")
        f.write("-" * 50 + "\n")
        for (a, b), w in stats["top_10_peso"]:
            f.write(f"  {w:>5}x  {a}  <->  {b}\n")

    print(f"  Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Red de co-ocurrencia de personas.")
    parser.add_argument("--compilado", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.compilado:
        compilado_path = Path(args.compilado)
    else:
        runs = sorted(Path("outputs").glob("run_*"), reverse=True)
        compilado_path = None
        for r in runs:
            cp = r / "compilado.xlsx"
            if cp.exists():
                compilado_path = cp
                break
        if not compilado_path:
            print("ERROR: No se encontro compilado.xlsx")
            return

    print(f"  Compilado: {compilado_path}")
    df = pd.read_excel(compilado_path)
    print(f"  {len(df):,} contratos")

    print("  Construyendo red...")
    G, stats = construir_red(df)

    output_dir = Path(args.output_dir) if args.output_dir else compilado_path.parent
    exportar(G, stats, output_dir)

    # Imprimir resumen
    print(f"\n  {'=' * 50}")
    print(f"  RED DE PERSONAS")
    print(f"  {'=' * 50}")
    print(f"  Nodos:     {stats['nodos']:,} personas unicas")
    print(f"  Aristas:   {stats['aristas']:,} co-ocurrencias")
    print(f"  Componente gigante: {stats['componente_gigante']:,} ({stats['pct_gigante']:.1f}%)")
    print(f"\n  Top 5 personas:")
    for p, c in stats["top_10_frecuencia"][:5]:
        print(f"    {c:>5}x  {p}")
    print(f"\n  Top 5 pares:")
    for (a, b), w in stats["top_10_peso"][:5]:
        print(f"    {w:>4}x  {a}  <->  {b}")


if __name__ == "__main__":
    main()
