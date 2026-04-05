"""
Integración con Label Studio para revisión de cajas de segmentación.

Subcomandos:
  import  — Crea proyecto en Label Studio y sube imágenes + pre-anotaciones
             desde contract_boxes_proposed.csv.
  export  — Descarga anotaciones revisadas y genera contract_boxes_reviewed.csv.
  serve   — Levanta servidor local de imágenes (necesario para que Label Studio
             las muestre).

Requisitos:
  pip install label-studio-sdk

Uso típico:
  # 1. Levantar servidor de imágenes (dejar corriendo en otra terminal)
  py scripts/labelstudio_sync.py serve --images-dir data/segmentation/images/train --port 8089

  # 2. Importar pre-labels heurísticas a Label Studio
  py scripts/labelstudio_sync.py import ^
      --proposed data/segmentation/prelabels/contract_boxes_heuristic.csv ^
      --images-dir data/segmentation/images/train ^
      --ls-url http://localhost:8080 ^
      --ls-email admin@local.dev --ls-password admin123 ^
      --image-server http://localhost:8089

  # 3. Revisar y corregir en Label Studio (http://localhost:8080)

  # 4. Exportar ground truth a labels/
  py scripts/labelstudio_sync.py export ^
      --project-id <id> ^
      --output data/segmentation/labels/ ^
      --ls-url http://localhost:8080 ^
      --ls-email admin@local.dev --ls-password admin123
"""

from __future__ import annotations

import argparse
import csv
import http.server
import json
import functools
from pathlib import Path
from urllib.parse import quote

import requests

# ---------------------------------------------------------------------------
# Configuración de etiquetado para Label Studio
# ---------------------------------------------------------------------------

LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"
         rotateControl="false" brightnessControl="true" contrastControl="true"/>
  <Header value="Página: $pagina | Diagnóstico: $diagnostico_estado (score $diagnostico_score)"/>
  <Header value="Flags: $diagnostico_flags"/>
  <RectangleLabels name="label" toName="image">
    <Label value="contrato" background="green"/>
  </RectangleLabels>
</View>
"""


# ---------------------------------------------------------------------------
# Lectura del CSV propuesto
# ---------------------------------------------------------------------------

def _leer_proposed(csv_path: Path) -> dict[str, list[dict]]:
    """Agrupa filas del CSV por página."""
    paginas: dict[str, list[dict]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pagina = row["pagina"]
            if pagina not in paginas:
                paginas[pagina] = []
            paginas[pagina].append(row)
    return paginas


def _construir_tareas(
    paginas: dict[str, list[dict]],
    image_server: str,
) -> list[dict]:
    """Convierte las cajas propuestas en tareas de Label Studio con pre-anotaciones."""
    tareas = []

    for pagina, rows in sorted(paginas.items()):
        first = rows[0]
        img_w = int(float(first["img_width"]))
        img_h = int(float(first["img_height"]))

        # URL de la imagen (servida por el servidor local)
        image_url = f"{image_server.rstrip('/')}/{quote(pagina)}"

        # Pre-anotaciones: convertir cajas a formato Label Studio (porcentajes)
        results = []
        for i, row in enumerate(rows):
            x0 = int(float(row["x0"]))
            y0 = int(float(row["y0"]))
            x1 = int(float(row["x1"]))
            y1 = int(float(row["y1"]))

            results.append({
                "id": f"box_{i}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": img_w,
                "original_height": img_h,
                "value": {
                    "x": (x0 / img_w) * 100,
                    "y": (y0 / img_h) * 100,
                    "width": ((x1 - x0) / img_w) * 100,
                    "height": ((y1 - y0) / img_h) * 100,
                    "rectanglelabels": ["contrato"],
                    "rotation": 0,
                },
            })

        tarea = {
            "data": {
                "image": image_url,
                "pagina": pagina,
                "diagnostico_estado": first.get("diagnostico_estado", ""),
                "diagnostico_score": first.get("diagnostico_score", ""),
                "diagnostico_flags": first.get("diagnostico_flags", ""),
                "img_width": img_w,
                "img_height": img_h,
            },
            "predictions": [
                {
                    "result": results,
                    "score": 1.0,
                }
            ],
        }
        tareas.append(tarea)

    return tareas


# ---------------------------------------------------------------------------
# Sesión autenticada con Label Studio (cookies, no token legacy)
# ---------------------------------------------------------------------------

def _ls_session(ls_url: str, email: str, password: str) -> requests.Session:
    """Inicia sesión en Label Studio y devuelve una Session con cookies."""
    s = requests.Session()
    # Obtener CSRF
    login_page = s.get(f"{ls_url}/user/login")
    login_page.raise_for_status()
    import re as _re
    csrf_match = _re.search(r'csrfmiddlewaretoken.*?value="([^"]+)"', login_page.text)
    if not csrf_match:
        raise RuntimeError("No se encontró CSRF token en la página de login")
    csrf = csrf_match.group(1)
    # Login
    resp = s.post(
        f"{ls_url}/user/login",
        data={"csrfmiddlewaretoken": csrf, "email": email, "password": password},
        headers={"Referer": f"{ls_url}/user/login"},
        allow_redirects=False,
    )
    if resp.status_code not in (200, 301, 302):
        raise RuntimeError(f"Login falló: {resp.status_code}")
    # CSRF para API calls
    s.headers.update({
        "X-CSRFToken": s.cookies.get("csrftoken", csrf),
        "Referer": ls_url,
    })
    return s


# ---------------------------------------------------------------------------
# Subcomando: import
# ---------------------------------------------------------------------------

def cmd_import(args: argparse.Namespace) -> None:
    """Crea proyecto en Label Studio y sube tareas con pre-anotaciones."""
    proposed = Path(args.proposed)
    if not proposed.exists():
        print(f"ERROR: No existe {proposed}")
        return

    paginas = _leer_proposed(proposed)
    print(f"Leídas {sum(len(v) for v in paginas.values())} cajas de {len(paginas)} páginas")

    tareas = _construir_tareas(paginas, args.image_server)

    # Conectar a Label Studio via session
    s = _ls_session(args.ls_url, args.ls_email, args.ls_password)

    # Crear proyecto
    resp = s.post(f"{args.ls_url}/api/projects", json={
        "title": args.project_name or f"Segmentación - {proposed.parent.name}",
        "label_config": LABEL_CONFIG,
    })
    resp.raise_for_status()
    project_id = resp.json()["id"]
    print(f"Proyecto creado: id={project_id}")

    # Importar tareas en lotes de 50
    batch_size = 50
    for i in range(0, len(tareas), batch_size):
        batch = tareas[i:i + batch_size]
        resp = s.post(
            f"{args.ls_url}/api/projects/{project_id}/import",
            json=batch,
        )
        resp.raise_for_status()
        print(f"  Importadas {min(i + batch_size, len(tareas))}/{len(tareas)} tareas")

    print(f"\nAbre Label Studio en {args.ls_url}/projects/{project_id} para revisar")


# ---------------------------------------------------------------------------
# Subcomando: export
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> None:
    """Exporta anotaciones de Label Studio a contract_boxes_reviewed.csv."""
    s = _ls_session(args.ls_url, args.ls_email, args.ls_password)

    # Descargar tareas con anotaciones
    resp = s.get(f"{args.ls_url}/api/projects/{args.project_id}/export?exportType=JSON")
    resp.raise_for_status()
    tasks = resp.json()
    print(f"Descargadas {len(tasks)} tareas")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pagina", "image_path", "box_id",
            "img_width", "img_height",
            "x0", "y0", "x1", "y1",
            "box_width", "box_height",
            "source",
            "diagnostico_estado", "diagnostico_score", "diagnostico_flags",
        ])

        for task in tasks:
            data = task["data"]
            pagina = data["pagina"]
            img_w = int(data["img_width"])
            img_h = int(data["img_height"])

            # Tomar la última anotación (la más reciente)
            annotations = task.get("annotations", [])
            if not annotations:
                continue
            annotation = annotations[-1]
            results = annotation.get("result", [])

            # Convertir resultados de vuelta a píxeles
            boxes = []
            for r in results:
                if r.get("type") != "rectanglelabels":
                    continue
                val = r["value"]
                x0 = int(val["x"] / 100 * img_w)
                y0 = int(val["y"] / 100 * img_h)
                w = int(val["width"] / 100 * img_w)
                h = int(val["height"] / 100 * img_h)
                boxes.append((x0, y0, x0 + w, y0 + h))

            # Ordenar por posición vertical
            boxes.sort(key=lambda b: (b[1], b[3]))

            for idx, (x0, y0, x1, y1) in enumerate(boxes, start=1):
                # Determinar source: si coincide con pre-anotación -> heuristic, sino -> manual
                source = "manual"  # conservador: si fue revisado en LS, marcarlo como manual
                writer.writerow([
                    pagina,
                    "",  # image_path se puede reconstruir después
                    idx,
                    img_w,
                    img_h,
                    x0, y0, x1, y1,
                    x1 - x0,
                    y1 - y0,
                    source,
                    data.get("diagnostico_estado", ""),
                    data.get("diagnostico_score", ""),
                    data.get("diagnostico_flags", ""),
                ])

    print(f"Exportado: {output} ({sum(1 for t in tasks for _ in t.get('annotations', [])[:1])} páginas)")


# ---------------------------------------------------------------------------
# Subcomando: serve (servidor de imágenes local)
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> None:
    """Levanta un servidor HTTP para servir imágenes a Label Studio."""
    images_dir = Path(args.images_dir).resolve()
    if not images_dir.is_dir():
        print(f"ERROR: No existe directorio {images_dir}")
        return

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(images_dir),
    )

    # Agregar CORS headers para que Label Studio pueda acceder
    original_end_headers = http.server.SimpleHTTPRequestHandler.end_headers

    class CORSHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(images_dir), **kw)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(200)
            self.end_headers()

    server = http.server.HTTPServer(("0.0.0.0", args.port), CORSHandler)
    print(f"Sirviendo imágenes de {images_dir} en http://localhost:{args.port}")
    print("Dejar corriendo mientras se usa Label Studio. Ctrl+C para parar.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Integración con Label Studio para revisión de cajas de segmentación"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- import ---
    p_imp = sub.add_parser("import", help="Importar cajas propuestas a Label Studio")
    p_imp.add_argument("--proposed", required=True, help="CSV con cajas propuestas")
    p_imp.add_argument("--images-dir", required=True, help="Directorio con imágenes")
    p_imp.add_argument("--ls-url", default="http://localhost:8080", help="URL de Label Studio")
    p_imp.add_argument("--ls-email", default="admin@local.dev", help="Email de usuario")
    p_imp.add_argument("--ls-password", default="admin123", help="Password de usuario")
    p_imp.add_argument("--image-server", default="http://localhost:8089",
                        help="URL del servidor local de imágenes")
    p_imp.add_argument("--project-name", default=None, help="Nombre del proyecto (opcional)")
    p_imp.set_defaults(func=cmd_import)

    # --- export ---
    p_exp = sub.add_parser("export", help="Exportar anotaciones revisadas a CSV")
    p_exp.add_argument("--project-id", required=True, type=int, help="ID del proyecto en Label Studio")
    p_exp.add_argument("--output", required=True, help="CSV de salida")
    p_exp.add_argument("--ls-url", default="http://localhost:8080", help="URL de Label Studio")
    p_exp.add_argument("--ls-email", default="admin@local.dev", help="Email de usuario")
    p_exp.add_argument("--ls-password", default="admin123", help="Password de usuario")
    p_exp.set_defaults(func=cmd_export)

    # --- serve ---
    p_srv = sub.add_parser("serve", help="Servidor local de imágenes para Label Studio")
    p_srv.add_argument("--images-dir", required=True, help="Directorio con imágenes")
    p_srv.add_argument("--port", type=int, default=8089, help="Puerto (default: 8089)")
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
