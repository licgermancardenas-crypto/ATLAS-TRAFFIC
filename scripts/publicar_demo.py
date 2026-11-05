"""Copia a demo/outputs/ los datos que usa el dashboard, para publicarlo en la nube.

outputs/ está fuera de git (videos derivados, detecciones crudas). El dashboard solo
necesita los consolidados, los conteos por clip y los recortes, que pesan poco y no
permiten leer patentes ni distinguir caras (vienen de video de 464 px de ancho).

Uso: python scripts/publicar_demo.py      (después de `atlas-traffic batch`)
"""

import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ORIGEN = RAIZ / "outputs"
DESTINO = RAIZ / "demo" / "outputs"

CONSOLIDADOS = ["summary.parquet", "tracks_todos.parquet", "cruces_todos.parquet",
                "eventos_todos.parquet"]  # fmt: skip


def main() -> None:
    if not (ORIGEN / "summary.parquet").exists():
        raise SystemExit("No hay outputs/summary.parquet: corré `atlas-traffic batch` primero.")
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir(parents=True)
    for nombre in CONSOLIDADOS:
        if (ORIGEN / nombre).exists():
            shutil.copy2(ORIGEN / nombre, DESTINO / nombre)
    n_imagenes = 0
    for corrida in sorted((ORIGEN / "runs").iterdir()):
        if not corrida.is_dir():
            continue
        dest = DESTINO / "runs" / corrida.name
        dest.mkdir(parents=True)
        if (corrida / "conteo.csv").exists():
            shutil.copy2(corrida / "conteo.csv", dest / "conteo.csv")
        for carpeta in ("thumbs", "eventos"):
            if (corrida / carpeta).is_dir():
                shutil.copytree(corrida / carpeta, dest / carpeta)
                n_imagenes += len(list((dest / carpeta).glob("*.jpg")))
    peso = sum(f.stat().st_size for f in DESTINO.rglob("*") if f.is_file()) / 1e6
    print(f"demo/outputs: {len(CONSOLIDADOS)} tablas, {n_imagenes} recortes, {peso:.1f} MB")


if __name__ == "__main__":
    main()
