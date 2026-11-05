"""Evalúa un modelo de detección contra los conteos manuales (primeros 20 s de cada clip).

Esos 20 s quedaron fuera del dataset de entrenamiento (ver dataset_finetuning.py), así
que la comparación es justa. Corre el pipeline completo con el modelo indicado en una
carpeta aparte (no pisa outputs/) y compara, línea por línea, con
docs/validacion_conteo_registros.csv y con el modelo actual.

Uso: python scripts/evaluar_modelo.py modelos/atlas_esquina_yolov8s.pt
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
VENTANA_S = 20


def correr(modelo: Path, etiqueta: str, clips: list[str]) -> Path:
    base = yaml.safe_load((RAIZ / "configs/default.yaml").read_text(encoding="utf-8"))
    salida = RAIZ / "outputs_eval" / etiqueta
    salida.mkdir(parents=True, exist_ok=True)
    base["deteccion"]["modelo"] = str(modelo)
    base["rutas"]["outputs"] = str(salida)
    base["rutas"]["calibracion"] = str(salida / "calibracion.yaml")
    config = salida / "config.yaml"
    config.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")
    for clip in clips:
        print(f"[{etiqueta}] procesando {clip} ({VENTANA_S} s)...", flush=True)
        subprocess.run([sys.executable, "-m", "atlas_traffic", "run", "--clip", clip,
                        "--max-segundos", str(VENTANA_S), "--config", str(config)],
                       check=True, cwd=RAIZ, capture_output=True)  # fmt: skip
    return salida / "runs"


def contar(runs: Path, clip: str) -> pd.DataFrame:
    c = pd.read_parquet(runs / clip / "cruces.parquet")
    return c[c["ts_cruce"] < VENTANA_S]


def main() -> None:
    modelo = Path(sys.argv[1])
    registros = pd.read_csv(RAIZ / "docs/validacion_conteo_registros.csv")
    clips = sorted(registros["clip_id"].unique())
    runs_nuevo = correr(modelo, modelo.stem, clips)
    runs_actual = RAIZ / "outputs" / "runs"  # modelo actual, ya procesado

    filas = []
    for r in registros.itertuples():
        nuevo = (contar(runs_nuevo, r.clip_id)["aproximacion"] == r.linea).sum()
        actual = (contar(runs_actual, r.clip_id)["aproximacion"] == r.linea).sum()
        filas.append((r.clip_id, r.linea, r.luz, r.manual, actual, nuevo))
    t = pd.DataFrame(filas, columns=["clip", "linea", "luz", "manual", "actual", "nuevo"])
    for col in ("actual", "nuevo"):
        t[f"err_{col}"] = ((t[col] - t["manual"]) / t["manual"].where(t["manual"] > 0)).map(
            lambda x: "" if pd.isna(x) else f"{x:+.0%}"
        )
    print("\nCruces de vehículos, primeros 20 s (manual = conteo de referencia):")
    print(t.to_string(index=False))
    tot = t[t["manual"] > 0]
    for col in ("actual", "nuevo"):
        recall = tot[[col, "manual"]].min(axis=1).sum() / tot["manual"].sum()
        print(f"  detección global {col}: {recall:.0%}")

    print("\nPeatones y motos detectados (tracks, primeros 20 s):")
    for clip in clips:
        partes = []
        for nombre, runs in (("actual", runs_actual), ("nuevo", runs_nuevo)):
            d = pd.read_parquet(runs / clip / "detecciones.parquet")
            d = d[d["ts"] < VENTANA_S]
            per = d[d["clase"] == "peaton"]["track_id"].nunique()
            motos = d[d["clase_detalle"] == "motorcycle"]["track_id"].nunique()
            partes.append(f"{nombre}: {per} peatones, {motos} motos")
        print(f"  {clip}: " + " | ".join(partes))


if __name__ == "__main__":
    main()
