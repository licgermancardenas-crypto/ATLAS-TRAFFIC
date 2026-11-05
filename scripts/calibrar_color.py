"""Calibración y validación del color (Fase 3b). Ver docs/validacion_color.md.

Las etiquetas manuales están en docs/validacion_color_muestras.csv, en dos muestras
disjuntas: "calibracion" (para elegir umbrales) y "validacion" (para medir una vez).

Uso:
    python scripts/calibrar_color.py buscar              # grilla de umbrales sobre calibración
    python scripts/calibrar_color.py validar             # mide la config actual en validación
"""

import itertools
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atlas_traffic.attributes import (
    COLORES,
    brillo_fondo,
    clasificar_pixeles,
    muestrear,
    region_color,
    votar_color,
)
from atlas_traffic.config import ColorCfg, load_app_config

RAIZ = Path(__file__).resolve().parents[1]
SATURADOS = {"rojo", "azul", "verde", "amarillo"}


def leer_frame(clip: str, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(RAIZ / f"data/raw/{clip}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    return img


def extraer(muestra: pd.DataFrame, cfg: ColorCfg) -> list:
    """Por track: lista de (hsv de la región, brillo de fondo) de los frames que votan."""
    salida, dets = [], {}
    for r in muestra.itertuples():
        if r.clip_id not in dets:
            dets[r.clip_id] = pd.read_parquet(
                RAIZ / f"outputs/runs/{r.clip_id}/detecciones.parquet"
            )
        det = dets[r.clip_id]
        m = muestrear(det[det.track_id == r.track_id], 464, 832, cfg)
        filas = []
        for d in m[m.votar].itertuples():
            frame = leer_frame(r.clip_id, d.frame)
            cajas = det[det.frame == d.frame][["x1", "y1", "x2", "y2"]].to_numpy(dtype=int)
            caja = (d.x1, d.y1, d.x2, d.y2)
            x1, y1, x2, y2 = region_color(caja, "vehiculo")
            hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            filas.append((hsv, brillo_fondo(frame, caja, cajas)))
        salida.append(filas)
    return salida


def predecir(frames: list, cfg: ColorCfg) -> str:
    votos = []
    for hsv, fondo in frames:
        et = clasificar_pixeles(hsv, cfg, fondo).ravel()
        et = et[et >= 0]
        if et.size:
            votos.append(COLORES[int(np.bincount(et, minlength=len(COLORES)).argmax())])
    return votar_color(votos, cfg.confianza_min)[0]


def puntaje(datos: list, etiquetas: list, cfg: ColorCfg) -> tuple[float, float]:
    """(acierto en saturados, acierto en acromáticos) sobre los tracks etiquetados."""
    res = {True: [], False: []}
    for frames, real in zip(datos, etiquetas, strict=True):
        if isinstance(real, str):
            res[real in SATURADOS].append(predecir(frames, cfg) == real)
    return float(np.mean(res[True])), float(np.mean(res[False]))


def main() -> None:
    modo = sys.argv[1] if len(sys.argv) > 1 else "validar"
    cfg_base = load_app_config(RAIZ / "configs/default.yaml").color
    etiquetas = pd.read_csv(RAIZ / "docs/validacion_color_muestras.csv")
    cache = RAIZ / "outputs/color_cache.pkl"
    if cache.exists():
        datos = pickle.loads(cache.read_bytes())
    else:
        datos = {m: extraer(g, cfg_base) for m, g in etiquetas.groupby("muestra")}
        cache.write_bytes(pickle.dumps(datos))
    lab = {m: list(g["color_manual"]) for m, g in etiquetas.groupby("muestra")}

    if modo == "validar":
        s, a = puntaje(datos["validacion"], lab["validacion"], cfg_base)
        print(f"Validación con la config actual: saturados {s:.0%}, acromáticos {a:.0%}")
        return
    resultados = []
    for rn, rb, sm in itertools.product(
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [1.1, 1.2, 1.3, 1.4, 1.6], [40, 50, 60, 70, 80]
    ):
        cfg = cfg_base.model_copy(update=dict(r_negro=rn, r_blanco=rb, sat_min=sm))
        s, a = puntaje(datos["calibracion"], lab["calibracion"], cfg)
        resultados.append((min(s, a), s + a, rn, rb, sm, s, a))
    for r in sorted(resultados, reverse=True)[:10]:
        print(f"r_negro={r[2]} r_blanco={r[3]} sat_min={r[4]}: saturados {r[5]:.0%}, "
              f"acromáticos {r[6]:.0%}")  # fmt: skip


if __name__ == "__main__":
    main()
