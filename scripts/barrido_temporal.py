"""Barrido temporal (slit-scan) por línea de conteo, independiente del detector.

Para cada frame procesado se muestrean los píxeles sobre la línea (en coordenadas de
referencia, llevados al frame con la inversa de la H de estabilización) y se apilan en
el tiempo. Cada vehículo que cruza deja una mancha.

Uso: python scripts/barrido_temporal.py <clip> [segundos] [carpeta_salida]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atlas_traffic.config import load_geometria
from atlas_traffic.render import homografias_por_frame

clip = sys.argv[1]
segundos = float(sys.argv[2]) if len(sys.argv) > 2 else 20
salida_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(".")
raiz = Path(__file__).resolve().parents[1]
geo = load_geometria(raiz / f"configs/geometria/encuadre_m{clip[-1]}.yaml")
hs = homografias_por_frame(pd.read_parquet(raiz / f"outputs/runs/{clip}/estabilizacion.parquet"))

cap = cv2.VideoCapture(str(raiz / f"data/raw/{clip}.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(segundos * fps)
# cada línea se ensancha a una franja de 5 px (perpendicular) para no perder vehículos finos
muestras = {}
for linea in geo.lineas:
    a, b = (np.array(p, float) for p in linea.puntos)
    largo = int(np.hypot(*(b - a)))
    t = np.linspace(0, 1, largo)
    nrm = np.array([-(b - a)[1], (b - a)[0]]) / largo
    pts = [a + np.outer(t, b - a) + k * nrm for k in (-2, -1, 0, 1, 2)]
    muestras[linea.id] = (np.stack(pts), [])

H = np.eye(3)
for i in range(n):
    ok, frame = cap.read()
    if not ok:
        break
    if i not in hs:
        continue
    H_inv = np.linalg.inv(hs[i])
    for pts, filas in muestras.values():
        p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H_inv).reshape(pts.shape)
        mx, my = p[..., 0].astype(np.float32), p[..., 1].astype(np.float32)
        fila = cv2.remap(frame, mx, my, cv2.INTER_LINEAR).mean(axis=0)  # promedio de la franja
        filas.append(fila.astype(np.uint8))
cap.release()

for lid, (_, filas) in muestras.items():
    img = np.stack(filas)  # (tiempo, posición, 3): tiempo hacia abajo
    img = cv2.resize(img, (img.shape[1] * 2, img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # marcas de tiempo cada 1 s (cada fila = 2 frames)
    for s in range(int(segundos) + 1):
        y = int(s * fps / 2)
        cv2.line(img, (0, y), (12, y), (0, 255, 255), 1)
        cv2.putText(img, str(s), (14, y + 4), 0, 0.35, (0, 255, 255), 1)
    cv2.imwrite(str(salida_dir / f"barrido_{clip}_{lid}.png"), img)
    print(lid, img.shape)
