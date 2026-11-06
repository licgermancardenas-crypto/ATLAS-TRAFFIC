"""Calibración métrica del suelo de cada encuadre (para velocidades en km/h).

1. El encuadre de referencia (video original del iPhone, el más nítido) se calibra solo
   con las cuatro sendas peatonales de la esquina (ver src/atlas_traffic/suelo.py).
2. Los demás encuadres muestran las mismas sendas desde otro ángulo: se calibran con las
   posiciones en metros de esas franjas, ya conocidas por la referencia.

Cada senda se da con un segmento que la recorre por el medio, atravesando las franjas,
en coordenadas del cuadro de referencia del clip, siempre en el mismo sentido:
- sendas que cruzan Córdoba ("x"): de la mano izquierda a la derecha de la avenida
  (mirando hacia donde circula);
- sendas que cruzan Rodríguez Peña ("y"): del lado de la cámara hacia el fondo.

Salidas: configs/suelo/<encuadre>.yaml y una vista desde arriba en outputs/suelo/.

Uso: python scripts/calibrar_suelo.py
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atlas_traffic.clips import load_clips
from atlas_traffic.config import load_app_config, load_geometria
from atlas_traffic.render import homografias_por_frame
from atlas_traffic.suelo import (
    CalibracionSuelo,
    Senda,
    a_metros,
    ajustar_suelo,
    centros_franjas,
    fondo_estabilizado,
    guardar_calibracion_suelo,
    transferir_suelo,
)

RAIZ = Path(__file__).resolve().parents[1]
# franjas de 0,50 m separadas 0,50 m (MSH de Vialidad Nacional, IV.3; verificado con el
# ancho de los carriles de Córdoba, ver docs/velocidad.md)
PERIODO_M = 1.0

REFERENCIA = "encuadre_n1"
SENDAS = {
    "encuadre_n1": [
        Senda("cordoba_frente", "x", (470, 1455), (1065, 1385)),
        Senda("cordoba_fondo", "x", (300, 1213), (790, 1201)),
        Senda("rp_izq", "y", (170, 1428), (163, 1298)),
        Senda("rp_der", "y", (1032, 1336), (903, 1240)),
    ],
    "encuadre_m1": [
        Senda("cordoba_frente", "x", (172, 527), (400, 475)),
        Senda("cordoba_fondo", "x", (92, 420), (277, 396)),
        Senda("rp_izq", "y", (30, 507), (27, 465)),
    ],
    "encuadre_m2": [
        Senda("cordoba_frente", "x", (52, 512), (277, 447)),
        Senda("cordoba_fondo", "x", (2, 415), (145, 387)),
        Senda("rp_der", "y", (280, 422), (205, 390)),
    ],
    "encuadre_m3": [
        Senda("cordoba_frente", "x", (270, 535), (425, 512)),  # solo la parte sin cola
        Senda("cordoba_fondo", "x", (112, 446), (300, 432)),
    ],
}
# posición aproximada (q, s0) en metros de cada senda respecto de la primera, solo como
# punto de partida del ajuste de la referencia
GUIA = {"cordoba_fondo": (18.0, 0.0), "rp_izq": (-3.0, 4.0), "rp_der": (17.0, 4.0)}


def clip_de(encuadre: str):
    cfg = load_app_config(RAIZ / "configs/default.yaml")
    for c in load_clips(RAIZ / cfg.rutas.clips):
        if load_geometria(RAIZ / c.geometria).encuadre == encuadre:
            return c, cfg
    raise SystemExit(f"no hay clips del encuadre {encuadre}")


def fondo_de(encuadre: str) -> np.ndarray:
    c, cfg = clip_de(encuadre)
    hs = homografias_por_frame(
        pd.read_parquet(RAIZ / cfg.rutas.outputs / "runs" / c.id / "estabilizacion.parquet")
    )
    geo = load_geometria(RAIZ / c.geometria)
    return fondo_estabilizado(RAIZ / c.archivo, hs, tuple(geo.resolucion))


def vista_desde_arriba(fondo: np.ndarray, H: np.ndarray, salida: Path) -> None:
    """Fondo llevado al suelo: 20 px por metro, de X=-10 a 30 m y de Y=-15 a 30 m."""
    px_m = 20
    A = np.array([[px_m, 0, 10 * px_m], [0, -px_m, 30 * px_m], [0, 0, 1]], dtype=float)
    img = cv2.warpPerspective(fondo, A @ H, (40 * px_m, 45 * px_m))
    for m in range(-10, 31, 5):  # cuadrícula cada 5 m
        cv2.line(img, (int((m + 10) * px_m), 0), (int((m + 10) * px_m), img.shape[0]),
                 (0, 200, 255), 1)  # fmt: skip
    for m in range(-15, 31, 5):
        y = int((30 - m) * px_m)
        cv2.line(img, (0, y), (img.shape[1], y), (0, 200, 255), 1)
        cv2.putText(img, f"{m} m", (3, y - 3), 0, 0.45, (0, 200, 255), 1)
    salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida), img)


def main() -> None:
    referencia = None
    for encuadre in [REFERENCIA] + [e for e in SENDAS if e != REFERENCIA]:
        fondo = fondo_de(encuadre)
        sendas = [centros_franjas(fondo, s) for s in SENDAS[encuadre]]
        n = sum(len(s.centros) for s in sendas)
        verif: dict[str, float] = {}
        if encuadre == REFERENCIA:
            H, residuo, posiciones = ajustar_suelo(sendas, PERIODO_M, GUIA)
            referencia = posiciones
            notas = "Ajuste propio con las cuatro sendas de la esquina."
        else:
            H, residuo, desfases = transferir_suelo(sendas, referencia, PERIODO_M)
            posiciones = {}
            usadas = ", ".join(f"{k} (desfase {v})" for k, v in desfases.items())
            notas = f"Transferida desde {REFERENCIA} con las sendas {usadas}."
        # verificación: período medido en el suelo, franja por franja
        periodos = []
        for s in sendas:
            m = a_metros(H, s.centros)
            periodos += list(np.hypot(*np.diff(m, axis=0).T) / np.diff(s.indices))
        verif["periodo_medio_m"] = round(float(np.mean(periodos)), 3)
        verif["periodo_desvio_m"] = round(float(np.std(periodos)), 3)
        cal = CalibracionSuelo(
            encuadre=encuadre, homografia=H.tolist(), periodo_m=PERIODO_M,
            residuo_px=round(residuo, 2), n_franjas=n, verificaciones=verif,
            sendas=posiciones, notas=notas,
        )  # fmt: skip
        guardar_calibracion_suelo(cal, RAIZ / "configs" / "suelo" / f"{encuadre}.yaml")
        vista_desde_arriba(fondo, H, RAIZ / "outputs" / "suelo" / f"{encuadre}.jpg")
        print(f"{encuadre}: {n} franjas, error {residuo:.2f} px, período "
              f"{verif['periodo_medio_m']} ± {verif['periodo_desvio_m']} m. {notas}")  # fmt: skip


if __name__ == "__main__":
    main()
