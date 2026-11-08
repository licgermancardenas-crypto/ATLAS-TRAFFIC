"""Vehículos detenidos donde no deben: sobre la senda peatonal o en la calzada.

1. Senda bloqueada: un vehículo detenido (menos de 3 km/h) con al menos el 30 % de su
   apoyo en el suelo sobre una senda peatonal, durante 2 s o más. Pasa sobre todo en la
   cola: el semáforo corta y el auto queda encima de la senda.
   - El apoyo es la parte baja de la caja (el 40 % inferior), llevada al suelo en metros.
     Un solo punto no alcanza: un auto de 4,5 m puede tener la cola antes de la senda y la
     trompa encima.
   - Las sendas son zonas en metros, comunes a todos los encuadres de la esquina
     (configs/camaras.yaml). Cada encuadre usa solo las sendas que se midieron en él: fuera
     de esa zona la calibración del suelo no vale.
2. Detención prolongada: un vehículo detenido 20 s o más en la calzada mientras el
   tránsito de al lado sigue circulando (no es una cola de semáforo). Puede ser doble
   fila, estacionamiento prohibido o ascenso y descenso: el sistema no sabe dónde está el
   carril de estacionamiento, así que la alerta es para revisar, no una infracción. Los
   colectivos se excluyen (paran en la parada).

Gravedad de la senda bloqueada: leve menos de 5 s, moderada 5 a 15 s, grave 15 s o más;
un nivel más si había peatones en la senda mientras estaba bloqueada.

Sin OpenCV. Validación a ojo en docs/infracciones.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas_traffic.config import InfraccionesCfg

__all__ = ["COLUMNAS_INFRACCIONES", "apoyo_en_sendas", "detectar_infracciones"]

COLUMNAS_INFRACCIONES = [
    "tipo",  # senda_bloqueada | detencion_prolongada
    "track_id",
    "track_ids",  # todos los seguimientos del mismo vehículo, si el tracker lo partió
    "senda",
    "ts",  # inicio
    "ts_fin",
    "duracion_s",
    "x_m",
    "y_m",
    "peatones",  # personas en la senda mientras estaba bloqueada
    "gravedad",
]
NIVELES = ["leve", "moderado", "grave"]

# puntos de la parte baja de la caja, en fracciones del ancho y del alto
_U, _V = np.meshgrid(np.linspace(0.15, 0.85, 5), np.linspace(0.6, 1.0, 3))
_U, _V = _U.ravel(), _V.ravel()


def _aplicar_h(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.c_[p, np.ones(len(p))] @ np.asarray(H, float).T
    return q[:, :2] / q[:, 2:3]


def apoyo_en_sendas(
    tray: pd.DataFrame,
    homografias: dict[int, np.ndarray],
    H_suelo,
    sendas: dict[str, tuple[tuple[float, float], tuple[float, float]]],
) -> pd.DataFrame:
    """Por detección: fracción del apoyo sobre cada senda -> columnas `senda` y `frac_senda`.

    `tray`: frame, x1..y2 (coordenadas del cuadro). `sendas`: id -> ((x0, x1), (y0, y1)) m.
    """
    if tray.empty or not sendas:
        return tray.assign(senda=None, frac_senda=0.0)
    n = len(_U)
    px = tray["x1"].to_numpy()[:, None] + _U[None] * (tray["x2"] - tray["x1"]).to_numpy()[:, None]
    py = tray["y1"].to_numpy()[:, None] + _V[None] * (tray["y2"] - tray["y1"]).to_numpy()[:, None]
    pts = np.c_[px.ravel(), py.ravel()]
    frames = np.repeat(tray["frame"].to_numpy(), n)
    ref = pts.copy()
    for f in np.unique(frames):
        H = homografias.get(int(f))
        if H is not None:
            m = frames == f
            ref[m] = _aplicar_h(H, pts[m])
    g = _aplicar_h(H_suelo, ref).reshape(len(tray), n, 2)
    fracs = {}
    for sid, ((x0, x1), (y0, y1)) in sendas.items():
        dentro = (g[..., 0] >= x0) & (g[..., 0] <= x1) & (g[..., 1] >= y0) & (g[..., 1] <= y1)
        fracs[sid] = dentro.mean(axis=1)
    f = pd.DataFrame(fracs, index=tray.index)
    return tray.assign(senda=f.idxmax(axis=1), frac_senda=f.max(axis=1))


def _tramos(ts: np.ndarray, hueco_s: float) -> list[tuple[int, int]]:
    """Índices (inicio, fin) de los tramos de `ts` (ordenado) sin huecos mayores a hueco_s."""
    if len(ts) == 0:
        return []
    cortes = np.where(np.diff(ts) > hueco_s)[0]
    return list(zip(np.r_[0, cortes + 1], np.r_[cortes, len(ts) - 1], strict=True))


def _unir(ev: pd.DataFrame, cfg: InfraccionesCfg) -> pd.DataFrame:
    """Une eventos del mismo tipo y lugar separados por poco tiempo (el tracker partió al
    vehículo en dos seguimientos)."""
    if ev.empty:
        return ev
    ev = ev.sort_values("ts").reset_index(drop=True)
    grupos: list[list[int]] = []
    for i, r in ev.iterrows():
        for gr in grupos:
            u = ev.loc[gr[-1]]
            cerca = np.hypot(u["x_m"] - r["x_m"], u["y_m"] - r["y_m"]) <= cfg.union_lugar_m
            seguido = r["ts"] - u["ts_fin"] <= cfg.union_hueco_s
            if u["tipo"] == r["tipo"] and u["senda"] == r["senda"] and cerca and seguido:
                gr.append(i)
                break
        else:
            grupos.append([i])
    filas = []
    for gr in grupos:
        g = ev.loc[gr]
        ids = list(dict.fromkeys(int(t) for t in g["track_id"]))
        filas.append({
            "tipo": g["tipo"].iat[0], "track_id": ids[0],
            "track_ids": ",".join(str(t) for t in ids), "senda": g["senda"].iat[0],
            "ts": float(g["ts"].min()), "ts_fin": float(g["ts_fin"].max()),
            "x_m": float(g["x_m"].mean()), "y_m": float(g["y_m"].mean()),
            "peatones": int(g["peatones"].sum()),
        })  # fmt: skip
    out = pd.DataFrame(filas)
    out["duracion_s"] = out["ts_fin"] - out["ts"]
    return out


def _gravedad_senda(dur: float, peatones: int) -> str:
    k = 0 if dur < 5 else 1 if dur < 15 else 2
    return NIVELES[min(2, k + (1 if peatones > 0 else 0))]


def detectar_infracciones(
    veh: pd.DataFrame,
    ped: pd.DataFrame,
    sendas: dict[str, tuple[tuple[float, float], tuple[float, float]]],
    clases: dict[int, str],
    cfg: InfraccionesCfg,
    zona_valida: tuple[tuple[float, float], tuple[float, float]] = ((-10.0, 30.0), (-15.0, 30.0)),
) -> pd.DataFrame:
    """Senda bloqueada y detención prolongada.

    `veh`: track_id, ts, x_m, y_m, vel_kmh, senda, frac_senda (ver apoyo_en_sendas), ya en
    la zona de análisis. `ped`: track_id, ts, x_m, y_m. `clases`: track_id -> clase_detalle.
    `zona_valida`: ((x0, x1), (y0, y1)) en metros donde vale la calibración del suelo.
    """
    veh = veh.dropna(subset=["x_m", "y_m", "vel_kmh"])
    if veh.empty:
        return pd.DataFrame(columns=COLUMNAS_INFRACCIONES)
    filas = []
    quieto = veh[veh["vel_kmh"] < cfg.vel_detenido_kmh]
    # 1. senda bloqueada
    sobre = quieto[quieto["frac_senda"] >= cfg.senda_frac_min]
    for (tid, sid), g in sobre.groupby(["track_id", "senda"]):
        g = g.sort_values("ts")
        ts = g["ts"].to_numpy()
        for a, b in _tramos(ts, cfg.hueco_s):
            if ts[b] - ts[a] < cfg.senda_min_s:
                continue
            (x0, x1), (y0, y1) = sendas[sid]
            # peatones dentro de la senda, vistos ahí varias veces (no un roce del borde)
            en_senda = ped["x_m"].between(x0, x1) & ped["y_m"].between(y0, y1)
            p = ped[ped["ts"].between(ts[a], ts[b]) & en_senda]
            vistos = p.groupby("track_id").size()
            p = p[p["track_id"].isin(vistos[vistos >= cfg.senda_min_detecciones_peaton].index)]
            filas.append({
                "tipo": "senda_bloqueada", "track_id": int(tid), "senda": sid,
                "ts": float(ts[a]), "ts_fin": float(ts[b]),
                "x_m": float(g["x_m"].iloc[a : b + 1].median()),
                "y_m": float(g["y_m"].iloc[a : b + 1].median()),
                "peatones": int(p["track_id"].nunique()),
            })  # fmt: skip
    # 2. detención prolongada con el tránsito de al lado circulando
    for tid, g in quieto.groupby("track_id"):
        if clases.get(int(tid)) in cfg.detencion_excluir:
            continue
        g = g.sort_values("ts")
        ts = g["ts"].to_numpy()
        for a, b in _tramos(ts, cfg.hueco_s):
            if ts[b] - ts[a] < cfg.detencion_min_s / 2:  # se unen tramos partidos más abajo
                continue
            x, y = g["x_m"].iloc[a : b + 1].median(), g["y_m"].iloc[a : b + 1].median()
            # solo donde el suelo está medido: más lejos, la posición en metros no vale
            if not (zona_valida[0][0] <= x <= zona_valida[0][1]
                    and zona_valida[1][0] <= y <= zona_valida[1][1]):  # fmt: skip
                continue
            v = veh[(veh["track_id"] != tid) & veh["ts"].between(ts[a], ts[b])]
            cerca = v[np.hypot(v["x_m"] - x, v["y_m"] - y) < cfg.detencion_radio_m]
            circulan = cerca.loc[cerca["vel_kmh"] > cfg.detencion_vel_vecinos_kmh, "track_id"]
            parados = cerca.loc[cerca["vel_kmh"] < cfg.vel_detenido_kmh, "track_id"]
            if (circulan.nunique() < cfg.detencion_min_circulan
                    or parados.nunique() > cfg.detencion_max_parados):  # fmt: skip
                continue
            filas.append({
                "tipo": "detencion_prolongada", "track_id": int(tid), "senda": None,
                "ts": float(ts[a]), "ts_fin": float(ts[b]), "x_m": float(x), "y_m": float(y),
                "peatones": 0,
            })  # fmt: skip
    if not filas:
        return pd.DataFrame(columns=COLUMNAS_INFRACCIONES)
    ev = _unir(pd.DataFrame(filas), cfg)
    det = ev["tipo"] == "detencion_prolongada"
    ev = ev[~det | (ev["duracion_s"] >= cfg.detencion_min_s)].copy()
    ev["gravedad"] = [
        _gravedad_senda(d, p) if t == "senda_bloqueada" else "leve"
        for t, d, p in zip(ev["tipo"], ev["duracion_s"], ev["peatones"], strict=True)
    ]
    return ev[COLUMNAS_INFRACCIONES].sort_values("ts", ignore_index=True)
