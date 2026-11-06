"""Matriz de giros: por dónde entra y por dónde sale cada vehículo.

Cada vehículo que cruza las líneas de calzada tiene un **origen** (la primera línea que
cruza entrando) y un **destino** (la última que cruza saliendo, después de entrar). Con
eso se arma la matriz origen -> destino: cuántos siguen derecho y cuántos doblan.

Muchos recorridos se ven a medias:
- el clip empieza o termina con el vehículo dentro del cruce;
- algo lo tapa en el medio (en muestra_3, un árbol tapa justo la salida por Rodríguez
  Peña) y el tracker lo parte en dos.

Para el primer caso se infiere el origen: un vehículo que aparece ya adentro, a pocos
metros pasada una entrada (la cola de Córdoba al arrancar el clip), entró por ahí.
Para el segundo se reconstruye el recorrido: un vehículo que entró y no se lo vio
salir se empareja con uno que salió y no se lo vio entrar, si el segundo aparece poco
después, cerca de donde se perdió el primero y en la dirección en que iba. Buscar solo
entre esos "huérfanos" deja pocos candidatos, a diferencia de una unión general de tracks.

Los porcentajes de giro se calculan con los recorridos completos (vistos o
reconstruidos), con intervalo de confianza de Wilson, y el volumen de cada movimiento se
estima aplicando ese reparto a todos los que entraron por cada origen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas_traffic.config import Geometria, Linea
from atlas_traffic.correction import wilson
from atlas_traffic.geometry import lado
from atlas_traffic.stitching import resumen_extremos

__all__ = ["COLUMNAS_GIROS", "movimientos", "matriz_giros", "reparto_giros"]

RECORRIDOS_COMPLETOS = ["completo", "origen_inferido", "reconstruido"]
COLUMNAS_GIROS = [
    "track_id", "origen", "destino", "estado", "ts_entrada", "ts_salida", "track_id_salida",
]  # fmt: skip


def _recorridos(cruces: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for tid, g in cruces.sort_values("ts_cruce").groupby("track_id"):
        ent = g[g["sentido"] == "entrante"]
        origen = ts_e = destino = ts_s = None
        if not ent.empty:
            origen, ts_e = ent["aproximacion"].iat[0], float(ent["ts_cruce"].iat[0])
        sal = g[g["sentido"] == "saliente"]
        if ts_e is not None:
            sal = sal[sal["ts_cruce"] > ts_e]
        if not sal.empty:
            destino, ts_s = sal["aproximacion"].iat[-1], float(sal["ts_cruce"].iat[-1])
        if origen is not None and origen == destino:
            # entra y sale por la misma línea: una caja que tiembla sobre ella, no un giro
            estado = "dudoso"
        elif origen and destino:
            estado = "completo"
        elif origen:
            estado = "solo_entrada"
        else:
            estado = "solo_salida"
        filas.append((int(tid), origen, destino, estado, ts_e, ts_s, int(tid)))
    return pd.DataFrame(filas, columns=COLUMNAS_GIROS)


def _distancia_a_segmento(linea: Linea, p: np.ndarray) -> float:
    a, b = (np.asarray(q, dtype=np.float64) for q in linea.puntos)
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
    return float(np.hypot(*(p - (a + t * (b - a)))))


def _origen_inferido(
    geo: Geometria, p: np.ndarray, v: np.ndarray, distancia_max_px: float
) -> str | None:
    """Entrada más cercana a p, si p quedó del lado de adentro, a menos de la distancia,
    y el vehículo (si se mueve) se aleja de ella hacia adentro."""
    mejor, dist_mejor = None, distancia_max_px
    for linea in geo.lineas:
        if "vehiculo" not in linea.clases or linea.sentido_permitido != "entrante":
            continue
        # lado de adentro: el que se alcanza al cruzar la línea entrando
        adentro = 1 if linea.sentido_positivo == "entrante" else -1
        if np.sign(lado(linea, p)) != adentro:
            continue
        # ¿se aleja de la línea hacia adentro? (un vehículo quieto en la cola pasa)
        rapidez = float(np.linalg.norm(v))
        if rapidez > 5:
            despues = lado(linea, p + v / rapidez)
            if np.sign(despues - lado(linea, p)) != adentro:
                continue
        d = _distancia_a_segmento(linea, p)
        if d <= dist_mejor:
            mejor, dist_mejor = linea.id, d
    return mejor


def movimientos(
    cruces: pd.DataFrame,
    det: pd.DataFrame,
    geo: Geometria,
    hueco_max_s: float = 3.0,
    distancia_max_px: float = 100.0,
    distancia_origen_px: float = 60.0,
) -> pd.DataFrame:
    """Un registro por vehículo con origen, destino y estado del recorrido.

    estado: completo (visto entero) | origen_inferido | reconstruido (dos pedazos
    emparejados) | solo_entrada | solo_salida | dudoso. Un recorrido reconstruido queda
    con el track_id del pedazo de entrada.
    """
    if det.empty:
        return pd.DataFrame(columns=COLUMNAS_GIROS)
    det = det[det["clase"] == "vehiculo"]
    cruces = cruces[cruces["clase"] == "vehiculo"] if not cruces.empty else cruces
    rec = _recorridos(cruces) if not cruces.empty else pd.DataFrame(columns=COLUMNAS_GIROS)
    ext = resumen_extremos(det).set_index("track_id")
    # vehículos sin ningún cruce: pueden tener origen inferido y ser la mitad de un recorrido
    sin_cruces = sorted(set(ext.index) - set(rec["track_id"]))
    extra = pd.DataFrame({"track_id": sin_cruces, "estado": "sin_cruces",
                          "track_id_salida": sin_cruces})  # fmt: skip
    rec = pd.concat([rec, extra], ignore_index=True) if sin_cruces else rec
    rec = rec.astype({"track_id": int}).set_index("track_id")
    for tid in rec.index[rec["origen"].isna()]:
        if tid not in ext.index:
            continue
        p = np.array([ext.at[tid, "x_ini"], ext.at[tid, "y_ini"]])
        origen = _origen_inferido(geo, p, ext.at[tid, "v_ini"], distancia_origen_px)
        if origen is None:
            continue
        rec.loc[tid, ["origen", "ts_entrada"]] = [origen, float(ext.at[tid, "t_ini"])]
        sale = rec.at[tid, "estado"] == "solo_salida"
        # un pedazo sin cruces no cuenta como entrada: solo sirve para emparejarlo
        rec.loc[tid, "estado"] = "origen_inferido" if sale else "fragmento"
        if origen == rec.at[tid, "destino"]:
            rec.loc[tid, "estado"] = "dudoso"
    rec = rec[rec["estado"] != "sin_cruces"]
    entradas = rec[rec["estado"].isin(["solo_entrada", "fragmento"])].reset_index()
    salidas = rec[rec["estado"] == "solo_salida"].reset_index()

    candidatos = []
    for e in entradas.itertuples():
        a = ext.loc[e.track_id]
        for s in salidas.itertuples():
            b = ext.loc[s.track_id]
            hueco = b.t_ini - a.t_fin
            if not 0 < hueco <= hueco_max_s:
                continue
            salto = np.array([b.x_ini - a.x_fin, b.y_ini - a.y_fin])
            dist = float(np.hypot(*salto))
            if dist > distancia_max_px:
                continue
            # si el primero se movía, el segundo tiene que aparecer hacia donde iba
            v = np.linalg.norm(a.v_fin)
            if v > 5 and dist > 5 and float(a.v_fin @ salto) / (v * dist) < 0.3:
                continue
            costo = dist / distancia_max_px + hueco / hueco_max_s
            candidatos.append((costo, e.track_id, s.track_id))

    usados_e: set[int] = set()
    usados_s: set[int] = set()
    for _, te, ts_ in sorted(candidatos):
        if te in usados_e or ts_ in usados_s:
            continue
        usados_e.add(te)
        usados_s.add(ts_)
        rec.loc[te, ["destino", "ts_salida", "estado", "track_id_salida"]] = [
            rec.loc[ts_, "destino"],
            rec.loc[ts_, "ts_salida"],
            "reconstruido",
            ts_,
        ]
        if rec.loc[te, "origen"] == rec.loc[te, "destino"]:
            rec.loc[te, "estado"] = "dudoso"
    rec = rec.drop(index=sorted(usados_s))
    rec = rec[rec["estado"] != "fragmento"]  # los que no se emparejaron se descartan
    return rec.reset_index()[COLUMNAS_GIROS]


def matriz_giros(mov: pd.DataFrame) -> pd.DataFrame:
    """Conteo origen x destino de los recorridos completos (vistos o reconstruidos)."""
    ok = mov[mov["estado"].isin(RECORRIDOS_COMPLETOS)]
    return pd.crosstab(ok["origen"], ok["destino"])


def reparto_giros(mov: pd.DataFrame, confianza: float = 0.95) -> pd.DataFrame:
    """Por movimiento: recorridos completos, % sobre su origen (IC Wilson) y volumen
    estimado (entradas totales por ese origen x %)."""
    columnas = ["origen", "destino", "observados", "pct", "pct_inf", "pct_sup",
                "entradas_origen", "volumen_estimado"]  # fmt: skip
    if mov.empty:
        return pd.DataFrame(columns=columnas)
    m = matriz_giros(mov)
    entradas = mov.dropna(subset=["origen"]).query("estado != 'dudoso'")
    entradas = entradas.groupby("origen").size()
    filas = []
    for origen, fila in m.iterrows():
        n = int(fila.sum())
        for destino, k in fila.items():
            if k == 0:
                continue
            p, lo, hi = wilson(int(k), n, confianza)
            tot = int(entradas.get(origen, n))
            filas.append((origen, destino, int(k), p, lo, hi, tot, p * tot))
    return pd.DataFrame(filas, columns=columnas)
