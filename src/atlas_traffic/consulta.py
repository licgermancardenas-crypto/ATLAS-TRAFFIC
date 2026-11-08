"""Lógica de datos del dashboard, separada de Streamlit para poder testearla.

- cargar: lee los consolidados de outputs/ (los genera `batch`).
- consultar: filtra los cruces y devuelve un registro por vehículo (pestaña Consulta).
- resúmenes para el Panorama: flujo por tipo de día, congestión por franja horaria.
- alertas: sentido contrario, riesgo peatonal y exceso de velocidad en una sola lista.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ORDEN_CONGESTION = ["LOW", "MODERATE", "HIGH"]


@dataclass(frozen=True)
class Datos:
    summary: pd.DataFrame
    tracks: pd.DataFrame
    cruces: pd.DataFrame
    eventos: pd.DataFrame
    outputs: Path
    conteos: pd.DataFrame = field(default_factory=pd.DataFrame)  # conteo.csv de cada clip
    giros: pd.DataFrame = field(default_factory=pd.DataFrame)  # origen -> destino por vehículo
    conflictos: pd.DataFrame = field(default_factory=pd.DataFrame)  # riesgo vehículo-peatón

    @property
    def vacio(self) -> bool:
        return self.summary.empty


def cargar(outputs: Path) -> Datos:
    def leer(nombre: str) -> pd.DataFrame:
        ruta = outputs / nombre
        return pd.read_parquet(ruta) if ruta.exists() else pd.DataFrame()

    conteos = []
    for ruta in sorted((outputs / "runs").glob("*/conteo.csv")):
        conteos.append(pd.read_csv(ruta).assign(clip_id=ruta.parent.name))
    return Datos(
        conteos=pd.concat(conteos, ignore_index=True) if conteos else pd.DataFrame(),
        summary=leer("summary.parquet"),
        tracks=leer("tracks_todos.parquet"),
        cruces=leer("cruces_todos.parquet"),
        eventos=leer("eventos_todos.parquet"),
        giros=leer("giros_todos.parquet"),
        conflictos=leer("conflictos_todos.parquet"),
        outputs=outputs,
    )


# --------------------------------------------------------------------------- consulta


@dataclass(frozen=True)
class Filtro:
    clips: list[str] = field(default_factory=list)  # vacío = todos
    fechas: tuple[dt.date, dt.date] | None = None
    segundos: tuple[float, float] | None = None  # ventana dentro del video
    clases: list[str] = field(default_factory=list)  # car, truck, bus, motorcycle, person
    colores: list[str] = field(default_factory=list)
    aproximaciones: list[str] = field(default_factory=list)
    sentidos: list[str] = field(default_factory=list)  # entrante / saliente
    # pares (aproximación, sentido): "entró por Córdoba frente" en un solo filtro
    movimientos: list[tuple[str, str]] = field(default_factory=list)
    confianza_color_min: float = 0.0
    # buscador: hora del día real (desde, hasta; si desde > hasta, cruza la medianoche)
    franja: tuple[dt.time, dt.time] | None = None
    calles: list[str] = field(default_factory=list)  # prefijo de línea: av_cordoba, ...
    encuadres: list[str] = field(default_factory=list)  # cámaras / barrios -> encuadres
    velocidad_min_kmh: float = 0.0


def consultar(cruces: pd.DataFrame, f: Filtro) -> pd.DataFrame:
    """Vehículos (un registro por clip + track) con al menos un cruce que cumple el filtro.

    De cada vehículo se devuelve el primer cruce que cumple, así el conteo es de
    vehículos únicos y no de cruces (un auto cruza dos líneas al atravesar el cruce).
    """
    if cruces.empty:
        return cruces
    m = pd.Series(True, index=cruces.index)
    if f.clips:
        m &= cruces["clip_id"].isin(f.clips)
    if f.fechas:
        fechas = pd.to_datetime(cruces["fecha"]).dt.date
        m &= (fechas >= f.fechas[0]) & (fechas <= f.fechas[1])
    if f.segundos:
        m &= cruces["ts_cruce"].between(*f.segundos)
    if f.clases:
        m &= cruces["clase_detalle"].isin(f.clases)
    if f.colores:
        m &= cruces["color"].isin(f.colores)
    if f.aproximaciones:
        m &= cruces["aproximacion"].isin(f.aproximaciones)
    if f.sentidos:
        m &= cruces["sentido"].isin(f.sentidos)
    if f.movimientos:
        pares = pd.Series(list(zip(cruces["aproximacion"], cruces["sentido"], strict=True)),
                          index=cruces.index)  # fmt: skip
        m &= pares.isin([tuple(x) for x in f.movimientos])
    if f.confianza_color_min > 0:
        m &= cruces["color_confianza"].fillna(0) >= f.confianza_color_min
    if f.franja and "hora" in cruces:
        h = pd.to_datetime(cruces["hora"]).dt.time
        a, b = f.franja
        m &= (h >= a) & (h <= b) if a <= b else (h >= a) | (h <= b)
    if f.calles:
        m &= cruces["aproximacion"].map(calle_de_linea).isin(f.calles)
    if f.encuadres and "encuadre" in cruces:
        m &= cruces["encuadre"].isin(f.encuadres)
    if f.velocidad_min_kmh > 0:
        if "velocidad_kmh" not in cruces:
            return cruces.iloc[0:0]
        m &= cruces["velocidad_kmh"].fillna(-1) >= f.velocidad_min_kmh
    res = cruces[m].sort_values(["clip_id", "ts_cruce"])
    return res.drop_duplicates(["clip_id", "track_id"]).reset_index(drop=True)


COLUMNAS_EXPORTAR = [
    "clip_id", "track_id", "hora", "ts_cruce", "aproximacion", "sentido", "clase_detalle",
    "color", "color_confianza", "velocidad_kmh", "fecha", "dia_semana", "tipo_dia", "clima",
    "encuadre", "thumb",
]  # fmt: skip


def calle_de_linea(linea: str) -> str:
    """av_cordoba_frente -> av_cordoba (las líneas se nombran calle_lado)."""
    return linea.rsplit("_", 1)[0] if isinstance(linea, str) else linea


SEPARADOR_MOVIMIENTO = " · "


def movimientos_disponibles(cruces: pd.DataFrame) -> list[str]:
    """Etiquetas "aproximación · sentido" de los cruces que existen en los datos."""
    if cruces.empty:
        return []
    pares = (
        cruces[["aproximacion", "sentido"]]
        .drop_duplicates()
        .sort_values(["aproximacion", "sentido"])
    )
    return [f"{a}{SEPARADOR_MOVIMIENTO}{s}" for a, s in pares.itertuples(index=False)]


def parsear_movimiento(etiqueta: str) -> tuple[str, str]:
    aprox, sentido = etiqueta.split(SEPARADOR_MOVIMIENTO)
    return aprox, sentido


def para_csv(resultado: pd.DataFrame) -> bytes:
    cols = [c for c in COLUMNAS_EXPORTAR if c in resultado.columns]
    return resultado[cols].to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- panorama


def flujo_por_tipo_dia(summary: pd.DataFrame) -> pd.DataFrame:
    """Flujo, densidad y velocidad medios por tipo de día (hábil / finde)."""
    cols = {"flujo_veh_min": "mean", "densidad_media": "mean", "clip_id": "nunique"}
    if "velocidad_norm" in summary:
        cols["velocidad_norm"] = "mean"
    out = summary.groupby("tipo_dia").agg(cols).rename(columns={"clip_id": "clips"})
    return out.reset_index()


def congestion_por_franja(summary: pd.DataFrame) -> pd.DataFrame:
    """Proporción de intervalos de cada nivel por hora del día."""
    d = summary.dropna(subset=["congestion"])
    if d.empty:
        return pd.DataFrame(columns=["hora_del_dia", "congestion", "intervalos", "proporcion"])
    t = d.groupby(["hora_del_dia", "congestion"]).size().rename("intervalos").reset_index()
    t["proporcion"] = t["intervalos"] / t.groupby("hora_del_dia")["intervalos"].transform("sum")
    return t


def tabla_clips(summary: pd.DataFrame) -> pd.DataFrame:
    """Una fila por clip: contexto + promedios + nivel de congestión más frecuente."""
    g = summary.groupby("clip_id")
    out = pd.DataFrame({
        "fecha": g["fecha"].first(),
        "hora_inicio": g["hora_inicio"].first(),
        "dia_semana": g["dia_semana"].first(),
        "tipo_dia": g["tipo_dia"].first(),
        "clima": g["clima"].first(),
        "duracion_s": g["duracion_s"].sum(),
        "flujo_veh_min": g["flujo_veh_min"].mean(),
        "densidad_media": g["densidad_media"].mean(),
        "congestion": g["congestion"].agg(
            lambda s: s.mode().iat[0] if s.notna().any() else None),
    })  # fmt: skip
    return out.reset_index()


# --------------------------------------------------------------------------- alertas

COLUMNAS_ALERTAS = ["clip_id", "hora", "ts", "tipo", "gravedad", "titulo", "detalle",
                    "track_id", "thumb", "video", "encuadre"]  # fmt: skip
ORDEN_GRAVEDAD = ["grave", "moderado", "leve"]


def _col(df: pd.DataFrame, nombre: str):
    return df[nombre] if nombre in df else None


def alertas(datos: Datos, limites_kmh: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Todas las alertas en una lista, la más reciente primero.

    `limites_kmh`: encuadre -> {calle: km/h} (de configs/camaras.yaml) para el exceso de
    velocidad. Sin límites no hay alertas de velocidad.
    """
    partes = []
    e = datos.eventos
    if not e.empty:
        partes.append(pd.DataFrame({
            "clip_id": e["clip_id"], "hora": _col(e, "hora"), "ts": e["ts"],
            "tipo": "Sentido contrario", "gravedad": "grave",
            "titulo": "Circulación a contramano",
            "detalle": [f"Vehículo #{t} ({c}) cruzó {a} en sentido {s}, no permitido"
                        for t, c, a, s in zip(e["track_id"], e["clase_detalle"],
                                              e["aproximacion"], e["sentido"], strict=True)],
            "track_id": e["track_id"], "thumb": _col(e, "thumb"), "video": _col(e, "video"),
            "encuadre": _col(e, "encuadre"),
        }))  # fmt: skip
    k = datos.conflictos
    if not k.empty:
        cuando = {"peaton": "después que él", "vehiculo": "antes que él"}
        partes.append(pd.DataFrame({
            "clip_id": k["clip_id"], "hora": _col(k, "hora"), "ts": k["ts"],
            "tipo": "Riesgo peatonal", "gravedad": k["gravedad"],
            "titulo": "Vehículo por donde cruzaba un peatón",
            "detalle": [f"Vehículo #{t} a {v:.0f} km/h pasó por el lugar de un peatón en la "
                        f"calzada, {p:.1f} s {cuando.get(q, '')}"
                        for t, v, p, q in zip(k["track_id"], k["vel_kmh"], k["pet_s"],
                                              k["primero"], strict=True)],
            "track_id": k["track_id"], "thumb": _col(k, "thumb"), "video": _col(k, "video"),
            "encuadre": _col(k, "encuadre"),
        }))  # fmt: skip
    partes.append(_excesos(datos, limites_kmh or {}))
    partes = [p for p in partes if not p.empty]
    if not partes:
        return pd.DataFrame(columns=COLUMNAS_ALERTAS)
    out = pd.concat(partes, ignore_index=True)[COLUMNAS_ALERTAS]
    return out.sort_values(["hora", "ts"], ascending=False, ignore_index=True)


def _excesos(datos: Datos, limites: dict[str, dict[str, float]]) -> pd.DataFrame:
    t, c = datos.tracks, datos.cruces
    vacio = pd.DataFrame(columns=COLUMNAS_ALERTAS)
    if t.empty or c.empty or "velocidad_max_kmh" not in t or not limites or "encuadre" not in t:
        return vacio
    # calle por la que circula: la de su primer cruce
    primer = c[c["clase"] == "vehiculo"].sort_values("ts_cruce").drop_duplicates(
        ["clip_id", "track_id"])  # fmt: skip
    cols_t = ["clip_id", "track_id", "velocidad_max_kmh", "encuadre", "thumb"]
    v = t[[x for x in cols_t if x in t]].merge(
        primer[["clip_id", "track_id", "aproximacion", "ts_cruce", "hora"]],
        on=["clip_id", "track_id"])  # fmt: skip
    v["calle"] = v["aproximacion"].map(calle_de_linea)
    v["limite"] = [limites.get(e, {}).get(k) for e, k in zip(v["encuadre"], v["calle"],
                                                             strict=True)]  # fmt: skip
    v = v.dropna(subset=["limite", "velocidad_max_kmh"])
    v = v[v["velocidad_max_kmh"] > v["limite"]]
    if v.empty:
        return vacio
    exceso = v["velocidad_max_kmh"] - v["limite"]
    foco = [f"runs/{cl}/foco/{tid}.mp4" for cl, tid in zip(v["clip_id"], v["track_id"],
                                                          strict=True)]  # fmt: skip
    return pd.DataFrame({
        "clip_id": v["clip_id"], "hora": v["hora"], "ts": v["ts_cruce"],
        "tipo": "Exceso de velocidad",
        "gravedad": ["grave" if x > 20 else "moderado" if x > 10 else "leve" for x in exceso],
        "titulo": "Velocidad por encima del máximo",
        "detalle": [f"Vehículo #{tid} a {vel:.0f} km/h (máximo {lim:.0f})"
                    for tid, vel, lim in zip(v["track_id"], v["velocidad_max_kmh"], v["limite"],
                                             strict=True)],
        "track_id": v["track_id"], "thumb": _col(v, "thumb"), "video": foco,
        "encuadre": v["encuadre"],
    })  # fmt: skip
