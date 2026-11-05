"""Lógica de datos del dashboard, separada de Streamlit para poder testearla.

- cargar: lee los consolidados de outputs/ (los genera `batch`).
- consultar: filtra los cruces y devuelve un registro por vehículo (pestaña Consulta).
- resúmenes para el Panorama: flujo por tipo de día, congestión por franja horaria.
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
    res = cruces[m].sort_values(["clip_id", "ts_cruce"])
    return res.drop_duplicates(["clip_id", "track_id"]).reset_index(drop=True)


COLUMNAS_EXPORTAR = [
    "clip_id", "track_id", "hora", "ts_cruce", "aproximacion", "sentido", "clase_detalle",
    "color", "color_confianza", "fecha", "dia_semana", "tipo_dia", "clima", "thumb",
]  # fmt: skip


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
