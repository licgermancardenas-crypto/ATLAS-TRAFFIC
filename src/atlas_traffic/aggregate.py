"""Consolidación multi-clip (brief §3.4) y calibración sobre el dataset completo.

- calibrar_dataset: junta intervalos y velocidades de todos los clips procesados,
  calcula los umbrales de congestión, los guarda y reclasifica cada clip.
- consolidar: une los resultados por clip con la metadata de clips.yaml:
    outputs/summary.parquet       intervalos + metadata (lo que consume el Panorama)
    outputs/tracks_todos.parquet  un registro por objeto, con hora real de cruce
    outputs/eventos_todos.parquet eventos de todos los clips
    outputs/cruces_todos.parquet  un registro por cruce de línea, con color, recorte y
                                  hora real del vehículo (base de la consulta)
  Las rutas de recortes (thumb) quedan relativas a outputs/.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from atlas_traffic.config import AppConfig, Calibracion, Clip, load_geometria, save_calibracion
from atlas_traffic.congestion import aplicar, calibrar

METADATA = ["fecha", "hora_inicio", "dia_semana", "tipo_dia", "clima", "encuadre", "notas"]


def _dir_corrida(cfg: AppConfig, clip: Clip) -> Path:
    return cfg.rutas.outputs / "runs" / clip.id


def _inicio(clip: Clip) -> dt.datetime:
    h, m = (int(x) for x in clip.hora_inicio.split(":"))
    return dt.datetime.combine(clip.fecha, dt.time(h, m))


def _metadata(clip: Clip) -> dict:
    encuadre = load_geometria(clip.geometria).encuadre if clip.geometria.exists() else None
    return {
        "fecha": pd.Timestamp(clip.fecha),
        "hora_inicio": clip.hora_inicio,
        "dia_semana": clip.dia_semana,
        "tipo_dia": clip.tipo_dia,
        "clima": clip.clima,
        "encuadre": encuadre,
        "notas": clip.notas,
    }


# --------------------------------------------------------------------------- calibración


@dataclass(frozen=True)
class ResultadoCalibracion:
    calibracion: Calibracion
    intervalos: pd.DataFrame  # intervalos de todos los clips, ya reclasificados
    omitidos: list[str]


def calibrar_dataset(cfg: AppConfig, clips: list[Clip]) -> ResultadoCalibracion:
    intervalos, velocidades, usados, encuadres, omitidos = [], {}, [], {}, []
    grupos: dict[str, str] = {}
    for c in clips:
        d = _dir_corrida(cfg, c)
        ruta_iv, ruta_vel = d / "intervalos.parquet", d / "velocidades_roi.parquet"
        if not (ruta_iv.exists() and ruta_vel.exists() and c.geometria.exists()):
            omitidos.append(c.id)
            continue
        geo = load_geometria(c.geometria)
        enc = geo.encuadre
        grupo = geo.grupo_normalizacion or enc
        encuadres[c.id] = enc
        grupos[enc] = grupo
        # se calibra por grupo: los encuadres del mismo grupo comparten flujo libre (p90)
        intervalos.append(pd.read_parquet(ruta_iv).assign(encuadre=grupo))
        velocidades.setdefault(grupo, []).append(pd.read_parquet(ruta_vel)["vel"].to_numpy())
        usados.append(c.id)
    if not intervalos:
        raise ValueError("no hay clips procesados con geometría: corré `run` o `batch` primero")
    todos = pd.concat(intervalos, ignore_index=True)
    cal = calibrar(todos, {e: np.concatenate(v) for e, v in velocidades.items()}, usados)
    p90_grupo = cal.velocidad_p90_por_encuadre
    cal = cal.model_copy(update={
        "velocidad_p90_por_encuadre": {e: p90_grupo[g] for e, g in grupos.items()},
        "grupos": dict(grupos),
    })  # fmt: skip
    save_calibracion(cal, cfg.rutas.calibracion)
    reclasificados = []
    for cid in usados:
        ruta = cfg.rutas.outputs / "runs" / cid / "intervalos.parquet"
        iv = aplicar(pd.read_parquet(ruta), encuadres[cid], cal)
        iv.to_parquet(ruta, index=False)
        reclasificados.append(iv)
    return ResultadoCalibracion(cal, pd.concat(reclasificados, ignore_index=True), omitidos)


# --------------------------------------------------------------------------- consolidación


@dataclass(frozen=True)
class ResultadoConsolidacion:
    summary: pd.DataFrame
    tracks: pd.DataFrame
    eventos: pd.DataFrame
    cruces: pd.DataFrame
    rutas: dict[str, Path]
    giros: pd.DataFrame = field(default_factory=pd.DataFrame)


def _prefijar_thumbs(df: pd.DataFrame, clip: Clip) -> pd.DataFrame:
    if "thumb" in df.columns:
        df = df.assign(thumb=df["thumb"].map(
            lambda t: f"runs/{clip.id}/{t}" if isinstance(t, str) else None))  # fmt: skip
    return df


def consolidar(cfg: AppConfig, clips: list[Clip]) -> ResultadoConsolidacion:
    summary, tracks, eventos, cruces, giros = [], [], [], [], []
    for c in clips:
        d = _dir_corrida(cfg, c)
        if not (d / "intervalos.parquet").exists():
            continue
        meta = _metadata(c)
        inicio = _inicio(c)
        iv = pd.read_parquet(d / "intervalos.parquet").assign(**meta)
        iv["inicio"] = inicio + pd.to_timedelta(iv["t_inicio_s"], unit="s")
        iv["hora_del_dia"] = iv["inicio"].dt.hour
        summary.append(iv)
        if (d / "tracks.parquet").exists():
            t = _prefijar_thumbs(pd.read_parquet(d / "tracks.parquet"), c)
            t = t.assign(clip_id=c.id, **meta)
            if "ts_cruce" in t.columns:
                t["hora_cruce"] = inicio + pd.to_timedelta(t["ts_cruce"], unit="s")
            tracks.append(t)
            if (d / "cruces.parquet").exists():
                cr = pd.read_parquet(d / "cruces.parquet")
                atributos = (
                    t[["track_id", "color", "color_confianza", "thumb"]]
                    if "color" in t
                    else t[["track_id"]]
                )
                cr = cr.merge(atributos, on="track_id", how="left").assign(clip_id=c.id, **meta)
                cr["hora"] = inicio + pd.to_timedelta(cr["ts_cruce"], unit="s")
                cruces.append(cr)
        if (d / "giros.parquet").exists():
            giros.append(pd.read_parquet(d / "giros.parquet").assign(clip_id=c.id, **meta))
        if (d / "eventos.parquet").exists():
            e = _prefijar_thumbs(pd.read_parquet(d / "eventos.parquet"), c).assign(**meta)
            e["hora"] = inicio + pd.to_timedelta(e["ts"], unit="s")
            eventos.append(e)
    if not summary:
        raise ValueError("no hay intervalos para consolidar: procesá los clips primero")
    salida = cfg.rutas.outputs
    res = ResultadoConsolidacion(
        summary=pd.concat(summary, ignore_index=True),
        tracks=pd.concat(tracks, ignore_index=True) if tracks else pd.DataFrame(),
        eventos=pd.concat(eventos, ignore_index=True) if eventos else pd.DataFrame(),
        cruces=pd.concat(cruces, ignore_index=True) if cruces else pd.DataFrame(),
        rutas={
            "summary": salida / "summary.parquet",
            "tracks": salida / "tracks_todos.parquet",
            "eventos": salida / "eventos_todos.parquet",
            "cruces": salida / "cruces_todos.parquet",
            "giros": salida / "giros_todos.parquet",
        },
        giros=pd.concat(giros, ignore_index=True) if giros else pd.DataFrame(),
    )
    res.summary.to_parquet(res.rutas["summary"], index=False)
    res.tracks.to_parquet(res.rutas["tracks"], index=False)
    res.eventos.to_parquet(res.rutas["eventos"], index=False)
    res.cruces.to_parquet(res.rutas["cruces"], index=False)
    res.giros.to_parquet(res.rutas["giros"], index=False)
    return res
