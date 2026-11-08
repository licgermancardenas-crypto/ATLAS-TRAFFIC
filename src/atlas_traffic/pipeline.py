"""Orquestación de una corrida sobre un clip.

Fase 1: video -> detector (protocolo) -> detecciones.parquet (una fila por detección y
frame) -> tracks.parquet (una fila por objeto).

Si la estabilización está activa, cada frame procesado se alinea con el frame de
referencia y los centroides se guardan también en coordenadas de referencia (cx_ref,
cy_ref). Todo lo geométrico (velocidad, ROI, cruces) usa esas coordenadas; x1..y2, cx, cy
quedan en coordenadas del frame original, para recortes y render. La H de cada frame va
a estabilizacion.parquet, para diagnóstico. Las fases siguientes agregan columnas a
tracks (cruce, sentido, color) y generan intervalos y eventos a partir de las mismas
detecciones crudas.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atlas_traffic.alerts import detectar_sentido_contrario, guardar_evidencia
from atlas_traffic.attributes import extraer_atributos
from atlas_traffic.config import (
    AppConfig,
    Clip,
    EstabilizacionCfg,
    load_calibracion,
    load_geometria,
)
from atlas_traffic.conflictos import COLUMNAS_CONFLICTOS, detectar_conflictos
from atlas_traffic.congestion import aplicar
from atlas_traffic.correction import (
    calcular_tasas,
    corregir_conteo,
    detectar_luz,
    flujo_entrante,
    leer_registros,
)
from atlas_traffic.counting import anotar_tracks, contar, resumen_conteo
from atlas_traffic.detect import Detector
from atlas_traffic.geometry import dentro_roi, filtrar_exclusiones
from atlas_traffic.metrics import calcular_intervalos, vehiculos_en_roi
from atlas_traffic.render import homografias_por_frame
from atlas_traffic.stabilize import (
    Estabilizador,
    mascara_sin_cajas,
    suavizar_homografias,
    transformar_puntos,
)
from atlas_traffic.stitching import aplicar_uniones, unir_tracks
from atlas_traffic.suelo import cargar_calibracion_suelo
from atlas_traffic.turning import movimientos
from atlas_traffic.velocidad import (
    en_suelo,
    velocidad_kmh,
    velocidad_por_intervalo,
    velocidad_por_track,
)

COLUMNAS_DETECCIONES = [
    "frame", "ts", "track_id", "clase", "clase_detalle", "confianza",
    "x1", "y1", "x2", "y2", "cx", "cy", "cx_ref", "cy_ref",
]  # fmt: skip
COLUMNAS_ESTABILIZACION = ["frame", "inliers", "ok"] + [
    f"h{i}{j}" for i in range(3) for j in range(3)
]


@dataclass(frozen=True)
class ResultadoDeteccion:
    detecciones: pd.DataFrame
    estabilizacion: pd.DataFrame  # vacío si la estabilización está apagada
    frames_leidos: int = 0  # frames del video recorridos (procesados o salteados)
    fps: float = 30.0


@dataclass(frozen=True)
class InfoVideo:
    fps: float
    n_frames: int
    ancho: int
    alto: int


def info_video(ruta: Path) -> InfoVideo:
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f"no se pudo abrir el video: {ruta}")
    try:
        return InfoVideo(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            ancho=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            alto=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def leer_frame(ruta: Path, indice: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(ruta))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"no se pudo leer el frame {indice} de {ruta}")
    return frame


def detectar_video(
    ruta: Path,
    detector: Detector,
    paso_frames: int = 1,
    max_segundos: float | None = None,
    progreso: Callable[[int, int], None] | None = None,
    estabilizacion: EstabilizacionCfg | None = None,
) -> ResultadoDeteccion:
    """Corre el detector sobre el video y devuelve las detecciones con track_id."""
    info = info_video(ruta)
    limite = info.n_frames
    if max_segundos is not None:
        limite = min(limite, int(max_segundos * info.fps))
    detector.reset()
    estab = None
    if estabilizacion is not None and estabilizacion.activa:
        estab = Estabilizador(
            leer_frame(ruta, estabilizacion.frame_referencia),
            max_puntos=estabilizacion.max_puntos,
            min_inliers=estabilizacion.min_inliers,
            umbral_ransac_px=estabilizacion.umbral_ransac_px,
        )
    filas: list[tuple] = []
    filas_estab: list[tuple] = []
    cap = cv2.VideoCapture(str(ruta))
    try:
        frame_idx = 0
        while frame_idx < limite:
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % paso_frames == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                dets = [d for d in detector.track(frame) if d.track_id is not None]
                # sin track_id no hay objeto único que contar
                if estab is not None:
                    mascara = mascara_sin_cajas(frame.shape[:2], (d.bbox for d in dets))
                    est = estab.estimar(frame, mascara)
                    filas_estab.append((frame_idx, est.inliers, est.ok, *est.H.ravel()))
                for d in dets:
                    x1, y1, x2, y2 = d.bbox
                    cx, cy = d.centroide
                    # cx_ref/cy_ref = cx/cy hasta que se aplique la estabilización suavizada
                    filas.append((
                        frame_idx, frame_idx / info.fps, d.track_id, d.clase, d.clase_detalle,
                        d.confianza, x1, y1, x2, y2, cx, cy, cx, cy,
                    ))  # fmt: skip
                if progreso:
                    progreso(frame_idx, limite)
            frame_idx += 1
    finally:
        cap.release()
    df = pd.DataFrame(filas, columns=COLUMNAS_DETECCIONES)
    df = df.astype({"frame": "int32", "track_id": "int32", "x1": "int32", "y1": "int32",
                    "x2": "int32", "y2": "int32"})  # fmt: skip
    df_estab = pd.DataFrame(filas_estab, columns=COLUMNAS_ESTABILIZACION)
    if estab is not None and not df_estab.empty:
        df_estab = _aplicar_estabilizacion(df, df_estab, estabilizacion.ventana_suavizado)
    return ResultadoDeteccion(df, df_estab, frames_leidos=frame_idx, fps=info.fps)


def _aplicar_estabilizacion(det: pd.DataFrame, estab: pd.DataFrame, ventana: int) -> pd.DataFrame:
    """Suaviza las H en el tiempo y pasa los centroides a coordenadas de referencia.

    Modifica `det` in place (cx_ref, cy_ref) y devuelve las H suavizadas. La estimación
    por frame tiembla ~0.5-1 px; el movimiento real de la mano es lento. Promediar en una
    ventana corta elimina el temblor sin dejar de seguir a la cámara.
    """
    cols_h = COLUMNAS_ESTABILIZACION[3:]
    H = suavizar_homografias(estab[cols_h].to_numpy().reshape(-1, 3, 3), ventana)
    estab = estab.copy()
    estab[cols_h] = H.reshape(-1, 9)
    por_frame = dict(zip(estab["frame"], H, strict=True))
    for frame, idx in det.groupby("frame").groups.items():
        ref = transformar_puntos(por_frame[frame], det.loc[idx, ["cx", "cy"]].to_numpy())
        det.loc[idx, ["cx_ref", "cy_ref"]] = ref
    return estab


def velocidad_ventana(frames: np.ndarray, xy: np.ndarray, ventana_frames: int) -> float:
    """Velocidad media en px/frame, medida como desplazamiento en ventanas de ~N frames.

    Sumar pasos frame a frame acumula el ruido de la caja y de la estabilización (un auto
    quieto "se mueve" ~0.5 px por frame). Medir el desplazamiento entre puntos separados
    por una ventana diluye ese ruido y no sesga un movimiento uniforme.
    """
    if len(frames) < 2:
        return 0.0
    destino = np.searchsorted(frames, frames + ventana_frames)
    validos = destino < len(frames)
    if validos.any():
        i, j = np.nonzero(validos)[0], destino[validos]
    else:  # track más corto que la ventana: punta a punta
        i, j = np.array([0]), np.array([len(frames) - 1])
    dist = np.hypot(*(xy[j] - xy[i]).T)
    return float(np.mean(dist / (frames[j] - frames[i])))


def tracks_desde_detecciones(det: pd.DataFrame, ventana_frames: int = 20) -> pd.DataFrame:
    """Una fila por track_id. Clase por mayoría; velocidad por ventanas (velocidad_ventana).

    La velocidad se mide en coordenadas de referencia (cx_ref, cy_ref) si existen: así el
    movimiento de la cámara no se confunde con movimiento del objeto.
    """
    columnas = ["track_id", "clase", "clase_detalle", "frame_inicio", "frame_fin",
                "n_detecciones", "confianza_media", "velocidad_px"]  # fmt: skip
    if det.empty:
        return pd.DataFrame(columns=columnas)
    det = det.sort_values(["track_id", "frame"])
    cols = ["cx_ref", "cy_ref"] if {"cx_ref", "cy_ref"} <= set(det.columns) else ["cx", "cy"]
    g = det.groupby("track_id")
    velocidad = {
        tid: velocidad_ventana(grupo["frame"].to_numpy(), grupo[cols].to_numpy(), ventana_frames)
        for tid, grupo in g
    }
    tracks = pd.DataFrame({
        "clase": g["clase"].agg(lambda s: s.mode().iat[0]),
        "clase_detalle": g["clase_detalle"].agg(lambda s: s.mode().iat[0]),
        "frame_inicio": g["frame"].min(),
        "frame_fin": g["frame"].max(),
        "n_detecciones": g.size(),
        "confianza_media": g["confianza"].mean(),
    })  # fmt: skip
    tracks["velocidad_px"] = pd.Series(velocidad)
    return tracks.reset_index()[columnas]


def correr_clip(
    clip: Clip,
    cfg: AppConfig,
    detector: Detector,
    max_segundos: float | None = None,
    progreso: Callable[[int, int], None] | None = None,
) -> Path:
    """Procesa un clip y escribe outputs/runs/<clip_id>/. Devuelve la carpeta de salida."""
    salida = cfg.rutas.outputs / "runs" / clip.id
    salida.mkdir(parents=True, exist_ok=True)
    imgsz = clip.imgsz or cfg.deteccion.imgsz
    if hasattr(detector, "ajustar"):
        detector.ajustar(imgsz=imgsz)
    estab = clip.estabilizacion(cfg.estabilizacion)
    res = detectar_video(
        clip.archivo, detector, cfg.video.paso_frames, max_segundos, progreso, estab
    )
    res.detecciones.to_parquet(salida / "detecciones.parquet", index=False)
    estab_path = salida / "estabilizacion.parquet"
    if not res.estabilizacion.empty:
        res.estabilizacion.to_parquet(estab_path, index=False)
    elif estab_path.exists():
        estab_path.unlink()  # no dejar el diagnóstico de una corrida anterior
    meta = {
        "clip_id": clip.id,
        "fps": res.fps,
        "paso_frames": cfg.video.paso_frames,
        "frames_leidos": res.frames_leidos,
        "duracion_s": res.frames_leidos / res.fps,
        "modelo": cfg.deteccion.modelo,
        "imgsz": imgsz,
        "frame_referencia": estab.frame_referencia,
    }
    (salida / "corrida.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return salida


def leer_corrida(salida: Path, clip: Clip, cfg: AppConfig, det: pd.DataFrame) -> dict:
    """Metadatos de la corrida. Si no hay corrida.json (corridas viejas), los estima."""
    ruta = salida / "corrida.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    fps = info_video(clip.archivo).fps
    leidos = int(det["frame"].max()) + 1 if not det.empty else 0
    return {"fps": fps, "paso_frames": cfg.video.paso_frames, "frames_leidos": leidos,
            "duracion_s": leidos / fps}  # fmt: skip


@dataclass(frozen=True)
class ResultadoAnalisis:
    salida: Path
    conteo: pd.DataFrame | None  # aproximación x sentido x clase; None si no hay geometría
    intervalos: pd.DataFrame | None = None  # una fila por ventana; None si no hay geometría
    eventos: pd.DataFrame | None = None  # alertas de sentido contrario; None sin geometría
    excluidos: tuple[int, ...] = ()  # tracks descartados por zonas de exclusión


def cargar_detecciones(salida: Path) -> pd.DataFrame:
    """detecciones.parquet con los tracks ya unidos como en el análisis (para dibujar)."""
    det = pd.read_parquet(salida / "detecciones.parquet")
    ruta = salida / "uniones.parquet"
    if ruta.exists():
        u = pd.read_parquet(ruta)
        det = aplicar_uniones(det, dict(zip(u["track_id_original"], u["track_id"], strict=True)))
    return det


def analizar_clip(clip: Clip, cfg: AppConfig) -> ResultadoAnalisis:
    """De detecciones.parquet a tracks.parquet + cruces.parquet + conteo.csv.

    Separado de la detección para poder recontar en segundos (por ejemplo, después de
    corregir la geometría) sin volver a correr el modelo.
    """
    salida = cfg.rutas.outputs / "runs" / clip.id
    ruta_det = salida / "detecciones.parquet"
    if not ruta_det.exists():
        raise FileNotFoundError(f"no hay detecciones para {clip.id}: corré `run` sin reusar")
    det = pd.read_parquet(ruta_det)
    excluidos: list[int] = []
    if clip.geometria.exists():
        det, excluidos = filtrar_exclusiones(det, load_geometria(clip.geometria))
    mapa: dict[int, int] = {}
    if cfg.union.activa:
        u = cfg.union
        mapa = unir_tracks(det, u.hueco_max_s, u.factor_distancia)
        det = aplicar_uniones(det, mapa)
        mapa_peat = unir_tracks(
            det, u.peaton_hueco_max_s, u.peaton_factor_distancia,
            clases=("peaton",), velocidad_min_px_s=u.peaton_velocidad_min_px_s,
        )  # fmt: skip
        det = aplicar_uniones(det, mapa_peat)
        # mapa total: original -> final (componer las dos pasadas)
        mapa = {k: mapa_peat.get(v, v) for k, v in mapa.items()} | {
            k: v for k, v in mapa_peat.items() if k not in mapa
        }
    uniones = pd.DataFrame({"track_id_original": list(mapa), "track_id": list(mapa.values())})
    uniones.to_parquet(salida / "uniones.parquet", index=False)
    meta = leer_corrida(salida, clip, cfg, det)
    ventana = max(1, round(cfg.metricas.ventana_velocidad_s * meta["fps"]))
    tracks = tracks_desde_detecciones(det, ventana)
    conteo = intervalos = eventos = None
    if clip.geometria.exists():
        geo = load_geometria(clip.geometria)
        cruces = contar(det, geo, cfg.conteo.margen_px, cfg.conteo.ida_vuelta_min_s)
        cruces.to_parquet(salida / "cruces.parquet", index=False)
        movimientos(cruces, det, geo).to_parquet(salida / "giros.parquet", index=False)
        conteo = resumen_conteo(cruces, geo)
        luz = clip.luz or detectar_luz(leer_frame(clip.archivo, 0))
        tasas = {}
        if cfg.correccion.activa:
            if clip.correccion:
                tasas = calcular_tasas(
                    leer_registros(cfg.correccion.registros), cfg.correccion.confianza
                )
            conteo = conteo.join(corregir_conteo(conteo, tasas, luz).drop(columns="vehiculo"))
        conteo.to_csv(salida / "conteo.csv")
        tracks = anotar_tracks(tracks, cruces)
        intervalos = calcular_intervalos(
            det, cruces, geo, clip.id,
            duracion_s=meta["duracion_s"],
            frames_procesados=np.arange(0, meta["frames_leidos"], meta["paso_frames"]),
            fps=meta["fps"],
            intervalo_s=cfg.metricas.intervalo_s,
            ventana_frames=ventana,
        )  # fmt: skip
        # velocidades instantáneas en la ROI: insumo de la calibración de congestión
        vel = vehiculos_en_roi(det, geo, ventana)[["frame", "track_id", "vel"]]
        vel.dropna().to_parquet(salida / "velocidades_roi.parquet", index=False)
        bordes = np.append(intervalos["t_inicio_s"].to_numpy(), intervalos["t_fin_s"].iat[-1])
        entrante = flujo_entrante(cruces, tasas, luz, bordes)
        intervalos = pd.concat([intervalos, entrante], axis=1).assign(luz=luz)
        ruta_estab = salida / "estabilizacion.parquet"
        hs = homografias_por_frame(pd.read_parquet(ruta_estab)) if ruta_estab.exists() else None
        # velocidad en km/h, si el encuadre tiene calibración métrica del suelo
        cal_suelo = cargar_calibracion_suelo(cfg.rutas.suelo / f"{geo.encuadre}.yaml")
        columnas = ["track_id", "frame", "ts", "cx_ref", "cy_ref", "x1", "y1", "x2", "y2"]
        trayectorias = det.loc[det["clase"] == "vehiculo", columnas]
        if cal_suelo is not None:
            veh = en_suelo(det[det["clase"] == "vehiculo"], hs or {}, cal_suelo.homografia)
            veh = veh.assign(
                vel_kmh=velocidad_kmh(veh, meta["fps"], cfg.metricas.ventana_velocidad_kmh_s)
            )
            # por vehículo, solo dentro de la ROI: lejos, la calle puede tener pendiente o
            # estar fuera de la zona donde se midieron las sendas
            en_roi = dentro_roi(geo.roi, veh[["cx_ref", "cy_ref"]].to_numpy())
            tracks = tracks.merge(velocidad_por_track(veh[en_roi]), on="track_id", how="left")
            intervalos = pd.concat([intervalos, velocidad_por_intervalo(veh, geo, bordes)], axis=1)
            trayectorias = veh[columnas + ["x_m", "y_m", "vel_kmh"]]
            peatones = _peatones_en_suelo(det, geo, hs or {}, cal_suelo.homografia)
            peatones.to_parquet(salida / "peatones.parquet", index=False)
            conflictos = pd.DataFrame(columns=COLUMNAS_CONFLICTOS)
            if cfg.conflictos.activa:
                es_peaton = trayectorias["track_id"].isin(peatones["track_id"]).to_numpy()
                en_zona = trayectorias[en_roi & ~es_peaton]
                conflictos = detectar_conflictos(en_zona, peatones, cfg.conflictos)
            conflictos = _evidencia_conflictos(clip.archivo, conflictos, det, salida / "conflictos")
            conflictos.assign(clip_id=clip.id).to_parquet(salida / "conflictos.parquet",
                                                          index=False)  # fmt: skip
        # para "seguir un vehículo" en el dashboard: recorridos y el cuadro de referencia
        trayectorias.to_parquet(salida / "trayectorias.parquet", index=False)
        ref = clip.estabilizacion(cfg.estabilizacion).frame_referencia
        referencia = leer_frame(clip.archivo, ref)
        escala_ref = min(1.0, 540 / referencia.shape[1])
        referencia = cv2.resize(referencia, None, fx=escala_ref, fy=escala_ref,
                                interpolation=cv2.INTER_AREA)  # fmt: skip
        cv2.imwrite(str(salida / "referencia.jpg"), referencia, [cv2.IMWRITE_JPEG_QUALITY, 80])
        # la imagen se achica: el dashboard necesita la resolución original de las coordenadas
        (salida / "referencia.json").write_text(
            json.dumps({"ancho": int(geo.resolucion[0]), "alto": int(geo.resolucion[1])}),
            encoding="utf-8",
        )
        if cfg.rutas.calibracion.exists():
            intervalos = aplicar(intervalos, geo.encuadre, load_calibracion(cfg.rutas.calibracion))
        intervalos.to_parquet(salida / "intervalos.parquet", index=False)
        eventos = detectar_sentido_contrario(det, geo, cfg.alertas, clip.id, cfg.conteo.margen_px)
        eventos = guardar_evidencia(clip.archivo, eventos, det, geo, salida / "eventos", hs)
        eventos = _video_eventos(clip.archivo, eventos, det, salida / "eventos")
        eventos.to_parquet(salida / "eventos.parquet", index=False)
    else:
        viejos = ("cruces.parquet", "conteo.csv", "intervalos.parquet", "eventos.parquet",
                  "giros.parquet", "conflictos.parquet", "peatones.parquet")  # fmt: skip
        for viejo in viejos:
            (salida / viejo).unlink(missing_ok=True)
    atributos = extraer_atributos(clip.archivo, det, cfg.color, salida / "thumbs")
    tracks = tracks.drop(columns=["color", "color_confianza", "thumb"], errors="ignore")
    tracks = tracks.merge(atributos, on="track_id", how="left")
    tracks.to_parquet(salida / "tracks.parquet", index=False)
    return ResultadoAnalisis(salida, conteo, intervalos, eventos, tuple(excluidos))


def _peatones_en_suelo(det: pd.DataFrame, geo, homografias: dict, H_suelo) -> pd.DataFrame:
    """Personas (tracks cuya clase más frecuente es peatón), en metros.

    Todo su recorrido, no solo la ROI (que es la calzada): casi todo el tiempo están en la
    vereda o la senda, y hace falta el recorrido entero para saber si se las siguió bien.
    """
    clase = det.groupby("track_id")["clase"].agg(lambda s: s.mode().iat[0])
    ids = clase[clase == "peaton"].index
    p = det[det["track_id"].isin(ids) & (det["clase"] == "peaton")]
    columnas = ["track_id", "frame", "ts", "cx_ref", "cy_ref", "x1", "y1", "x2", "y2"]
    return en_suelo(p[columnas], homografias, H_suelo)


ROJO, AMARILLO = (60, 60, 255), (0, 220, 255)


def _evidencia_conflictos(video: Path, conf: pd.DataFrame, det: pd.DataFrame,
                          carpeta: Path) -> pd.DataFrame:  # fmt: skip
    """Foto y video corto de cada situación de riesgo (vehículo en rojo, peatón en amarillo)."""
    from atlas_traffic.foco import video_evidencia

    carpeta.mkdir(parents=True, exist_ok=True)
    for viejo in [*carpeta.glob("*.jpg"), *carpeta.glob("*.mp4")]:
        viejo.unlink()
    conf = conf.assign(thumb=None, video=None)
    for k, c in conf.iterrows():
        nombre = f"{int(c['track_id'])}_{int(c['track_id_peaton'])}"
        a, b = sorted((c["ts"], c["ts_peaton"]))
        video_evidencia(
            video, det,
            [(int(c["track_id"]), ROJO, f"#{int(c['track_id'])} {c['vel_kmh']:.0f} km/h"),
             (int(c["track_id_peaton"]), AMARILLO, "peaton")],
            a - 2.0, b + 2.0, c["ts"], carpeta / f"{nombre}.mp4", carpeta / f"{nombre}.jpg",
        )  # fmt: skip
        if (carpeta / f"{nombre}.jpg").exists():
            conf.at[k, "thumb"] = f"{carpeta.name}/{nombre}.jpg"
        if (carpeta / f"{nombre}.mp4").exists():
            conf.at[k, "video"] = f"{carpeta.name}/{nombre}.mp4"
    return conf


def _video_eventos(video: Path, eventos: pd.DataFrame, det: pd.DataFrame,
                   carpeta: Path) -> pd.DataFrame:  # fmt: skip
    """Video corto de cada evento de sentido contrario (además de la foto)."""
    from atlas_traffic.foco import video_evidencia

    eventos = eventos.assign(video=None)
    for k, e in eventos.iterrows():
        nombre = f"{int(e['track_id'])}_{e['aproximacion']}"
        mp4 = carpeta / f"{nombre}.mp4"
        video_evidencia(video, det, [(int(e["track_id"]), ROJO, "contramano")],
                        e["ts"] - 2.5, e["ts"] + 2.5, e["ts"], mp4, None)  # fmt: skip
        if mp4.exists():
            eventos.at[k, "video"] = f"{carpeta.name}/{nombre}.mp4"
    return eventos
