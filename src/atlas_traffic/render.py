"""Video anotado: cajas, IDs, ROI, líneas de conteo y contador por aproximación.

Fase 1: render de debug de cajas + track_id para verificar estabilidad de IDs.
Fase 1b: ROI y líneas de conteo, con la flecha del cruce positivo. En el video de debug
la geometría (definida en el frame de referencia) se lleva a cada frame con la inversa
de la homografía de estabilización: si queda pegada a la calle, todo encaja.
Fase 5 completa el overlay (contadores por aproximación).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atlas_traffic.config import Geometria, Linea

COLOR_CLASE = {"vehiculo": (0, 200, 255), "peaton": (255, 120, 0)}  # BGR
COLOR_ROI = (80, 220, 80)
COLOR_LINEA = {"entrante": (60, 60, 255), "saliente": (255, 160, 40)}
COLOR_PENDIENTE = (0, 255, 255)
COLUMNAS_H = [f"h{i}{j}" for i in range(3) for j in range(3)]


def _color_id(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def _texto(img: np.ndarray, texto: str, org: tuple[int, int], color, escala: float = 0.45) -> None:
    grosor = 1 if escala < 0.55 else 2  # texto grande con trazo fino se lee mal
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 2,
                cv2.LINE_AA)  # fmt: skip
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def _transformar(H: np.ndarray | None, puntos) -> np.ndarray:
    p = np.asarray(puntos, dtype=np.float64).reshape(-1, 2)
    if H is None or len(p) == 0:
        return p
    return cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2)


def dibujar_cajas(frame: np.ndarray, dets: pd.DataFrame) -> np.ndarray:
    out = frame.copy()
    for r in dets.itertuples(index=False):
        color = _color_id(int(r.track_id))
        cv2.rectangle(out, (r.x1, r.y1), (r.x2, r.y2), color, 2)
        etiqueta = f"{r.track_id} {r.clase_detalle}"
        cv2.putText(out, etiqueta, (r.x1, max(10, r.y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)  # fmt: skip
    return out


def _flecha_normal(img, a: np.ndarray, b: np.ndarray, H: np.ndarray | None, color) -> None:
    """Flecha desde el punto medio hacia el lado positivo (ver convención en geometry.py)."""
    largo = max(18.0, 0.25 * float(np.hypot(*(b - a))))
    n = np.array([-(b[1] - a[1]), b[0] - a[0]], dtype=np.float64)
    n /= np.linalg.norm(n)
    medio = (a + b) / 2
    p0, p1 = _transformar(H, [medio, medio + largo * n])
    cv2.arrowedLine(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                    color, 2, cv2.LINE_AA, tipLength=0.35)  # fmt: skip


def dibujar_geometria(
    frame: np.ndarray,
    roi: Sequence[tuple[int, int]],
    lineas: Sequence[Linea],
    pendiente: Sequence[tuple[int, int]] = (),
    roi_cerrada: bool = True,
    H: np.ndarray | None = None,
    exclusiones=(),
) -> np.ndarray:
    """Dibuja ROI, líneas (con id, sentido y flecha de cruce positivo) y la línea en curso.

    `H` lleva coordenadas de referencia al frame que se dibuja (None = son iguales).
    """
    out = frame.copy()
    if exclusiones:
        out = dibujar_exclusiones(out, exclusiones, H)
    if len(roi) >= 3 and roi_cerrada:
        poligono = np.round(_transformar(H, roi)).astype(np.int32)
        capa = out.copy()
        cv2.fillPoly(capa, [poligono], COLOR_ROI)
        out = cv2.addWeighted(capa, 0.18, out, 0.82, 0)
        cv2.polylines(out, [poligono], True, COLOR_ROI, 2, cv2.LINE_AA)
    elif roi:
        pts = np.round(_transformar(H, roi)).astype(np.int32)
        cv2.polylines(out, [pts], False, COLOR_ROI, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, tuple(p), 4, COLOR_ROI, -1)

    for linea in lineas:
        a, b = (np.asarray(p, dtype=np.float64) for p in linea.puntos)
        color = COLOR_LINEA[linea.sentido_positivo]
        pa, pb = np.round(_transformar(H, [a, b])).astype(int)
        cv2.line(out, tuple(pa), tuple(pb), color, 3, cv2.LINE_AA)
        cv2.circle(out, tuple(pa), 4, color, -1)
        _flecha_normal(out, a, b, H, color)
        _texto(out, f"{linea.id} ({linea.sentido_positivo})", (int(pa[0]), int(pa[1]) - 6), color)

    if pendiente:
        pts = [np.asarray(p, dtype=np.float64) for p in pendiente]
        for p in np.round(_transformar(H, pts)).astype(int):
            cv2.circle(out, tuple(p), 5, COLOR_PENDIENTE, -1)
        if len(pts) == 2:
            pa, pb = np.round(_transformar(H, pts)).astype(int)
            cv2.line(out, tuple(pa), tuple(pb), COLOR_PENDIENTE, 2, cv2.LINE_AA)
            _flecha_normal(out, pts[0], pts[1], H, COLOR_PENDIENTE)
    return out


def dibujar_contadores(
    frame: np.ndarray, lineas: Sequence[Linea], conteos: dict[tuple[str, str], int]
) -> np.ndarray:
    """Panel arriba a la izquierda: cruces acumulados por línea (entrantes / salientes)."""
    out = frame.copy()
    alto_fila = 18
    ancho = 210
    capa = out.copy()
    cv2.rectangle(capa, (4, 4), (4 + ancho, 10 + alto_fila * (len(lineas) + 1)), (0, 0, 0), -1)
    out = cv2.addWeighted(capa, 0.55, out, 0.45, 0)
    _texto(out, "linea      entra  sale", (10, 4 + alto_fila), (255, 255, 255), 0.42)
    for k, linea in enumerate(lineas, start=2):
        e = conteos.get((linea.id, "entrante"), 0)
        s = conteos.get((linea.id, "saliente"), 0)
        _texto(out, f"{linea.id[:10]:<10} {e:>5} {s:>5}", (10, 4 + alto_fila * k),
               COLOR_LINEA[linea.sentido_positivo], 0.42)  # fmt: skip
    return out


def homografias_por_frame(estabilizacion: pd.DataFrame) -> dict[int, np.ndarray]:
    """frame -> H (frame -> referencia), desde estabilizacion.parquet."""
    H = estabilizacion[COLUMNAS_H].to_numpy().reshape(-1, 3, 3)
    return dict(zip(estabilizacion["frame"].astype(int), H, strict=True))


def video_debug(
    video: Path,
    detecciones: pd.DataFrame,
    salida: Path,
    geometria: Geometria | None = None,
    estabilizacion: pd.DataFrame | None = None,
    cruces: pd.DataFrame | None = None,
) -> Path:
    """mp4 con cajas e IDs; si hay geometría, ROI y líneas siguiendo a la cámara.

    Con `cruces`, suma un contador acumulado por línea y sentido: sirve para el chequeo
    manual de la Fase 2 (pausar el video y comparar con lo que uno ve).
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ultimo = int(detecciones["frame"].max()) if not detecciones.empty else -1
    por_frame = dict(tuple(detecciones.groupby("frame")))
    hs = homografias_por_frame(estabilizacion) if estabilizacion is not None else {}
    eventos = (
        list(cruces[["frame", "aproximacion", "sentido"]].itertuples(index=False, name=None))
        if cruces is not None and not cruces.empty
        else []
    )
    eventos.sort()
    conteos: dict[tuple[str, str], int] = {}
    writer = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        vigentes = detecciones.iloc[0:0]
        H_inv = None
        for i in range(ultimo + 1):
            ok, frame = cap.read()
            if not ok:
                break
            # con paso_frames > 1 hay frames sin inferencia: se mantienen las últimas cajas
            vigentes = por_frame.get(i, vigentes)
            if i in hs:
                H_inv = np.linalg.inv(hs[i])  # referencia -> este frame
            if geometria is not None:
                frame = dibujar_geometria(frame, geometria.roi, geometria.lineas, H=H_inv,
                                          exclusiones=geometria.exclusiones)  # fmt: skip
            frame = dibujar_cajas(frame, vigentes)
            while eventos and eventos[0][0] <= i:
                _, linea_id, sentido = eventos.pop(0)
                conteos[(linea_id, sentido)] = conteos.get((linea_id, sentido), 0) + 1
            if geometria is not None and cruces is not None:
                frame = dibujar_contadores(frame, geometria.lineas, conteos)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    return salida


def video_estabilizado(video: Path, estabilizacion: pd.DataFrame, salida: Path) -> Path:
    """Escribe un mp4 con cada frame procesado alineado al frame de referencia.

    Sirve para verificar a ojo la estabilización: los edificios deberían quedar quietos.
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hs = homografias_por_frame(estabilizacion)
    ultimo = max(hs) if hs else -1
    writer = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        H = np.eye(3)
        for i in range(ultimo + 1):
            ok, frame = cap.read()
            if not ok:
                break
            H = hs.get(i, H)
            writer.write(cv2.warpPerspective(frame, H, (w, h)))
    finally:
        cap.release()
        writer.release()
    return salida


# =========================================================================== presentación
# Video anotado para mostrar (Fase 5). A diferencia del de debug: color por clase, cajas
# que se ponen verdes al ser contadas, flecha del sentido de circulación permitido (no
# del cruce positivo), panel de conteo por aproximación, insignia de congestión del
# intervalo en curso y aviso de eventos. Se dibuja sobre el frame agrandado (`escala`)
# para que el texto sea legible en videos chicos.

COLOR_CONTADO = (80, 200, 80)
COLOR_CONGESTION = {"LOW": (80, 180, 60), "MODERATE": (0, 170, 255), "HIGH": (40, 40, 220)}
COLOR_EVENTO = (0, 0, 255)
COLOR_LINEA_PRESENTACION = (0, 220, 255)


def _flecha_permitida(img, linea: Linea, H: np.ndarray | None, escala: float) -> None:
    """Flecha perpendicular a la línea, hacia donde circula el tránsito permitido."""
    if linea.sentido_permitido == "ambos":
        return
    a, b = (np.asarray(p, dtype=np.float64) for p in linea.puntos)
    n = np.array([-(b[1] - a[1]), b[0] - a[0]])
    n /= np.linalg.norm(n)
    if linea.sentido_permitido != linea.sentido_positivo:
        n = -n
    medio = (a + b) / 2
    largo = max(14.0, 0.18 * float(np.hypot(*(b - a))))
    p0, p1 = _transformar(H, [medio - 0.5 * largo * n, medio + 0.5 * largo * n]) * escala
    cv2.arrowedLine(
        img,
        tuple(np.round(p0).astype(int)),
        tuple(np.round(p1).astype(int)),
        (255, 255, 255),
        max(2, int(2 * escala)),
        cv2.LINE_AA,
        tipLength=0.4,
    )


COLOR_EXCLUSION = (160, 160, 160)


def dibujar_exclusiones(
    img: np.ndarray, zonas, H: np.ndarray | None = None, escala: float = 1.0
) -> np.ndarray:
    """Zonas de exclusión: polígono gris con rayado (acá se ignoran las detecciones).

    Trabaja solo sobre el recuadro de cada zona: se dibuja en cada frame del video y
    procesar la imagen entera lo hacía varias veces más lento.
    """
    out = img
    alto, ancho = img.shape[:2]
    for zona in zonas:
        pts = np.round(_transformar(H, zona) * escala).astype(np.int32)
        x0, y0, w, h = cv2.boundingRect(pts)
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(ancho, x0 + w), min(alto, y0 + h)
        if x2 <= x1 or y2 <= y1:
            continue  # zona fuera del cuadro en este frame
        if out is img:
            out = img.copy()
        recorte = out[y1:y2, x1:x2]
        mascara = np.zeros(recorte.shape[:2], np.uint8)
        cv2.fillPoly(mascara, [pts - [x1, y1]], 255)
        rayado = recorte.copy()
        paso = max(4, int(8 * escala))
        rh, rw = recorte.shape[:2]
        for d in range(-rh, rw + rh, paso):
            cv2.line(rayado, (d, 0), (d + rh, rh), COLOR_EXCLUSION, 1, cv2.LINE_AA)
        mezcla = cv2.addWeighted(rayado, 0.6, recorte, 0.4, 0)
        recorte[mascara > 0] = mezcla[mascara > 0]
        cv2.polylines(out, [pts], True, COLOR_EXCLUSION, 1, cv2.LINE_AA)
    return out


def dibujar_escena(
    frame: np.ndarray,
    geo: Geometria,
    H: np.ndarray | None,
    dets: pd.DataFrame,
    contados: set[int],
    escala: float,
) -> np.ndarray:
    """ROI, líneas con flecha de sentido permitido y cajas por clase (verde = contado)."""
    out = cv2.resize(frame, None, fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR)
    poligono = np.round(_transformar(H, geo.roi) * escala).astype(np.int32)
    capa = out.copy()
    cv2.fillPoly(capa, [poligono], COLOR_ROI)
    out = cv2.addWeighted(capa, 0.10, out, 0.90, 0)
    cv2.polylines(out, [poligono], True, COLOR_ROI, 1, cv2.LINE_AA)
    out = dibujar_exclusiones(out, geo.exclusiones, H, escala)
    for linea in geo.lineas:
        pa, pb = np.round(_transformar(H, linea.puntos) * escala).astype(int)
        cv2.line(out, tuple(pa), tuple(pb), (0, 0, 0), max(4, int(5 * escala)), cv2.LINE_AA)
        cv2.line(out, tuple(pa), tuple(pb), COLOR_LINEA_PRESENTACION, max(2, int(3 * escala)),
                 cv2.LINE_AA)  # fmt: skip
        _flecha_permitida(out, linea, H, escala)
    grosor = max(1, int(round(1.5 * escala)))
    for r in dets.itertuples(index=False):
        tid = int(r.track_id)
        color = COLOR_CONTADO if tid in contados else COLOR_CLASE.get(r.clase, (200, 200, 200))
        p1 = (int(r.x1 * escala), int(r.y1 * escala))
        p2 = (int(r.x2 * escala), int(r.y2 * escala))
        cv2.rectangle(out, p1, p2, color, grosor, cv2.LINE_AA)
        _texto(out, str(tid), (p1[0], max(12, p1[1] - 4)), color, 0.35 * escala)
    return out


def dibujar_panel(
    img: np.ndarray,
    titulo: str,
    t: float,
    geo: Geometria,
    conteos: dict[tuple[str, str], int],
    ventana: dict | None,
    escala: float,
) -> np.ndarray:
    """Panel superior: título, tiempo, cruces por aproximación y nivel de congestión."""
    fs = 0.42 * escala
    fila = int(20 * escala)
    x = int(8 * escala)
    alto = fila * (len(geo.lineas) + 3) + int(8 * escala)
    out = img.copy()
    capa = out.copy()
    cv2.rectangle(capa, (0, 0), (int(300 * escala), alto), (20, 20, 20), -1)
    out = cv2.addWeighted(capa, 0.65, out, 0.35, 0)
    reloj = f"{int(t // 60):02d}:{t % 60:04.1f}"
    _texto(out, f"ATLAS Traffic  {titulo}  {reloj}", (x, fila), (255, 255, 255), fs)
    _texto(out, "aproximacion          entran  salen", (x, 2 * fila), (180, 180, 180), fs * 0.9)
    for k, linea in enumerate(geo.lineas, start=3):
        e = conteos.get((linea.id, "entrante"), 0)
        s = conteos.get((linea.id, "saliente"), 0)
        texto = f"{linea.id[:20]:<20} {e:>6} {s:>6}"
        _texto(out, texto, (x, k * fila), COLOR_LINEA_PRESENTACION, fs * 0.9)
    nivel = (ventana or {}).get("congestion")
    if isinstance(nivel, str) and nivel in COLOR_CONGESTION:
        y = (len(geo.lineas) + 3) * fila
        (w, h), _ = cv2.getTextSize(nivel, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.rectangle(out, (x - 3, y - h - 5), (x + w + 5, y + 5), COLOR_CONGESTION[nivel], -1)
        cv2.putText(out, nivel, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2,
                    cv2.LINE_AA)  # fmt: skip
        detalle = f"densidad {ventana['densidad_media']:.1f} veh"
        _texto(out, detalle, (x + w + int(12 * escala), y), (220, 220, 220), fs * 0.85)
    return out


def video_anotado(
    video: Path,
    detecciones: pd.DataFrame,
    geometria: Geometria,
    salida: Path,
    titulo: str = "",
    estabilizacion: pd.DataFrame | None = None,
    cruces: pd.DataFrame | None = None,
    intervalos: pd.DataFrame | None = None,
    eventos: pd.DataFrame | None = None,
    escala: float = 1.5,
) -> Path:
    """mp4 de presentación con la escena anotada y los contadores en pantalla."""
    from atlas_traffic.geometry import filtrar_exclusiones

    # mismas detecciones que usó el análisis: sin los objetos fijos excluidos
    detecciones, _ = filtrar_exclusiones(detecciones, geometria)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tam = (int(w * escala), int(h * escala))
    ultimo = int(detecciones["frame"].max()) if not detecciones.empty else -1
    por_frame = dict(tuple(detecciones.groupby("frame")))
    hs = homografias_por_frame(estabilizacion) if estabilizacion is not None else {}
    lista_cruces = []
    if cruces is not None and not cruces.empty:
        columnas = ["frame", "track_id", "aproximacion", "sentido"]
        lista_cruces = sorted(cruces[columnas].itertuples(index=False, name=None))
    ventanas = intervalos.to_dict("records") if intervalos is not None else []
    momentos_evento = np.array([])
    if eventos is not None and not eventos.empty:
        momentos_evento = eventos["frame"].to_numpy()
    conteos: dict[tuple[str, str], int] = {}
    contados: set[int] = set()
    writer = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps, tam)
    try:
        vigentes = detecciones.iloc[0:0]
        H_inv = None
        for i in range(ultimo + 1):
            ok, frame = cap.read()
            if not ok:
                break
            vigentes = por_frame.get(i, vigentes)
            if i in hs:
                H_inv = np.linalg.inv(hs[i])
            while lista_cruces and lista_cruces[0][0] <= i:
                _, tid, linea_id, sentido = lista_cruces.pop(0)
                conteos[(linea_id, sentido)] = conteos.get((linea_id, sentido), 0) + 1
                contados.add(int(tid))
            t = i / fps
            ventana = next(
                (v for v in ventanas if v["t_inicio_s"] <= t < v["t_fin_s"]),
                ventanas[-1] if ventanas else None,
            )
            img = dibujar_escena(frame, geometria, H_inv, vigentes, contados, escala)
            img = dibujar_panel(img, titulo, t, geometria, conteos, ventana, escala)
            if momentos_evento.size and np.any(np.abs(momentos_evento - i) <= fps):
                cv2.rectangle(img, (0, 0), (tam[0] - 1, tam[1] - 1), COLOR_EVENTO, int(6 * escala))
                pos = (int(10 * escala), tam[1] - int(15 * escala))
                _texto(img, "SENTIDO CONTRARIO", pos, COLOR_EVENTO, 0.7 * escala)
            writer.write(img)
    finally:
        cap.release()
        writer.release()
    return salida


def gif_desde_video(
    video: Path,
    salida: Path,
    inicio_s: float = 0.0,
    duracion_s: float = 8.0,
    ancho: int = 360,
    fps: float = 10.0,
    colores: int = 64,
) -> Path:
    """GIF liviano de un tramo del video (para el README). Requiere Pillow.

    Ancho chico a propósito: además de pesar poco, no deja leer patentes ni distinguir caras.
    """
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    paso = max(1, round(fps_video / fps))
    ini, fin = int(inicio_s * fps_video), int((inicio_s + duracion_s) * fps_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ini)
    cuadros = []
    try:
        for i in range(ini, fin):
            ok, frame = cap.read()
            if not ok:
                break
            if (i - ini) % paso:
                continue
            alto = int(frame.shape[0] * ancho / frame.shape[1])
            chico = cv2.resize(frame, (ancho, alto), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(cv2.cvtColor(chico, cv2.COLOR_BGR2RGB))
            cuadros.append(img)
    finally:
        cap.release()
    if not cuadros:
        raise ValueError(f"no hay cuadros entre {inicio_s} y {inicio_s + duracion_s} s")
    # una sola paleta para todo el GIF (sacada del cuadro del medio): comprime mucho mejor
    # que una paleta por cuadro y evita el parpadeo de colores entre cuadros
    paleta = cuadros[len(cuadros) // 2].quantize(colors=colores, method=Image.Quantize.MEDIANCUT)
    indexados = [c.quantize(palette=paleta, dither=Image.Dither.NONE) for c in cuadros]
    indexados[0].save(
        salida, save_all=True, append_images=indexados[1:], duration=int(1000 / fps), loop=0,
        optimize=True,
    )  # fmt: skip
    return salida
