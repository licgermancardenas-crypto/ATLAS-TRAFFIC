"""Entrypoint de línea de comandos.

Subcomandos: run, batch, geometry, calibrate, dashboard.
"""

from __future__ import annotations

from pathlib import Path

import typer

from atlas_traffic.config import DEFAULT_CONFIG_PATH, load_app_config

app = typer.Typer(
    name="atlas-traffic",
    help="ATLAS Traffic — conteo, congestión y consulta de tránsito sobre video.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOpt = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Config general (YAML).")


@app.command()
def run(
    clip: str = typer.Option(..., "--clip", help="id del clip en clips.yaml."),
    config: Path = ConfigOpt,
    max_segundos: float | None = typer.Option(
        None, "--max-segundos", help="Procesar solo los primeros N segundos."
    ),
    modelo: str | None = typer.Option(None, "--modelo", help="Pisa deteccion.modelo."),
    imgsz: int | None = typer.Option(None, "--imgsz", help="Pisa deteccion.imgsz."),
    debug_video: bool = typer.Option(
        False, "--debug-video", help="Escribe debug.mp4 con cajas e IDs para revisar a ojo."
    ),
    render: bool = typer.Option(
        False, "--render", help="Escribe anotado.mp4 (presentación) y muestra.gif."
    ),
    reusar_detecciones: bool = typer.Option(
        False,
        "--reusar-detecciones",
        help="No correr el modelo: recontar sobre las detecciones de la corrida anterior.",
    ),
) -> None:
    """Procesa un clip: detección, tracking, conteo, métricas y eventos."""
    from atlas_traffic.clips import get_clip, load_clips

    cfg = _cfg_con_cambios(config, modelo, imgsz)
    c = get_clip(load_clips(cfg.rutas.clips), clip)
    _procesar_clip(c, cfg, reusar_detecciones, max_segundos, render, debug_video, detallado=True)


def _cfg_con_cambios(config: Path, modelo: str | None, imgsz: int | None):
    cfg = load_app_config(config)
    cambios = {k: v for k, v in {"modelo": modelo, "imgsz": imgsz}.items() if v is not None}
    if cambios:
        cfg = cfg.model_copy(update={"deteccion": cfg.deteccion.model_copy(update=cambios)})
    return cfg


def _procesar_clip(
    c,
    cfg,
    reusar_detecciones: bool,
    max_segundos: float | None = None,
    render: bool = False,
    debug_video: bool = False,
    detallado: bool = False,
    detector=None,
):
    """Detección (salvo que se reusen las detecciones) + análisis de un clip."""
    import time

    from atlas_traffic.detect import crear_detector
    from atlas_traffic.pipeline import analizar_clip, correr_clip

    t0 = time.perf_counter()
    if not reusar_detecciones:
        if detector is None:
            detector = crear_detector(cfg.deteccion.detector, **cfg.deteccion.kwargs_motor())
        with typer.progressbar(length=1, label=f"Procesando {c.id}") as barra:

            def progreso(i: int, total: int) -> None:
                barra.length = total
                barra.update(i - barra.pos)

            correr_clip(c, cfg, detector, max_segundos, progreso)
    res = analizar_clip(c, cfg)
    salida = res.salida
    typer.echo(f"{c.id}: listo en {time.perf_counter() - t0:.0f}s -> {salida}")
    if res.conteo is None:
        typer.secho(
            f"Sin geometría ({c.geometria}): no hay conteo. Marcala con "
            f"`atlas-traffic geometry --clip {c.id}`.",
            fg=typer.colors.YELLOW,
        )
    elif detallado:
        typer.echo("\nConteo por aproximación (cruces únicos):")
        typer.echo(res.conteo.to_string())
    if detallado and res.intervalos is not None:
        cols = ["t_inicio_s", "t_fin_s", "flujo_veh_min", "flujo_peat_min", "densidad_media",
                "velocidad_media_px"]  # fmt: skip
        typer.echo(f"\nIntervalos de {cfg.metricas.intervalo_s} s:")
        typer.echo(res.intervalos[cols].round(2).to_string(index=False))
    if res.excluidos:
        typer.echo(f"Tracks descartados por zonas de exclusión: {list(res.excluidos)}")
    if res.eventos is not None and (detallado or not res.eventos.empty):
        typer.echo(f"Eventos de sentido contrario: {len(res.eventos)}")
        if detallado and not res.eventos.empty:
            cols = ["ts", "track_id", "aproximacion", "sentido", "clase_detalle", "thumb"]
            typer.echo(res.eventos[cols].round(1).to_string(index=False))

    if render:
        _render_presentacion(c, cfg, salida)

    if debug_video:
        import pandas as pd

        from atlas_traffic.config import load_geometria
        from atlas_traffic.render import video_debug, video_estabilizado

        det = pd.read_parquet(salida / "detecciones.parquet")
        ruta_estab = salida / "estabilizacion.parquet"
        estab = pd.read_parquet(ruta_estab) if ruta_estab.exists() else None
        geo = load_geometria(c.geometria) if c.geometria.exists() else None
        ruta_cruces = salida / "cruces.parquet"
        cruces = pd.read_parquet(ruta_cruces) if ruta_cruces.exists() else None
        ruta = video_debug(c.archivo, det, salida / "debug.mp4", geo, estab, cruces)
        typer.echo(f"Video de debug -> {ruta}")
        if estab is not None:
            ruta = video_estabilizado(c.archivo, estab, salida / "estabilizado.mp4")
            typer.echo(f"Video estabilizado -> {ruta}")
    return res


def _render_presentacion(c, cfg, salida: Path) -> None:
    import pandas as pd

    from atlas_traffic.config import load_geometria
    from atlas_traffic.render import gif_desde_video, video_anotado

    if not c.geometria.exists():
        typer.secho("Sin geometría: no hay video anotado.", fg=typer.colors.YELLOW)
        return

    def leer(nombre: str):
        ruta = salida / nombre
        return pd.read_parquet(ruta) if ruta.exists() else None

    mp4 = video_anotado(
        c.archivo, leer("detecciones.parquet"), load_geometria(c.geometria),
        salida / "anotado.mp4", titulo=c.id, estabilizacion=leer("estabilizacion.parquet"),
        cruces=leer("cruces.parquet"), intervalos=leer("intervalos.parquet"),
        eventos=leer("eventos.parquet"), escala=cfg.render.escala,
    )  # fmt: skip
    typer.echo(f"Video anotado -> {mp4}")
    r = cfg.render
    gif = gif_desde_video(mp4, salida / "muestra.gif", r.gif_inicio_s, r.gif_duracion_s,
                          r.gif_ancho, r.gif_fps, r.gif_colores)  # fmt: skip
    typer.echo(f"GIF -> {gif} ({gif.stat().st_size / 1e6:.1f} MB)")


@app.command()
def batch(
    config: Path = ConfigOpt,
    reusar_detecciones: bool = typer.Option(
        False,
        "--reusar-detecciones",
        help="No correr el modelo: reanalizar sobre las detecciones existentes.",
    ),
    render: bool = typer.Option(False, "--render", help="Video anotado y GIF de cada clip."),
    modelo: str | None = typer.Option(None, "--modelo", help="Pisa deteccion.modelo."),
    imgsz: int | None = typer.Option(None, "--imgsz", help="Pisa deteccion.imgsz."),
) -> None:
    """Reprocesa todo el dataset de punta a punta.

    1. detección y análisis de cada clip de clips.yaml;
    2. calibración de congestión con todos los intervalos y reclasificación;
    3. (opcional) video anotado y GIF;
    4. consolidación en outputs/summary.parquet, tracks_todos y eventos_todos.
    Un clip que falla no frena al resto; el comando termina con error si alguno falló.
    """
    import time

    from atlas_traffic.aggregate import calibrar_dataset, consolidar
    from atlas_traffic.clips import load_clips
    from atlas_traffic.detect import crear_detector

    t0 = time.perf_counter()
    cfg = _cfg_con_cambios(config, modelo, imgsz)
    clips = load_clips(cfg.rutas.clips)
    typer.echo(f"Batch: {len(clips)} clips en {cfg.rutas.clips}")
    detector = None
    if not reusar_detecciones:
        detector = crear_detector(cfg.deteccion.detector, **cfg.deteccion.kwargs_motor())
    fallidos: dict[str, str] = {}
    procesados = []
    for c in clips:
        if not c.archivo.exists():
            fallidos[c.id] = f"no existe el video {c.archivo}"
            typer.secho(f"{c.id}: {fallidos[c.id]}", fg=typer.colors.RED)
            continue
        try:
            _procesar_clip(c, cfg, reusar_detecciones, detector=detector)
            procesados.append(c)
        except Exception as e:  # un clip roto no frena el batch
            fallidos[c.id] = f"{type(e).__name__}: {e}"
            typer.secho(f"{c.id}: falló ({fallidos[c.id]})", fg=typer.colors.RED)

    if procesados:
        cal = calibrar_dataset(cfg, procesados)
        typer.echo(f"\nCalibración: {cal.calibracion.n_intervalos} intervalos -> "
                   f"{cfg.rutas.calibracion}")  # fmt: skip
        if render:
            for c in procesados:
                _render_presentacion(c, cfg, cfg.rutas.outputs / "runs" / c.id)
        res = consolidar(cfg, procesados)
        tabla = res.summary.groupby(["clip_id", "congestion"], dropna=False).size()
        n_clips = res.summary["clip_id"].nunique()
        typer.echo(f"\nsummary.parquet: {len(res.summary)} intervalos de {n_clips} clips "
                   f"-> {res.rutas['summary']}")  # fmt: skip
        typer.echo(tabla.rename("intervalos").to_string())
        typer.echo(f"tracks_todos.parquet: {len(res.tracks)} objetos | "
                   f"eventos_todos.parquet: {len(res.eventos)} eventos")  # fmt: skip
    typer.echo(f"\nBatch terminado en {time.perf_counter() - t0:.0f}s: "
               f"{len(procesados)} ok, {len(fallidos)} con error")  # fmt: skip
    if fallidos:
        raise typer.Exit(code=1)


@app.command()
def geometry(
    clip: str = typer.Option(..., "--clip", help="id del clip cuyo encuadre se marca."),
    config: Path = ConfigOpt,
    ver: bool = typer.Option(
        False,
        "--ver",
        help="No editar: dibujar la geometría guardada sobre el frame de referencia.",
    ),
) -> None:
    """Abre el frame de referencia del clip para marcar ROI y líneas de conteo con el mouse.

    La geometría se guarda en el archivo que indica `geometria` en clips.yaml y se
    comparte entre los clips filmados con el mismo encuadre.
    """
    import cv2

    from atlas_traffic.clips import get_clip, load_clips
    from atlas_traffic.config import load_geometria
    from atlas_traffic.geometry import EditorGeometria, editar_interactivo
    from atlas_traffic.pipeline import leer_frame
    from atlas_traffic.render import dibujar_geometria

    cfg = load_app_config(config)
    c = get_clip(load_clips(cfg.rutas.clips), clip)
    ref = cfg.estabilizacion.frame_referencia if cfg.estabilizacion.activa else 0
    frame = leer_frame(c.archivo, ref)
    alto, ancho = frame.shape[:2]

    if ver:
        geo = load_geometria(c.geometria)
        if tuple(geo.resolucion) != (ancho, alto):
            typer.secho(f"Ojo: la geometría es {geo.resolucion} y el video {ancho}x{alto}.",
                        fg=typer.colors.YELLOW)  # fmt: skip
        salida = cfg.rutas.outputs / "runs" / c.id / "geometria.png"
        salida.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(salida), dibujar_geometria(frame, geo.roi, geo.lineas,
                                                   exclusiones=geo.exclusiones))  # fmt: skip
        typer.echo(f"{len(geo.lineas)} líneas, ROI de {len(geo.roi)} vértices -> {salida}")
        return

    if c.geometria.exists():
        editor = EditorGeometria.desde_geometria(load_geometria(c.geometria))
        typer.echo(f"Editando {c.geometria} existente (r = empezar de cero).")
    else:
        editor = EditorGeometria(encuadre=c.geometria.stem, resolucion=(ancho, alto))
    typer.echo(f"Frame de referencia: {ref} de {c.archivo} ({ancho}x{alto}).")
    if not editar_interactivo(frame, editor, c.geometria):
        typer.echo("Salida sin guardar.")


@app.command()
def calibrate(config: Path = ConfigOpt) -> None:
    """Calcula umbrales de congestión por percentiles y los guarda en calibracion.yaml.

    Usa los intervalos y las velocidades de todos los clips ya procesados con `run`, y
    después reclasifica la congestión de cada uno con los umbrales nuevos.
    """
    from atlas_traffic.aggregate import calibrar_dataset
    from atlas_traffic.clips import load_clips

    cfg = load_app_config(config)
    try:
        res = calibrar_dataset(cfg, load_clips(cfg.rutas.clips))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    cal = res.calibracion
    for cid in res.omitidos:
        typer.secho(f"  {cid}: sin procesar o sin geometría, se omite", fg=typer.colors.YELLOW)
    typer.echo(f"Calibración con {cal.n_intervalos} intervalos de {len(cal.clips)} clips "
               f"-> {cfg.rutas.calibracion}")  # fmt: skip
    p90 = {k: round(v, 2) for k, v in cal.velocidad_p90_por_encuadre.items()}
    typer.echo(f"  velocidad de flujo libre (p90) por encuadre: {p90}")
    typer.echo(f"  densidad: p33={cal.densidad.p33:.2f} p66={cal.densidad.p66:.2f}")
    typer.echo(f"  velocidad normalizada: p33={cal.velocidad_norm.p33:.2f} "
               f"p66={cal.velocidad_norm.p66:.2f}")  # fmt: skip
    cols = ["clip_id", "t_inicio_s", "t_fin_s", "densidad_media", "velocidad_norm", "congestion"]
    typer.echo("\nCongestión por intervalo:")
    typer.echo(res.intervalos[cols].round(2).to_string(index=False))


@app.command()
def dashboard(
    config: Path = ConfigOpt,
    probar: bool = typer.Option(False, "--probar", help="Validar sin levantar el servidor."),
) -> None:
    """Levanta el dashboard Streamlit (Panorama, Eventos y Consulta)."""
    import subprocess
    import sys

    cfg = load_app_config(config)
    # en la carpeta de trabajo (Docker, repo) o junto al código (instalación editable)
    candidatos = [
        Path("dashboard/app.py"),
        Path(__file__).resolve().parents[2] / "dashboard/app.py",
    ]
    app_py = next((c for c in candidatos if c.exists()), None)
    if app_py is None:
        raise typer.BadParameter(f"no se encontró el dashboard en {candidatos}")
    if not (cfg.rutas.outputs / "summary.parquet").exists():
        typer.secho("Ojo: no hay outputs/summary.parquet. Corré `atlas-traffic batch` primero.",
                    fg=typer.colors.YELLOW)  # fmt: skip
    if probar:
        typer.echo(f"Dashboard en {app_py} (config {config}); no se levanta (--probar).")
        return
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_py), "--", "--config", str(config)]
    raise typer.Exit(subprocess.call(cmd))


if __name__ == "__main__":
    app()
