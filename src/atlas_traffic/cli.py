"""Entrypoint de línea de comandos.

Subcomandos: run, batch, geometry, calibrate, dashboard.
En la Fase 0 son esqueletos: validan la configuración y anuncian en qué fase se implementan.
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


def _pendiente(fase: str) -> None:
    typer.secho(f"Todavía no implementado — llega en la {fase}.", fg=typer.colors.YELLOW)


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
) -> None:
    """Procesa un clip: detección, tracking, conteo, métricas y eventos."""
    import time

    from atlas_traffic.clips import get_clip, load_clips
    from atlas_traffic.detect import crear_detector
    from atlas_traffic.pipeline import correr_clip

    cfg = load_app_config(config)
    cambios = {k: v for k, v in {"modelo": modelo, "imgsz": imgsz}.items() if v is not None}
    if cambios:
        cfg = cfg.model_copy(update={"deteccion": cfg.deteccion.model_copy(update=cambios)})
    c = get_clip(load_clips(cfg.rutas.clips), clip)
    detector = crear_detector(cfg.deteccion.detector, **cfg.deteccion.kwargs_motor())

    t0 = time.perf_counter()
    with typer.progressbar(length=1, label=f"Procesando {c.id}") as barra:

        def progreso(i: int, total: int) -> None:
            barra.length = total
            barra.update(i - barra.pos)

        salida = correr_clip(c, cfg, detector, max_segundos, progreso)
    typer.echo(f"Listo en {time.perf_counter() - t0:.0f}s -> {salida}")

    if debug_video:
        import pandas as pd

        from atlas_traffic.render import video_debug, video_estabilizado

        det = pd.read_parquet(salida / "detecciones.parquet")
        typer.echo(f"Video de debug -> {video_debug(c.archivo, det, salida / 'debug.mp4')}")
        estab = salida / "estabilizacion.parquet"
        if estab.exists():
            ruta = video_estabilizado(
                c.archivo, pd.read_parquet(estab), salida / "estabilizado.mp4"
            )
            typer.echo(f"Video estabilizado -> {ruta}")


@app.command()
def batch(config: Path = ConfigOpt) -> None:
    """Procesa todos los clips de clips.yaml y consolida summary.parquet."""
    load_app_config(config)
    _pendiente("Fase 6")


@app.command()
def geometry(
    clip: str = typer.Option(..., "--clip", help="id del clip cuyo encuadre se marca."),
    config: Path = ConfigOpt,
) -> None:
    """Abre un frame del clip para marcar ROI y líneas de conteo con el mouse."""
    load_app_config(config)
    _pendiente("Fase 1b")


@app.command()
def calibrate(config: Path = ConfigOpt) -> None:
    """Calcula umbrales de congestión por percentiles y los guarda en calibracion.yaml."""
    load_app_config(config)
    _pendiente("Fase 4")


@app.command()
def dashboard(config: Path = ConfigOpt) -> None:
    """Levanta el dashboard Streamlit."""
    load_app_config(config)
    _pendiente("Fase 7")


if __name__ == "__main__":
    app()
