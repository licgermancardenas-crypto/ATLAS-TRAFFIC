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
) -> None:
    """Procesa un clip: detección, tracking, conteo, métricas y eventos."""
    load_app_config(config)
    _pendiente("Fase 1")


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
