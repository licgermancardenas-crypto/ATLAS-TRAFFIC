import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from atlas_traffic.cli import app

RAIZ = Path(__file__).resolve().parents[1]
SUBCOMANDOS = ("run", "batch", "geometry", "calibrate", "dashboard")


def test_python_m_help_lista_subcomandos():
    res = subprocess.run(
        [sys.executable, "-m", "atlas_traffic", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=RAIZ,
        env={**os.environ, "COLUMNS": "200", "PYTHONIOENCODING": "utf-8"},
    )
    assert res.returncode == 0, res.stderr
    for sub in SUBCOMANDOS:
        assert sub in res.stdout


def test_subcomandos_validan_config_y_salen_ok(monkeypatch):
    monkeypatch.chdir(RAIZ)
    runner = CliRunner()
    for args in (["dashboard"],):  # batch y calibrate trabajan de verdad: tienen sus tests
        res = runner.invoke(app, args)
        assert res.exit_code == 0, res.output


def test_config_inexistente_falla(tmp_path):
    res = CliRunner().invoke(app, ["batch", "--config", str(tmp_path / "nope.yaml")])
    assert res.exit_code != 0


def test_run_y_geometry_con_clip_inexistente_fallan(monkeypatch):
    monkeypatch.chdir(RAIZ)
    for comando in ("run", "geometry"):
        res = CliRunner().invoke(app, [comando, "--clip", "no-existe"])
        assert res.exit_code != 0
        assert "no está en clips.yaml" in str(res.exception)
