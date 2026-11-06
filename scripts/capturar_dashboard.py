"""Capturas del dashboard para el README, con Chrome headless vía DevTools Protocol.

Requiere el dashboard corriendo (atlas-traffic dashboard) y Chrome instalado.
Uso: python scripts/capturar_dashboard.py [url_base] [carpeta_salida]

Streamlit renderiza por websocket después de cargar la página, así que `chrome
--screenshot` captura el esqueleto de carga. Acá se espera tiempo real antes de capturar.
"""

import asyncio
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]
PUERTO = 9233
# vista -> (parámetros de la URL, alto de la captura). La consulta es la del criterio de
# aceptación del brief: autos blancos que entraron por Córdoba entre los segundos 10 y 30.
VISTAS = {
    "panorama": ("vista=panorama", 1500),
    "seguir": ("vista=seguir&clip=muestra_2&vehiculo=134", 1750),
    "consulta": (
        "vista=consulta&color=blanco&movimiento=av_cordoba_frente%20%C2%B7%20entrante"
        "&desde=10&hasta=30",
        1000,
    ),
}


async def capturar(ws_url: str, url: str, salida: Path, alto: int) -> None:
    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        n = 0

        async def cdp(metodo: str, **params):
            nonlocal n
            n += 1
            await ws.send(json.dumps({"id": n, "method": metodo, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == n:
                    return msg.get("result", {})

        await cdp("Emulation.setDeviceMetricsOverride", width=1400, height=alto,
                  deviceScaleFactor=1, mobile=False)  # fmt: skip
        await cdp("Page.navigate", url=url)
        await asyncio.sleep(12)  # Streamlit: websocket + gráficos de Vega
        res = await cdp("Page.captureScreenshot", format="png")
        salida.write_bytes(base64.b64decode(res["data"]))


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
    carpeta = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/assets")
    chrome = next(c for c in CHROME if c and Path(c).exists())
    perfil = tempfile.mkdtemp()
    proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                             f"--remote-debugging-port={PUERTO}", f"--user-data-dir={perfil}",
                             "about:blank"])  # fmt: skip
    try:
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PUERTO}/json") as r:
                    paginas = [p for p in json.load(r) if p["type"] == "page"]
                if paginas:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        ws_url = paginas[0]["webSocketDebuggerUrl"]
        # calentamiento: la primera carga del servidor lee los datos y tarda más
        asyncio.run(
            capturar(ws_url, f"{base}/?vista=panorama", carpeta / "_calentamiento.png", 800)
        )
        (carpeta / "_calentamiento.png").unlink()
        for vista, (parametros, alto) in VISTAS.items():
            destino = carpeta / f"dashboard_{vista}.png"
            asyncio.run(capturar(ws_url, f"{base}/?{parametros}", destino, alto))
            print(destino, destino.stat().st_size // 1024, "KB")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
