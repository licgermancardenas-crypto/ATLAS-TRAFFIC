# ATLAS Traffic

Pipeline de visión por computadora para analizar tránsito en una bocacalle de avenida
(CABA, Recoleta) filmada desde un punto fijo. Cuenta vehículos y peatones de forma única
y por aproximación, clasifica la congestión por **densidad + velocidad** y consolida
varios clips en un dataset comparable por día y horario.

> Estado: **Fase 0 / 0b** — scaffolding y protocolo de detección. Spec completa en
> [`docs/ATLAS-Traffic-Brief.pdf`](docs/ATLAS-Traffic-Brief.pdf).

## Instalación

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"              # núcleo + tests (sin motor de detección)
uv pip install -e ".[detect,dashboard]" # YOLO + ByteTrack, Streamlit
```

## Uso

```bash
python -m atlas_traffic --help
atlas-traffic run --clip <id>       # procesa un clip
atlas-traffic batch                 # todos los clips de configs/clips.yaml
atlas-traffic geometry --clip <id>  # marcar ROI y líneas de conteo
atlas-traffic calibrate             # umbrales de congestión por percentiles
atlas-traffic dashboard             # Streamlit
```

## Licencia

Código propio: MIT. El motor Ultralytics (extra `detect`) es AGPL-3.0; el detector está
abstraído detrás de un protocolo (`atlas_traffic.detect.Detector`) para poder migrar a
motores Apache 2.0 (RF-DETR, RT-DETR, YOLOX) sin reescribir el pipeline.
