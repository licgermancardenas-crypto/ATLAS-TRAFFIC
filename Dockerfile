# ATLAS Traffic — pipeline + dashboard en CPU.
#
#   docker build -t atlas-traffic .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" \
#       -v "$PWD/configs:/app/configs" atlas-traffic batch --render
#   docker run --rm -p 8501:8501 -v "$PWD/outputs:/app/outputs" \
#       -v "$PWD/configs:/app/configs" atlas-traffic dashboard
#
# Ultralytics es AGPL-3.0 (ver README). torch se instala en su variante CPU para no
# arrastrar CUDA (varios GB) a una imagen que corre en CPU.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias primero (capa cacheable), código después.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install ".[detect,dashboard]"

COPY configs ./configs
COPY dashboard ./dashboard
COPY modelos ./modelos
RUN mkdir -p data/raw outputs

EXPOSE 8501
ENTRYPOINT ["atlas-traffic"]
CMD ["--help"]
