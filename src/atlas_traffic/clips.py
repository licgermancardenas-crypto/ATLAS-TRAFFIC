"""Lectura y validación de configs/clips.yaml — la fuente de verdad del contexto de cada clip."""

from __future__ import annotations

from pathlib import Path

from atlas_traffic.config import Clip, ClipsFile, read_yaml


def load_clips(path: Path) -> list[Clip]:
    return ClipsFile.model_validate(read_yaml(path)).clips


def get_clip(clips: list[Clip], clip_id: str) -> Clip:
    for clip in clips:
        if clip.id == clip_id:
            return clip
    disponibles = ", ".join(c.id for c in clips) or "(ninguno)"
    raise KeyError(f"clip {clip_id!r} no está en clips.yaml. Disponibles: {disponibles}")


def archivos_faltantes(clips: list[Clip], base: Path = Path(".")) -> list[Path]:
    """Videos o geometrías referenciados que no existen en disco."""
    faltan: list[Path] = []
    for clip in clips:
        for ruta in (clip.archivo, clip.geometria):
            if not (base / ruta).exists():
                faltan.append(ruta)
    return faltan
