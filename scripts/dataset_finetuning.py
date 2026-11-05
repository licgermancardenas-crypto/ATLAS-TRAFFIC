"""Arma el dataset para ajustar YOLO a esta esquina (fine-tuning), con etiquetas automáticas.

Etiquetado por "modelo maestro": el mismo YOLO, pero mirando cada cuadro (a) entero a 1280
px y (b) en 6 partes ampliadas que se solapan. Así ve objetos chicos (motos lejanas,
peatones de 10 px) que a 640 se pierde, pero es demasiado lento para el uso diario.
Entrenar al modelo normal con estas etiquetas le enseña a ver como el maestro a la
velocidad de siempre (destilación). Las etiquetas se pueden corregir a mano después en
Label Studio o CVAT (formato YOLO estándar) para mejorarlo más.

Para no hacer trampa en la evaluación:
- Los primeros 20 s de cada clip NO se usan: ahí están los conteos manuales con los que se
  mide el modelo entrenado (docs/validacion_conteo_registros.csv).
- El tramo final de cada clip va a validación del entrenamiento, el resto a entrenamiento.

Las detecciones que caen en zonas de exclusión (valla, vidriera, techo de la parada) se
descartan: si no, el modelo aprendería que esos objetos fijos son vehículos.

Uso: python scripts/dataset_finetuning.py   ->  entrenamiento/dataset_esquina.zip
"""

import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from atlas_traffic.clips import load_clips
from atlas_traffic.config import load_app_config, load_geometria
from atlas_traffic.geometry import dentro_roi
from atlas_traffic.render import homografias_por_frame

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "entrenamiento"
CLASES = ["person", "car", "motorcycle", "bus", "truck"]  # orden = índice del dataset
RESERVA_EVALUACION_S = 20.0  # los conteos manuales cubren 0-20 s
CADA_S = 1 / 3  # un cuadro cada 1/3 s
FRACCION_VAL = 0.25  # tramo final de la parte usable de cada clip
CONF_MAESTRO = 0.35  # umbral para aceptar una etiqueta del maestro


def nms_por_clase(cajas: np.ndarray, conf: np.ndarray, clases: np.ndarray, iou: float = 0.5):
    """Non-maximum suppression por clase. cajas (N, 4) x1 y1 x2 y2. Devuelve índices."""
    elegidos = []
    for c in np.unique(clases):
        idx = np.nonzero(clases == c)[0]
        rects = [
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in cajas[idx]
        ]
        keep = cv2.dnn.NMSBoxes(rects, conf[idx].tolist(), 0.0, iou)
        elegidos += [int(idx[k]) for k in np.array(keep).reshape(-1)]
    return sorted(elegidos)


def maestro(modelo, frame: np.ndarray, nombres: dict[int, str]):
    """Detecciones del modelo maestro: cuadro entero a 1280 + 2 x 3 partes ampliadas."""
    alto, ancho = frame.shape[:2]
    cajas, confs, clases = [], [], []

    def agregar(res, dx=0, dy=0):
        for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                           res.boxes.cls.cpu().numpy().astype(int), strict=True):  # fmt: skip
            nombre = nombres[int(k)]
            if nombre in CLASES and c >= CONF_MAESTRO:
                cajas.append(b + [dx, dy, dx, dy])
                confs.append(float(c))
                clases.append(CLASES.index(nombre))

    agregar(modelo.predict(frame, imgsz=1280, conf=CONF_MAESTRO, verbose=False)[0])
    columnas, filas, solape = 2, 3, 0.25
    tw, th = int(ancho / columnas * (1 + solape)), int(alto / filas * (1 + solape))
    for i in range(columnas):
        for j in range(filas):
            x0 = min(int(i * ancho / columnas), ancho - tw)
            y0 = min(int(j * alto / filas), alto - th)
            parte = frame[y0 : y0 + th, x0 : x0 + tw]
            agregar(modelo.predict(parte, imgsz=640, conf=CONF_MAESTRO, verbose=False)[0], x0, y0)
    if not cajas:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    cajas, confs, clases = np.array(cajas), np.array(confs), np.array(clases)
    k = nms_por_clase(cajas, confs, clases)
    return cajas[k], confs[k], clases[k]


def main() -> None:
    from ultralytics import YOLO

    cfg = load_app_config(RAIZ / "configs/default.yaml")
    modelo = YOLO(cfg.deteccion.modelo)
    nombres = modelo.names
    dataset = SALIDA / "dataset_esquina"
    if dataset.exists():
        shutil.rmtree(dataset)
    for parte in ("train", "val"):
        (dataset / "images" / parte).mkdir(parents=True)
        (dataset / "labels" / parte).mkdir(parents=True)
    resumen = []
    vistas = []
    excluidas = 0
    for clip in load_clips(RAIZ / cfg.rutas.clips):
        video = RAIZ / clip.archivo
        geo = load_geometria(RAIZ / clip.geometria)
        estab = RAIZ / cfg.rutas.outputs / "runs" / clip.id / "estabilizacion.parquet"
        hs = homografias_por_frame(pd.read_parquet(estab)) if estab.exists() else {}
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        inicio, paso = int(RESERVA_EVALUACION_S * fps), max(2, int(round(CADA_S * fps / 2)) * 2)
        cuadros = list(range(inicio, n, paso))
        corte_val = cuadros[int(len(cuadros) * (1 - FRACCION_VAL))] if cuadros else n
        for f in cuadros:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                break
            cajas, confs, clases = maestro(modelo, frame, nombres)
            if len(cajas) and geo.exclusiones:  # sacar objetos fijos (valla, vidriera, parada)
                if not hs:
                    raise SystemExit(
                        f"{clip.id}: falta estabilizacion.parquet para ubicar las zonas "
                        "de exclusión; corré `atlas-traffic run` primero"
                    )
                # la estabilización guarda la H de los cuadros procesados (pares): la más cercana
                H = hs[min(hs, key=lambda k: abs(k - f))]
                centros = np.column_stack(
                    [(cajas[:, 0] + cajas[:, 2]) / 2, (cajas[:, 1] + cajas[:, 3]) / 2]
                )
                ref = cv2.perspectiveTransform(centros.reshape(-1, 1, 2), H).reshape(-1, 2)
                fuera = np.ones(len(cajas), dtype=bool)
                for zona in geo.exclusiones:
                    fuera &= ~dentro_roi(zona, ref)
                # solo se descartan vehículos: una persona esperando bajo la parada es real, y
                # sin etiqueta el modelo aprendería a no verla
                fuera |= clases == CLASES.index("person")
                excluidas += int((~fuera).sum())
                cajas, confs, clases = cajas[fuera], confs[fuera], clases[fuera]
            parte = "val" if f >= corte_val else "train"
            nombre = f"{clip.id}_{f:05d}"
            cv2.imwrite(str(dataset / "images" / parte / f"{nombre}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])  # fmt: skip
            alto, ancho = frame.shape[:2]
            with open(dataset / "labels" / parte / f"{nombre}.txt", "w") as fh:
                for (x1, y1, x2, y2), c in zip(cajas, clases, strict=True):
                    fh.write(f"{c} {(x1 + x2) / 2 / ancho:.6f} {(y1 + y2) / 2 / alto:.6f} "
                             f"{(x2 - x1) / ancho:.6f} {(y2 - y1) / alto:.6f}\n")  # fmt: skip
            for c in clases:
                resumen.append((clip.id, parte, CLASES[c]))
            if len(vistas) < 6 and (f - inicio) % (paso * 25) == 0:
                vistas.append((frame.copy(), cajas, clases))
            print(f"\r{clip.id} cuadro {f}/{n}  ", end="", file=sys.stderr)
        cap.release()
    print(file=sys.stderr)
    (dataset / "data.yaml").write_text(yaml.safe_dump({
        "path": ".", "train": "images/train", "val": "images/val",
        "names": dict(enumerate(CLASES)),
    }, sort_keys=False), encoding="utf-8")  # fmt: skip

    # vista previa para revisar las etiquetas a ojo
    colores = [(255, 120, 0), (0, 200, 255), (255, 0, 255), (0, 0, 255), (0, 255, 0)]
    previas = []
    for frame, cajas, clases in vistas:
        img = frame.copy()
        for (x1, y1, x2, y2), c in zip(cajas.astype(int), clases, strict=True):
            cv2.rectangle(img, (x1, y1), (x2, y2), colores[c], 1)
        previas.append(cv2.resize(img, (232, 416)))
    if previas:
        cv2.imwrite(str(SALIDA / "vista_previa_etiquetas.jpg"), np.hstack(previas))

    zip_ruta = SALIDA / "dataset_esquina.zip"
    with zipfile.ZipFile(zip_ruta, "w", zipfile.ZIP_DEFLATED) as z:
        for f in dataset.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(dataset))
    t = pd.DataFrame(resumen, columns=["clip", "parte", "clase"])
    print(
        "imágenes:",
        {p: len(list((dataset / "images" / p).glob("*.jpg"))) for p in ("train", "val")},
    )
    print("etiquetas por clase y parte:")
    print(t.groupby(["parte", "clase"]).size().unstack(fill_value=0).to_string())
    print(f"etiquetas descartadas por zonas de exclusión: {excluidas}")
    print(f"-> {zip_ruta} ({zip_ruta.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
