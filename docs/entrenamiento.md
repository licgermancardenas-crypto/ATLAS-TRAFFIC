# Modelo ajustado a la esquina (fine-tuning)

El detector por defecto es `modelos/atlas_esquina_yolov8s.pt`: un `yolov8s` general
reentrenado con imágenes de Av. Córdoba y Rodríguez Peña, para que reconozca cómo se ven
acá los autos, motos, colectivos, camiones y personas (chicos, comprimidos y vistos desde
un sexto piso).

## Cómo se hizo

1. **Dataset** (`scripts/dataset_finetuning.py`): 179 cuadros (133 de entrenamiento y 46
   de validación) de los tres clips, uno cada 1/3 s.
   - **Los primeros 20 s de cada clip no se usan.** Son los que tienen conteo manual, y
     quedan reservados para evaluar sin trampa.
2. **Etiquetas automáticas de un "modelo maestro":** el mismo `yolov8s` mirando cada
   cuadro entero a 1280 px y además en 6 partes ampliadas. Así ve objetos chicos que a
   640 px se pierden, pero es demasiado lento para el uso diario. El modelo entrenado
   aprende a ver como el maestro a la velocidad normal (destilación).
   - Se descartan los vehículos en zonas de exclusión (valla, vidriera, techo de la
     parada). Las personas se conservan.
   - Las cajas repetidas se filtran **sin importar la clase**: si no, la misma camioneta
     quedaba como "auto" y "camión", y el modelo aprendía a dibujar dos cajas por
     vehículo (ver más abajo).
3. **Entrenamiento** en Google Colab con GPU gratis (`entrenamiento/entrenar_en_colab.ipynb`):
   80 épocas, 640 px. mAP50 de validación = 0.58. Ese número mide cuánto imita al maestro;
   la prueba que importa es la siguiente.
4. **Evaluación** (`scripts/evaluar_modelo.py`): el pipeline completo con cada modelo
   sobre los 20 s reservados, comparado línea por línea con el conteo manual.

## Resultado (cruces de vehículos, primeros 20 s)

| Clip | Línea | Manual | `yolov8s` general | **Modelo de la esquina** |
|---|---|---|---|---|
| muestra_2 | frente | 22 | 21 | **22** |
| muestra_2 | fondo | 18 | 17 | **17** |
| muestra_3 | frente | 18 | 16 | **19** |
| muestra_3 | fondo | 10 (5 motos) | 5 | **7** |
| muestra_3 | R. Peña izq | 2 | 1 | **2** |
| muestra_1 (noche) | frente | 14 | 9 | **10** |
| muestra_1 (noche) | fondo | 23 | 18 | **19** |
| muestra_1 (noche) | R. Peña izq | 2 | 1 | **1** |
| **Detección global** | | | **81 %** | **88 %** |

- **Gana o empata en todas las líneas.**
- **Motos:** las detecta (3 y 7 en los clips de día), y el modelo general no detectaba
  ninguna.
- **Peatones:** detecta más en muestra_3 (23 contra 7) pero menos en muestra_2 (2 contra
  10). No hay conteo manual de peatones para saber cuál acierta.

## Problemas que aparecieron y cómo se resolvieron

- **Cajas dobles:** en la primera evaluación el modelo contaba +39 % en la cola de
  muestra_3. Dibujaba "auto" y "camión" sobre la misma camioneta: 2.223 pares de cajas
  casi idénticas, contra 82 del modelo general.
  - Se activó el filtro de cajas repetidas sin importar la clase al detectar
    (`nms_agnostico`, también mejora al modelo general).
  - Se corrigió la causa en el armado del dataset.
- **Ida y vuelta instantánea:** un pedazo de un colectivo tapado por un árbol "cruzaba y
  volvía" la línea en el mismo instante y sumaba dos cruces falsos.
  - Ahora, cruzar y volver a cruzar en menos de 1 s no cuenta (`ida_vuelta_min_s`).
  - Una vuelta real, más lenta, sigue contando en los dos sentidos, como pide el brief.
- **Conteo de referencia corregido:** al revisar esos cruces apareció un auto rojo,
  tapado por un árbol, que el conteo manual no había registrado. Se verificó cuadro por
  cuadro antes de corregir el registro (Rodríguez Peña izquierda en muestra_3: de 1 a 2).

## Cómo mejorarlo

- **Corregir etiquetas a mano.** El dataset está en formato YOLO estándar
  (`entrenamiento/dataset_esquina.zip`) y se puede importar en Label Studio o CVAT. El
  maestro deja pasar algunos vehículos evidentes y pocas motos (26 en total).
- **Más video y otras condiciones:** noche, lluvia, fines de semana.
- **Licencia:** como deriva de YOLO (Ultralytics), el modelo es AGPL-3.0. Para un
  producto comercial cerrado, el mismo procedimiento se puede repetir sobre un modelo
  Apache 2.0 (RF-DETR, RT-DETR); el sistema cambia de motor sin tocar el resto.
