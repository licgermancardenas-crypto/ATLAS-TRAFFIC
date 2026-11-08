# Riesgo peatonal: vehículos que pasan por donde cruza un peatón

## Qué se mide

El sistema usa el **PET** (*post-encroachment time*), el indicador estándar de los estudios
de seguridad vial con video. Es el tiempo entre que un usuario de la calle deja un lugar y
otro llega a ese mismo lugar.

No hace falta que choquen ni que estén al mismo tiempo en el mismo lugar. Un auto que pasa
por donde un peatón estaba un segundo antes estuvo a un segundo de atropellarlo.

- **"Mismo lugar":** a menos de 2 m en el suelo. Se usa la calibración métrica de las
  sendas peatonales (ver [velocidad.md](velocidad.md)).
- **Gravedad:**
  - grave: PET menor a 1 s;
  - moderada: de 1 a 2 s;
  - leve: de 2 a 3 s.
  - Con más de 3 s no se considera una situación de riesgo.

Código: `src/atlas_traffic/conflictos.py`. Parámetros: sección `conflictos:` de
`configs/default.yaml`.

## Filtros (validados a ojo)

Sin filtros, la regla dio 8 candidatos en los 4 clips. Se revisaron uno por uno con tiras
de cuadros y con la evidencia en video. Siete eran falsos, por estas razones:

| Causa del falso positivo | Casos | Filtro que lo descarta |
|---|---|---|
| Un "peatón" detectado apenas un instante (2 a 7 detecciones en 0,1–0,2 s) en medio de la calle: un poste que se proyecta sobre la calzada o un reflejo con lluvia | 5 | El peatón tiene que haberse seguido al menos 1 s (15 detecciones) |
| Una persona parada en el cordón, a ~2 m del carril, mientras pasan los autos | 1 | El peatón tiene que estar en la calzada: por ese punto pasan al menos 3 vehículos en el clip |
| Un objeto fijo de la vereda que el detector toma por vehículo y cuya caja "tiembla" | 1 | El vehículo tiene que circular (más de 8 km/h) y desplazarse al menos 10 m |

Con los filtros queda **1 situación de riesgo** en los 4 clips, y es real:

- **Clip:** `esquina_20261105_1946`, 19:46:17.
- **Qué pasó:** el auto #237 cruzó la senda peatonal de Córdoba a 36 km/h. 2,7 s después,
  el peatón #284 empezó a cruzar por el mismo lugar.
- **Clasificación:** leve.

## Evidencia

Cada situación guarda dos archivos en `outputs/runs/<clip>/conflictos/`:
- una foto;
- un video corto (unos 2 s antes y después).

En los dos, el vehículo está marcado en rojo con su velocidad y el peatón en amarillo. Las
alertas de sentido contrario también guardan ahora su video, en `eventos/`.

## Limitaciones

- **Pocos peatones bien seguidos.** En estos clips los peatones se detectan poco: son
  chicos, el video pasó por WhatsApp y hay muchas personas que se tapan entre sí. Por eso
  el filtro de 1 s descarta mucho. Es preferible perder una situación real antes que
  mostrar una falsa. Con video original y cámara fija van a aparecer más.
- **El vehículo es un punto.** Se usa el centro del borde inferior de la caja, donde el
  vehículo toca el suelo, y no su contorno completo. Un auto mide ~1,8 × 4,5 m, así que el
  radio de 2 m es una aproximación.
- **Sin semáforo.** El sistema no sabe de qué color estaba el semáforo. No distingue si el
  peatón cruzaba con luz verde o si el auto pasó en rojo; solo mide la cercanía.
- **Revisión a ojo.** Las alertas son para revisarlas, no para sancionar automáticamente.
