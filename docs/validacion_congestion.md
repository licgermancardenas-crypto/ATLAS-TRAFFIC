# Validación de la congestión (Fase 4)

**Criterio del brief:** el clip de hora pico da HIGH y el de domingo temprano da LOW,
sin tocar los umbrales a mano.

Todavía no hay clips de hora pico ni de domingo, así que se usa el análogo con los clips
disponibles: muestra_3 (cola de autos detenidos) debería dar HIGH, y muestra_1 (noche,
poco tránsito) y muestra_2 (tránsito fluido) deberían dar LOW.

## Método (brief §4)

1. **Velocidad normalizada por encuadre:** velocidad media / p90 de las velocidades
   instantáneas del encuadre, que representa el flujo libre de esa vista.
2. **Umbrales con los datos:** densidad y velocidad normalizada se cortan en los
   percentiles 33 y 66 de todos los intervalos.
3. **Matriz explícita**, la del brief, en `congestion.py`.
4. **Persistencia:** `configs/calibracion.yaml`, versionado en el repo para que los
   umbrales sean auditables. Se regenera con `atlas-traffic calibrate`.

## Resultado

Calibración con 5 intervalos de 30 s de los 3 clips (detección con `yolov8s`):

| Clip | Ventana | Densidad | Velocidad normalizada | Congestión |
|---|---|---|---|---|
| muestra_1 | 0–22 s | 3.9 | 0.56 | LOW ✓ |
| muestra_2 | 0–30 s | 6.9 | 0.53 | LOW ✓ |
| muestra_2 | 30–50 s | 3.7 | 0.44 | LOW ✓ |
| muestra_3 | 0–30 s | 18.1 | 0.48 | HIGH ✓ |
| muestra_3 | 30–46 s | 18.1 | 0.39 | HIGH ✓ |

**Umbrales calculados:**
- Densidad: p33 = 4.88, p66 = 14.05.
- Velocidad normalizada: p33 = 0.46, p66 = 0.51.

Cumple el criterio sin ajustar nada a mano.

## Problema: el supuesto de flujo libre no se cumple

El brief advierte que la normalización por encuadre supone que cada encuadre tiene al
menos un momento de flujo libre. Acá **cada encuadre tiene un solo clip**, y el de
muestra_3 es casi todo cola:

| Encuadre | p90 (flujo libre estimado) |
|---|---|
| encuadre_m1 | 1.77 px/frame |
| encuadre_m2 | 2.72 px/frame |
| encuadre_m3 | **0.58 px/frame** |

El p90 de muestra_3 es una cola lenta, no flujo libre. Al dividir por él, sus
velocidades se inflan y terminan iguales a las de los clips fluidos. Todas las
velocidades normalizadas quedan entre 0.39 y 0.56, y los cortes p33 y p66 quedan a
0.05 de distancia. En la práctica, **el HIGH de muestra_3 lo decide la densidad**, y la
dimensión de velocidad casi no aporta.

### Comparación con una referencia común

Con un único p90 para los tres encuadres (1.54 px/frame), la velocidad sí distingue:

| Clip | Ventana | Velocidad cruda | Normalizada por encuadre | Normalizada común |
|---|---|---|---|---|
| muestra_1 | 0–22 s | 0.99 | 0.56 | 0.64 |
| muestra_2 | 0–30 s | 1.43 | 0.53 | 0.93 |
| muestra_2 | 30–50 s | 1.20 | 0.44 | 0.78 |
| muestra_3 | 0–30 s | 0.28 | 0.48 | **0.18** |
| muestra_3 | 30–46 s | 0.23 | 0.39 | **0.15** |

No se adoptó porque supone que los tres encuadres tienen una escala en píxeles
comparable. Acá es razonable (mismo balcón, mismo teléfono), pero no está garantizado.

## Recomendación

Grabar todos los clips con el **mismo encuadre fijo** (brief §8: "menos encuadres, menos
supuestos"). Con varios clips de distintos horarios en un mismo encuadre, el p90 incluye
momentos de flujo libre y la normalización funciona como se diseñó.

## Otras limitaciones

- **Con 5 intervalos, los percentiles son poco estables.** Los umbrales se van a mover
  bastante al sumar clips; por eso se recalculan con `calibrate` y quedan versionados.
- **La densidad no se normaliza.** Con zonas de análisis de distinto tamaño físico, las
  densidades de distintos encuadres no son del todo comparables.
