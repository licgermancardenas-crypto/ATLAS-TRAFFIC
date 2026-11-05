# Validación del conteo (Fase 2)

**Criterio del brief:** el conteo automático por aproximación tiene que coincidir con un
conteo manual del clip de 20 s, con un margen de ±10 % en cada línea.

## Método

El conteo de referencia no usa el detector, para no comparar el sistema contra sí mismo.
Para cada línea se arma un **barrido temporal** (`scripts/barrido_temporal.py`): en cada
frame se toman los píxeles que caen sobre la línea (en coordenadas estabilizadas, así la
línea queda fija sobre la calle aunque la cámara se mueva) y se apilan en el tiempo. Cada
vehículo que cruza deja una mancha. El conteo se hizo **a ciegas** sobre esas imágenes y
recién después se comparó, cruce por cruce, con `cruces.parquet`. Los casos dudosos se
verificaron mirando los frames originales.

> El conteo de referencia lo hizo Claude (Claude Code) mirando las imágenes, no una
> persona. Conviene repetirlo a mano en algún clip.

**Casos de borde:**
- Los vehículos que en el primer frame ya pasaron la línea no cuentan.
- Los que la cruzan después del segundo 20 tampoco.
- Las cuñas negras del barrido son el borde de la imagen cuando la cámara se corre, no vehículos.

## Resultado — muestra_2, primeros 20 s

| Línea | Manual | Automático | Diferencia | Falsos positivos |
|---|---|---|---|---|
| av_cordoba_frente | 22 | 20 | −9.1 % ✓ | 0 |
| av_cordoba_fondo | 18 | 17 | −5.6 % ✓ | 0 |
| rodriguez_pena_der | 0 | 0 | ✓ | 0 |

**Cruces perdidos:**
- **Una moto**, en ambas líneas: la detección de motos con `yolov8n` a 464 px es débil.
- **Un auto del carril izquierdo**, en el borde inferior del cuadro (línea del frente).
  Se detectó recién cuando ya había cruzado.

No hubo ningún cruce falso, y todos los cruces de la avenida salieron en el sentido de
circulación (Córdoba es mano única).

## Correcciones que salieron de la validación

La primera comparación daba −19 % en el frente y −15 % en el fondo. Las causas eran dos:

1. **Línea del frente demasiado cerca del borde inferior.** Los autos del carril izquierdo
   se detectaban recién después de cruzarla. Se subió ~45 px hacia el cruce.
2. **Segmento del fondo demasiado corto.** Los autos que salían por el borde izquierdo
   pasaban por fuera del segmento. Se alargó hasta el borde de la imagen.

Además, un track que aparece dentro de la banda muerta de ±4 px toma como lado inicial
el de su primer punto. Antes, esos vehículos nunca contaban.

**Lección para marcar geometrías:**
- Las líneas tienen que quedar donde los vehículos ya están detectados de forma estable,
  lejos de los bordes del cuadro.
- Tienen que cubrir todo el ancho de la calzada.

## Pendiente

- Validar muestra_1 (noche) y muestra_3 (cola de autos) con el mismo método. Sus líneas
  del frente están en una posición parecida a la que falló en muestra_2.
- Mejorar la detección de motos: probar `yolov8s`, más resolución de inferencia o, mejor,
  video original sin la compresión de WhatsApp.
