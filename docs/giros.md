# Matriz de giros

Para cada vehículo, el sistema registra por dónde entra al cruce y por dónde sale
(`src/atlas_traffic/turning.py`). Con eso arma la matriz origen → destino: cuántos siguen
derecho y cuántos doblan.

En Córdoba y Rodríguez Peña hay cuatro movimientos posibles: Córdoba es mano única,
alejándose de la cámara, y Rodríguez Peña va de derecha a izquierda.

| Desde \ Hacia | Córdoba (fondo) | Rodríguez Peña (izquierda) |
|---|---|---|
| **Córdoba** (frente) | sigue derecho | dobla a la izquierda |
| **Rodríguez Peña** (derecha) | dobla a la derecha | sigue derecho |

## Cómo se arma

1. **Origen y destino:** el origen es la primera línea que el vehículo cruza entrando, y
   el destino la última que cruza saliendo.
2. **Recorridos a medias:** muchos vehículos se ven solo entrando o solo saliendo. El
   clip empieza o termina con ellos dentro del cruce, o algo los tapa en el medio. Hay
   dos formas de completarlos:
   - **Origen inferido:** si un vehículo aparece ya adentro, a menos de 60 px pasada una
     entrada y alejándose de ella (por ejemplo, la cola de Córdoba al arrancar el clip),
     entró por ahí.
   - **Recorrido reconstruido:** un vehículo que entró y no se lo vio salir se empareja
     con uno que salió y no se lo vio entrar. El segundo tiene que aparecer a menos de
     3 s y 100 px de donde se perdió el primero, hacia donde iba. Buscar solo entre esos
     "huérfanos" deja pocos candidatos, a diferencia de una unión general de tracks.
3. **Reparto:** el porcentaje de cada giro sale de los recorridos completos (vistos,
   inferidos o reconstruidos), con intervalo de confianza de Wilson al 95 %. El volumen
   estimado aplica ese porcentaje a todos los que entraron por ese origen.
4. **Casos descartados:** si un vehículo entra y sale por la misma línea, es una caja que
   tiembla sobre ella y se marca `dudoso`.

Salidas: `outputs/runs/<clip>/giros.parquet`, una fila por vehículo con su estado, y
`outputs/giros_todos.parquet`. El dashboard lo muestra en la pestaña Panorama.

## Resultado con los tres clips de muestra

| Movimiento | Recorridos completos | % del origen (IC 95 %) |
|---|---|---|
| Córdoba → sigue por Córdoba | 72 | 99 % (93–100 %) |
| Córdoba → dobla a Rodríguez Peña | 1 | 1 % (0–7 %) |

Se ve el recorrido completo de 73 de 108 vehículos que entraron.

**Verificación a ojo:** se dibujaron las trayectorias de todos los recorridos inferidos y
reconstruidos sobre la imagen:
- **Córdoba → Rodríguez Peña:** es el único giro, y es un auto rojo de muestra_3. Sale
  de la cola de Córdoba, dobla y pasa detrás del árbol entre los 11 y 13 s. Se reconstruyó
  a partir de dos pedazos (tracks 6 → 185).
- **Errores corregidos:** en una primera versión, con un margen de 120 px, a dos
  vehículos que aparecieron en medio de Córdoba se les asignó origen "Rodríguez Peña".
  Con 60 px y la condición de alejarse de la entrada, quedaron sin origen.

## Limitaciones

- **Rodríguez Peña casi no se ve en estos clips.** La entrada por la derecha queda en el
  borde de la imagen y la salida por la izquierda, detrás de un árbol. Ningún vehículo de
  Rodríguez Peña tuvo un recorrido completo, así que esos dos movimientos no se pudieron
  medir. Por ejemplo, el colectivo que sale por la izquierda en muestra_3 ya estaba en
  el cruce cuando empezó el video.
- **Muestra chica:** 1 giro en 73 recorridos da un intervalo de 0–7 %. Hacen falta videos
  más largos para un porcentaje útil.
- **Mejor encuadre:** que se vea Rodríguez Peña completo, de borde a borde, con la
  cámara fija.
