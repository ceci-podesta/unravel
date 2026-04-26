# Experimento 03 — Head CNN

## Hipótesis
El RNN modela dependencias secuenciales del idioma fuente (inglés).
Para palabras españolas con secuencias de caracteres inexistentes en
inglés, esa "memoria" del idioma podría perjudicar más que ayudar.
El CNN puro (sin contexto secuencial bidireccional) trabaja localmente
sobre features visuales — capaz se adapta mejor a un dominio distinto.

## Configuración
Idéntica al baseline (preprocessing en esquina sup-izq) + flag
`--head cnn`. En la combinación de heads del modelo `head_type='both'`,
usamos sólo el output del path CNN (output[1]) en lugar del RNN
(output[0]).

## Resultados
Ver `summary.json`.

## Lectura
[para completar juntas con los números a la vista]
