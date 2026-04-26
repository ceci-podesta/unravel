# Experimento 02 — Imagen centrada en el canvas

## Hipótesis
El modelo IAM se entrenó con líneas de texto que ocupan todo el ancho
(1024 px). En el baseline padeamos a la esquina superior izquierda,
dejando ~370 px de fondo padeado a la derecha. Si centramos la palabra
(con fondo simétrico a izquierda y derecha), capaz el modelo se confunde
menos al no ver una "línea" tan inusual.

## Configuración
Idéntica al baseline (head=rnn, mismos pesos, mismo dataset) + flag
`--center`. La función `preprocess` calcula padding simétrico horizontal
y vertical en lugar de pegar la imagen a la esquina sup-izq.

## Resultados
Ver `summary.json`.

## Lectura
[para completar juntas con los números a la vista]
