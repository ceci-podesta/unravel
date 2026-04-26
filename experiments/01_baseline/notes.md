# Experimento 01 — Baseline zero-shot

## Hipótesis
El modelo IAM, sin fine-tuning ni adaptación de preprocessing, va a
tener WER alto sobre el dataset español pero **no 100%** — esperamos
algunos aciertos en palabras cortas con caracteres comunes (dígitos
sueltos, letras a-z minúsculas).

## Configuración
- Pesos: `htrnet.pt` del repo `HTR-best-practices`.
- Vocabulario: `classes.npy` del mismo repo (79 caracteres IAM, sin
  tildes ni ñ).
- Preprocessing: grayscale + polaridad invertida + resize preservando
  aspect ratio + padding mediano hasta (128, 1024). Imagen escalada
  ubicada en la esquina superior izquierda (border 8 px).
- Head usado en inferencia: `rnn` (output[0] del modelo `head_type='both'`).
- Decode: greedy CTC.

## Resultados (3284 imágenes, 12s en RTX 3060)
- WER global: 0.9982
- CER_micro global: 1.0664
- CER_macro global: 1.1789
- CER por bucket de longitud: ver `summary.json`

## Lectura
La hipótesis era WER alto pero no 100% — lo que vimos fue
prácticamente 100% (sólo 6 aciertos sobre 3.284). Inspección del
`predictions.csv` mostró que la predicción más frecuente es ` # `
(~36% de los inputs), variantes de `#` suman ~50%. El modelo "se
rinde" hacia un carácter que en IAM se usa como marker de zona ambigua.

Hipótesis para los siguientes experimentos: el sesgo a `#` se debe a
**mismatch de dominio severo** (palabras sueltas + mucho fondo padeado
≠ líneas IAM). Vamos a probar dos modificaciones de preprocessing
(centrar la palabra) y dos modificaciones de inferencia (heads) para
ver cuánto del gap cierra cada cosa.
