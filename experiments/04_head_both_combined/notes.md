# Experimento 04 — Combinación RNN + CNN (promedio de logits)

## Hipótesis
Si los dos heads tienen sesgos distintos en cómo confunden caracteres
(ej. RNN se apoya en contexto inglés, CNN en forma visual local),
combinarlos puede compensar errores y dar predicciones más robustas.
Es ensembling clásico aplicado a los dos heads del mismo modelo.

## Configuración
Idéntica al baseline + flag `--head both`. La combinación se hace
promediando logits: `logits = (rnn_out + cnn_out) / 2`, después argmax
y decode CTC sobre la secuencia combinada.

## Resultados
Ver `summary.json`.

## Lectura
[para completar juntas con los números a la vista]
