"""Vocabulario unificado: IAM (79 chars) + caracteres específicos del español.

Convención del vocabulario:
- Índice 0: reservado para el blank de CTC (no es un carácter).
- Índices 1..79: caracteres del IAM (en el orden de classes.npy).
- Índices 80..85: caracteres específicos del español (ñ, á, é, í, ó, ú)
  en orden estable.

Esta convención es importante porque debe matchear con el orden en que
se hace `extend_model_vocabulary(net, n_extra=6)`: los outputs nuevos
del modelo (índices 80..85) corresponden exactamente a los 6 chars del
español, en el mismo orden que `SPANISH_EXTRA_CHARS`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Caracteres del español que aparecen en datos_entrenamiento real pero NO
# en el vocabulario IAM. Decisión de diseño: orden estable y explícito.
SPANISH_EXTRA_CHARS: list[str] = ["ñ", "á", "é", "í", "ó", "ú"]


def build_unified_vocab(
    iam_classes_path: Path | str,
    extra_chars: list[str] | None = None,
) -> dict:
    """Construye el vocabulario unificado IAM + extras.

    Args:
        iam_classes_path: ruta a `classes.npy` del repo HTR-best-practices.
        extra_chars: caracteres a agregar al final. Default
            `SPANISH_EXTRA_CHARS`. Cada char no debe ya estar en IAM.

    Returns:
        dict con:
        - 'classes': np.ndarray de los chars del vocabulario unificado
          (sin blank).
        - 'c2i': dict char -> idx (1..N, idx 0 es blank).
        - 'i2c': dict idx -> char (1..N).
        - 'n_classes': N (cantidad de chars).
        - 'blank_id': 0.
    """
    iam_classes = np.load(iam_classes_path)
    iam_chars: list[str] = list(iam_classes.tolist())

    if extra_chars is None:
        extra_chars = SPANISH_EXTRA_CHARS

    # Validar que los extras no se solapen con IAM
    iam_set = set(iam_chars)
    duplicados = [c for c in extra_chars if c in iam_set]
    if duplicados:
        raise ValueError(
            f"Los caracteres {duplicados} ya están en el vocabulario IAM. "
            "No se pueden agregar como extras."
        )

    unified_chars = iam_chars + list(extra_chars)
    c2i = {c: i + 1 for i, c in enumerate(unified_chars)}
    i2c = {i + 1: c for i, c in enumerate(unified_chars)}

    return {
        "classes": np.array(unified_chars),
        "c2i": c2i,
        "i2c": i2c,
        "n_classes": len(unified_chars),
        "blank_id": 0,
    }
