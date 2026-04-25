"""Métricas de evaluación HTR: CER y WER.

Implementación con normalización NFC unicode y strip de whitespace exterior.
NO se aplica lowercase ni se eliminan tildes — esos son errores reales que
queremos medir.

Ver `notas-para-informe.md` sección 3.6.1 para el fundamento de diseño.
"""
from __future__ import annotations

import unicodedata


def _normalize(s: str) -> str:
    """Aplica NFC + strip. Nada más."""
    return unicodedata.normalize("NFC", s).strip()


def edit_distance(a: str, b: str) -> int:
    """Distancia de Levenshtein entre dos strings (después de normalizar).

    Cuenta el mínimo número de inserciones, borrados y sustituciones para
    convertir `a` en `b`. Implementado con programación dinámica O(n*m).
    """
    a = _normalize(a)
    b = _normalize(b)
    n, m = len(a), len(b)

    if n == 0:
        return m
    if m == 0:
        return n

    # dp[i][j] = edit distance entre a[:i] y b[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # borrar de a
                    dp[i][j - 1],      # insertar en a
                    dp[i - 1][j - 1],  # sustituir
                )
    return dp[n][m]


def cer(prediction: str, reference: str) -> float:
    """Character Error Rate de una predicción vs su referencia.

    CER = edit_distance(pred, ref) / len(ref).

    Raises:
        ValueError: si la referencia normalizada es vacía.
    """
    pred_norm = _normalize(prediction)
    ref_norm = _normalize(reference)
    if len(ref_norm) == 0:
        raise ValueError("La referencia no puede ser vacía para calcular CER")
    return edit_distance(pred_norm, ref_norm) / len(ref_norm)


def cer_micro(predictions: list[str], references: list[str]) -> float:
    """CER agregado micro-average sobre un set de predicciones.

    Suma todas las edit distances y divide por la suma de longitudes de
    las referencias. Las palabras largas pesan proporcionalmente más.
    Es la métrica usada por convención en literatura HTR.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions y references tienen distinta cantidad: "
            f"{len(predictions)} vs {len(references)}"
        )
    total_dist = sum(
        edit_distance(p, r) for p, r in zip(predictions, references, strict=True)
    )
    total_len = sum(len(_normalize(r)) for r in references)
    if total_len == 0:
        raise ValueError("Todas las referencias normalizadas tienen longitud 0")
    return total_dist / total_len


def cer_macro(predictions: list[str], references: list[str]) -> float:
    """CER agregado macro-average sobre un set de predicciones.

    Promedia CER por palabra. Cada palabra cuenta igual, independiente
    de su longitud. Métrica complementaria a micro.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions y references tienen distinta cantidad: "
            f"{len(predictions)} vs {len(references)}"
        )
    if len(predictions) == 0:
        raise ValueError("No se puede calcular macro CER sobre lista vacía")
    return sum(
        cer(p, r) for p, r in zip(predictions, references, strict=True)
    ) / len(predictions)


def wer(predictions: list[str], references: list[str]) -> float:
    """Word Error Rate sobre un set de predicciones word-level.

    Para cada par (pred, ref), pred cuenta como error si y solo si
    pred normalizado != ref normalizado. WER = errores / total.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions y references tienen distinta cantidad: "
            f"{len(predictions)} vs {len(references)}"
        )
    if len(predictions) == 0:
        raise ValueError("No se puede calcular WER sobre lista vacía")
    errores = sum(
        1
        for p, r in zip(predictions, references, strict=True)
        if _normalize(p) != _normalize(r)
    )
    return errores / len(predictions)
