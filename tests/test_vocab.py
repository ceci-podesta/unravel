"""Tests para vocab.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unravel.vocab import SPANISH_EXTRA_CHARS, build_unified_vocab

CLASSES_PATH = Path.home() / "projects/HTR-best-practices/saved_models/classes.npy"
SKIP_REASON = "classes.npy de IAM no disponible"


@pytest.mark.skipif(not CLASSES_PATH.is_file(), reason=SKIP_REASON)
def test_vocab_size_correcto() -> None:
    vocab = build_unified_vocab(CLASSES_PATH)
    # IAM = 79, extras = 6 → total 85
    assert vocab["n_classes"] == 79 + 6


@pytest.mark.skipif(not CLASSES_PATH.is_file(), reason=SKIP_REASON)
def test_blank_id_es_cero() -> None:
    vocab = build_unified_vocab(CLASSES_PATH)
    assert vocab["blank_id"] == 0


@pytest.mark.skipif(not CLASSES_PATH.is_file(), reason=SKIP_REASON)
def test_iam_chars_ocupan_indices_iniciales() -> None:
    """IAM en 1..79, extras en 80..85 — el orden DEBE matchear con el
    extend_model_vocabulary para que los nuevos outputs del modelo
    correspondan al mismo char.
    """
    vocab = build_unified_vocab(CLASSES_PATH)
    iam_classes = np.load(CLASSES_PATH).tolist()
    for i, c in enumerate(iam_classes):
        assert vocab["c2i"][c] == i + 1
    for i, c in enumerate(SPANISH_EXTRA_CHARS):
        assert vocab["c2i"][c] == len(iam_classes) + i + 1


@pytest.mark.skipif(not CLASSES_PATH.is_file(), reason=SKIP_REASON)
def test_c2i_y_i2c_son_inversas() -> None:
    vocab = build_unified_vocab(CLASSES_PATH)
    for c, i in vocab["c2i"].items():
        assert vocab["i2c"][i] == c


@pytest.mark.skipif(not CLASSES_PATH.is_file(), reason=SKIP_REASON)
def test_caracteres_españoles_presentes() -> None:
    vocab = build_unified_vocab(CLASSES_PATH)
    for c in "ñáéíóú":
        assert c in vocab["c2i"], f"{c!r} debería estar en el vocab unificado"


@pytest.mark.skipif(not CLASSES_PATH.is_file(), reason=SKIP_REASON)
def test_levanta_si_extra_solapa_con_iam() -> None:
    """Pedir un extra que ya está en IAM debe fallar."""
    with pytest.raises(ValueError):
        build_unified_vocab(CLASSES_PATH, extra_chars=["a", "b"])
