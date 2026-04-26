"""Tests para SpanishHTRTrainDataset."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unravel.train_dataset import SpanishHTRTrainDataset

DATASET_ROOT = Path.home() / "datasets/spanish-htr/datos_entrenamiento/PERFECT_CUT_a_z_1_9"
SKIP_REASON = "dataset español de training no disponible"


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason=SKIP_REASON)
def test_train_y_val_no_se_solapan() -> None:
    train = SpanishHTRTrainDataset(DATASET_ROOT, subset="train")
    val = SpanishHTRTrainDataset(DATASET_ROOT, subset="val")
    train_paths = {s[0] for s in train.samples}
    val_paths = {s[0] for s in val.samples}
    assert train_paths.isdisjoint(val_paths), "train y val NO deben compartir muestras"


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason=SKIP_REASON)
def test_seed_fija_es_reproducible() -> None:
    a = SpanishHTRTrainDataset(DATASET_ROOT, subset="train", seed=42)
    b = SpanishHTRTrainDataset(DATASET_ROOT, subset="train", seed=42)
    assert [s[0] for s in a.samples] == [s[0] for s in b.samples]


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason=SKIP_REASON)
def test_seed_distinta_genera_orden_distinto() -> None:
    a = SpanishHTRTrainDataset(DATASET_ROOT, subset="train", seed=42)
    b = SpanishHTRTrainDataset(DATASET_ROOT, subset="train", seed=99)
    assert [s[0] for s in a.samples] != [s[0] for s in b.samples]


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason=SKIP_REASON)
def test_proporcion_aproximada_90_10() -> None:
    train = SpanishHTRTrainDataset(DATASET_ROOT, subset="train")
    val = SpanishHTRTrainDataset(DATASET_ROOT, subset="val")
    total = len(train) + len(val)
    assert 0.85 < len(train) / total < 0.95
    assert 0.05 < len(val) / total < 0.15


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason=SKIP_REASON)
def test_devuelve_tensor_correcto() -> None:
    train = SpanishHTRTrainDataset(DATASET_ROOT, subset="train")
    tensor, palabra, longitud = train[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 128, 1024)
    assert isinstance(palabra, str)
    assert longitud == len(palabra)


def test_subset_invalido_levanta() -> None:
    # No requiere dataset — testea validación de input
    with pytest.raises(ValueError):
        SpanishHTRTrainDataset("/tmp/cualquier_path_que_no_existe", subset="banana")
