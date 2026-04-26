"""Tests para ctc_utils.py."""
from __future__ import annotations

import pytest
import torch

from unravel.ctc_utils import collate_for_ctc, compute_ctc_loss


def make_batch(palabras: list[str], h: int = 16, w: int = 64) -> list[tuple[torch.Tensor, str, int]]:
    """Construye un batch sintético chico (no usa el dataset real)."""
    return [(torch.zeros(1, h, w), p, len(p)) for p in palabras]


# === collate ===

def test_collate_shapes_basicos() -> None:
    c2i = {"a": 1, "b": 2, "c": 3}
    batch = make_batch(["a", "ab", "abc"])
    out = collate_for_ctc(batch, c2i)
    assert out["images"].shape == (3, 1, 16, 64)
    # targets concatenados: 1 + 2 + 3 = 6 elementos
    assert out["targets"].shape == (6,)
    assert out["target_lengths"].tolist() == [1, 2, 3]


def test_collate_indices_correctos() -> None:
    c2i = {"a": 1, "b": 2, "c": 3}
    batch = make_batch(["abc"])
    out = collate_for_ctc(batch, c2i)
    assert out["targets"].tolist() == [1, 2, 3]


def test_collate_levanta_si_char_no_en_vocab() -> None:
    c2i = {"a": 1, "b": 2}
    batch = make_batch(["abz"])
    with pytest.raises(KeyError):
        collate_for_ctc(batch, c2i)


def test_collate_preserva_metadatos() -> None:
    c2i = {"a": 1, "b": 2, "c": 3}
    batch = make_batch(["a", "ab", "abc"])
    out = collate_for_ctc(batch, c2i)
    assert out["palabras"] == ["a", "ab", "abc"]
    assert out["longitudes"] == [1, 2, 3]


# === compute_ctc_loss ===

def test_loss_devuelve_escalar_con_head_both() -> None:
    """Output simulado: T=20, B=3, C=4. Targets coherentes."""
    torch.manual_seed(0)
    c2i = {"a": 1, "b": 2, "c": 3}
    batch_data = make_batch(["a", "ab", "abc"])
    batch = collate_for_ctc(batch_data, c2i)
    rnn_out = torch.randn(20, 3, 4)
    cnn_out = torch.randn(20, 3, 4)
    loss = compute_ctc_loss((rnn_out, cnn_out), batch, blank_id=0, head="both")
    assert loss.dim() == 0  # escalar
    assert torch.isfinite(loss)


def test_loss_head_rnn_y_cnn_dan_distinto() -> None:
    torch.manual_seed(0)
    c2i = {"a": 1, "b": 2, "c": 3}
    batch_data = make_batch(["abc", "ab"])
    batch = collate_for_ctc(batch_data, c2i)
    rnn_out = torch.randn(20, 2, 4)
    cnn_out = torch.randn(20, 2, 4) * 2  # distribución distinta a propósito
    loss_rnn = compute_ctc_loss((rnn_out, cnn_out), batch, head="rnn")
    loss_cnn = compute_ctc_loss((rnn_out, cnn_out), batch, head="cnn")
    assert not torch.allclose(loss_rnn, loss_cnn)


def test_loss_head_invalido_levanta() -> None:
    c2i = {"a": 1}
    batch = collate_for_ctc(make_batch(["a"]), c2i)
    rnn_out = torch.randn(20, 1, 2)
    cnn_out = torch.randn(20, 1, 2)
    with pytest.raises(ValueError):
        compute_ctc_loss((rnn_out, cnn_out), batch, head="banana")


def test_loss_acepta_output_no_tupla() -> None:
    """Si output es un solo tensor (head='rnn' single), también funciona."""
    c2i = {"a": 1, "b": 2}
    batch = collate_for_ctc(make_batch(["ab"]), c2i)
    out = torch.randn(20, 1, 3)
    loss = compute_ctc_loss(out, batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
