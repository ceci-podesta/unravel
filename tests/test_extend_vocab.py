"""Tests para extend_vocab.py."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from unravel.extend_vocab import _extend_conv2d, _extend_linear, extend_model_vocabulary
from unravel.htr_model import HTRNet, default_arch_cfg


# === unit tests de las funciones helper ===

def test_extend_linear_preserva_pesos_originales() -> None:
    old = nn.Linear(64, 10)
    old_weight = old.weight.detach().clone()
    old_bias = old.bias.detach().clone()
    new = _extend_linear(old, n_extra=3)
    assert new.in_features == 64
    assert new.out_features == 13
    assert torch.allclose(new.weight[:10], old_weight)
    assert torch.allclose(new.bias[:10], old_bias)


def test_extend_linear_los_nuevos_no_son_cero() -> None:
    """Los outputs nuevos deben tener init random (no quedar en 0)."""
    old = nn.Linear(64, 10)
    new = _extend_linear(old, n_extra=3)
    assert not torch.all(new.weight[10:] == 0), "los nuevos pesos NO deberían ser todos cero"


def test_extend_conv2d_preserva_pesos_originales() -> None:
    old = nn.Conv2d(32, 10, kernel_size=(1, 3), padding=(0, 1))
    old_weight = old.weight.detach().clone()
    old_bias = old.bias.detach().clone()
    new = _extend_conv2d(old, n_extra=3)
    assert new.in_channels == 32
    assert new.out_channels == 13
    assert torch.allclose(new.weight[:10], old_weight)
    assert torch.allclose(new.bias[:10], old_bias)


# === tests de la función pública ===

def test_extend_model_vocabulary_aumenta_outputs_correctamente() -> None:
    net = HTRNet(default_arch_cfg(), 80)
    extend_model_vocabulary(net, n_extra=6)
    # head='both' tiene fnl (Linear) y cnn (Conv2d) — ambos deben pasar a 86
    assert net.top.fnl[-1].out_features == 86
    assert net.top.cnn[-1].out_channels == 86


def test_extend_model_vocabulary_predice_idéntico_para_indices_originales() -> None:
    """Test crítico: los logits de los índices viejos deben ser exactamente
    los mismos antes y después de extender. Si esto rompe, el modelo
    olvidó lo que ya sabía.
    """
    torch.manual_seed(0)
    n_old = 80
    n_extra = 6
    net = HTRNet(default_arch_cfg(), n_old)
    net.eval()
    x = torch.randn(1, 1, 128, 1024)
    with torch.no_grad():
        out_old = net(x)
    extend_model_vocabulary(net, n_extra=n_extra)
    net.eval()
    with torch.no_grad():
        out_new = net(x)
    # head='both' devuelve tupla (rnn_out, cnn_out)
    assert isinstance(out_old, tuple) and isinstance(out_new, tuple)
    old_rnn, old_cnn = out_old
    new_rnn, new_cnn = out_new
    # Los primeros n_old índices del último eje deben coincidir
    assert torch.allclose(new_rnn[..., :n_old], old_rnn, atol=1e-6), \
        "RNN: los logits de los índices viejos cambiaron"
    assert torch.allclose(new_cnn[..., :n_old], old_cnn, atol=1e-6), \
        "CNN: los logits de los índices viejos cambiaron"
    # Y la dimensión total debe haber crecido a n_old + n_extra
    assert new_rnn.shape[-1] == n_old + n_extra
    assert new_cnn.shape[-1] == n_old + n_extra


def test_extend_model_vocabulary_levanta_si_n_extra_invalido() -> None:
    net = HTRNet(default_arch_cfg(), 80)
    with pytest.raises(ValueError):
        extend_model_vocabulary(net, n_extra=0)
    with pytest.raises(ValueError):
        extend_model_vocabulary(net, n_extra=-3)
