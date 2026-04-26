"""Tests para lora_setup.py."""
from __future__ import annotations

import torch

from unravel.extend_vocab import extend_model_vocabulary
from unravel.htr_model import HTRNet, default_arch_cfg
from unravel.lora_setup import DEFAULT_TARGET_MODULES, apply_lora


def _build_extended_net() -> HTRNet:
    net = HTRNet(default_arch_cfg(), 80)
    extend_model_vocabulary(net, n_extra=6)
    return net


def test_apply_lora_reporta_pocos_params_entrenables() -> None:
    """LoRA con r=8 sobre 2 cabezas debería dejar ~1-2% trainable."""
    net = _build_extended_net()
    _, stats = apply_lora(net, r=8, alpha=16)
    assert stats["trainable_params"] > 0
    assert stats["total_params"] > stats["trainable_params"]
    assert stats["percent_trainable"] < 5.0, \
        f"Esperaba <5% trainable, obtuve {stats['percent_trainable']:.2f}%"


def test_apply_lora_forward_devuelve_shapes_correctos() -> None:
    """El modelo con LoRA debe seguir produciendo (T, B, 86) en cada head."""
    net = _build_extended_net()
    peft_model, _ = apply_lora(net, r=8, alpha=16)
    peft_model.eval()
    x = torch.zeros(1, 1, 128, 1024)
    with torch.no_grad():
        out = peft_model(x)
    assert isinstance(out, tuple), "head='both' debería devolver tupla"
    rnn_out, cnn_out = out
    assert rnn_out.shape[-1] == 86
    assert cnn_out.shape[-1] == 86


def test_apply_lora_solo_lora_params_son_trainable() -> None:
    """Solo los pesos LoRA deben tener requires_grad=True; el resto frozen."""
    net = _build_extended_net()
    peft_model, _ = apply_lora(net)
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            assert "lora" in name.lower(), \
                f"El parámetro {name!r} es trainable pero no es LoRA"


def test_apply_lora_default_target_modules() -> None:
    """Verifica que los target_modules por default sean los esperados."""
    assert DEFAULT_TARGET_MODULES == ["top.fnl.1", "top.cnn.1"]
