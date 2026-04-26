"""Wrapper de PEFT/LoRA para HTRNet.

Aplica LoRA a las cabezas finales del modelo (`top.fnl.1` Linear y
`top.cnn.1` Conv2d) usando la librería PEFT. Por default todo el resto
del modelo queda congelado (LSTM + backbone CNN no se actualizan).

Decisión documentada en notas-para-informe.md sección Etapa 4 →
hiperparámetros → target_modules ("Medio revisado").
"""
from __future__ import annotations

import torch.nn as nn
from peft import LoraConfig, get_peft_model

from unravel.htr_model import HTRNet


DEFAULT_TARGET_MODULES: list[str] = ["top.fnl.1", "top.cnn.1"]


def apply_lora(
    net: HTRNet,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_module_names: list[str] | None = None,
) -> tuple[nn.Module, dict]:
    """Aplica LoRA a las cabezas finales de HTRNet usando peft.

    Args:
        net: HTRNet con vocabulario ya extendido si corresponde.
        r: rango de las matrices low-rank.
        alpha: factor de escala (`α/r` multiplica la actualización).
        dropout: dropout dentro del módulo LoRA, regularización.
        target_module_names: nombres exactos de módulos a adaptar.
            Default ["top.fnl.1", "top.cnn.1"] — las dos cabezas finales.

    Returns:
        (peft_model, stats) donde stats es un dict con:
        - 'trainable_params': cantidad de parámetros entrenables (LoRA).
        - 'total_params': cantidad total del modelo.
        - 'percent_trainable': porcentaje (debería ser ~1-2%).
        - 'target_modules': la lista de target_modules usada.
    """
    if target_module_names is None:
        target_module_names = list(DEFAULT_TARGET_MODULES)

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_module_names,
        lora_dropout=dropout,
        bias="none",
    )
    peft_model = get_peft_model(net, config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    stats = {
        "trainable_params": trainable,
        "total_params": total,
        "percent_trainable": 100.0 * trainable / total,
        "target_modules": target_module_names,
    }
    return peft_model, stats
