"""Extender el vocabulario del modelo HTR.

Toma un HTRNet ya construido (típicamente con los pesos IAM cargados) y
agrega `n_extra` salidas adicionales en las capas finales, preservando
los pesos preentrenados de los índices originales en los primeros
slots. Los nuevos slots quedan con la inicialización default de PyTorch
(Kaiming uniform), lista para que LoRA aprenda a "encenderlos" durante
el fine-tuning.

Soporta los dos tipos de head que tenemos implementados:
- CTCtopR (head='rnn'):  extiende `fnl[-1]` (Linear)
- CTCtopB (head='both'): extiende `fnl[-1]` (Linear) y `cnn[-1]` (Conv2d)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from unravel.htr_model import CTCtopB, CTCtopR, HTRNet


def _extend_linear(old: nn.Linear, n_extra: int) -> nn.Linear:
    """Linear con `n_extra` outputs adicionales; los primeros outputs
    mantienen los pesos y bias originales.
    """
    new = nn.Linear(
        in_features=old.in_features,
        out_features=old.out_features + n_extra,
        bias=(old.bias is not None),
    )
    with torch.no_grad():
        new.weight[: old.out_features].copy_(old.weight)
        if old.bias is not None:
            new.bias[: old.out_features].copy_(old.bias)
    return new


def _extend_conv2d(old: nn.Conv2d, n_extra: int) -> nn.Conv2d:
    """Conv2d con `n_extra` out_channels adicionales; los primeros canales
    mantienen los pesos y bias originales.
    """
    new = nn.Conv2d(
        in_channels=old.in_channels,
        out_channels=old.out_channels + n_extra,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    with torch.no_grad():
        new.weight[: old.out_channels].copy_(old.weight)
        if old.bias is not None:
            new.bias[: old.out_channels].copy_(old.bias)
    return new


def extend_model_vocabulary(net: HTRNet, n_extra: int) -> HTRNet:
    """Modifica `net` in-place para que sus capas de salida tengan
    `n_extra` clases adicionales. Devuelve el mismo `net` por
    conveniencia (encadenable).

    Args:
        net: HTRNet con los pesos preentrenados ya cargados.
        n_extra: número de clases nuevas a agregar.
    """
    if n_extra <= 0:
        raise ValueError(f"n_extra debe ser > 0, no {n_extra}")
    top = net.top
    if isinstance(top, CTCtopB):
        # head='both': dos salidas (Linear de RNN + Conv2d de CNN)
        top.fnl[-1] = _extend_linear(top.fnl[-1], n_extra)
        top.cnn[-1] = _extend_conv2d(top.cnn[-1], n_extra)
    elif isinstance(top, CTCtopR):
        top.fnl[-1] = _extend_linear(top.fnl[-1], n_extra)
    else:
        raise ValueError(f"head no soportado: {type(top).__name__}")
    return net
