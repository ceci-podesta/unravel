"""Extender el vocabulario del modelo HTR.

Toma un HTRNet ya construido (típicamente con los pesos IAM cargados) y
agrega `n_extra` salidas adicionales en las capas finales, preservando
los pesos preentrenados de los índices originales en los primeros
slots.

Por default, los nuevos slots quedan con la inicialización default de
PyTorch (Kaiming uniform). Opcionalmente, `init_from` permite copiar
pesos de slots existentes — útil para inicializar chars nuevos a partir
de sus equivalentes ASCII (ej: ñ ← n, á ← a) y evitar shock random
durante full fine-tuning.

Soporta los dos tipos de head que tenemos implementados:
- CTCtopR (head='rnn'):  extiende `fnl[-1]` (Linear)
- CTCtopB (head='both'): extiende `fnl[-1]` (Linear) y `cnn[-1]` (Conv2d)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from unravel.htr_model import CTCtopB, CTCtopR, HTRNet


def _extend_linear(old: nn.Linear, n_extra: int,
                   init_from: dict[int, int] | None = None) -> nn.Linear:
    """Linear con `n_extra` outputs adicionales; los primeros outputs
    mantienen los pesos y bias originales.

    Si `init_from` se pasa, debe ser un dict {new_idx: source_idx} donde
    new_idx ∈ [old.out_features, old.out_features + n_extra) y source_idx
    es un índice válido del Linear original. Para cada par, se copian los
    pesos y bias de la fila source_idx a la fila new_idx (sobrescribiendo
    la inicialización default).
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

        if init_from:
            for new_idx, source_idx in init_from.items():
                if not (old.out_features <= new_idx < old.out_features + n_extra):
                    raise ValueError(
                        f"new_idx={new_idx} fuera del rango "
                        f"[{old.out_features}, {old.out_features + n_extra})"
                    )
                if not (0 <= source_idx < old.out_features):
                    raise ValueError(
                        f"source_idx={source_idx} fuera del rango "
                        f"[0, {old.out_features})"
                    )
                new.weight[new_idx].copy_(old.weight[source_idx])
                if old.bias is not None:
                    new.bias[new_idx].copy_(old.bias[source_idx])
    return new


def _extend_conv2d(old: nn.Conv2d, n_extra: int,
                   init_from: dict[int, int] | None = None) -> nn.Conv2d:
    """Conv2d con `n_extra` out_channels adicionales; los primeros canales
    mantienen los pesos y bias originales.

    Si `init_from` se pasa, debe ser un dict {new_idx: source_idx} donde
    new_idx ∈ [old.out_channels, old.out_channels + n_extra). Mismas
    semánticas que `_extend_linear`.
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

        if init_from:
            for new_idx, source_idx in init_from.items():
                if not (old.out_channels <= new_idx < old.out_channels + n_extra):
                    raise ValueError(
                        f"new_idx={new_idx} fuera del rango "
                        f"[{old.out_channels}, {old.out_channels + n_extra})"
                    )
                if not (0 <= source_idx < old.out_channels):
                    raise ValueError(
                        f"source_idx={source_idx} fuera del rango "
                        f"[0, {old.out_channels})"
                    )
                new.weight[new_idx].copy_(old.weight[source_idx])
                if old.bias is not None:
                    new.bias[new_idx].copy_(old.bias[source_idx])
    return new


def extend_model_vocabulary(net: HTRNet, n_extra: int,
                            init_from: dict[int, int] | None = None) -> HTRNet:
    """Modifica `net` in-place para que sus capas de salida tengan
    `n_extra` clases adicionales. Devuelve el mismo `net` por
    conveniencia (encadenable).

    Args:
        net: HTRNet con los pesos preentrenados ya cargados.
        n_extra: número de clases nuevas a agregar.
        init_from: opcional, dict {new_idx: source_idx} para inicializar
            los nuevos slots copiando pesos de slots existentes.
            Ej: para vocab unificado con ñ en posición 80, n en posición 14,
            init_from={80: 14} hace que el output del modelo para ñ arranque
            similar al de n. Si es None (default), los nuevos slots quedan
            con Kaiming uniform random — comportamiento original.
    """
    if n_extra <= 0:
        raise ValueError(f"n_extra debe ser > 0, no {n_extra}")
    top = net.top
    if isinstance(top, CTCtopB):
        top.fnl[-1] = _extend_linear(top.fnl[-1], n_extra, init_from)
        top.cnn[-1] = _extend_conv2d(top.cnn[-1], n_extra, init_from)
    elif isinstance(top, CTCtopR):
        top.fnl[-1] = _extend_linear(top.fnl[-1], n_extra, init_from)
    else:
        raise ValueError(f"head no soportado: {type(top).__name__}")
    return net
