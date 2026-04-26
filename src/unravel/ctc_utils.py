"""Utilidades para training con CTC loss.

- `collate_for_ctc`: convierte un batch de (image, palabra, longitud) en
  los tensores que necesita la CTC loss (imágenes apiladas + targets
  concatenados + target_lengths).
- `compute_ctc_loss`: helper que toma el output del modelo y un batch
  del collate, y devuelve la loss escalar. Soporta los 3 modos de head
  (rnn, cnn, both).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def collate_for_ctc(
    batch: list[tuple[torch.Tensor, str, int]],
    c2i: dict[str, int],
) -> dict:
    """Collate function para batches del Spanish HTR dataset.

    Args:
        batch: lista de (image_tensor, palabra, longitud) — formato del
            __getitem__ de SpanishHTRTrainDataset.
        c2i: dict char -> idx (idx 0 reservado para blank de CTC).

    Returns:
        dict con:
        - 'images': tensor (B, 1, H, W).
        - 'targets': tensor 1D (sum_target_lengths,) con todos los
          índices concatenados.
        - 'target_lengths': tensor (B,) con la longitud de cada target.
        - 'palabras': lista de strings originales (para logging).
        - 'longitudes': lista de int (longitud de cada palabra, para
          análisis por bucket).

    Raises:
        KeyError: si alguna palabra contiene chars fuera de c2i.
            Levantamos en lugar de skip silenciosamente — preferimos
            fallar ruidosamente que entrenar con datos incompletos.
    """
    images = torch.stack([b[0] for b in batch])
    palabras = [b[1] for b in batch]
    longitudes = [b[2] for b in batch]

    targets_list: list[int] = []
    target_lengths: list[int] = []
    for palabra in palabras:
        try:
            indices = [c2i[c] for c in palabra]
        except KeyError as e:
            raise KeyError(
                f"Carácter {e.args[0]!r} de palabra {palabra!r} no está en "
                "el vocabulario. El vocab debe incluir todos los chars del dataset."
            ) from e
        targets_list.extend(indices)
        target_lengths.append(len(indices))

    return {
        "images": images,
        "targets": torch.tensor(targets_list, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "palabras": palabras,
        "longitudes": longitudes,
    }


def compute_ctc_loss(
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    batch: dict,
    blank_id: int = 0,
    head: str = "both",
) -> torch.Tensor:
    """Calcula CTC loss desde el output del modelo y un batch del collate.

    Args:
        output: tensor (T, B, C) o tupla (rnn_out, cnn_out) si head='both'.
        batch: dict producido por `collate_for_ctc`.
        blank_id: índice del blank en el vocabulario (default 0).
        head: 'rnn' (usa solo output[0]), 'cnn' (solo output[1]),
            o 'both' (promedia las dos losses).

    Returns:
        Tensor escalar con la loss.
    """
    if isinstance(output, tuple):
        rnn_out, cnn_out = output
        if head == "rnn":
            outputs_to_use = [rnn_out]
        elif head == "cnn":
            outputs_to_use = [cnn_out]
        elif head == "both":
            outputs_to_use = [rnn_out, cnn_out]
        else:
            raise ValueError(f"head desconocido: {head!r}")
    else:
        outputs_to_use = [output]

    targets = batch["targets"]
    target_lengths = batch["target_lengths"]

    total_loss = torch.zeros((), device=targets.device, dtype=torch.float32)
    for out in outputs_to_use:
        T, B, _ = out.shape
        log_probs = F.log_softmax(out, dim=2)
        input_lengths = torch.full((B,), T, dtype=torch.long, device=out.device)
        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank_id,
            zero_infinity=True,
        )
        total_loss = total_loss + loss
    return total_loss / len(outputs_to_use)
