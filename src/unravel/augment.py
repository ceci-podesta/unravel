"""Augmentations on-the-fly para HTR sobre imágenes manuscritas reales.

Aplica con probabilidad p_apply al menos una transformación. Si se aplica,
se eligen 1 o 2 transformaciones random entre las 4 disponibles:

  - Rotación ±5°
  - Shift pequeño (±5px x, ±3px y)
  - Dilatación leve (trazo más grueso)
  - Erosión leve (trazo más fino)

Las transformaciones se aplican sobre la imagen original (después de
load_image, antes de preprocess), usando el RNG de torch — con
torch.manual_seed(seed) y worker_init_fn las augs son reproducibles.

Transformaciones descartadas (ver notas_informe_lora_manual.md):
  - Brillo / contraste: no aportan sobre escaneos limpios.
  - Ruido gaussiano: no representa variabilidad real para HTR.
  - Rotaciones grandes / reflejos: confunden caracteres (b/d, p/q, 6/9).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import shift as nd_shift
from skimage.morphology import dilation, erosion, footprint_rectangle
_FOOTPRINT_2 = footprint_rectangle((2, 2))
from skimage.transform import rotate as sk_rotate


def _aug_rotate(img: np.ndarray) -> np.ndarray:
    """Rotación uniforme entre -5° y +5°."""
    angle = (torch.rand(1).item() * 2 - 1) * 5.0
    out = sk_rotate(img, angle=angle, mode="constant", cval=0.0, preserve_range=True)
    return out.astype(np.float32)


def _aug_shift(img: np.ndarray) -> np.ndarray:
    """Shift entre ±5px en x y ±3px en y."""
    dx = (torch.rand(1).item() * 2 - 1) * 5.0
    dy = (torch.rand(1).item() * 2 - 1) * 3.0
    out = nd_shift(img, (dy, dx), cval=0.0, order=1)
    return out.astype(np.float32)


def _aug_dilate(img: np.ndarray) -> np.ndarray:
    """Dilata el texto (trazo más grueso). La imagen está invertida en polaridad
    (texto = valores claros), entonces dilation directa engrosa el texto."""
    return dilation(img, _FOOTPRINT_2).astype(np.float32)


def _aug_erode(img: np.ndarray) -> np.ndarray:
    """Erosiona el texto (trazo más fino)."""
    return erosion(img, _FOOTPRINT_2).astype(np.float32)


_AUG_FNS = [_aug_rotate, _aug_shift, _aug_dilate, _aug_erode]


def random_augment_image(img: np.ndarray, prob_apply: float = 0.7) -> np.ndarray:
    """Aplica con probabilidad prob_apply una composición de 1 o 2 augs random.

    Args:
        img: imagen post `load_image`, polaridad invertida, valores en [0, 1].
        prob_apply: probabilidad de aplicar al menos una aug.

    Returns:
        np.ndarray transformado, o img sin tocar si no se aplica aug.
    """
    if torch.rand(1).item() >= prob_apply:
        return img.astype(np.float32)
    n_augs = 1 if torch.rand(1).item() < 0.5 else 2
    perm = torch.randperm(len(_AUG_FNS))[:n_augs].tolist()
    out = img
    for idx in perm:
        out = _AUG_FNS[idx](out)
    return out.astype(np.float32)
