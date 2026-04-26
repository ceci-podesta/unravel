"""Preprocessing de imágenes para HTR.

Replica el preprocessing del repo HTR-best-practices: grayscale +
polaridad invertida + resize preservando aspect ratio + padding mediano.

Fix sobre el repo original: el `load_image` de ellos asume que la imagen
final está en uint8 [0, 255], pero `rgb2gray` ya devuelve float [0, 1].
Acá normalizamos a [0, 1] explícitamente según el tipo, después invertimos.

Extensión: `preprocess` acepta un flag `center` (default False) para elegir
si la imagen escalada se ubica en la esquina superior izquierda (default,
matchea el repo) o centrada en el canvas (experimento 02).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import skimage.color as img_color
import skimage.io as img_io
from skimage.transform import resize


def load_image(image_path: Path | str) -> np.ndarray:
    """Carga imagen, convierte a grayscale, normaliza con polaridad invertida."""
    image = img_io.imread(str(image_path))
    if image.ndim == 3:
        image = img_color.rgb2gray(image)        # → [0, 1]
    else:
        image = image.astype(np.float32) / 255.  # → [0, 1]
    return 1.0 - image


def preprocess(
    img: np.ndarray,
    input_size: tuple[int, int],
    border_size: int = 8,
    center: bool = False,
) -> np.ndarray:
    """Resize preservando aspect ratio + padding mediano hasta `input_size`.

    Si `center=False` (default), padea a la esquina superior izquierda con
    border 8 px (replica exacta del repo de referencia).
    Si `center=True`, padea simétricamente para que la imagen escalada
    quede centrada en el canvas (experimento 02).
    """
    h_target, w_target = input_size
    n_height = min(h_target - 2 * border_size, img.shape[0])
    scale = n_height / img.shape[0]
    n_width = min(w_target - 2 * border_size, int(scale * img.shape[1]))
    img = resize(image=img, output_shape=(n_height, n_width)).astype(np.float32)

    if center:
        pad_top = (h_target - n_height) // 2
        pad_bottom = h_target - n_height - pad_top
        pad_left = (w_target - n_width) // 2
        pad_right = w_target - n_width - pad_left
    else:
        pad_top = border_size
        pad_bottom = h_target - n_height - border_size
        pad_left = border_size
        pad_right = w_target - n_width - border_size

    return np.pad(
        img,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="median",
    )
