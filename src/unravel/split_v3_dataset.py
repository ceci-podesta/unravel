"""Dataset wrapper sobre SplitV2Dataset que aplica augmentations on-the-fly.

Si apply_aug=True (solo para subset='train'), aplica random_augment_image
entre load_image() y preprocess(). Si apply_aug=False, comportamiento
idéntico a SplitV2Dataset (val/test sin alteración).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from unravel.augment import random_augment_image
from unravel.preproc import load_image, preprocess
from unravel.split_dataset import SplitV2Dataset


class SplitV3Dataset(Dataset):
    """Wrapper del SplitV2Dataset con flag apply_aug.

    Args:
        split_path: path al JSON del split (split_v2.json).
        subset: 'train', 'val', o 'test'.
        fixed_size: tamaño objetivo del preprocessing (default 128x1024).
        apply_aug: si True, aplica augs on-the-fly. Solo recomendado para
            subset='train'.
        prob_apply: probabilidad de aplicar al menos una aug.
    """

    def __init__(
        self,
        split_path: Path | str,
        subset: str = "train",
        fixed_size: tuple[int, int] = (128, 1024),
        apply_aug: bool = False,
        prob_apply: float = 0.7,
    ) -> None:
        self.base = SplitV2Dataset(split_path, subset, fixed_size)
        self.apply_aug = apply_aug
        self.prob_apply = prob_apply

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, str, int]:
        path, palabra, longitud = self.base.samples[i]
        img = load_image(path)
        if self.apply_aug:
            img = random_augment_image(img, prob_apply=self.prob_apply)
        img = preprocess(img, self.base.fixed_size)
        return torch.from_numpy(img).float().unsqueeze(0), palabra, longitud
