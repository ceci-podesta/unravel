"""Dataset class para el split de testing del dataset español (Kaggle verack)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from unravel.preproc import load_image, preprocess


class SpanishHTRTestDataset(Dataset):
    """Test set del dataset español, organizado por longitud de palabra."""

    def __init__(
        self,
        root: Path | str,
        fixed_size: tuple[int, int] = (128, 1024),
        center: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.fixed_size = fixed_size
        self.center = center
        self.samples: list[tuple[Path, str, int]] = []
        carpetas = sorted(
            (p for p in self.root.iterdir() if p.is_dir()),
            key=lambda p: int(p.name),
        )
        for carpeta in carpetas:
            longitud = int(carpeta.name)
            gt_path = carpeta / f"gt_{longitud}.txt"
            if not gt_path.is_file():
                continue
            with gt_path.open(encoding="utf-8") as f:
                for linea in f:
                    parts = linea.rstrip("\n").split("\t", 1)
                    if len(parts) != 2:
                        continue
                    filename, palabra = parts
                    img_path = carpeta / filename
                    if img_path.is_file():
                        self.samples.append((img_path, palabra, longitud))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, int]:
        img_path, palabra, longitud = self.samples[index]
        img = load_image(img_path)
        img = preprocess(img, self.fixed_size, center=self.center)
        tensor = torch.from_numpy(img).float().unsqueeze(0)
        return tensor, palabra, longitud
