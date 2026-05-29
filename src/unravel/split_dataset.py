"""Dataset que lee el split estratificado v2 desde data/split_v2.json."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from unravel.preproc import load_image, preprocess


class SplitV2Dataset(Dataset):
    """Lee de data/split_v2.json el subset solicitado (train|val|test)."""

    def __init__(
        self,
        split_path: Path | str,
        subset: str = "train",
        fixed_size: tuple[int, int] = (128, 1024),
    ) -> None:
        if subset not in {"train", "val", "test"}:
            raise ValueError(f"subset debe ser train|val|test, no {subset!r}")
        self.fixed_size = fixed_size
        with Path(split_path).open(encoding="utf-8") as f:
            data = json.load(f)
        self.samples = [(Path(s["path"]), s["palabra"], s["longitud"])
                        for s in data[subset]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, str, int]:
        path, palabra, longitud = self.samples[i]
        img = load_image(path)
        img = preprocess(img, self.fixed_size)
        return torch.from_numpy(img).float().unsqueeze(0), palabra, longitud
