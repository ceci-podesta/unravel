"""Dataset de training/validation sobre el split real español.

Carga `datos_entrenamiento/PERFECT_CUT_a_z_1_9/` (~6.100 imágenes con
JSON de anotaciones filename → palabra) y particiona 90/10 en train/val
con seed fija para reproducibilidad.

El split de testing (`datos_testing/`) queda intocado durante training —
solo se usa para la evaluación final post-LoRA.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from unravel.preproc import load_image, preprocess


class SpanishHTRTrainDataset(Dataset):
    """Split real español, particionado en train (90%) / val (10%).

    Args:
        root: ruta a `datos_entrenamiento/PERFECT_CUT_a_z_1_9/`.
        subset: 'train' o 'val'.
        fixed_size: tamaño objetivo (h, w) para el preprocessing.
        seed: seed para la partición. Default 42 → reproducibilidad.
        val_fraction: fracción para validación (default 0.1).
    """

    def __init__(
        self,
        root: Path | str,
        subset: str = "train",
        fixed_size: tuple[int, int] = (128, 1024),
        seed: int = 42,
        val_fraction: float = 0.1,
    ) -> None:
        if subset not in {"train", "val"}:
            raise ValueError(f"subset debe ser 'train' o 'val', no {subset!r}")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction inválido: {val_fraction}")
        self.root = Path(root).resolve()
        self.fixed_size = fixed_size
        self.subset = subset

        # Cargar JSON de anotaciones (debería haber uno solo en la carpeta)
        json_files = sorted(self.root.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No se encontró JSON de anotaciones en {self.root}")
        with json_files[0].open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"JSON {json_files[0]} no es un dict")

        # Construir lista (path, palabra), validando que cada path resuelva
        # dentro de root (defensa contra path traversal aunque el dataset sea confiable)
        all_samples: list[tuple[Path, str]] = []
        for filename, palabra in data.items():
            if not isinstance(palabra, str):
                continue
            img_path = (self.root / filename).resolve()
            try:
                img_path.relative_to(self.root)  # raises si está afuera de root
            except ValueError:
                continue
            if img_path.is_file():
                all_samples.append((img_path, palabra))

        # Partición determinística train/val con seed fija
        rng = random.Random(seed)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)
        n_val = int(len(indices) * val_fraction)
        val_indices = set(indices[:n_val])

        if subset == "train":
            self.samples = [s for i, s in enumerate(all_samples) if i not in val_indices]
        else:  # val
            self.samples = [s for i, s in enumerate(all_samples) if i in val_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, int]:
        img_path, palabra = self.samples[index]
        img = load_image(img_path)
        img = preprocess(img, self.fixed_size)
        tensor = torch.from_numpy(img).float().unsqueeze(0)
        return tensor, palabra, len(palabra)
