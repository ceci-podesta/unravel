"""Inspecciona las predicciones del LoRA entrenado sobre el val set.

Carga el modelo IAM, lo extiende, le aplica el adapter LoRA del
best_lora checkpoint, y genera predicciones sobre el val. Muestra:
- Aciertos (palabras donde el modelo acertó exactamente)
- Una muestra de errores (real vs predicho)
- Distribución de las predicciones más frecuentes
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel

from unravel.ctc_utils import collate_for_ctc
from unravel.extend_vocab import extend_model_vocabulary
from unravel.htr_model import HTRNet, default_arch_cfg
from unravel.metrics import _normalize
from unravel.train_dataset import SpanishHTRTrainDataset
from unravel.vocab import build_unified_vocab

HTR_REPO = Path.home() / "projects/HTR-best-practices"
DATASET = Path.home() / "datasets/spanish-htr/datos_entrenamiento/PERFECT_CUT_a_z_1_9"
LORA_CKPT = Path.home() / "projects/unravel/experiments/06_lora_real_only/best_lora"


def decode_ctc_greedy(seq, icdict, blank_id=0):
    colapsada = [v for j, v in enumerate(seq) if j == 0 or v != seq[j - 1]]
    return "".join(icdict[t] for t in colapsada if t != blank_id and t in icdict)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] device: {device}")

# Vocabulario
vocab = build_unified_vocab(HTR_REPO / "saved_models/classes.npy")
n_iam = vocab["n_classes"] - 6

# Modelo: IAM cargado + vocab extendido + LoRA aplicado desde checkpoint
net = HTRNet(default_arch_cfg(), n_iam + 1)
state = torch.load(HTR_REPO / "saved_models/htrnet.pt", map_location=device, weights_only=True)
net.load_state_dict(state, strict=True)
net = extend_model_vocabulary(net, n_extra=6).to(device)
net = PeftModel.from_pretrained(net, str(LORA_CKPT))
net.eval()
print(f"[INFO] modelo cargado con LoRA desde {LORA_CKPT}")

# Val set
val_set = SpanishHTRTrainDataset(DATASET, subset="val", seed=42)
print(f"[INFO] val set: {len(val_set)} muestras")

# Predecir todo (no es mucho, son 610)
def collate(b):
    return collate_for_ctc(b, vocab["c2i"])

from torch.utils.data import DataLoader
loader = DataLoader(val_set, batch_size=16, shuffle=False, collate_fn=collate)

all_real, all_pred, all_lengths = [], [], []
with torch.no_grad():
    for batch in loader:
        imgs = batch["images"].to(device)
        out = net(imgs)
        logits = out[0] if isinstance(out, tuple) else out  # rama RNN
        pred_indices = logits.argmax(2).permute(1, 0).cpu().numpy()
        for seq, real, longitud in zip(pred_indices, batch["palabras"], batch["longitudes"]):
            pred = decode_ctc_greedy(seq, vocab["i2c"], vocab["blank_id"])
            all_real.append(real)
            all_pred.append(pred)
            all_lengths.append(longitud)

# Stats globales
filtered = [(p, r, l) for p, r, l in zip(all_pred, all_real, all_lengths) if _normalize(r)]
aciertos = [(p, r, l) for p, r, l in filtered if _normalize(p) == _normalize(r)]
errores  = [(p, r, l) for p, r, l in filtered if _normalize(p) != _normalize(r)]

print(f"\n=== Stats globales sobre val ({len(filtered)} muestras válidas) ===")
print(f"  Aciertos word-level: {len(aciertos)} ({100*len(aciertos)/len(filtered):.1f}%)")
print(f"  Errores: {len(errores)}")

print(f"\n=== Aciertos: TODOS los {len(aciertos)} ===")
for pred, real, longitud in aciertos[:50]:
    print(f"  L{longitud:>2}  real={real!r:>15}  pred={pred!r}")
if len(aciertos) > 50:
    print(f"  ... y {len(aciertos) - 50} más")

print(f"\n=== Muestra de 30 errores ===")
for pred, real, longitud in errores[:30]:
    print(f"  L{longitud:>2}  real={real!r:>15}  pred={pred!r}")

print(f"\n=== Top 10 predicciones más frecuentes (en errores) ===")
pred_counts = Counter(p for p, _, _ in errores)
for pred, count in pred_counts.most_common(10):
    print(f"  {count:>4}x  {pred!r}")
