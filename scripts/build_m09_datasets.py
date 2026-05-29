"""
Consolida los archivos previos en dos JSONs listos para usar:
1. data/split_m09.json — train extendido (split_v2 train + sintéticos no-OOV),
   val y test idénticos a split_v2. Mismo formato que split_v2 → train_lora_manual.py
   puede leerlo pasándolo como --dataset sin cambios.
2. data/oov_test_for_eval.json — wrapper con {"train":[], "val":[], "test": <oov samples>}
   listo para evaluate_lora_manual.py.

NO modifica nada que ya exista. Verifica conteos y overlap.
"""

import json
from pathlib import Path

REPO = Path.home() / "projects" / "unravel"
DATA = REPO / "data"

split_v2 = json.load(open(DATA / "split_v2.json"))
train_extra = json.load(open(DATA / "train_extra_synthetic.json"))
oov_test = json.load(open(DATA / "oov_test.json"))

# ---- split_m09.json: train_v2 + sintéticos no-OOV ----
split_m09 = {
    "train": split_v2["train"] + train_extra["samples"],
    "val": split_v2["val"],
    "test": split_v2["test"],
}

print("split_m09:")
print(f"  train: {len(split_v2['train']):,} reales + {len(train_extra['samples']):,} sintéticos = {len(split_m09['train']):,}")
print(f"  val:   {len(split_m09['val']):,} (idéntico a split_v2)")
print(f"  test:  {len(split_m09['test']):,} (idéntico a split_v2)")

# ---- oov_test_for_eval.json: wrapper con estructura split-compatible ----
oov_test_for_eval = {
    "train": [],
    "val": [],
    "test": oov_test["samples"],
}

print(f"\noov_test_for_eval:")
print(f"  test: {len(oov_test_for_eval['test']):,} muestras OOV")

# ---- Sanity: verificar que ningún path se duplica entre train de m09 y oov ----
train_paths = set(s["path"] for s in split_m09["train"])
oov_paths = set(s["path"] for s in oov_test_for_eval["test"])
overlap = train_paths & oov_paths
if overlap:
    raise SystemExit(f"[ERROR] {len(overlap)} paths duplicados entre split_m09.train y oov_test")
print(f"\nSanity: 0 overlap entre split_m09.train y oov_test (OK)")

# ---- Guardar ----
out_split = DATA / "split_m09.json"
out_eval = DATA / "oov_test_for_eval.json"
out_split.write_text(json.dumps(split_m09, indent=2, ensure_ascii=False))
out_eval.write_text(json.dumps(oov_test_for_eval, indent=2, ensure_ascii=False))

print(f"\nGuardado:")
print(f"  {out_split}")
print(f"  {out_eval}")

