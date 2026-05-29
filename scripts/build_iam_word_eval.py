"""
Convierte el formato IAM word-level (val_gt.txt) al formato split-compatible
que usa evaluate_full_finetune.py.

Lee val_gt.txt línea a línea (formato: path\ttranscripcion) y construye un
JSON con la misma estructura que split_v2.json:
    {"train": [], "val": [], "test": [{path, palabra, longitud}, ...]}

donde test contiene todas las samples de val_gt.txt (que en IAM word-level
es el conjunto de evaluación holdout — no hay test set aparte).
"""

import json
from pathlib import Path

IAM_DIR = Path("/home/cecilia/datasets/iam_word_level")
GT_PATH = IAM_DIR / "val_gt.txt"
WORDS_DIR = IAM_DIR / "words"
OUT_PATH = Path.home() / "projects/unravel/data/iam_word_val.json"

samples = []
n_lines = 0
n_skipped = 0
with open(GT_PATH, encoding="utf-8") as f:
    for line in f:
        n_lines += 1
        line = line.rstrip("\n")
        if not line:
            n_skipped += 1
            continue
        # Split en path + transcripcion (separados por tab, pero usar split() general)
        parts = line.split("\t") if "\t" in line else line.split(None, 1)
        if len(parts) < 2:
            n_skipped += 1
            continue
        rel_path, transcr = parts[0], parts[1]
        # Construir path absoluto
        abs_path = str(IAM_DIR / rel_path)
        # Verificar que el archivo existe (los IAM tienen algunas imágenes "rotas"
        # marcadas en el dataset que conviene skippear)
        if not Path(abs_path).exists():
            n_skipped += 1
            continue
        samples.append({
            "path": abs_path,
            "palabra": transcr,
            "longitud": len(transcr),
        })

print(f"Lineas leidas: {n_lines}")
print(f"Skipped: {n_skipped}")
print(f"Samples válidas: {len(samples)}")

# Distribución por longitud
from collections import Counter
by_len = Counter(s["longitud"] for s in samples)
print(f"\nDistribución por longitud (top 15):")
for L in sorted(by_len)[:15]:
    print(f"  L={L:>2}: {by_len[L]:>5}")
print(f"  L máx: {max(by_len)}")

# Chars únicos
chars = set()
for s in samples:
    chars.update(s["palabra"])
print(f"\nChars únicos en transcripciones: {len(chars)}")
print(f"Chars: {sorted(chars)}")

# Guardar
out = {"train": [], "val": [], "test": samples}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\nGuardado: {OUT_PATH}")
print(f"  test: {len(samples)} samples")
