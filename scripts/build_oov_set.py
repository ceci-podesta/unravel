"""
Arma el set OOV para M09 usando el JSON de anotaciones del aug como fuente de verdad.

Decisiones 2026-05-06:
- Fuente de labels: synthetic_annotation.json (no extracción del filename).
- Descartar archivos NO en el JSON (~16K archivos cuya palabra no se puede
  determinar de forma confiable; tenían encoding raro `+â-¦`/`+â-+` en el filename).
- Descartar palabras con caracteres FUERA del vocab del modelo (en particular
  `ü`, que no está en el vocab — el modelo no puede predecirlo).
- Cortar OOV en L=18 (inclusive). Longitudes mayores no son representativas.

Genera:
1. data/oov_test.json
2. data/train_extra_synthetic.json
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path.home() / "projects" / "unravel"
sys.path.insert(0, str(REPO / "src"))

from unravel.vocab import build_unified_vocab

DATASETS = Path("/home/cecilia/datasets/spanish-htr")
AUG_DIR = DATASETS / "datos_entrenamiento_augmented" / "PERFECT_CUT_a_z_1_9_aug_SYNTHETIC"
SYNTHETIC_JSON = AUG_DIR / "synthetic_annotation.json"
SPLIT_V2_PATH = REPO / "data" / "split_v2.json"
OUT_DIR = REPO / "data"
HTR_REPO = Path.home() / "projects" / "HTR-best-practices"
CLASSES_PATH = HTR_REPO / "saved_models" / "classes.npy"

WORDS_PER_LENGTH_TARGET = 50
MAX_LENGTH = 18
SEED = 42

random.seed(SEED)

# ---------- 0. Cargar vocab del modelo ----------
vocab = build_unified_vocab(CLASSES_PATH)
c2i = vocab["c2i"]
print(f"Vocab del modelo: {vocab['n_classes']} chars (incluye {[c for c in c2i if not c.isascii()]})")

def word_in_vocab(w: str) -> bool:
    return all(c in c2i for c in w)

# ---------- 1. Cargar JSON de anotaciones (fuente de verdad) ----------
ann = json.load(open(SYNTHETIC_JSON))
print(f"Entries en synthetic_annotation.json: {len(ann):,}")

# ---------- 2. Filtrar entries por:
#    - palabra dentro del vocab del modelo
#    - archivo existe en disco
# ----------
ann_valid = {}
n_no_vocab = 0
n_no_file = 0
chars_rejected = set()

for filename, palabra in ann.items():
    # Validar vocab
    if not word_in_vocab(palabra):
        n_no_vocab += 1
        for c in palabra:
            if c not in c2i:
                chars_rejected.add(c)
        continue
    # Validar que el archivo existe
    full_path = AUG_DIR / filename
    if not full_path.exists():
        n_no_file += 1
        continue
    ann_valid[filename] = palabra

print(f"Entries válidas (palabra en vocab + archivo existe): {len(ann_valid):,}")
print(f"  rechazadas por chars fuera del vocab: {n_no_vocab:,} (chars problemáticos: {sorted(chars_rejected)})")
print(f"  rechazadas por archivo inexistente: {n_no_file:,}")

# ---------- 3. Mapear palabra -> lista de paths ----------
word_to_paths = defaultdict(list)
for filename, palabra in ann_valid.items():
    full_path = str(AUG_DIR / filename)
    word_to_paths[palabra].append(full_path)

print(f"Palabras únicas (válidas): {len(word_to_paths):,}")

# ---------- 4. Cargar split_v2 ----------
split = json.load(open(SPLIT_V2_PATH))
known_words = (set(s["palabra"] for s in split["train"])
               | set(s["palabra"] for s in split["val"])
               | set(s["palabra"] for s in split["test"]))
print(f"Palabras conocidas en split_v2 (train ∪ val ∪ test): {len(known_words)}")

# ---------- 5. Filtrar candidatos OOV ----------
oov_candidates = {w: paths for w, paths in word_to_paths.items() if w not in known_words}
print(f"\nPalabras OOV candidatas (sintéticas válidas no en split_v2): {len(oov_candidates):,}")

# ---------- 6. Agrupar por longitud ----------
by_len = defaultdict(list)
for w in oov_candidates:
    by_len[len(w)].append(w)

# ---------- 7. Sample N por longitud ----------
oov_selected = []
print(f"\nSampling palabras OOV (target {WORDS_PER_LENGTH_TARGET} por longitud, "
      f"MAX_LENGTH={MAX_LENGTH}, seed {SEED}):")
print(f"  {'L':>3} {'candidatos':>10} {'tomadas':>8}")
for L in sorted(by_len):
    if L > MAX_LENGTH:
        print(f"  {L:>3} {len(by_len[L]):>10} {'skip':>8}  (L > {MAX_LENGTH})")
        continue
    candidates = sorted(by_len[L])
    n_take = min(WORDS_PER_LENGTH_TARGET, len(candidates))
    sampled = random.sample(candidates, n_take)
    oov_selected.extend(sampled)
    print(f"  {L:>3} {len(candidates):>10} {n_take:>8}")

oov_selected_set = set(oov_selected)
print(f"\nTotal palabras OOV seleccionadas: {len(oov_selected)}")

# ---------- 8. Armar oov_test.json ----------
oov_test_samples = []
for w in oov_selected:
    for p in word_to_paths[w]:
        oov_test_samples.append({"path": p, "palabra": w, "longitud": len(w)})
print(f"Total muestras en oov_test: {len(oov_test_samples)}")

# ---------- 9. Armar train_extra ----------
train_extra_samples = []
for w, paths in word_to_paths.items():
    if w in oov_selected_set:
        continue
    for p in paths:
        train_extra_samples.append({"path": p, "palabra": w, "longitud": len(w)})
print(f"Total muestras en train_extra (sintéticos válidos no-OOV): {len(train_extra_samples)}")

# ---------- 10. Anti-leakage verification ----------
val_paths = set(s["path"] for s in split["val"])
test_paths = set(s["path"] for s in split["test"])
train_paths_real = set(s["path"] for s in split["train"])
oov_paths = set(s["path"] for s in oov_test_samples)
train_extra_paths = set(s["path"] for s in train_extra_samples)

print(f"\n=== Anti-leakage verification ===")
checks = [
    ("train_extra ∩ val_v2 (real)", train_extra_paths & val_paths),
    ("train_extra ∩ test_v2 (real)", train_extra_paths & test_paths),
    ("train_extra ∩ oov_test", train_extra_paths & oov_paths),
    ("oov_test ∩ train_v2 (real)", oov_paths & train_paths_real),
    ("oov_test ∩ val_v2 (real)", oov_paths & val_paths),
    ("oov_test ∩ test_v2 (real)", oov_paths & test_paths),
]
all_ok = True
for label, intersection in checks:
    n = len(intersection)
    status = "OK" if n == 0 else "FAIL"
    print(f"  {label}: {n} ({status})")
    if n > 0:
        all_ok = False
if not all_ok:
    raise SystemExit("[ERROR] Anti-leakage FAIL — no escribir archivos.")

oov_words_in_known = oov_selected_set & known_words
print(f"  oov_words ∩ known_words (palabras): {len(oov_words_in_known)} (esperado 0)")
if oov_words_in_known:
    raise SystemExit(f"[ERROR] palabras OOV que también están en split_v2")

# Sanity: validar que TODAS las muestras finales tienen palabra en vocab
for sample_set, name in [(oov_test_samples, "oov_test"), (train_extra_samples, "train_extra")]:
    invalid = [s for s in sample_set if not word_in_vocab(s["palabra"])]
    if invalid:
        raise SystemExit(f"[ERROR] {len(invalid)} muestras de {name} con chars fuera del vocab")
print(f"  Sanity vocab: 0 muestras inválidas en oov_test ni train_extra (OK)")

# ---------- 11. Guardar ----------
oov_path = OUT_DIR / "oov_test.json"
train_extra_path = OUT_DIR / "train_extra_synthetic.json"

oov_payload = {
    "samples": oov_test_samples,
    "metadata": {
        "n_samples": len(oov_test_samples),
        "n_unique_words": len(oov_selected),
        "words_by_length_selected": {str(L): sum(1 for w in oov_selected if len(w) == L) for L in sorted(by_len) if L <= MAX_LENGTH},
        "seed": SEED,
        "target_per_length": WORDS_PER_LENGTH_TARGET,
        "max_length": MAX_LENGTH,
        "source": "synthetic_annotation.json del aug existente",
        "filtered_chars_outside_vocab": sorted(chars_rejected),
    },
}
oov_path.write_text(json.dumps(oov_payload, indent=2, ensure_ascii=False))

train_extra_payload = {
    "samples": train_extra_samples,
    "metadata": {
        "n_samples": len(train_extra_samples),
        "n_unique_words": len(set(s["palabra"] for s in train_extra_samples)),
        "source": "synthetic_annotation.json del aug existente",
    },
}
train_extra_path.write_text(json.dumps(train_extra_payload, indent=2, ensure_ascii=False))

print(f"\nArchivos guardados:")
print(f"  {oov_path}")
print(f"  {train_extra_path}")

