"""
Análisis de cobertura textual del split ORIGINAL (M01-M04):
- Train: /home/cecilia/datasets/spanish-htr/datos_entrenamiento/PERFECT_CUT_a_z_1_9/ (con JSON de anotaciones)
- Test:  /home/cecilia/datasets/spanish-htr/datos_testing/ (11 carpetas por longitud, gt_*.txt por carpeta)

NO modifica archivos, solo lee. Compara con la cobertura ya conocida del split_v2 (M05+).
"""

import json
from pathlib import Path

DATASETS = Path("/home/cecilia/datasets/spanish-htr")
TRAIN_DIR = DATASETS / "datos_entrenamiento" / "PERFECT_CUT_a_z_1_9"
TEST_DIR = DATASETS / "datos_testing"

# ------------------------------------------------------------------
# FASE 1 — detectar archivos relevantes
# ------------------------------------------------------------------
print("=" * 60)
print("FASE 1 — detectar archivos")
print("=" * 60)

if not TRAIN_DIR.exists():
    print(f"ERROR: no existe {TRAIN_DIR}")
    exit(1)
if not TEST_DIR.exists():
    print(f"ERROR: no existe {TEST_DIR}")
    exit(1)

json_candidates = sorted(TRAIN_DIR.glob("*.json"))
print(f"\nJSONs encontrados en TRAIN_DIR ({len(json_candidates)}):")
for j in json_candidates:
    print(f"  {j.name}")

test_subdirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
print(f"\nSubdirs en TEST_DIR ({len(test_subdirs)}):")
for d in test_subdirs:
    txts = sorted(d.glob("*.txt"))
    print(f"  {d.name}: {[t.name for t in txts]}")

# ------------------------------------------------------------------
# FASE 2 — extraer palabras de train y test originales
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 2 — extracción de palabras")
print("=" * 60)

# Train: leer todos los JSON y juntar values
train_words = set()
total_train_samples = 0
for json_path in json_candidates:
    with open(json_path) as f:
        annotations = json.load(f)
    print(f"\n{json_path.name}: {len(annotations)} entries")
    print(f"  sample: {dict(list(annotations.items())[:2])}")
    train_words.update(str(v) for v in annotations.values())
    total_train_samples += len(annotations)

# Test: leer todos los gt_*.txt
test_words = set()
total_test_samples = 0
for d in test_subdirs:
    for txt_path in sorted(d.glob("*.txt")):
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Asume TSV: imagen.jpg<TAB>palabra (puede haber espacios también)
                if "\t" in line:
                    parts = line.split("\t")
                else:
                    parts = line.split(None, 1)
                if len(parts) >= 2:
                    test_words.add(parts[1])
                    total_test_samples += 1

print(f"\n=== Train original ===")
print(f"  total samples (suma de JSONs): {total_train_samples}")
print(f"  palabras únicas: {len(train_words)}")
print(f"  muestra: {list(sorted(train_words))[:10]}")

print(f"\n=== Test original (datos_testing) ===")
print(f"  total samples: {total_test_samples}")
print(f"  palabras únicas: {len(test_words)}")
print(f"  muestra: {list(sorted(test_words))[:10]}")

# ------------------------------------------------------------------
# FASE 3 — cobertura test ⊆ train
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 3 — cobertura textual")
print("=" * 60)

test_in_train = test_words & train_words
solo_en_test = test_words - train_words

print(f"\nCobertura ORIGINAL (M01-M04):")
print(f"  palabras de test que están en train: {len(test_in_train)} / {len(test_words)} = {100*len(test_in_train)/len(test_words):.1f}%")
print(f"  palabras de test NUEVAS (no aparecen en train): {len(solo_en_test)} = {100*len(solo_en_test)/len(test_words):.1f}%")

if solo_en_test:
    print(f"\n  Primeras 20 palabras nuevas de test (no estaban en train):")
    for w in sorted(solo_en_test)[:20]:
        print(f"    '{w}'")

# Referencia conocida
print(f"\n=== Referencia split_v2 (M05+) ===")
print(f"  cobertura test ⊆ train: 100% (502/502)")

# Verdict
print(f"\n=== Lectura ===")
gap = 100 - 100*len(test_in_train)/len(test_words)
print(f"  Cobertura M01-M04: {100*len(test_in_train)/len(test_words):.1f}%")
print(f"  Cobertura M05+:    100.0%")
print(f"  Diferencia: {gap:.1f} puntos porcentuales que el modelo NO veía en train para M01-M04 y SÍ ve en M05+")

