"""
Análisis: ¿cuántas palabras del test_v2 son cubiertas por el train_v2 actual,
y cuántas serían cubiertas si sumamos los sintéticos puros del aug existente?

NO modifica archivos. Solo lee:
- ~/projects/unravel/data/split_v2.json
- ~/projects/unravel/data/datos_entrenamiento_augmented/PERFECT_CUT_a_z_1_9_aug_SYNTHETIC/

Asume estructura de split_v2.json: {"train": [{path, palabra, longitud}, ...], "val": [...], "test": [...]}
Asume que los sintéticos puros tienen filename "synthetic-*.jpg".
"""

import json
import re
from pathlib import Path

REPO = Path.home() / "projects" / "unravel"
SPLIT_PATH = REPO / "data" / "split_v2.json"
AUG_DIR = Path("/home/cecilia/datasets/spanish-htr/datos_entrenamiento_augmented/PERFECT_CUT_a_z_1_9_aug_SYNTHETIC")

# 1. Cargar split_v2.json
print("=" * 60)
print("FASE 1 — verificar estructura de archivos")
print("=" * 60)

with open(SPLIT_PATH) as f:
    split = json.load(f)

print(f"\nsplit_v2.json: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
print(f"Ejemplo de muestra del train: {split['train'][0]}")
print(f"Ejemplo de muestra del test: {split['test'][0]}")

# 2. Listar sintéticos puros del aug
if not AUG_DIR.exists():
    print(f"\nERROR: AUG_DIR no existe en {AUG_DIR}")
    print("Ajustá la ruta al directorio del aug y volvé a correr.")
    exit(1)

all_files = list(AUG_DIR.iterdir())
synthetic_files = [f for f in all_files if f.name.startswith("synthetic-")]
print(f"\nAUG_DIR total archivos: {len(all_files)}")
print(f"AUG_DIR sintéticos puros (synthetic-*): {len(synthetic_files)}")
print(f"\nMuestra de 5 filenames de sintéticos puros:")
for f in synthetic_files[:5]:
    print(f"  {f.name}")

# 3. Extraer palabra de los filenames sintéticos
# Heurística: el formato típico es "synthetic-PALABRA-N.jpg" o similar.
# Si la palabra contiene guiones, esto puede pifiar — por eso mostramos ejemplos primero.
def extract_word_from_synthetic(filename):
    name = filename.replace(".jpg", "").replace(".png", "")
    # Quitar prefijo "synthetic-"
    if name.startswith("synthetic-"):
        name = name[len("synthetic-"):]
    # El sufijo es "-N" donde N es un número. Lo quitamos.
    m = re.match(r"^(.+)-\d+$", name)
    if m:
        return m.group(1)
    return name

print(f"\nMuestra de palabras extraídas (primeras 5):")
for f in synthetic_files[:5]:
    word = extract_word_from_synthetic(f.name)
    print(f"  {f.name} → palabra='{word}'")

# 4. Si la extracción se ve razonable, calcular cobertura
print("\n" + "=" * 60)
print("FASE 2 — análisis de cobertura textual")
print("=" * 60)

train_words = set(s["palabra"] for s in split["train"])
test_words = set(s["palabra"] for s in split["test"])
val_words = set(s["palabra"] for s in split["val"])
synthetic_words = set(extract_word_from_synthetic(f.name) for f in synthetic_files)

print(f"\nPalabras únicas:")
print(f"  train_v2: {len(train_words)}")
print(f"  val_v2:   {len(val_words)}")
print(f"  test_v2:  {len(test_words)}")
print(f"  sintéticos puros: {len(synthetic_words)}")

# Cobertura baseline (M07/M08): test ∩ train
test_in_train = test_words & train_words
print(f"\nCobertura BASELINE (M07/M08):")
print(f"  palabras de test que aparecen en train_v2: {len(test_in_train)} / {len(test_words)} = {100*len(test_in_train)/len(test_words):.1f}%")

# Cobertura M09: test ∩ (train ∪ sintéticos)
extended_train = train_words | synthetic_words
test_in_extended = test_words & extended_train
print(f"\nCobertura M09 (train + sintéticos puros):")
print(f"  palabras de test que aparecen en train+sintéticos: {len(test_in_extended)} / {len(test_words)} = {100*len(test_in_extended)/len(test_words):.1f}%")

# Cuántas palabras de test pasarían de "no cubiertas" a "cubiertas"
new_coverage = (test_words & synthetic_words) - train_words
print(f"\nGanancia de M09:")
print(f"  palabras de test cubiertas SOLO por sintéticos (no en train_v2): {len(new_coverage)}")
print(f"  delta de cobertura: +{100*len(new_coverage)/len(test_words):.1f} puntos porcentuales")

# Sanity check: verificar que ninguna palabra del val/test aparece en los sintéticos como path
# (aunque esto sería leakage por palabra, no por imagen — distinto al de proximidad)
val_in_synthetic = val_words & synthetic_words
test_in_synthetic = test_words & synthetic_words
print(f"\nSANITY: palabras de val que aparecen también en sintéticos: {len(val_in_synthetic)} ({100*len(val_in_synthetic)/len(val_words):.1f}%)")
print(f"SANITY: palabras de test que aparecen también en sintéticos: {len(test_in_synthetic)} ({100*len(test_in_synthetic)/len(test_words):.1f}%)")
print("(Esto es ESPERADO — no es leakage por imagen, los sintéticos son plantillas distintas)")

