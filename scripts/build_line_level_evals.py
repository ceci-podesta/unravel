"""Arma JSONs de eval para IAM line-level y Rodrigo line-level.

Para Rodrigo, normaliza chars del español antiguo a equivalentes modernos
(ç→c, ē→e, etc.) y filtra líneas con chars verdaderamente raros que el
modelo no podría predecir (þ, ƚ, ք, etc.).
"""

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path.home() / "projects" / "unravel"
sys.path.insert(0, str(REPO / "src"))
from unravel.vocab import build_unified_vocab

HTR_REPO = Path.home() / "projects/HTR-best-practices"
vocab = build_unified_vocab(HTR_REPO / "saved_models/classes.npy")
VOCAB_CHARS = set(vocab["c2i"].keys())
print(f"Vocab del modelo: {len(VOCAB_CHARS)} chars")

# ============================================================
# IAM line-level
# ============================================================
print("\n=== IAM line-level ===")
IAM_TEST = HTR_REPO / "data/IAM/processed_lines/test"
gt_path = IAM_TEST / "gt.txt"
samples_iam = []
skipped_iam_chars = []
with open(gt_path, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        img_id, transcr = parts[0], parts[1]
        img_path = IAM_TEST / f"{img_id}.png"
        if not img_path.exists():
            continue
        # Filtrar chars fuera del vocab
        if any(c not in VOCAB_CHARS for c in transcr):
            skipped_iam_chars.append(transcr)
            continue
        samples_iam.append({
            "path": str(img_path),
            "palabra": transcr,
            "longitud": len(transcr),
        })

print(f"Samples válidas (palabra completa en vocab): {len(samples_iam)}")
print(f"Skipped por chars fuera de vocab: {len(skipped_iam_chars)}")
if skipped_iam_chars:
    # Mostrar chars únicos problemáticos
    problem_chars = set()
    for t in skipped_iam_chars:
        for c in t:
            if c not in VOCAB_CHARS:
                problem_chars.add(c)
    print(f"Chars problemáticos en IAM: {sorted(problem_chars)}")

iam_out = REPO / "data/iam_line_test.json"
iam_out.write_text(json.dumps(
    {"train": [], "val": [], "test": samples_iam},
    ensure_ascii=False, indent=2
))
print(f"Guardado: {iam_out}")

# ============================================================
# Rodrigo (español antiguo, line-level) — con mapeo + filtro
# ============================================================
print("\n=== Rodrigo line-level ===")
RODRIGO = Path("/home/cecilia/datasets/rodrigo/Rodrigo corpus 1.0.0")
trans_path = RODRIGO / "text/transcriptions.txt"
test_partition_path = RODRIGO / "partitions/test.txt"
images_dir = RODRIGO / "images"

# Mapeo de chars del español antiguo → modernos
CHAR_MAP = {
    'ç': 'c', 'Ç': 'C',
    'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u',
    'ę': 'e', 'ħ': 'h', 'ł': 'l', 'ř': 'r',
    'ś': 's', 'š': 's', 'đ': 'd',
    '–': '-', '"': '"', '“': '"',
    # Chars verdaderamente raros (no mapeables) — los listamos abajo en SKIP_CHARS para filtrar líneas
}
# Chars que si aparecen, descartamos la línea entera (no son representables)
SKIP_CHARS = set("þƚȓʠքႲḡṕỹ¶℣♦₉|\\")

def normalize_rodrigo(s):
    """Aplica mapeo char por char. Devuelve None si tiene chars en SKIP_CHARS."""
    if any(c in SKIP_CHARS for c in s):
        return None
    return "".join(CHAR_MAP.get(c, c) for c in s)

# Cargar test partition IDs
test_ids = set()
with open(test_partition_path) as f:
    for line in f:
        line = line.strip()
        if line:
            test_ids.add(line)
print(f"IDs en test partition: {len(test_ids)}")

# Cargar transcripciones y filtrar por test
samples_rodrigo = []
skipped_rodrigo_chars = 0
skipped_rodrigo_vocab = 0
skipped_rodrigo_missing = 0
rejected_chars_seen = Counter()
with open(trans_path, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        img_id, transcr_orig = parts[0], parts[1]
        if img_id not in test_ids:
            continue
        # Verificar imagen existe
        img_path = images_dir / f"{img_id}.png"
        if not img_path.exists():
            skipped_rodrigo_missing += 1
            continue
        # Normalizar (puede devolver None si tiene chars raros)
        transcr = normalize_rodrigo(transcr_orig)
        if transcr is None:
            skipped_rodrigo_chars += 1
            for c in transcr_orig:
                if c in SKIP_CHARS:
                    rejected_chars_seen[c] += 1
            continue
        # Verificar que tras normalización todos los chars están en vocab
        if any(c not in VOCAB_CHARS for c in transcr):
            skipped_rodrigo_vocab += 1
            continue
        samples_rodrigo.append({
            "path": str(img_path),
            "palabra": transcr,
            "longitud": len(transcr),
        })

print(f"Total test partition: {len(test_ids)}")
print(f"Samples válidas (con normalización): {len(samples_rodrigo)}")
print(f"Skipped por chars muy raros (filtro): {skipped_rodrigo_chars}")
print(f"Skipped por chars fuera de vocab tras normalizar: {skipped_rodrigo_vocab}")
print(f"Skipped por imagen faltante: {skipped_rodrigo_missing}")
if rejected_chars_seen:
    print(f"Chars que disparan filtro (top 10): {rejected_chars_seen.most_common(10)}")

rodrigo_out = REPO / "data/rodrigo_line_test.json"
rodrigo_out.write_text(json.dumps(
    {"train": [], "val": [], "test": samples_rodrigo},
    ensure_ascii=False, indent=2
))
print(f"Guardado: {rodrigo_out}")

# Resumen final
print("\n=== Resumen ===")
print(f"IAM line-level test:   {len(samples_iam):,} samples")
print(f"Rodrigo test (filtrado): {len(samples_rodrigo):,} samples (de {len(test_ids):,} en partition)")

