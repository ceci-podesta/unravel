"""
Análisis: cómo predijo M01-M04 las palabras OOV (test-only) vs las palabras de igual
longitud que sí estaban en train original.
NO modifica archivos.
"""
import csv
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

DATASETS = Path("/home/cecilia/datasets/spanish-htr")
TRAIN_DIR = DATASETS / "datos_entrenamiento" / "PERFECT_CUT_a_z_1_9"
TEST_DIR = DATASETS / "datos_testing"
EXPERIMENTS_DIR = Path("/home/cecilia/projects/unravel/experiments")

# 1. Calcular OOV
print("=" * 60)
print("FASE 1 — calcular set OOV del split original")
print("=" * 60)

train_words = set()
for json_path in sorted(TRAIN_DIR.glob("*.json")):
    with open(json_path) as f:
        annotations = json.load(f)
    train_words.update(str(v) for v in annotations.values())

test_words = set()
for d in sorted([d for d in TEST_DIR.iterdir() if d.is_dir()]):
    for txt_path in sorted(d.glob("*.txt")):
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split(None, 1)
                if len(parts) >= 2:
                    test_words.add(parts[1])

oov_words = test_words - train_words
print(f"Train palabras únicas: {len(train_words)}")
print(f"Test palabras únicas: {len(test_words)}")
print(f"OOV (test - train): {len(oov_words)}")
print(f"OOV: {sorted(oov_words)}")

# 2. Detectar predictions.csv del split original
preds_files = sorted(EXPERIMENTS_DIR.rglob("predictions.csv"))
m1_to_m4_preds = [f for f in preds_files if "eval_test_v2" not in str(f)]

# 3. Análisis
print("\n" + "=" * 60)
print("FASE 2 — análisis OOV vs in-vocab por longitud")
print("=" * 60)

# Mapeo de columnas reales del CSV
GT_CANDIDATES = {'real', 'palabra_real', 'gt', 'truth', 'label', 'palabra'}
PRED_CANDIDATES = {'predicho', 'palabra_pred', 'pred', 'prediccion', 'prediction'}
CER_CANDIDATES = {'cer_palabra', 'cer'}
WER_CANDIDATES = {'error_word', 'wer'}  # error_word es 0/1 = WER por palabra

def find_col(cols, candidates):
    for c in cols or []:
        if c.lower() in candidates:
            return c
    return None

for pred_file in m1_to_m4_preds:
    print(f"\n{'='*60}")
    print(f"=== {pred_file.relative_to(EXPERIMENTS_DIR)} ===")
    print(f"{'='*60}")

    with open(pred_file, newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)

    GT = find_col(cols, GT_CANDIDATES)
    PRED = find_col(cols, PRED_CANDIDATES)
    CER = find_col(cols, CER_CANDIDATES)
    WER = find_col(cols, WER_CANDIDATES)
    print(f"  GT={GT} PRED={PRED} CER={CER} WER={WER} | total filas={len(rows)}")

    if not GT or not CER:
        print(f"  WARN: faltan columnas críticas, salto")
        continue

    oov_rows = [r for r in rows if r[GT] in oov_words]
    in_rows = [r for r in rows if r[GT] not in oov_words]

    print(f"\n  Filas OOV: {len(oov_rows)}  |  in-vocab: {len(in_rows)}")

    # Globales
    cer_oov_global = mean(float(r[CER]) for r in oov_rows) if oov_rows else None
    cer_in_global = mean(float(r[CER]) for r in in_rows) if in_rows else None
    wer_oov_global = mean(float(r[WER]) for r in oov_rows) if (WER and oov_rows) else None
    wer_in_global = mean(float(r[WER]) for r in in_rows) if (WER and in_rows) else None
    print(f"\n  GLOBAL:")
    print(f"    CER OOV: {cer_oov_global:.3f}" if cer_oov_global is not None else "    CER OOV: -")
    print(f"    CER inV: {cer_in_global:.3f}" if cer_in_global is not None else "    CER inV: -")
    if wer_oov_global is not None:
        print(f"    WER OOV: {wer_oov_global:.3f}")
        print(f"    WER inV: {wer_in_global:.3f}")

    # Por longitud (solo las longitudes donde hay OOV)
    oov_lens = sorted({len(r[GT]) for r in oov_rows})
    if oov_lens:
        print(f"\n  POR LONGITUD (solo Ls con OOV):")
        print(f"  {'L':>3} {'n_OOV':>6} {'CER_OOV':>9} {'WER_OOV':>9} {'n_inV':>6} {'CER_inV':>9} {'WER_inV':>9}")
        for L in oov_lens:
            oov_L = [r for r in oov_rows if len(r[GT]) == L]
            in_L = [r for r in in_rows if len(r[GT]) == L]
            cer_oov = mean(float(r[CER]) for r in oov_L) if oov_L else 0
            cer_in = mean(float(r[CER]) for r in in_L) if in_L else 0
            wer_oov = mean(float(r[WER]) for r in oov_L) if (WER and oov_L) else 0
            wer_in = mean(float(r[WER]) for r in in_L) if (WER and in_L) else 0
            print(f"  {L:>3} {len(oov_L):>6} {cer_oov:>9.3f} {wer_oov:>9.3f} {len(in_L):>6} {cer_in:>9.3f} {wer_in:>9.3f}")

    # Detalle de OOV no triviales (>= 2 chars)
    long_oov = [r for r in oov_rows if len(r[GT]) >= 2]
    if long_oov:
        print(f"\n  Predicciones para OOV >=2 chars (despues, tambien):")
        for r in long_oov:
            real = r[GT]
            pred = r.get(PRED, "?")
            cer = r.get(CER, "?")
            print(f"    real='{real}' pred='{pred}' cer={cer}")

