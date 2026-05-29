"""
Diagnóstico: para los filenames sintéticos cuyas palabras tienen caracteres
fuera del vocab (â, ¦, etc.), inspeccionar los BYTES raw del filename y ver
si es mojibake (Latin-1 leído como UTF-8) o palabras realmente raras.
"""

import os
import re
from pathlib import Path

AUG_DIR = Path("/home/cecilia/datasets/spanish-htr/datos_entrenamiento_augmented/PERFECT_CUT_a_z_1_9_aug_SYNTHETIC")

# Caracteres "sospechosos" típicos de mojibake Latin-1 → UTF-8
SUSPECT_CHARS = set("âôãçÃÂ¦Ã±")

def extract_word(filename: str) -> str:
    name = filename.replace(".jpg", "").replace(".png", "")
    if name.startswith("synthetic-"):
        name = name[len("synthetic-"):]
    m = re.match(r"^(.+)-\d+$", name)
    return m.group(1) if m else name

# Buscar archivos con caracteres sospechosos en el filename
print("Buscando filenames con caracteres sospechosos...")
samples = []
for f in AUG_DIR.iterdir():
    if any(c in f.name for c in SUSPECT_CHARS):
        samples.append(f)
        if len(samples) >= 20:
            break

print(f"Encontrados {len(samples)} (mostramos hasta 20):\n")

for f in samples:
    name_str = f.name
    word_extracted = extract_word(name_str)

    # Bytes raw del filename (usando os.fsencode → siempre da los bytes "como están en el filesystem")
    name_bytes = os.fsencode(name_str)

    # Intentar decode como Latin-1 (si los bytes son un Latin-1 válido, esto da la versión correcta)
    try:
        as_latin1 = name_bytes.decode('latin-1')
    except Exception as e:
        as_latin1 = f"<error: {e}>"

    # Intentar decode como UTF-8 desde los bytes
    try:
        as_utf8 = name_bytes.decode('utf-8')
    except Exception as e:
        as_utf8 = f"<error: {e}>"

    # Detectar mojibake: a veces los bytes son "ya decodificados como UTF-8 desde una fuente Latin-1",
    # y la palabra "real" se obtiene re-encoding como Latin-1 → decode como UTF-8.
    try:
        as_remix = name_str.encode('latin-1').decode('utf-8')
    except Exception:
        as_remix = "<no aplica>"

    print(f"=== {name_str}")
    print(f"  word extracted:           '{word_extracted}'")
    print(f"  bytes raw:                {name_bytes}")
    print(f"  decoded latin-1:          '{as_latin1}'")
    print(f"  decoded utf-8 (=actual):  '{as_utf8}'")
    print(f"  encode-latin1 → decode-utf8: '{as_remix}'  ← si esto se ve como palabra normal, ES MOJIBAKE")
    print()

