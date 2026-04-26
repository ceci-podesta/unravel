"""Exploración del dataset español de Kaggle (verack).

Recorre las carpetas del dataset y reporta:
- Conteo de archivos por extensión
- Estructura de archivos JSON / TXT de anotaciones
- Ejemplos de pares (imagen → palabra)
- Distribución de longitudes y vocabulario
- Tamaños de imágenes (muestra)

Uso:
    uv run python -m unravel.explore /home/cecilia/datasets/spanish-htr/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


def listar_extensiones(carpeta: Path) -> Counter[str]:
    """Cuenta archivos por extensión (recursivo)."""
    return Counter(p.suffix.lower() for p in carpeta.rglob("*") if p.is_file())


def inspeccionar_json(path: Path) -> None:
    """Abre un JSON de anotaciones y reporta su estructura."""
    print(f"\n  → JSON: {path.relative_to(path.anchor) if path.is_absolute() else path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    print(f"    tipo: {type(data).__name__}, len: {len(data)}")
    if isinstance(data, dict):
        items = list(data.items())
        print(f"    primeras 3 entradas:")
        for k, v in items[:3]:
            print(f"      {k!r}: {v!r}")
        # Vocabulario y longitudes
        palabras = [v for v in data.values() if isinstance(v, str)]
        if palabras:
            longitudes = [len(p) for p in palabras]
            print(
                f"    longitud de palabras: min={min(longitudes)}, "
                f"max={max(longitudes)}, mean={sum(longitudes) / len(longitudes):.1f}"
            )
            vocab: Counter[str] = Counter()
            for p in palabras:
                vocab.update(p)
            print(f"    vocabulario: {len(vocab)} caracteres únicos")
            print(f"    top 10 caracteres: {dict(vocab.most_common(10))}")
            tildes = sorted(c for c in vocab if c in "áéíóúüñÁÉÍÓÚÜÑ")
            print(f"    tildes/ñ presentes: {tildes}")
            print(f"    todos los caracteres: {''.join(sorted(vocab.keys()))!r}")


def inspeccionar_gt_txt(path: Path) -> None:
    """Abre un gt_N.txt y muestra primeras líneas."""
    print(f"\n  → TXT: {path.name}")
    with path.open(encoding="utf-8") as f:
        lineas = f.readlines()
    print(f"    {len(lineas)} líneas")
    print(f"    primeras 3:")
    for linea in lineas[:3]:
        print(f"      {linea.rstrip()!r}")


def medir_imagenes(carpeta: Path, n: int = 50) -> None:
    """Reporta modo y tamaño de hasta N imágenes encontradas."""
    jpgs = list(carpeta.rglob("*.jpg"))[:n]
    if not jpgs:
        return
    print(f"\n  → muestra de {len(jpgs)} imágenes:")
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    for jpg in jpgs:
        with Image.open(jpg) as img:
            sizes[img.size] += 1
            modes[img.mode] += 1
    print(f"    modos: {dict(modes)}")
    print(f"    top 5 tamaños (W × H): {dict(sizes.most_common(5))}")
    todos_w = [s[0] for s, c in sizes.items() for _ in range(c)]
    todos_h = [s[1] for s, c in sizes.items() for _ in range(c)]
    print(
        f"    width:  min={min(todos_w)}, max={max(todos_w)}, "
        f"mean={sum(todos_w) / len(todos_w):.1f}"
    )
    print(
        f"    height: min={min(todos_h)}, max={max(todos_h)}, "
        f"mean={sum(todos_h) / len(todos_h):.1f}"
    )


def inspeccionar_carpeta(carpeta: Path) -> None:
    """Reporte completo de una carpeta."""
    print(f"\n========== {carpeta} ==========")
    if not carpeta.is_dir():
        print("  (no existe o no es carpeta)")
        return
    subs = sorted(p for p in carpeta.iterdir() if p.is_dir())
    if subs:
        nombres = [p.name for p in subs[:15]]
        cola = " ..." if len(subs) > 15 else ""
        print(f"  subcarpetas ({len(subs)}): {nombres}{cola}")
    exts = listar_extensiones(carpeta)
    print(f"  archivos por extensión: {dict(exts.most_common())}")
    # Primer JSON
    for json_path in carpeta.rglob("*.json"):
        inspeccionar_json(json_path)
        break
    # Primer gt_*.txt
    for txt_path in sorted(carpeta.rglob("gt_*.txt")):
        inspeccionar_gt_txt(txt_path)
        break
    medir_imagenes(carpeta, n=50)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exploración dataset HTR español")
    parser.add_argument(
        "root",
        type=Path,
        help="Raíz del dataset (ej. /home/cecilia/datasets/spanish-htr/)",
    )
    args = parser.parse_args()
    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: {root} no existe o no es una carpeta")
        return 1
    print(f"Explorando dataset en: {root}")
    print(f"\n--- TOP LEVEL ---")
    print(f"contenido: {[p.name for p in sorted(root.iterdir())]}")
    for nombre in ("datos_entrenamiento", "datos_entrenamiento_augmented", "datos_testing"):
        sub = root / nombre
        if sub.is_dir():
            inspeccionar_carpeta(sub)
        else:
            print(f"\n!! No encontrada: {sub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
