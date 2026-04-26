"""Zero-shot evaluation: modelo IAM sobre dataset español de test.

Flags:
    --center           si se pasa, centra la palabra en el canvas (default: esquina sup-izq)
    --head {rnn,cnn,both}  qué head usar en inferencia (default: rnn)
    --outputs DIR      carpeta donde guardar predictions.csv y summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from unravel.dataset import SpanishHTRTestDataset
from unravel.htr_model import HTRNet, default_arch_cfg
from unravel.metrics import _normalize, cer_macro, cer_micro, edit_distance, wer

HTR_REPO = Path.home() / "projects/HTR-best-practices"
DEFAULT_DATASET = Path.home() / "datasets/spanish-htr/datos_testing"
DEFAULT_OUTPUTS = Path.home() / "projects/unravel/outputs/zero_shot_test"


def cargar_vocabulario(classes_path: Path) -> tuple[np.ndarray, dict[int, str]]:
    classes = np.load(classes_path)
    icdict = {(i + 1): c for i, c in enumerate(classes)}
    return classes, icdict


def cargar_modelo(weights_path: Path, n_classes: int, device: str) -> torch.nn.Module:
    arch_cfg = default_arch_cfg()
    net = HTRNet(arch_cfg, n_classes + 1)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        print("[INFO] pesos cargados con weights_only=True (modo seguro)")
    except Exception as e:
        print(f"[WARN] weights_only=True falló ({e}); cayendo a weights_only=False")
        state = torch.load(weights_path, map_location=device, weights_only=False)
    net.load_state_dict(state, strict=True)
    net.to(device)
    net.eval()
    return net


def combinar_heads(output, head_mode: str) -> torch.Tensor:
    """Selecciona / combina los outputs del modelo `head_type='both'`."""
    if not isinstance(output, tuple):
        return output
    rnn_out, cnn_out = output
    if head_mode == "rnn":
        return rnn_out
    if head_mode == "cnn":
        return cnn_out
    if head_mode == "both":
        # Promedio de logits — combinación estándar de ensembling
        return (rnn_out + cnn_out) / 2
    raise ValueError(f"head_mode desconocido: {head_mode}")


def decode_ctc_greedy(seq: np.ndarray, icdict: dict[int, str], blank_id: int = 0) -> str:
    colapsada = [v for j, v in enumerate(seq) if j == 0 or v != seq[j - 1]]
    return "".join(icdict[t] for t in colapsada if t != blank_id and t in icdict)


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-shot eval: modelo IAM en test español")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--weights", type=Path, default=HTR_REPO / "saved_models/htrnet.pt")
    parser.add_argument("--classes", type=Path, default=HTR_REPO / "saved_models/classes.npy")
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--center",
        action="store_true",
        help="Centrar la palabra en el canvas (default: esquina sup-izq).",
    )
    parser.add_argument(
        "--head",
        choices=["rnn", "cnn", "both"],
        default="rnn",
        help="Qué head del modelo usar en inferencia (default: rnn).",
    )
    args = parser.parse_args()

    args.outputs.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")
    print(f"[INFO] config: center={args.center}, head={args.head}, outputs={args.outputs}")

    classes, icdict = cargar_vocabulario(args.classes)
    print(f"[INFO] vocabulario IAM: {len(classes)} caracteres")

    net = cargar_modelo(args.weights, len(classes), device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[INFO] modelo cargado, {n_params:,} parámetros")

    dataset = SpanishHTRTestDataset(args.dataset, center=args.center)
    print(f"[INFO] test español: {len(dataset)} imágenes")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),
            [b[1] for b in batch],
            [b[2] for b in batch],
        ),
    )

    rows: list[dict] = []
    skipped_empty = 0
    start = time.time()
    predictions_path = args.outputs / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["longitud", "real", "predicho", "edit_dist", "cer_palabra", "error_word"]
        )
        writer.writeheader()
        with torch.no_grad():
            for batch_idx, (imgs, palabras, longitudes) in enumerate(loader):
                imgs = imgs.to(device)
                output = net(imgs)
                output = combinar_heads(output, args.head)
                pred_indices = output.argmax(2).permute(1, 0).cpu().numpy()
                for seq, palabra_real, longitud in zip(pred_indices, palabras, longitudes):
                    palabra_pred = decode_ctc_greedy(seq, icdict)
                    real_norm = _normalize(palabra_real)
                    pred_norm = _normalize(palabra_pred)
                    if not real_norm:
                        skipped_empty += 1
                        continue
                    dist = edit_distance(pred_norm, real_norm)
                    cer_p = dist / len(real_norm)
                    error_word = int(real_norm != pred_norm)
                    row = {
                        "longitud": longitud,
                        "real": palabra_real,
                        "predicho": palabra_pred,
                        "edit_dist": dist,
                        "cer_palabra": round(cer_p, 4),
                        "error_word": error_word,
                    }
                    rows.append(row)
                    writer.writerow(row)
                if batch_idx % 20 == 0:
                    pct = 100.0 * (batch_idx + 1) * args.batch_size / len(dataset)
                    print(f"  batch {batch_idx + 1}/{len(loader)} (~{pct:.0f}%)")

    elapsed = time.time() - start
    print(f"\n[INFO] predicciones guardadas en {predictions_path} ({len(rows)} filas)")
    print(f"[INFO] skipped_empty: {skipped_empty}")
    print(f"[INFO] tiempo total: {elapsed:.1f}s")

    print("\n=== Métricas zero-shot ===")
    todos_pred = [r["predicho"] for r in rows]
    todos_real = [r["real"] for r in rows]
    summary = {
        "config": {"center": args.center, "head": args.head},
        "n_samples": len(rows),
        "skipped_empty": skipped_empty,
        "elapsed_seconds": elapsed,
        "global": {
            "cer_micro": cer_micro(todos_pred, todos_real),
            "cer_macro": cer_macro(todos_pred, todos_real),
            "wer": wer(todos_pred, todos_real),
        },
        "por_longitud": {},
    }
    g = summary["global"]
    print(f"Global: CER_micro={g['cer_micro']:.4f}  CER_macro={g['cer_macro']:.4f}  WER={g['wer']:.4f}")

    print(f"\n{'longitud':>10} {'n':>6} {'CER_micro':>10} {'CER_macro':>10} {'WER':>10}")
    grupos: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        grupos[r["longitud"]].append(r)
    for longitud in sorted(grupos):
        grupo = grupos[longitud]
        preds = [r["predicho"] for r in grupo]
        reals = [r["real"] for r in grupo]
        cm = cer_micro(preds, reals)
        cM = cer_macro(preds, reals)
        w = wer(preds, reals)
        summary["por_longitud"][longitud] = {"n": len(grupo), "cer_micro": cm, "cer_macro": cM, "wer": w}
        print(f"{longitud:>10} {len(grupo):>6} {cm:>10.4f} {cM:>10.4f} {w:>10.4f}")

    summary_path = args.outputs / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] summary guardado en {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
