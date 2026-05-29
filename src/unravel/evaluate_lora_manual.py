"""Evaluación del modelo HTR con LoRA manual sobre el test set español.

Carga el modelo IAM, extiende el vocabulario, aplica LoRA con la misma config
que el training, carga los pesos LoRA del checkpoint, y evalúa sobre el test
set español (datos_testing/) generando predicciones detalladas para análisis
de errores.

Uso:
    uv run python -m unravel.evaluate_lora_manual \\
        --lora-checkpoint experiments/M03_lora_lstm_extendido/best_lora.pt \\
        --target-modules top.fnl.1 top.cnn.1 top.rec \\
        --outputs experiments/M03_lora_lstm_extendido/eval_test/
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from unravel.dataset import SpanishHTRTestDataset
from unravel.evaluate_zero_shot import (
    cargar_modelo,
    combinar_heads,
    decode_ctc_greedy,
)
from unravel.extend_vocab import extend_model_vocabulary
from unravel.lora_manual import apply_lora_manual
from unravel.metrics import _normalize, cer_macro, cer_micro, edit_distance, wer
from unravel.vocab import build_unified_vocab

HTR_REPO = Path.home() / "projects/HTR-best-practices"
DEFAULT_DATASET = Path.home() / "datasets/spanish-htr/datos_testing"


def has_n_tilde(palabra: str) -> bool:
    return "ñ" in palabra or "Ñ" in palabra


def has_acento(palabra: str) -> bool:
    return any(c in palabra for c in "áéíóúÁÉÍÓÚ")


def stats_subset(rows: list[dict]) -> dict:
    """Calcula CER/WER sobre un subconjunto de filas."""
    if not rows:
        return {"n": 0, "cer_micro": float("nan"), "cer_macro": float("nan"), "wer": float("nan")}
    preds = [r["predicho"] for r in rows]
    reals = [r["real"] for r in rows]
    return {
        "n": len(rows),
        "cer_micro": cer_micro(preds, reals),
        "cer_macro": cer_macro(preds, reals),
        "wer": wer(preds, reals),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval del modelo HTR con LoRA manual sobre test español")
    parser.add_argument("--lora-checkpoint", type=Path, required=True,
                        help="Path al best_lora.pt generado por train_lora_manual.")
    parser.add_argument("--target-modules", nargs="+", required=True,
                        help="Mismos targets que se usaron en training. Ej: top.fnl.1 top.cnn.1 top.rec")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--weights", type=Path, default=HTR_REPO / "saved_models/htrnet.pt")
    parser.add_argument("--classes", type=Path, default=HTR_REPO / "saved_models/classes.npy")
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--head", choices=["rnn", "cnn", "both"], default="both")
    args = parser.parse_args()

    args.outputs.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")
    print(f"[INFO] checkpoint: {args.lora_checkpoint}")
    print(f"[INFO] targets: {args.target_modules}")

    # Vocabulario unificado (IAM + 6 chars españoles).
    vocab = build_unified_vocab(args.classes)
    n_iam = vocab["n_classes"] - 6  # 79
    print(f"[INFO] vocab unificado: {vocab['n_classes']} chars")

    # Modelo: cargar base IAM, extender vocab, aplicar LoRA, cargar pesos LoRA.
    net = cargar_modelo(args.weights, n_iam, device)
    net = extend_model_vocabulary(net, n_extra=6).to(device)
    net, lora_stats = apply_lora_manual(
        net, target_modules=args.target_modules,
        r=args.r, alpha=args.alpha, dropout=args.lora_dropout,
    )
    saved = torch.load(args.lora_checkpoint, map_location=device, weights_only=True)
    missing, unexpected = net.load_state_dict(saved, strict=False)
    print(f"[INFO] LoRA cargado: {len(saved)} keys")
    if unexpected:
        print(f"[WARN] keys inesperadas en checkpoint: {unexpected[:3]}...")
    print(f"[INFO] LoRA stats: {lora_stats['percent_trainable']:.2f}% entrenable")
    net.eval()

    # Test set por longitud.
    if str(args.dataset).endswith(".json"):
        from unravel.split_dataset import SplitV2Dataset
        dataset = SplitV2Dataset(args.dataset, subset="test")
    else:
        dataset = SpanishHTRTestDataset(args.dataset, center=args.center)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b], [x[2] for x in b]),
    )
    print(f"[INFO] test set: {len(dataset)} imágenes")

    rows: list[dict] = []
    start = time.time()
    with torch.no_grad():
        for batch_idx, (imgs, palabras, longitudes) in enumerate(loader):
            imgs = imgs.to(device)
            output = net(imgs)
            output = combinar_heads(output, args.head)
            pred_indices = output.argmax(2).permute(1, 0).cpu().numpy()
            for seq, palabra_real, longitud in zip(pred_indices, palabras, longitudes):
                palabra_pred = decode_ctc_greedy(seq, vocab["i2c"], vocab["blank_id"])
                real_n = _normalize(palabra_real)
                pred_n = _normalize(palabra_pred)
                if not real_n:
                    continue
                dist = edit_distance(pred_n, real_n)
                rows.append({
                    "longitud": longitud,
                    "real": palabra_real,
                    "predicho": palabra_pred,
                    "edit_dist": dist,
                    "cer_palabra": round(dist / len(real_n), 4),
                    "error_word": int(real_n != pred_n),
                    "has_ntilde": int(has_n_tilde(palabra_real)),
                    "has_acento": int(has_acento(palabra_real)),
                })
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx + 1}/{len(loader)}")

    elapsed = time.time() - start
    print(f"\n[INFO] {len(rows)} predicciones, tiempo: {elapsed:.1f}s")

    # Guardar predictions.csv.
    pred_path = args.outputs / "predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] predictions guardadas en {pred_path}")

    # Análisis por dimensiones.
    summary = {
        "config": {"target_modules": args.target_modules, "r": args.r, "alpha": args.alpha,
                   "head": args.head, "center": args.center,
                   "checkpoint": str(args.lora_checkpoint)},
        "n_samples": len(rows),
        "elapsed_seconds": elapsed,
        "global": stats_subset(rows),
        "por_longitud": {l: stats_subset([r for r in rows if r["longitud"] == l])
                          for l in sorted({r["longitud"] for r in rows})},
        "con_ntilde": stats_subset([r for r in rows if r["has_ntilde"]]),
        "sin_ntilde": stats_subset([r for r in rows if not r["has_ntilde"]]),
        "con_acento": stats_subset([r for r in rows if r["has_acento"]]),
        "sin_acento": stats_subset([r for r in rows if not r["has_acento"]]),
    }

    # Print resumen.
    g = summary["global"]
    print(f"\n=== GLOBAL ===  n={g['n']}  CER_micro={g['cer_micro']:.4f}  WER={g['wer']:.4f}")
    print(f"\n{'longitud':>10} {'n':>6} {'CER_micro':>10} {'WER':>10}")
    for L, s in summary["por_longitud"].items():
        print(f"{L:>10} {s['n']:>6} {s['cer_micro']:>10.4f} {s['wer']:>10.4f}")
    print(f"\nCon ñ:    n={summary['con_ntilde']['n']:>4}  CER={summary['con_ntilde']['cer_micro']:.4f}  WER={summary['con_ntilde']['wer']:.4f}")
    print(f"Sin ñ:    n={summary['sin_ntilde']['n']:>4}  CER={summary['sin_ntilde']['cer_micro']:.4f}  WER={summary['sin_ntilde']['wer']:.4f}")
    print(f"Con tilde: n={summary['con_acento']['n']:>4}  CER={summary['con_acento']['cer_micro']:.4f}  WER={summary['con_acento']['wer']:.4f}")
    print(f"Sin tilde: n={summary['sin_acento']['n']:>4}  CER={summary['sin_acento']['cer_micro']:.4f}  WER={summary['sin_acento']['wer']:.4f}")

    sum_path = args.outputs / "summary.json"
    with sum_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] summary guardado en {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
