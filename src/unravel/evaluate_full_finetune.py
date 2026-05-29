"""Evaluación de un modelo full fine-tuned (M10) sobre test_v2 u OOV.

Diferencias clave vs `evaluate_lora_manual.py`:
- Carga el state_dict COMPLETO del modelo (no solo LoRA adapters).
- No requiere args de LoRA (r, alpha, dropout, target_modules).
- Aplica init_custom de los 6 chars (necesario aunque después se sobrescriban
  todos los pesos al cargar state_dict — preserva consistencia de la firma).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from unravel.ctc_utils import collate_for_ctc, compute_ctc_loss
from unravel.extend_vocab import extend_model_vocabulary
from unravel.htr_model import HTRNet, default_arch_cfg
from unravel.metrics import _normalize, cer_micro, wer
from unravel.split_dataset import SplitV2Dataset
from unravel.vocab import build_unified_vocab

HTR_REPO = Path.home() / "projects/HTR-best-practices"
DEFAULT_DATASET = Path.home() / "projects/unravel/data/split_v2.json"


def decode_ctc_greedy(seq, icdict, blank_id=0):
    colapsada = [v for j, v in enumerate(seq) if j == 0 or v != seq[j - 1]]
    return "".join(icdict[t] for t in colapsada if t != blank_id and t in icdict)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación full FT M10")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path al state_dict guardado (best_model.pt típico)")
    parser.add_argument("--weights", type=Path, default=HTR_REPO / "saved_models/htrnet.pt",
                        help="Path al modelo IAM original (solo para construir la arquitectura)")
    parser.add_argument("--classes", type=Path, default=HTR_REPO / "saved_models/classes.npy")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head", choices=["rnn", "cnn", "both"], default="both")
    args = parser.parse_args()

    args.outputs.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")
    print(f"[INFO] checkpoint: {args.checkpoint}")

    # Construir el modelo con la misma arquitectura que en training
    vocab = build_unified_vocab(args.classes)
    print(f"[INFO] vocab unificado: {vocab['n_classes']} chars")
    n_iam_classes = vocab["n_classes"] - 6

    net = HTRNet(default_arch_cfg(), n_iam_classes + 1)
    base_state = torch.load(args.weights, map_location=device, weights_only=True)
    net.load_state_dict(base_state, strict=True)
    # Init custom (mismo que en training)
    init_from = {}
    SPANISH_ASCII = {'ñ': 'n', 'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u'}
    for sp, asc in SPANISH_ASCII.items():
        if sp in vocab["c2i"] and asc in vocab["c2i"]:
            init_from[vocab["c2i"][sp]] = vocab["c2i"][asc]
    net = extend_model_vocabulary(net, n_extra=6, init_from=init_from)

    # Cargar el state_dict completo del modelo entrenado
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    net.load_state_dict(state, strict=True)
    net = net.to(device)
    net.eval()
    print(f"[INFO] modelo cargado")

    # Dataset (test del archivo provisto)
    test_set = SplitV2Dataset(args.dataset, subset="test")
    print(f"[INFO] test set: {len(test_set)} imágenes")

    def collate(b):
        return collate_for_ctc(b, vocab["c2i"])

    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    # Evaluar
    preds, refs, longs, has_n, has_t = [], [], [], [], []
    start = time.time()
    n_batches = len(test_loader)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 20 == 0:
                print(f"  batch {i+1}/{n_batches}")
            images = batch["images"].to(device)
            output = net(images)
            logits = output[0] if isinstance(output, tuple) else output
            pred_indices = logits.argmax(2).permute(1, 0).cpu().numpy()
            for seq, palabra_real in zip(pred_indices, batch["palabras"]):
                pred = decode_ctc_greedy(seq, vocab["i2c"], vocab["blank_id"])
                preds.append(pred)
                refs.append(palabra_real)
                longs.append(len(palabra_real))
                has_n.append("ñ" in palabra_real)
                has_t.append(any(c in palabra_real for c in "áéíóú"))
    elapsed = time.time() - start
    print(f"[INFO] {len(preds)} predicciones, tiempo: {elapsed:.1f}s")

    # Guardar predicciones
    pred_path = args.outputs / "predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["longitud", "real", "predicho", "edit_dist", "cer_palabra", "error_word", "has_ntilde", "has_acento"])
        for L, real, pred, n, t in zip(longs, refs, preds, has_n, has_t):
            from unravel.metrics import _normalize
            from unravel.metrics import cer_micro as _cer
            ref_n = _normalize(real)
            pred_n = _normalize(pred)
            if ref_n:
                cer_w = _cer([pred_n], [ref_n])
                err_w = 1 if pred_n != ref_n else 0
            else:
                cer_w, err_w = float("nan"), 1
            ed = abs(len(ref_n) - len(pred_n))
            w.writerow([L, real, pred, ed, round(cer_w, 4), err_w, int(n), int(t)])
    print(f"[INFO] predictions guardadas en {pred_path}")

    # Métricas globales y por longitud
    filtered = [(p, r, L, n, t) for p, r, L, n, t in zip(preds, refs, longs, has_n, has_t) if _normalize(r)]
    if filtered:
        preds_f = [_normalize(p) for p, _, _, _, _ in filtered]
        refs_f = [_normalize(r) for _, r, _, _, _ in filtered]
        cer_g = cer_micro(preds_f, refs_f)
        wer_g = wer(preds_f, refs_f)
    else:
        cer_g = wer_g = float("nan")

    print(f"=== GLOBAL ===  n={len(preds)}  CER_micro={cer_g:.4f}  WER={wer_g:.4f}")

    # Por longitud
    print(f"  {'longitud':>8} {'n':>6} {'CER_micro':>10} {'WER':>10}")
    by_len = {}
    for p, r, L, _, _ in filtered:
        by_len.setdefault(L, []).append((p, r))
    for L in sorted(by_len):
        p_l = [x[0] for x in by_len[L]]
        r_l = [x[1] for x in by_len[L]]
        c = cer_micro(p_l, r_l)
        w_v = wer(p_l, r_l)
        print(f"  {L:>8} {len(p_l):>6} {c:>10.4f} {w_v:>10.4f}")

    # Caracteres especiales
    with_n = [(p, r) for p, r, _, n, _ in filtered if n]
    without_n = [(p, r) for p, r, _, n, _ in filtered if not n]
    with_t = [(p, r) for p, r, _, _, t in filtered if t]
    without_t = [(p, r) for p, r, _, _, t in filtered if not t]
    def m(items):
        if not items: return float("nan"), float("nan")
        ps, rs = zip(*items)
        return cer_micro(list(ps), list(rs)), wer(list(ps), list(rs))
    cn, wn = m(with_n); csn, wsn = m(without_n)
    ct, wt = m(with_t); cst, wst = m(without_t)
    print(f"Con ñ:    n={len(with_n):>4}  CER={cn:.4f}  WER={wn:.4f}")
    print(f"Sin ñ:    n={len(without_n):>4}  CER={csn:.4f}  WER={wsn:.4f}")
    print(f"Con tilde: n={len(with_t):>4}  CER={ct:.4f}  WER={wt:.4f}")
    print(f"Sin tilde: n={len(without_t):>4}  CER={cst:.4f}  WER={wst:.4f}")

    summary = {
        "config": {"checkpoint": str(args.checkpoint), "head": args.head},
        "n_samples": len(preds),
        "elapsed_seconds": elapsed,
        "global": {"n": len(preds), "cer_micro": cer_g, "wer": wer_g},
        "por_longitud": {
            str(L): {
                "n": len(by_len[L]),
                "cer_micro": cer_micro([x[0] for x in by_len[L]], [x[1] for x in by_len[L]]),
                "wer": wer([x[0] for x in by_len[L]], [x[1] for x in by_len[L]]),
            } for L in sorted(by_len)
        },
        "con_ntilde": {"n": len(with_n), "cer_micro": cn, "wer": wn},
        "sin_ntilde": {"n": len(without_n), "cer_micro": csn, "wer": wsn},
        "con_acento": {"n": len(with_t), "cer_micro": ct, "wer": wt},
        "sin_acento": {"n": len(without_t), "cer_micro": cst, "wer": wst},
    }
    with (args.outputs / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(f"[INFO] summary guardado en {args.outputs / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
