"""Training script para LoRA fine-tuning del modelo HTR.

Carga el modelo IAM, extiende vocabulario (+6 caracteres del español),
aplica LoRA sobre las cabezas finales, entrena sobre el split real
español (train 90%, val 10%, test intocado), y guarda métricas y
gráficos.

Uso:
    uv run python -m unravel.train_lora                    # config default
    uv run python -m unravel.train_lora --epochs 10 --lr 5e-4

Para un dry-run rápido (verificar que arranca todo):
    uv run python -m unravel.train_lora --epochs 1 --max-train-batches 2 --max-val-batches 2
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend no-interactivo
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from unravel.ctc_utils import collate_for_ctc, compute_ctc_loss
from unravel.extend_vocab import extend_model_vocabulary
from unravel.htr_model import HTRNet, default_arch_cfg
from unravel.lora_setup import apply_lora
from unravel.metrics import _normalize, cer_micro, wer
from unravel.train_dataset import SpanishHTRTrainDataset
from unravel.vocab import build_unified_vocab


HTR_REPO = Path.home() / "projects/HTR-best-practices"
DEFAULT_DATASET = Path.home() / "datasets/spanish-htr/datos_entrenamiento/PERFECT_CUT_a_z_1_9"
DEFAULT_OUTPUTS = Path.home() / "projects/unravel/experiments/06_lora_real_only"


def decode_ctc_greedy(seq: np.ndarray, icdict: dict[int, str], blank_id: int = 0) -> str:
    colapsada = [v for j, v in enumerate(seq) if j == 0 or v != seq[j - 1]]
    return "".join(icdict[t] for t in colapsada if t != blank_id and t in icdict)


def _move_batch_to_device(batch: dict, device: str) -> dict:
    return {
        **batch,
        "images": batch["images"].to(device),
        "targets": batch["targets"].to(device),
        "target_lengths": batch["target_lengths"].to(device),
    }


def evaluate_on_val(model, loader, vocab, device, head, max_batches=None) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    preds: list[str] = []
    refs: list[str] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = _move_batch_to_device(batch, device)
            output = model(batch["images"])
            loss = compute_ctc_loss(output, batch, blank_id=vocab["blank_id"], head=head)
            total_loss += loss.item()
            n_batches += 1

            logits = output[0] if isinstance(output, tuple) else output
            pred_indices = logits.argmax(2).permute(1, 0).cpu().numpy()
            for seq, palabra_real in zip(pred_indices, batch["palabras"]):
                pred = decode_ctc_greedy(seq, vocab["i2c"], vocab["blank_id"])
                preds.append(pred)
                refs.append(palabra_real)

    avg_loss = total_loss / max(n_batches, 1)
    filtered = [(p, r) for p, r in zip(preds, refs) if _normalize(r)]
    if filtered:
        preds_f = [p for p, _ in filtered]
        refs_f = [r for _, r in filtered]
        cer = cer_micro(preds_f, refs_f)
        wer_val = wer(preds_f, refs_f)
    else:
        cer = float("nan")
        wer_val = float("nan")
    return {"loss": avg_loss, "cer_micro": cer, "wer": wer_val}


def plot_curves(history: dict, outputs: Path) -> None:
    epochs = list(range(len(history["train_loss"])))
    # Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="train", marker="o")
    ax.plot(epochs, history["val_loss"], label="val", marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("CTC loss")
    ax.set_title("LoRA training — loss curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(outputs / "loss_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    # CER y WER
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["val_cer_micro"], label="CER_micro (val)", marker="o")
    ax.plot(epochs, history["val_wer"], label="WER (val)", marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("error rate")
    ax.set_title("LoRA training — CER and WER on validation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(outputs / "cer_wer_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning sobre dataset español real")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--weights", type=Path, default=HTR_REPO / "saved_models/htrnet.pt")
    parser.add_argument("--classes", type=Path, default=HTR_REPO / "saved_models/classes.npy")
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--r", type=int, default=8, help="rank LoRA")
    parser.add_argument("--alpha", type=int, default=16, help="alpha LoRA (típicamente 2*r)")
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--head", choices=["rnn", "cnn", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Si se setea, corta el train epoch a N batches (para dry-run)")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Si se setea, corta el val epoch a N batches (para dry-run)")
    args = parser.parse_args()

    args.outputs.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")
    print(f"[INFO] config: r={args.r} alpha={args.alpha} lr={args.lr} epochs={args.epochs} head={args.head}")

    # Vocabulario unificado
    vocab = build_unified_vocab(args.classes)
    n_iam_classes = vocab["n_classes"] - 6  # 79
    print(f"[INFO] vocab: {vocab['n_classes']} chars (= {n_iam_classes} IAM + 6 español)")

    # Cargar modelo IAM y extender vocabulario
    net = HTRNet(default_arch_cfg(), n_iam_classes + 1)  # +1 por blank → 80
    state = torch.load(args.weights, map_location=device, weights_only=True)
    net.load_state_dict(state, strict=True)
    net = extend_model_vocabulary(net, n_extra=6)
    net = net.to(device)

    # Aplicar LoRA
    peft_model, lora_stats = apply_lora(net, r=args.r, alpha=args.alpha, dropout=args.lora_dropout)
    print(f"[INFO] LoRA: {lora_stats['trainable_params']:,} / {lora_stats['total_params']:,} "
          f"trainable ({lora_stats['percent_trainable']:.2f}%)")

    # Datasets
    train_set = SpanishHTRTrainDataset(args.dataset, subset="train", seed=args.seed)
    val_set = SpanishHTRTrainDataset(args.dataset, subset="val", seed=args.seed)
    print(f"[INFO] train: {len(train_set)}  val: {len(val_set)}")

    def collate(b):
        return collate_for_ctc(b, vocab["c2i"])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    optimizer = torch.optim.Adam(
        [p for p in peft_model.parameters() if p.requires_grad], lr=args.lr,
    )

    # CSV writers
    step_path = args.outputs / "metrics_per_step.csv"
    epoch_path = args.outputs / "metrics_per_epoch.csv"
    step_f = step_path.open("w", encoding="utf-8", newline="")
    epoch_f = epoch_path.open("w", encoding="utf-8", newline="")
    step_writer = csv.DictWriter(step_f, fieldnames=["epoch", "step", "loss", "lr"])
    step_writer.writeheader()
    epoch_writer = csv.DictWriter(
        epoch_f, fieldnames=["epoch", "train_loss", "val_loss", "val_cer_micro", "val_wer"],
    )
    epoch_writer.writeheader()

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_cer_micro": [], "val_wer": [],
    }
    best_val_cer = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        # ---- Train ----
        peft_model.train()
        train_losses: list[float] = []
        for step, batch in enumerate(train_loader):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break
            batch = _move_batch_to_device(batch, device)
            output = peft_model(batch["images"])
            loss = compute_ctc_loss(output, batch, blank_id=vocab["blank_id"], head=args.head)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            step_writer.writerow({"epoch": epoch, "step": step,
                                   "loss": round(loss.item(), 6), "lr": args.lr})
            if step % 20 == 0:
                print(f"  E{epoch} step {step}/{len(train_loader)}  loss={loss.item():.4f}")

        avg_train_loss = sum(train_losses) / max(len(train_losses), 1)

        # ---- Validate ----
        val_metrics = evaluate_on_val(
            peft_model, val_loader, vocab, device, args.head,
            max_batches=args.max_val_batches,
        )
        epoch_writer.writerow({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_cer_micro": round(val_metrics["cer_micro"], 6),
            "val_wer": round(val_metrics["wer"], 6),
        })
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_cer_micro"].append(val_metrics["cer_micro"])
        history["val_wer"].append(val_metrics["wer"])

        print(f"=== Epoch {epoch}: train_loss={avg_train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_CER={val_metrics['cer_micro']:.4f}  "
              f"val_WER={val_metrics['wer']:.4f} ===")

        # Checkpoint best
        if val_metrics["cer_micro"] < best_val_cer:
            best_val_cer = val_metrics["cer_micro"]
            checkpoint_dir = args.outputs / "best_lora"
            peft_model.save_pretrained(str(checkpoint_dir))
            print(f"  [INFO] checkpoint guardado en {checkpoint_dir} "
                  f"(val_CER={best_val_cer:.4f})")

    step_f.close()
    epoch_f.close()
    elapsed = time.time() - start_time

    # Plots
    plot_curves(history, args.outputs)

    # Summary
    summary = {
        "config": {
            "r": args.r, "alpha": args.alpha, "lora_dropout": args.lora_dropout,
            "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
            "head": args.head, "seed": args.seed,
            "target_modules": lora_stats["target_modules"],
        },
        "lora_stats": {k: lora_stats[k] for k in ["trainable_params", "total_params", "percent_trainable"]},
        "history": history,
        "best_val_cer_micro": best_val_cer,
        "elapsed_seconds": elapsed,
        "n_train": len(train_set), "n_val": len(val_set),
    }
    with (args.outputs / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] tiempo total: {elapsed:.1f}s")
    print(f"[INFO] outputs en {args.outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
