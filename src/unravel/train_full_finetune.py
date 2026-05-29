"""Training script para FULL fine-tuning del modelo HTR (M10).

Diferencias clave vs `train_lora_manual.py`:
- Todos los pesos son entrenables (no se aplica LoRA).
- BN se congela durante training.
- Inicialización custom de los 6 chars españoles (ñ←n, á←a, é←e, í←i, ó←o, ú←u)
  para evitar shock de chars random durante full FT.
- Linear warmup del lr opcional.
- Gradient clipping defensivo.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from unravel.ctc_utils import collate_for_ctc, compute_ctc_loss
from unravel.extend_vocab import extend_model_vocabulary
from unravel.htr_model import HTRNet, default_arch_cfg
from unravel.metrics import _normalize, cer_micro, wer
from unravel.split_dataset import SplitV2Dataset
from unravel.split_v3_dataset import SplitV3Dataset
from unravel.vocab import build_unified_vocab

HTR_REPO = Path.home() / "projects/HTR-best-practices"
DEFAULT_DATASET = Path.home() / "projects/unravel/data/split_v2.json"
DEFAULT_OUTPUTS = Path.home() / "projects/unravel/experiments/M10_full_finetune"

# Mapping de chars españoles a chars ASCII similares para inicialización.
# Evita el "shock random" cuando se extiende el vocab: en lugar de pesos
# Kaiming uniform random, copiamos del char ASCII más cercano.
SPANISH_ASCII_INIT = {
    'ñ': 'n', 'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
}


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
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="train", marker="o")
    ax.plot(epochs, history["val_loss"], label="val", marker="o")
    ax.set_xlabel("epoch"); ax.set_ylabel("CTC loss")
    ax.set_title("Full FT — loss curves"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(outputs / "loss_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["val_cer_micro"], label="CER_micro (val)", marker="o")
    ax.plot(epochs, history["val_wer"], label="WER (val)", marker="o")
    ax.set_xlabel("epoch"); ax.set_ylabel("error rate")
    ax.set_title("Full FT — CER and WER on validation"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(outputs / "cer_wer_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _freeze_bn_running_stats(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()


def make_warmup_lambda(warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        return 1.0
    return lr_lambda


def build_init_from_map(vocab: dict) -> dict[int, int]:
    """Construye el dict {new_idx: source_idx} para inicializar los chars
    españoles a partir de chars ASCII similares.

    Usa SPANISH_ASCII_INIT y el c2i del vocab unificado. Solo incluye pares
    donde tanto el char español como el ASCII están en c2i.
    """
    c2i = vocab["c2i"]
    init_from = {}
    for spanish_char, ascii_char in SPANISH_ASCII_INIT.items():
        if spanish_char in c2i and ascii_char in c2i:
            init_from[c2i[spanish_char]] = c2i[ascii_char]
    return init_from


def _save_full_checkpoint(path, model, optimizer, scheduler, epoch, best_val_cer,
                          epochs_without_improvement, history, args, optimizer_step_count):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "optimizer_step_count": optimizer_step_count,
        "epoch": epoch + 1,
        "best_val_cer": best_val_cer,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "config": {
            "lr": args.lr, "head": args.head, "seed": args.seed,
            "batch_size": args.batch_size, "grad_accum_steps": args.grad_accum_steps,
            "clip_grad_norm": args.clip_grad_norm, "warmup_steps": args.warmup_steps,
            "init_custom_chars": args.init_custom_chars,
            "apply_aug": args.apply_aug, "aug_prob": args.aug_prob,
        },
    }
    torch.save(ckpt, path)


def _load_full_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_resume_args(args, ckpt_config):
    must_match = ["lr", "head", "seed", "batch_size", "grad_accum_steps",
                  "clip_grad_norm", "warmup_steps", "init_custom_chars",
                  "apply_aug", "aug_prob"]
    mismatches = []
    for k in must_match:
        cur = getattr(args, k)
        saved = ckpt_config.get(k)
        if cur != saved:
            mismatches.append(f"  {k}: actual={cur}  ckpt={saved}")
    if mismatches:
        raise SystemExit("[ERROR] Args no coinciden con checkpoint:\n" + "\n".join(mismatches))


def main() -> int:
    parser = argparse.ArgumentParser(description="FULL fine-tuning sobre dataset español real (M10)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--weights", type=Path, default=HTR_REPO / "saved_models/htrnet.pt")
    parser.add_argument("--classes", type=Path, default=HTR_REPO / "saved_models/classes.npy")
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Optimizer steps para linear warmup. 0 = sin warmup.")
    parser.add_argument("--init-custom-chars", action="store_true",
                        help="Si está, inicializa los 6 chars españoles copiando de ASCII similares (ñ←n, etc.) en lugar de Kaiming uniform random.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head", choices=["rnn", "cnn", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--apply-aug", action="store_true")
    parser.add_argument("--aug-prob", type=float, default=0.7)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)

    args = parser.parse_args()
    args.outputs.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    effective_bs = args.batch_size * args.grad_accum_steps
    print(f"[INFO] device: {device}")
    print(f"[INFO] config: lr={args.lr} epochs={args.epochs} head={args.head}")
    print(f"[INFO] batch_size={args.batch_size}, grad_accum={args.grad_accum_steps}, effective_batch={effective_bs}")
    print(f"[INFO] clip_grad_norm={args.clip_grad_norm}")
    print(f"[INFO] warmup_steps={args.warmup_steps}")
    print(f"[INFO] init_custom_chars={args.init_custom_chars}")

    vocab = build_unified_vocab(args.classes)
    n_iam_classes = vocab["n_classes"] - 6
    print(f"[INFO] vocab: {vocab['n_classes']} chars (= {n_iam_classes} IAM + 6 español)")

    net = HTRNet(default_arch_cfg(), n_iam_classes + 1)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    net.load_state_dict(state, strict=True)

    # Inicialización custom de los 6 chars españoles (si --init-custom-chars).
    init_from = None
    if args.init_custom_chars:
        init_from = build_init_from_map(vocab)
        print(f"[INFO] init_from (español → ASCII): "
              f"{ {vocab['i2c'][k]: vocab['i2c'][v] for k, v in init_from.items()} }")
        print(f"[INFO] init_from (índices): {init_from}")
    net = extend_model_vocabulary(net, n_extra=6, init_from=init_from)
    net = net.to(device)

    for p in net.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in net.parameters())
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[INFO] Full FT: {n_trainable:,} / {n_params:,} trainable (100%)")
    print(f"[INFO] BN running stats CONGELADAS durante training (M04+ pattern)")

    train_set = SplitV3Dataset(args.dataset, subset="train", apply_aug=args.apply_aug, prob_apply=args.aug_prob)
    val_set = SplitV2Dataset(args.dataset, subset="val")
    print(f"[INFO] train: {len(train_set)}  val: {len(val_set)}")

    def collate(b):
        return collate_for_ctc(b, vocab["c2i"])

    def _worker_init_fn(worker_id):
        torch.manual_seed(args.seed + worker_id)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        worker_init_fn=_worker_init_fn if args.apply_aug else None,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_warmup_lambda(args.warmup_steps))

    optimizer_step_count = 0
    start_epoch = 0
    history = {"train_loss": [], "train_cer_micro": [], "train_wer": [],
               "val_loss": [], "val_cer_micro": [], "val_wer": []}
    best_val_cer = float("inf")
    epochs_without_improvement = 0

    if args.resume_from is not None:
        print(f"[INFO] reanudando desde checkpoint: {args.resume_from}")
        ckpt = _load_full_checkpoint(args.resume_from)
        _validate_resume_args(args, ckpt["config"])
        net.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        optimizer_step_count = ckpt.get("optimizer_step_count", 0)
        torch.set_rng_state(ckpt["rng_state_torch"])
        if torch.cuda.is_available() and ckpt["rng_state_cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_state_cuda"])
        start_epoch = int(ckpt["epoch"])
        best_val_cer = float(ckpt["best_val_cer"])
        epochs_without_improvement = int(ckpt["epochs_without_improvement"])
        history = ckpt["history"]
        print(f"[INFO] resume desde epoch {start_epoch}, best_val_cer={best_val_cer:.4f}")

    step_path = args.outputs / "metrics_per_step.csv"
    epoch_path = args.outputs / "metrics_per_epoch.csv"
    csv_mode = "a" if args.resume_from else "w"
    write_header = args.resume_from is None
    step_f = step_path.open(csv_mode, encoding="utf-8", newline="")
    epoch_f = epoch_path.open(csv_mode, encoding="utf-8", newline="")
    step_writer = csv.DictWriter(step_f, fieldnames=["epoch", "step", "loss", "lr"])
    if write_header: step_writer.writeheader()
    epoch_writer = csv.DictWriter(epoch_f, fieldnames=["epoch", "train_loss", "train_cer_micro", "train_wer", "val_loss", "val_cer_micro", "val_wer"])
    if write_header: epoch_writer.writeheader()

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        net.train()
        _freeze_bn_running_stats(net)

        train_losses = []
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break

            batch = _move_batch_to_device(batch, device)
            output = net(batch["images"])
            loss = compute_ctc_loss(output, batch, blank_id=vocab["blank_id"], head=args.head)

            (loss / args.grad_accum_steps).backward()

            if (step + 1) % args.grad_accum_steps == 0:
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step_count += 1

            current_lr = scheduler.get_last_lr()[0]
            train_losses.append(loss.item())
            step_writer.writerow({"epoch": epoch, "step": step,
                                  "loss": round(loss.item(), 6),
                                  "lr": round(current_lr, 8)})

            if step % 20 == 0:
                print(f"  E{epoch} step {step}/{len(train_loader)}  loss={loss.item():.4f}  lr={current_lr:.2e}")

        if (step + 1) % args.grad_accum_steps != 0:
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_step_count += 1

        avg_train_loss = sum(train_losses) / max(len(train_losses), 1)

        train_eval_metrics = evaluate_on_val(net, train_loader, vocab, device, args.head, max_batches=15)
        val_metrics = evaluate_on_val(net, val_loader, vocab, device, args.head, max_batches=args.max_val_batches)

        epoch_writer.writerow({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "train_cer_micro": round(train_eval_metrics["cer_micro"], 6),
            "train_wer": round(train_eval_metrics["wer"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_cer_micro": round(val_metrics["cer_micro"], 6),
            "val_wer": round(val_metrics["wer"], 6),
        })
        history["train_loss"].append(avg_train_loss)
        history["train_cer_micro"].append(train_eval_metrics["cer_micro"])
        history["train_wer"].append(train_eval_metrics["wer"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_cer_micro"].append(val_metrics["cer_micro"])
        history["val_wer"].append(val_metrics["wer"])

        print(f"=== Epoch {epoch}: train_loss={avg_train_loss:.4f}  "
              f"train_CER={train_eval_metrics['cer_micro']:.4f}  "
              f"train_WER={train_eval_metrics['wer']:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_CER={val_metrics['cer_micro']:.4f}  "
              f"val_WER={val_metrics['wer']:.4f}  "
              f"opt_steps={optimizer_step_count} ===")

        if val_metrics["cer_micro"] < best_val_cer:
            best_val_cer = val_metrics["cer_micro"]
            epochs_without_improvement = 0
            torch.save(net.state_dict(), args.outputs / "best_model.pt")
            _save_full_checkpoint(args.outputs / "best_full.pt", net, optimizer, scheduler, epoch,
                                  best_val_cer, epochs_without_improvement, history, args, optimizer_step_count)
            print(f"  [INFO] checkpoint guardado en best_model.pt (val_CER={best_val_cer:.4f})")
        else:
            epochs_without_improvement += 1
            if args.patience > 0:
                print(f"  [INFO] sin mejora en val_CER ({epochs_without_improvement}/{args.patience} epochs)")
                if epochs_without_improvement >= args.patience:
                    print(f"[INFO] Early stopping en epoch {epoch}")
                    break

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(net.state_dict(), args.outputs / f"epoch_{epoch + 1}.pt")
            _save_full_checkpoint(args.outputs / f"epoch_{epoch + 1}_full.pt", net, optimizer, scheduler, epoch,
                                  best_val_cer, epochs_without_improvement, history, args, optimizer_step_count)
            print(f"  [INFO] checkpoint periódico guardado en epoch_{epoch+1}.pt (+ full)")

    step_f.close()
    epoch_f.close()
    elapsed = time.time() - start_time
    plot_curves(history, args.outputs)

    summary = {
        "config": {
            "lr": args.lr, "batch_size": args.batch_size, "grad_accum_steps": args.grad_accum_steps,
            "effective_batch": effective_bs, "clip_grad_norm": args.clip_grad_norm,
            "warmup_steps": args.warmup_steps, "init_custom_chars": args.init_custom_chars,
            "epochs": args.epochs, "head": args.head, "seed": args.seed,
        },
        "trainable_stats": {"trainable_params": n_trainable, "total_params": n_params, "percent_trainable": 100.0},
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
