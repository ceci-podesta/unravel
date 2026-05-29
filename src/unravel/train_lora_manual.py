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

Para reanudar una corrida interrumpida:
    uv run python -m unravel.train_lora --resume-from experiments/M09/epoch_25_full.pt --outputs experiments/M09/ ...
    (el resto de los args debe coincidir con la corrida original)
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
from unravel.lora_manual import apply_lora_manual, lora_state_dict
from unravel.metrics import _normalize, cer_micro, wer
from unravel.split_dataset import SplitV2Dataset
from unravel.split_v3_dataset import SplitV3Dataset
from unravel.vocab import build_unified_vocab

HTR_REPO = Path.home() / "projects/HTR-best-practices"
DEFAULT_DATASET = Path.home() / "projects/unravel/data/split_v2.json"
DEFAULT_OUTPUTS = Path.home() / "projects/unravel/experiments/M01_lora_manual_replica_peft"


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


def _freeze_bn_running_stats(model: torch.nn.Module) -> None:
    """Pone BatchNorm en eval mode para que sus running stats NO se actualicen.

    Sin esto, durante el training de LoRA los running_mean / running_var de
    BN se adaptan al dominio nuevo en cada forward (al estar en train mode).
    Esa adaptación NO se persiste en el checkpoint (que solo guarda LoRA),
    así que al cargar el modelo se pierde, creando un gap entre el val_CER
    reportado durante training y el reproducible al cargar.
    """
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()


# ============================================================================
# Resume support (added 2026-05-06).
# Guardamos/cargamos buffers de BN explícitamente para protegernos del bug
# histórico de M03: si los running stats de BN se mueven y no se persisten,
# el checkpoint reproduce un val_CER distinto al del training.
# ============================================================================

def _bn_buffers_dict(model: torch.nn.Module) -> dict:
    """Extrae buffers de BN (running_mean/var/num_batches_tracked) por nombre."""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            out[name] = {
                "running_mean": m.running_mean.detach().cpu().clone() if m.running_mean is not None else None,
                "running_var": m.running_var.detach().cpu().clone() if m.running_var is not None else None,
                "num_batches_tracked": m.num_batches_tracked.detach().cpu().clone() if m.num_batches_tracked is not None else None,
            }
    return out


def _load_bn_buffers(model: torch.nn.Module, bn_dict: dict) -> tuple[list, list]:
    """Carga buffers de BN al modelo. Devuelve (loaded, missing_in_ckpt)."""
    loaded = []
    missing_in_ckpt = []
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if name not in bn_dict:
                missing_in_ckpt.append(name)
                continue
            saved = bn_dict[name]
            if saved["running_mean"] is not None and m.running_mean is not None:
                m.running_mean.copy_(saved["running_mean"].to(m.running_mean.device))
            if saved["running_var"] is not None and m.running_var is not None:
                m.running_var.copy_(saved["running_var"].to(m.running_var.device))
            if saved["num_batches_tracked"] is not None and m.num_batches_tracked is not None:
                m.num_batches_tracked.copy_(saved["num_batches_tracked"].to(m.num_batches_tracked.device))
            loaded.append(name)
    return loaded, missing_in_ckpt


def _verify_bn_buffers_match(model: torch.nn.Module, bn_dict: dict) -> list:
    """Verifica bitwise que los buffers de BN del modelo coinciden con bn_dict.
    Devuelve lista de mismatches (vacía si todo OK)."""
    mismatches = []
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if name not in bn_dict:
                mismatches.append(f"{name}: ausente en checkpoint")
                continue
            saved = bn_dict[name]
            for buf_name in ["running_mean", "running_var", "num_batches_tracked"]:
                saved_buf = saved[buf_name]
                model_buf = getattr(m, buf_name)
                if saved_buf is None and model_buf is None:
                    continue
                if saved_buf is None or model_buf is None:
                    mismatches.append(f"{name}.{buf_name}: uno es None y el otro no")
                    continue
                if not torch.equal(model_buf.detach().cpu(), saved_buf.cpu()):
                    mismatches.append(f"{name}.{buf_name}: valores no coinciden")
    return mismatches


def _save_full_checkpoint(path: Path, peft_model, optimizer, epoch: int,
                          best_val_cer: float, epochs_without_improvement: int,
                          history: dict, args) -> None:
    """Guarda checkpoint completo para reanudar. epoch guardado es 1-indexed
    (= número de epochs completados, también el sufijo del archivo)."""
    ckpt = {
        "lora_state": lora_state_dict(peft_model),
        "bn_buffers": _bn_buffers_dict(peft_model),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch + 1,
        "best_val_cer": best_val_cer,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "config": {
            "r": args.r,
            "alpha": args.alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": list(args.target_modules),
            "head": args.head,
            "seed": args.seed,
            "apply_aug": args.apply_aug,
            "aug_prob": args.aug_prob,
        },
    }
    torch.save(ckpt, path)


def _load_full_checkpoint(path: Path) -> dict:
    """Carga checkpoint completo. Solo se debe llamar con paths del propio
    experimento — no aceptar paths externos al repo."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_resume_args(args, ckpt_config: dict) -> None:
    """Verifica que los args actuales coinciden con los del checkpoint para
    los campos que afectan reconstrucción del modelo. Aborta si difieren."""
    must_match = ["r", "alpha", "lora_dropout", "target_modules", "head", "seed",
                  "apply_aug", "aug_prob"]
    mismatches = []
    for k in must_match:
        cur = getattr(args, k)
        if isinstance(cur, list):
            cur = list(cur)
        saved = ckpt_config.get(k)
        if cur != saved:
            mismatches.append(f"  {k}: actual={cur}  ckpt={saved}")
    if mismatches:
        msg = "[ERROR] Los args actuales no coinciden con los del checkpoint a reanudar:\n" + "\n".join(mismatches)
        raise SystemExit(msg)


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
    parser.add_argument(
        "--target-modules", nargs="+",
        default=["top.fnl.1", "top.cnn.1"],
        help="Paths dotted a los módulos LoRA. Ej: top.fnl.1 top.cnn.1 top.rec",
    )
    parser.add_argument(
        "--patience", type=int, default=0,
        help="Early stopping: cortar si val_CER no mejora en N epochs consecutivos. 0 = deshabilitado.",
    )
    parser.add_argument(
        "--save-every", type=int, default=0,
        help="Si N > 0, guarda un checkpoint cada N epochs como epoch_<N>.pt además del best. Default 0 = solo best.",
    )
    parser.add_argument(
        "--apply-aug", action="store_true",
        help="Si está, aplica augmentations on-the-fly al train (rotación, shift, dilatación, erosión). Default: sin aug (M01-M07 reproducibles).",
    )
    parser.add_argument(
        "--aug-prob", type=float, default=0.7,
        help="Probabilidad de aplicar al menos una aug por imagen del train (default 0.7). Solo aplica si --apply-aug está.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Si se setea, corta el train epoch a N batches (para dry-run)")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Si se setea, corta el val epoch a N batches (para dry-run)")
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Ruta a un epoch_<N>_full.pt (o best_full.pt) previamente guardado. "
             "Si se setea, reanuda el training desde el epoch siguiente con todo el "
             "estado restaurado (LoRA, BN, optimizer, RNG, history). Los args r, alpha, "
             "lora_dropout, target_modules, head, seed, apply_aug, aug_prob deben "
             "coincidir con los del checkpoint (validación interna).",
    )

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
    peft_model, lora_stats = apply_lora_manual(net, target_modules=args.target_modules, r=args.r, alpha=args.alpha, dropout=args.lora_dropout)
    print(f"[INFO] LoRA: {lora_stats['trainable_params']:,} / {lora_stats['total_params']:,} "
          f"trainable ({lora_stats['percent_trainable']:.2f}%)")

    # Datasets
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

    optimizer = torch.optim.Adam(
        [p for p in peft_model.parameters() if p.requires_grad], lr=args.lr,
    )

    # ---- Resume support: restaurar estado si --resume-from ----
    start_epoch = 0
    history: dict[str, list[float]] = {
        "train_loss": [], "train_cer_micro": [], "train_wer": [],
        "val_loss": [], "val_cer_micro": [], "val_wer": [],
    }
    best_val_cer = float("inf")
    epochs_without_improvement = 0

    if args.resume_from is not None:
        print(f"[INFO] reanudando desde checkpoint: {args.resume_from}")
        ckpt = _load_full_checkpoint(args.resume_from)
        _validate_resume_args(args, ckpt["config"])

        # Restaurar pesos LoRA (strict=False: el state guardado solo trae LoRA,
        # los pesos del IAM base ya están cargados al inicializar el modelo).
        missing_keys, unexpected_keys = peft_model.load_state_dict(
            ckpt["lora_state"], strict=False,
        )
        if unexpected_keys:
            raise SystemExit(f"[ERROR] keys inesperadas al cargar LoRA: {unexpected_keys}")

        # Restaurar buffers de BN (protección anti-bug-M03).
        loaded_bn, missing_bn = _load_bn_buffers(peft_model, ckpt["bn_buffers"])
        if missing_bn:
            raise SystemExit(f"[ERROR] BN faltantes en checkpoint: {missing_bn}")
        bn_check = _verify_bn_buffers_match(peft_model, ckpt["bn_buffers"])
        if bn_check:
            raise SystemExit(f"[ERROR] BN buffers no coinciden tras carga: {bn_check}")
        print(f"[INFO] BN buffers restaurados y verificados ({len(loaded_bn)} módulos)")

        # Restaurar optimizer.
        optimizer.load_state_dict(ckpt["optimizer_state"])

        # Restaurar RNG state.
        torch.set_rng_state(ckpt["rng_state_torch"])
        if torch.cuda.is_available() and ckpt["rng_state_cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_state_cuda"])

        # Restaurar metadatos del training.
        start_epoch = int(ckpt["epoch"])
        best_val_cer = float(ckpt["best_val_cer"])
        epochs_without_improvement = int(ckpt["epochs_without_improvement"])
        history = ckpt["history"]
        print(f"[INFO] resume desde epoch {start_epoch}, best_val_cer={best_val_cer:.4f}, "
              f"epochs_without_improvement={epochs_without_improvement}")

    # CSV writers (modo append si resume, write si fresh).
    step_path = args.outputs / "metrics_per_step.csv"
    epoch_path = args.outputs / "metrics_per_epoch.csv"
    csv_mode = "a" if args.resume_from else "w"
    write_header = args.resume_from is None
    step_f = step_path.open(csv_mode, encoding="utf-8", newline="")
    epoch_f = epoch_path.open(csv_mode, encoding="utf-8", newline="")
    step_writer = csv.DictWriter(step_f, fieldnames=["epoch", "step", "loss", "lr"])
    if write_header:
        step_writer.writeheader()
    epoch_writer = csv.DictWriter(
        epoch_f, fieldnames=["epoch", "train_loss", "train_cer_micro", "train_wer", "val_loss", "val_cer_micro", "val_wer"],
    )
    if write_header:
        epoch_writer.writeheader()

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # ---- Train ----
        peft_model.train()
        # BN en eval mode: solo LoRA aprende, los running stats quedan estables.
        _freeze_bn_running_stats(peft_model)
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

        # ---- Train CER/WER (sobre 15 batches del train para no duplicar tiempo) ----
        train_eval_metrics = evaluate_on_val(
            peft_model, train_loader, vocab, device, args.head, max_batches=15,
        )

        # ---- Validate ----
        val_metrics = evaluate_on_val(
            peft_model, val_loader, vocab, device, args.head,
            max_batches=args.max_val_batches,
        )

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
              f"val_WER={val_metrics['wer']:.4f} ===")

        # Checkpoint best + tracking de early stopping.
        if val_metrics["cer_micro"] < best_val_cer:
            best_val_cer = val_metrics["cer_micro"]
            epochs_without_improvement = 0
            checkpoint_path = args.outputs / "best_lora.pt"
            torch.save(lora_state_dict(peft_model), checkpoint_path)
            # Full checkpoint del best (permite resume desde el best).
            best_full_path = args.outputs / "best_full.pt"
            _save_full_checkpoint(best_full_path, peft_model, optimizer, epoch,
                                  best_val_cer, epochs_without_improvement, history, args)
            print(f"  [INFO] checkpoint guardado en {checkpoint_path} "
                  f"(val_CER={best_val_cer:.4f})")
        else:
            epochs_without_improvement += 1
            if args.patience > 0:
                print(f"  [INFO] sin mejora en val_CER ({epochs_without_improvement}/{args.patience} epochs)")
                if epochs_without_improvement >= args.patience:
                    print(f"[INFO] Early stopping en epoch {epoch}: sin mejora en val_CER por {args.patience} epochs consecutivos.")
                    break

        # Checkpoint periódico (si save_every > 0). Guardamos los dos formatos:
        # epoch_N.pt (solo LoRA, retro-compatible) y epoch_N_full.pt (para resume).
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            periodic_path = args.outputs / f"epoch_{epoch + 1}.pt"
            torch.save(lora_state_dict(peft_model), periodic_path)
            periodic_full_path = args.outputs / f"epoch_{epoch + 1}_full.pt"
            _save_full_checkpoint(periodic_full_path, peft_model, optimizer, epoch,
                                  best_val_cer, epochs_without_improvement, history, args)
            print(f"  [INFO] checkpoint periódico guardado en {periodic_path} "
                  f"(+ full en {periodic_full_path})")

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
