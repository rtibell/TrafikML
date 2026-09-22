"""Train the MLP traffic-status classifier.

Example (a GTX 1080 or an Apple-Silicon GPU is picked up automatically if available):

    python train.py --data raw-trafic-data.json --epochs 100 --batch-size 256 \\
        --num-layers 3 --activation leaky_relu

The epoch-loop core (`run_epoch`) and full train-a-config routine (`run_training`) are
imported by tune-rnd.py/tune-ga.py so hyperparameter search reuses exactly this code path rather than
a parallel copy.

Implements the task's "Features to implement" ML paradigms:
- **Gradient descent** + **backpropagation**: `torch.optim.SGD` + `loss.backward()`.
- **MiniBatch**: `DataLoader(..., batch_size=args.batch_size, shuffle=True)`.
- **Early stopping**: training halts once validation *balanced accuracy* (macro-average
  recall -- see `run_epoch` below) hasn't improved for `--early-stopping-patience`
  epochs (0 disables it). Balanced accuracy, not raw accuracy, drives model selection
  because the status target is heavily imbalanced (~86% freeflow): raw accuracy rewards
  a model that just always predicts freeflow, balanced accuracy does not.
- **Adjustable learning rate and momentum**: `--lr`, `--momentum`, and an optional
  per-epoch multiplicative decay `--lr-decay` (`torch.optim.lr_scheduler.ExponentialLR`).
- **Random initialization of weights**: see model.py::TabularMLPClassifier._init_weights,
  made reproducible per run by `--seed` via `set_seed` below.

Softmax: the model returns raw logits; `model.py::predict_status` applies
`torch.softmax` + `argmax` to turn them into a predicted status class, used below for
every accuracy computation. See model.py's module docstring for why training itself
uses `nn.CrossEntropyLoss` (which applies `log_softmax` internally) rather than a
separate `nn.Softmax` layer.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    RANDOM_SEED,
    SegmentSpeedDataset,
    compute_class_weights,
    fit_scalers,
    load_dataframe,
    split_train_test,
)
from features import STATUS_VALUES
from model import ACTIVATIONS, TabularMLPClassifier, predict_status


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device(preferred: str | None) -> torch.device:
    """cuda (e.g. a GTX 1080) > mps (Apple-Silicon GPU, e.g. an M2 MacBook) > cpu,
    unless `--device` overrides the choice explicitly.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f" ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return " (Apple GPU via Metal Performance Shaders)"
    return ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="raw-trafic-data.json")
    p.add_argument("--val-fraction", type=float, default=0.1, help="held out from the 90%% train split, for model selection")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100, help="maximum epochs; early stopping may end training sooner")

    # Model architecture
    p.add_argument("--num-layers", type=int, default=2, help="number of hidden layers in the MLP (0 = plain linear/softmax regression)")
    p.add_argument("--hidden-dim", type=int, default=64, help="width of each hidden layer")
    p.add_argument("--activation", choices=sorted(ACTIVATIONS), default="relu")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--class-weights", choices=["balanced", "none"], default="balanced",
        help="'balanced' (default) weights the cross-entropy loss inversely to training-set class frequency -- "
             "the status target is heavily imbalanced (~86%% freeflow, ~0.3%% impossible), see Solution.md",
    )

    # Gradient descent / optimization paradigms
    p.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--lr-decay", type=float, default=1.0, help="multiplicative per-epoch LR decay (1.0 = constant LR)")
    p.add_argument("--grad-clip", type=float, default=5.0)

    # Early stopping
    p.add_argument("--early-stopping-patience", type=int, default=10, help="stop after this many epochs with no val balanced-accuracy improvement; 0 disables")
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4, help="minimum val balanced-accuracy improvement to reset patience")

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--device", default=None, help="override auto-detected device, e.g. cuda:0, mps, or cpu")
    p.add_argument("--quiet", action="store_true", help="suppress per-epoch printing (used by tune-rnd.py/tune-ga.py)")
    return p.parse_args(argv)


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_epoch(model, loader, device, ce_loss, optimizer=None, grad_clip: float = 5.0) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    n_rows = 0
    n_batches = 0
    num_classes = model.net[-1].out_features
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)
    class_predicted = np.zeros(num_classes, dtype=np.int64)
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch_to_device(batch, device)
            logits = model(batch["features"])
            loss = ce_loss(logits, batch["status"])

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            pred, _ = predict_status(logits.detach())
            true = batch["status"]
            is_correct = (pred == true).cpu().numpy()
            correct += int(is_correct.sum())
            n_rows += true.shape[0]
            total_loss += loss.item()
            n_batches += 1

            true_np = true.cpu().numpy()
            pred_np = pred.cpu().numpy()
            for c in range(num_classes):
                mask = true_np == c
                class_total[c] += int(mask.sum())
                class_correct[c] += int(is_correct[mask].sum())
                class_predicted[c] += int((pred_np == c).sum())

    per_class_recall = {
        STATUS_VALUES[c]: (class_correct[c] / class_total[c] if class_total[c] > 0 else float("nan"))
        for c in range(num_classes)
    }
    # Macro-average recall ("balanced accuracy"): unlike raw accuracy, a model that
    # always predicts the majority class (freeflow, ~86% of rows -- see Solution.md)
    # scores ~0.25 here instead of ~0.86, so it -- not raw accuracy -- is what
    # model selection/early stopping optimizes for below.
    recalls = [v for v in per_class_recall.values() if not np.isnan(v)]
    balanced_accuracy = float(np.mean(recalls)) if recalls else float("nan")

    # Macro precision/F1/F2, using the same zero_division=0 convention as evaluate.py
    # (a class the model never predicts, or that never occurs, scores 0 rather than
    # NaN/undefined) -- computed here, not just in evaluate.py, so tune-ga.py's fitness
    # log can report these per candidate config without a second forward pass over the
    # data.
    def _safe_div(numerator: float, denominator: float) -> float:
        return float(numerator) / denominator if denominator > 0 else 0.0

    precision_per_class = {STATUS_VALUES[c]: _safe_div(class_correct[c], class_predicted[c]) for c in range(num_classes)}
    recall0_per_class = {STATUS_VALUES[c]: _safe_div(class_correct[c], class_total[c]) for c in range(num_classes)}
    f1_per_class = {}
    f2_per_class = {}
    for name in STATUS_VALUES:
        p, r = precision_per_class[name], recall0_per_class[name]
        f1_per_class[name] = _safe_div(2 * p * r, p + r)
        beta_sq = 4.0  # F2: recall weighted twice as heavily as precision
        f2_per_class[name] = _safe_div((1 + beta_sq) * p * r, beta_sq * p + r)

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": correct / max(n_rows, 1),
        "balanced_accuracy": balanced_accuracy,
        "per_class_recall": per_class_recall,
        "precision": float(np.mean(list(precision_per_class.values()))),
        "f1": float(np.mean(list(f1_per_class.values()))),
        "f2": float(np.mean(list(f2_per_class.values()))),
        "per_class_precision": precision_per_class,
        "per_class_f2": f2_per_class,
    }


def run_training(args: argparse.Namespace) -> dict:
    """Train one configuration end to end; returns a result dict with the best
    validation balanced accuracy, whether early stopping fired, the checkpoint path
    (if any) and the full epoch history. Shared by train.py's CLI and tune-rnd.py's/
    tune-ga.py's search loops -- both pass `checkpoint_dir=None` to skip writing
    checkpoints for trials they aren't keeping.
    """
    set_seed(args.seed)
    device = select_device(args.device)
    if not args.quiet:
        print(f"Using device: {device}{describe_device(device)}")

    df = load_dataframe(args.data)
    all_idx = np.arange(len(df))
    train_idx, test_idx = split_train_test(all_idx, test_size=0.1, seed=args.seed)
    if args.val_fraction > 0:
        train_idx, val_idx = split_train_test(train_idx, test_size=args.val_fraction, seed=args.seed)
    else:
        val_idx = test_idx  # monitor on test only if val disabled; never used for model selection then
    if not args.quiet:
        print(f"rows: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    scalers = fit_scalers(df, train_idx)
    train_ds = SegmentSpeedDataset(df, train_idx, scalers)
    val_ds = SegmentSpeedDataset(df, val_idx, scalers)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TabularMLPClassifier(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)

    if args.class_weights == "balanced":
        weights = compute_class_weights(df, train_idx).to(device)
    else:
        weights = None
    ce_loss = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_balanced_accuracy = -1.0
    best_val_metrics = None
    epochs_without_improvement = 0
    stopped_early = False
    stopped_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, ce_loss, optimizer, args.grad_clip)
        val_metrics = run_epoch(model, val_loader, device, ce_loss)
        dt = time.time() - t0

        if not args.quiet:
            print(
                f"epoch {epoch:3d}/{args.epochs} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} train_bal_acc={train_metrics['balanced_accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
                f"[{dt:.1f}s]"
            )
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"], "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"], "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_per_class_recall": val_metrics["per_class_recall"],
        })

        is_best = val_metrics["balanced_accuracy"] > best_val_balanced_accuracy + args.early_stopping_min_delta
        if is_best:
            best_val_balanced_accuracy = val_metrics["balanced_accuracy"]
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if checkpoint_dir is not None:
            state = {
                "model_state": model.state_dict(),
                "args": vars(args),
                "days_scaler": {"mean": scalers["days_until_holiday"].mean_.tolist(), "scale": scalers["days_until_holiday"].scale_.tolist()},
            }
            torch.save(state, checkpoint_dir / "last.pt")
            if is_best:
                torch.save(state, checkpoint_dir / "best.pt")

        scheduler.step()

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            if not args.quiet:
                print(f"Early stopping at epoch {epoch}: no val balanced-accuracy improvement for {args.early_stopping_patience} epochs")
            break

    if checkpoint_dir is not None:
        with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return {
        "best_val_balanced_accuracy": best_val_balanced_accuracy,
        "best_val_metrics": best_val_metrics,
        "history": history,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "stopped_early": stopped_early,
        "stopped_epoch": stopped_epoch,
    }


def main() -> None:
    args = parse_args()
    result = run_training(args)
    stopped_note = f" (early-stopped at epoch {result['stopped_epoch']})" if result["stopped_early"] else ""
    print(f"Training complete{stopped_note}. Best val balanced accuracy: {result['best_val_balanced_accuracy']:.4f}. Checkpoints in {result['checkpoint_dir']}/")


if __name__ == "__main__":
    main()
