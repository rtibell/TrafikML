"""Train the MLP traffic-speed regressor.

Example (a GTX 1080 or an Apple-Silicon GPU is picked up automatically if available):

    python train.py --data raw-trafic-data.json --epochs 100 --batch-size 256 \\
        --num-layers 3 --activation leaky_relu --log-target

The epoch-loop core (`run_epoch`) and full train-a-config routine (`run_training`) are
imported by tune.py so hyperparameter search reuses exactly this code path rather than
a parallel copy.

Implements the task's "Features to implement" ML paradigms:
- **Gradient descent** + **backpropagation**: `torch.optim.SGD` + `loss.backward()`.
- **MiniBatch**: `DataLoader(..., batch_size=args.batch_size, shuffle=True)`.
- **Early stopping**: training halts once validation MAE (km/h) hasn't improved by at
  least `--early-stopping-min-delta` for `--early-stopping-patience` epochs (0 disables
  it).
- **Adjustable learning rate and momentum**: `--lr`, `--momentum`, and an optional
  per-epoch multiplicative decay `--lr-decay` (`torch.optim.lr_scheduler.ExponentialLR`).
- **Random initialization of weights**: see model.py::TabularMLPRegressor._init_weights,
  made reproducible per run by `--seed` via `set_seed` below.

Regression loss is `nn.MSELoss` on the (optionally log1p'd, always standardized) target
-- see dataset.py::TargetTransform for the `--log-target` switch the task asks us to
evaluate. Every reported/early-stopping metric is converted back to km/h via
`TargetTransform.inverse_transform` first, so `--log-target` doesn't change the units
error is reported in.
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
    fit_scalers,
    load_dataframe,
    split_train_test,
)
from model import ACTIVATIONS, TabularMLPRegressor


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

    # Model architecture (task's "Instruction" CLI requirements)
    p.add_argument("--num-layers", type=int, default=2, help="number of hidden layers in the MLP (0 = plain linear regression)")
    p.add_argument("--hidden-dim", type=int, default=64, help="width of each hidden layer")
    p.add_argument("--activation", choices=sorted(ACTIVATIONS), default="relu")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--log-target", action="store_true",
        help="train on log1p(speed) instead of raw speed before standardizing -- see Solution.md",
    )

    # Gradient descent / optimization paradigms
    p.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--lr-decay", type=float, default=1.0, help="multiplicative per-epoch LR decay (1.0 = constant LR)")
    p.add_argument("--grad-clip", type=float, default=5.0)

    # Early stopping
    p.add_argument("--early-stopping-patience", type=int, default=10, help="stop after this many epochs with no val MAE improvement; 0 disables")
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-3, help="minimum val MAE (km/h) improvement to reset patience")

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--device", default=None, help="override auto-detected device, e.g. cuda:0, mps, or cpu")
    p.add_argument("--quiet", action="store_true", help="suppress per-epoch printing (used by tune.py)")
    return p.parse_args(argv)


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_epoch(model, loader, device, target_transform, mse, optimizer=None, grad_clip: float = 5.0) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    n_rows = 0
    n_batches = 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch_to_device(batch, device)
            pred_scaled = model(batch["features"])
            loss = mse(pred_scaled, batch["target_scaled"])

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            pred_kmh = target_transform.inverse_transform(pred_scaled.detach().cpu().numpy())
            true_kmh = batch["target_kmh"].cpu().numpy()
            err = pred_kmh - true_kmh
            abs_err_sum += float(np.abs(err).sum())
            sq_err_sum += float((err ** 2).sum())
            n_rows += true_kmh.shape[0]
            total_loss += loss.item()
            n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae_kmh": abs_err_sum / max(n_rows, 1),
        "rmse_kmh": float(np.sqrt(sq_err_sum / max(n_rows, 1))),
    }


def run_training(args: argparse.Namespace) -> dict:
    """Train one configuration end to end; returns a result dict with the best
    validation MAE, whether early stopping fired, the checkpoint path (if any) and the
    full epoch history. Shared by train.py's CLI and tune.py's search loop -- tune.py
    passes `checkpoint_dir=None` to skip writing checkpoints for trials it isn't
    keeping.
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

    scalers = fit_scalers(df, train_idx, args.log_target)
    train_ds = SegmentSpeedDataset(df, train_idx, scalers)
    val_ds = SegmentSpeedDataset(df, val_idx, scalers)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TabularMLPRegressor(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)

    mse = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    stopped_early = False
    stopped_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, scalers["target"], mse, optimizer, args.grad_clip)
        val_metrics = run_epoch(model, val_loader, device, scalers["target"], mse)
        dt = time.time() - t0

        if not args.quiet:
            print(
                f"epoch {epoch:3d}/{args.epochs} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"train_loss={train_metrics['loss']:.4f} train_mae={train_metrics['mae_kmh']:.2f}km/h "
                f"val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae_kmh']:.2f}km/h val_rmse={val_metrics['rmse_kmh']:.2f}km/h "
                f"[{dt:.1f}s]"
            )
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"], "train_mae_kmh": train_metrics["mae_kmh"], "train_rmse_kmh": train_metrics["rmse_kmh"],
            "val_loss": val_metrics["loss"], "val_mae_kmh": val_metrics["mae_kmh"], "val_rmse_kmh": val_metrics["rmse_kmh"],
        })

        is_best = val_metrics["mae_kmh"] < best_val_mae - args.early_stopping_min_delta
        if is_best:
            best_val_mae = val_metrics["mae_kmh"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if checkpoint_dir is not None:
            state = {
                "model_state": model.state_dict(),
                "args": vars(args),
                "target_transform": scalers["target"].state_dict(),
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
                print(f"Early stopping at epoch {epoch}: no val MAE improvement for {args.early_stopping_patience} epochs")
            break

    if checkpoint_dir is not None:
        with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return {
        "best_val_mae_kmh": best_val_mae,
        "history": history,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "stopped_early": stopped_early,
        "stopped_epoch": stopped_epoch,
    }


def main() -> None:
    args = parse_args()
    result = run_training(args)
    stopped_note = f" (early-stopped at epoch {result['stopped_epoch']})" if result["stopped_early"] else ""
    print(f"Training complete{stopped_note}. Best val MAE: {result['best_val_mae_kmh']:.2f} km/h. Checkpoints in {result['checkpoint_dir']}/")


if __name__ == "__main__":
    main()
