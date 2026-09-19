"""Train the CNN traffic speed regressor.

Example (a GTX 1080 is picked up automatically via CUDA if available):

    python train.py --data raw-trafic-data.json --epochs 30 --batch-size 256

The epoch-loop core (`run_epoch`) and full train-a-config routine (`run_training`) are
imported by tune.py so hyperparameter search reuses exactly this code path rather than
a parallel copy.
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
from model import TabularCNNRegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="raw-trafic-data.json")
    p.add_argument("--val-fraction", type=float, default=0.1, help="held out from the 90%% train split, for model selection")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--conv-channels", default="32,64", help="comma-separated Conv1d output channels, e.g. 32,64,64")
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--status-embed-dim", type=int, default=4)
    p.add_argument("--holiday-type-embed-dim", type=int, default=2)
    p.add_argument("--holiday-num-embed-dim", type=int, default=4)
    p.add_argument("--log-target", action="store_true", help="train on log1p(speed) instead of raw speed (see Solution.md)")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--device", default=None, help="override auto-detected device, e.g. cuda:0 or cpu")
    p.add_argument("--quiet", action="store_true", help="suppress per-epoch printing (used by tune.py)")
    return p.parse_args(argv)


def parse_conv_channels(spec: str) -> tuple[int, ...]:
    return tuple(int(x) for x in spec.split(","))


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_epoch(model, loader, device, target_transform, mse, optimizer=None, grad_clip: float = 5.0) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    abs_err_sum = 0.0
    n_rows = 0
    n_batches = 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch_to_device(batch, device)
            pred_scaled = model(batch["cont"], batch["status"], batch["holiday_type"], batch["holiday_num"])
            loss = mse(pred_scaled, batch["target_scaled"])

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            pred_kmh = target_transform.inverse_transform(pred_scaled.detach().cpu().numpy())
            true_kmh = batch["target_kmh"].cpu().numpy()
            abs_err_sum += float(np.abs(pred_kmh - true_kmh).sum())
            n_rows += true_kmh.shape[0]
            total_loss += loss.item()
            n_batches += 1

    return {"loss": total_loss / max(n_batches, 1), "mae_kmh": abs_err_sum / max(n_rows, 1)}


def run_training(args: argparse.Namespace) -> dict:
    """Train one configuration end to end; returns a result dict with the best
    validation MAE, the checkpoint path (if any) and the full epoch history. Shared by
    train.py's CLI and tune.py's search loop -- tune.py passes `checkpoint_dir=None`
    to skip writing checkpoints for trials it isn't keeping.
    """
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not args.quiet:
        name = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
        print(f"Using device: {device}{name}")

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

    model = TabularCNNRegressor(
        conv_channels=parse_conv_channels(args.conv_channels),
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        status_embed_dim=args.status_embed_dim,
        holiday_type_embed_dim=args.holiday_type_embed_dim,
        holiday_num_embed_dim=args.holiday_num_embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, scalers["target"], mse, optimizer, args.grad_clip)
        val_metrics = run_epoch(model, val_loader, device, scalers["target"], mse)
        dt = time.time() - t0

        if not args.quiet:
            print(
                f"epoch {epoch:3d}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.4f} train_mae={train_metrics['mae_kmh']:.2f}km/h "
                f"val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae_kmh']:.2f}km/h "
                f"[{dt:.1f}s]"
            )
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})

        is_best = val_metrics["mae_kmh"] < best_val_mae
        if is_best:
            best_val_mae = val_metrics["mae_kmh"]

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

    if checkpoint_dir is not None:
        with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return {"best_val_mae_kmh": best_val_mae, "history": history, "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None}


def main() -> None:
    args = parse_args()
    result = run_training(args)
    print(f"Training complete. Best val MAE: {result['best_val_mae_kmh']:.2f} km/h. Checkpoints in {result['checkpoint_dir']}/")


if __name__ == "__main__":
    main()
