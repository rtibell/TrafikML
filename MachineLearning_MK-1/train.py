"""Train the seq2seq traffic speed/status forecaster.

Example (GTX 1080 will be picked up automatically via CUDA if available):

    python train.py --data raw-trafic-data.json --epochs 30 --batch-size 256
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
    build_windows,
    fit_scalers,
    load_dataframe,
    split_train_test,
    TrafficWindowDataset,
)
from model import Seq2SeqTrafficForecaster


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="raw-trafic-data.json")
    p.add_argument("--t-in", type=int, default=60, help="minutes of history fed to the encoder")
    p.add_argument("--t-out", type=int, default=15, help="minutes ahead to forecast")
    p.add_argument("--stride", type=int, default=5, help="step between consecutive windows of the same segment")
    p.add_argument("--val-fraction", type=float, default=0.1, help="held out from the 90% train split, for early stopping")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seg-embed-dim", type=int, default=16)
    p.add_argument("--status-loss-weight", type=float, default=0.3)
    p.add_argument("--teacher-forcing-ratio", type=float, default=0.5)
    p.add_argument("--tf-decay-to-zero", action="store_true", help="linearly decay teacher forcing to 0 over training")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="mixed precision (limited benefit on Pascal/GTX1080, saves VRAM)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--device", default=None, help="override auto-detected device, e.g. cuda:0 or cpu")
    return p.parse_args()


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_epoch(model, loader, device, optimizer, mse, ce, status_weight, teacher_forcing_ratio, grad_clip, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_speed_loss = total_status_loss = 0.0
    n_batches = 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                out = model(
                    batch,
                    target_speed_scaled=batch["target_speed_scaled"],
                    target_status=batch["target_status"],
                    teacher_forcing_ratio=teacher_forcing_ratio if is_train else 0.0,
                )
                speed_loss = mse(out["speed_scaled"], batch["target_speed_scaled"])
                status_loss = ce(
                    out["status_logits"].reshape(-1, out["status_logits"].shape[-1]),
                    batch["target_status"].reshape(-1),
                )
                loss = speed_loss + status_weight * status_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            total_loss += loss.item()
            total_speed_loss += speed_loss.item()
            total_status_loss += status_loss.item()
            n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "speed_loss": total_speed_loss / max(n_batches, 1),
        "status_loss": total_status_loss / max(n_batches, 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    df = load_dataframe(args.data)
    windows, seg_index, group_cache = build_windows(df, args.t_in, args.t_out, args.stride)
    print(f"Built {len(windows)} windows across {len(seg_index)} segments")

    train_windows, test_windows = split_train_test(windows, test_size=0.1, seed=args.seed)
    if args.val_fraction > 0:
        train_windows, val_windows = split_train_test(train_windows, test_size=args.val_fraction, seed=args.seed)
    else:
        val_windows = test_windows  # monitor on test loss only if val disabled; never used for model selection then
    print(f"train={len(train_windows)} val={len(val_windows)} test={len(test_windows)}")

    scalers = fit_scalers(train_windows + val_windows, group_cache, args.t_in, args.t_out)

    train_ds = TrafficWindowDataset(train_windows, group_cache, args.t_in, args.t_out, scalers)
    val_ds = TrafficWindowDataset(val_windows, group_cache, args.t_in, args.t_out, scalers)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = Seq2SeqTrafficForecaster(
        num_segments=len(seg_index),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        cell=args.cell,
        seg_embed_dim=args.seg_embed_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    amp_scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        tf_ratio = args.teacher_forcing_ratio
        if args.tf_decay_to_zero:
            tf_ratio *= max(0.0, 1.0 - (epoch - 1) / max(args.epochs - 1, 1))

        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, optimizer, mse, ce, args.status_loss_weight, tf_ratio, args.grad_clip, amp_scaler)
        val_metrics = run_epoch(model, val_loader, device, None, mse, ce, args.status_loss_weight, 0.0, args.grad_clip, None)
        dt = time.time() - t0

        print(
            f"epoch {epoch:3d}/{args.epochs} tf={tf_ratio:.2f} "
            f"train_loss={train_metrics['loss']:.4f} (speed={train_metrics['speed_loss']:.4f} status={train_metrics['status_loss']:.4f}) "
            f"val_loss={val_metrics['loss']:.4f} (speed={val_metrics['speed_loss']:.4f} status={val_metrics['status_loss']:.4f}) "
            f"[{dt:.1f}s]"
        )
        history.append({"epoch": epoch, "tf_ratio": tf_ratio, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})

        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
        state = {
            "model_state": model.state_dict(),
            "args": vars(args),
            "seg_index": seg_index.id_to_idx,
            "scalers": {k: {"mean": v.mean_.tolist(), "scale": v.scale_.tolist()} for k, v in scalers.items()},
        }
        torch.save(state, checkpoint_dir / "last.pt")
        if is_best:
            torch.save(state, checkpoint_dir / "best.pt")

    with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}. Checkpoints in {checkpoint_dir}/")


if __name__ == "__main__":
    main()
