"""Hyperparameter tuning: random search over the MLP's architecture/optimization
hyperparameters (layers, width, activation, dropout, learning rate, momentum, LR decay,
batch size, and the log-target switch), reusing train.py's `run_training` for every
trial (short epoch budget, early stopping still active, no checkpoint written) so
ranking uses exactly the same training code path as a standalone `train.py` run. The
winning configuration (by best validation MAE, km/h -- lower is better) is then
retrained for a full epoch budget and checkpointed.

    python tune.py --data raw-trafic-data.json --trials 20 --trial-epochs 20 --final-epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from dataset import RANDOM_SEED
from model import ACTIVATIONS
from train import run_training

ACTIVATION_OPTIONS = sorted(ACTIVATIONS)


def sample_config(rng: random.Random) -> dict:
    return {
        "num_layers": rng.choice([1, 2, 3, 4]),
        "hidden_dim": rng.choice([32, 64, 128, 256]),
        "activation": rng.choice(ACTIVATION_OPTIONS),
        "dropout": rng.choice([0.0, 0.1, 0.2, 0.3]),
        "lr": rng.choice([1e-3, 3e-3, 1e-2, 3e-2]),
        "momentum": rng.choice([0.0, 0.8, 0.9, 0.95]),
        "lr_decay": rng.choice([1.0, 0.99, 0.97]),
        "batch_size": rng.choice([128, 256, 512, 1024]),
        "log_target": rng.choice([True, False]),
    }


def build_args(base_args: argparse.Namespace, config: dict, epochs: int, checkpoint_dir: str | None, quiet: bool) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(base_args))
    for key, value in config.items():
        setattr(ns, key, value)
    ns.epochs = epochs
    ns.checkpoint_dir = checkpoint_dir
    ns.quiet = quiet
    return ns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="raw-trafic-data.json")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--trial-epochs", type=int, default=20, help="max epoch budget used to rank candidate configs (early stopping may end a trial sooner)")
    p.add_argument("--final-epochs", type=int, default=100, help="max epoch budget for retraining the winning config")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--early-stopping-patience", type=int, default=5, help="applied to every trial and the final retrain")
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="seeds the data split/model init; held fixed across trials so they differ only in hyperparameters")
    p.add_argument("--search-seed", type=int, default=RANDOM_SEED, help="seeds which configs random search samples")
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="tuning")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.search_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_args = argparse.Namespace(
        data=args.data,
        val_fraction=args.val_fraction,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )

    results = []
    best = None
    for trial in range(1, args.trials + 1):
        config = sample_config(rng)
        trial_args = build_args(base_args, config, args.trial_epochs, checkpoint_dir=None, quiet=True)
        result = run_training(trial_args)
        row = {**config, "trial": trial, "val_mae_kmh": result["best_val_mae_kmh"], "stopped_epoch": result["stopped_epoch"] or args.trial_epochs}
        results.append(row)
        print(f"trial {trial:3d}/{args.trials}  val_mae={row['val_mae_kmh']:.2f}km/h  stopped@{row['stopped_epoch']:3d}  {config}")
        if best is None or row["val_mae_kmh"] < best["val_mae_kmh"]:
            best = row

    with open(output_dir / "tuning_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nBest trial: {best}")

    with open(output_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"\nRetraining winning config for up to {args.final_epochs} epochs...")
    final_checkpoint_dir = output_dir / "best_checkpoint"
    final_config = {k: v for k, v in best.items() if k not in ("trial", "val_mae_kmh", "stopped_epoch")}
    final_args = build_args(base_args, final_config, args.final_epochs, checkpoint_dir=str(final_checkpoint_dir), quiet=False)
    final_result = run_training(final_args)
    print(f"Final model val MAE: {final_result['best_val_mae_kmh']:.2f} km/h. Checkpoint: {final_checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
