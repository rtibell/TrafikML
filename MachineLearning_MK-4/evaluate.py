"""Evaluate a trained checkpoint on the held-out 10% test split.

The test split is *recomputed* deterministically (same seed as training, read back
from the checkpoint's saved training args) rather than stored, so this only reproduces
the original held-out rows if `raw-trafic-data.json` is unchanged since training.

    python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FixedScaler, SegmentSpeedDataset, TargetTransform, load_dataframe, split_train_test
from model import TabularMLPRegressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data", default=None, help="defaults to the path stored in the checkpoint's training args")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="checkpoints/eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from train import select_device  # local import avoids a hard dependency for callers that only need the model

    device = select_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    data_path = args.data or train_args["data"]

    df = load_dataframe(data_path)
    all_idx = np.arange(len(df))
    _, test_idx = split_train_test(all_idx, test_size=0.1, seed=train_args["seed"])
    print(f"Evaluating on {len(test_idx)} held-out rows")

    target_transform = TargetTransform.from_checkpoint(**ckpt["target_transform"])
    scalers = {
        "days_until_holiday": FixedScaler(**ckpt["days_scaler"]),
        "target": target_transform,
    }
    test_ds = SegmentSpeedDataset(df, test_idx, scalers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = TabularMLPRegressor(
        num_layers=train_args["num_layers"],
        hidden_dim=train_args["hidden_dim"],
        activation=train_args["activation"],
        dropout=train_args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred_scaled = model(batch["features"])
            pred_kmh = target_transform.inverse_transform(pred_scaled.cpu().numpy())
            all_pred.append(pred_kmh)
            all_true.append(batch["target_kmh"].cpu().numpy())

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    err = pred - true

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.clip(true, 1.0, None)  # avoid blow-up near standstill (speed ~ 0)
    mape = float(np.mean(np.abs(err) / denom) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"Test set (n={len(true)}): MAE={mae:.2f} km/h  RMSE={rmse:.2f} km/h  MAPE={mape:.2f}%  R2={r2:.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mae_kmh", "rmse_kmh", "mape_pct", "r2", "n"])
        writer.writeheader()
        writer.writerow({"mae_kmh": mae, "rmse_kmh": rmse, "mape_pct": mape, "r2": r2, "n": len(true)})
    print(f"Metrics written to {output_dir}/metrics.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.scatter(true, pred, s=4, alpha=0.15)
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        ax1.plot(lims, lims, color="tab:red", linewidth=1)
        ax1.set_xlabel("true speed (km/h)")
        ax1.set_ylabel("predicted speed (km/h)")
        ax1.set_title("Predicted vs. true")

        ax2.hist(err, bins=50)
        ax2.set_xlabel("prediction error, pred - true (km/h)")
        ax2.set_ylabel("count")
        ax2.set_title("Residuals")

        fig.tight_layout()
        fig.savefig(output_dir / "prediction_plots.png", dpi=150)
        print(f"Plots written to {output_dir}/prediction_plots.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
