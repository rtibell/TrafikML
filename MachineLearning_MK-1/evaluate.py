"""Evaluate a trained checkpoint on the held-out 10% test split.

The test split is *recomputed* deterministically (same seed, t_in/t_out/stride as
training) rather than stored, so this only reproduces the original split if
raw-trafic-data.json and those hyperparameters are unchanged since training.

    python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import FixedScaler, build_windows, load_dataframe, split_train_test, TrafficWindowDataset
from model import Seq2SeqTrafficForecaster


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data", default=None, help="defaults to the path stored in the checkpoint's training args")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="checkpoints/eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    data_path = args.data or train_args["data"]

    df = load_dataframe(data_path)
    windows, seg_index, group_cache = build_windows(df, train_args["t_in"], train_args["t_out"], train_args["stride"])
    if set(seg_index.id_to_idx) != set(ckpt["seg_index"]):
        print("WARNING: segment id set differs from training; embedding lookup may be misaligned.")
    _, test_windows = split_train_test(windows, test_size=0.1, seed=train_args["seed"])
    print(f"Evaluating on {len(test_windows)} test windows")

    scalers = {
        "speed": FixedScaler(**ckpt["scalers"]["speed"]),
        "days_until_holiday": FixedScaler(**ckpt["scalers"]["days_until_holiday"]),
    }
    test_ds = TrafficWindowDataset(test_windows, group_cache, train_args["t_in"], train_args["t_out"], scalers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = Seq2SeqTrafficForecaster(
        num_segments=len(ckpt["seg_index"]),
        hidden_size=train_args["hidden_size"],
        num_layers=train_args["num_layers"],
        cell=train_args["cell"],
        seg_embed_dim=train_args["seg_embed_dim"],
        dropout=train_args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    t_out = train_args["t_out"]
    all_speed_pred, all_speed_true = [], []
    all_status_pred, all_status_true = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)  # no targets passed -> fully autoregressive, matches real deployment use
            speed_pred_kmh = scalers["speed"].inverse_transform(out["speed_scaled"].cpu().numpy())
            all_speed_pred.append(speed_pred_kmh)
            all_speed_true.append(batch["target_speed_kmh"].cpu().numpy())
            all_status_pred.append(out["status_logits"].argmax(dim=-1).cpu().numpy())
            all_status_true.append(batch["target_status"].cpu().numpy())

    speed_pred = np.concatenate(all_speed_pred, axis=0)
    speed_true = np.concatenate(all_speed_true, axis=0)
    status_pred = np.concatenate(all_status_pred, axis=0)
    status_true = np.concatenate(all_status_true, axis=0)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for t in range(t_out):
        err = speed_pred[:, t] - speed_true[:, t]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        denom = np.clip(speed_true[:, t], 1.0, None)  # avoid blow-up near standstill (speed ~ 0)
        mape = float(np.mean(np.abs(err) / denom) * 100)
        acc = float(np.mean(status_pred[:, t] == status_true[:, t]))
        f1 = float(f1_score(status_true[:, t], status_pred[:, t], average="macro", zero_division=0))
        rows.append({"horizon_min": t + 1, "speed_mae_kmh": mae, "speed_rmse_kmh": rmse, "speed_mape_pct": mape, "status_acc": acc, "status_macro_f1": f1})
        print(f"t+{t + 1:>2}min  MAE={mae:6.2f} km/h  RMSE={rmse:6.2f} km/h  MAPE={mape:6.2f}%  status_acc={acc:5.1%}  status_F1={f1:.3f}")

    overall_mae = float(np.mean(np.abs(speed_pred - speed_true)))
    overall_acc = float(np.mean(status_pred == status_true))
    print(f"\nOverall: speed MAE={overall_mae:.2f} km/h, status accuracy={overall_acc:.1%}")

    with open(Path(args.output_dir) / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-horizon metrics written to {args.output_dir}/metrics.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        horizons = [r["horizon_min"] for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(horizons, [r["speed_mae_kmh"] for r in rows], marker="o")
        ax1.set_xlabel("forecast horizon (min)")
        ax1.set_ylabel("speed MAE (km/h)")
        ax2.plot(horizons, [r["status_acc"] for r in rows], marker="o", color="tab:orange")
        ax2.set_xlabel("forecast horizon (min)")
        ax2.set_ylabel("status accuracy")
        fig.tight_layout()
        fig.savefig(Path(args.output_dir) / "horizon_metrics.png", dpi=150)
        print(f"Plot written to {args.output_dir}/horizon_metrics.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
