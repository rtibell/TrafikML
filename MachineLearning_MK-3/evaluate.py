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

from dataset import FixedScaler, SegmentSpeedDataset, load_dataframe, split_train_test
from features import STATUS_VALUES
from model import TabularMLPClassifier, predict_status


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

    scalers = {"days_until_holiday": FixedScaler(**ckpt["days_scaler"])}
    test_ds = SegmentSpeedDataset(df, test_idx, scalers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = TabularMLPClassifier(
        num_layers=train_args["num_layers"],
        hidden_dim=train_args["hidden_dim"],
        activation=train_args["activation"],
        dropout=train_args["dropout"],
        holiday_type_embed_dim=train_args["holiday_type_embed_dim"],
        holiday_num_embed_dim=train_args["holiday_num_embed_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    num_classes = len(STATUS_VALUES)
    all_pred, all_true, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["cont"], batch["holiday_type"], batch["holiday_num"])
            pred, probs = predict_status(logits)
            all_pred.append(pred.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_true.append(batch["status"].cpu().numpy())

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    accuracy = float((pred == true).mean())
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)  # rows=true, cols=predicted
    for t, p in zip(true, pred):
        confusion[t, p] += 1

    per_class = {}
    for c, name in enumerate(STATUS_VALUES):
        tp = confusion[c, c]
        support = confusion[c, :].sum()
        predicted_c = confusion[:, c].sum()
        # sklearn's zero_division=0 convention: a class the model never predicts (or
        # that never occurs) scores 0, not NaN -- otherwise a macro average would
        # silently drop it instead of penalizing the model for missing it entirely.
        precision = tp / predicted_c if predicted_c > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(support)}

    macro_f1 = float(np.mean([m["f1"] for m in per_class.values()]))
    # Same definition train.py's model selection uses: macro-average recall. Reported
    # here for comparability with `--early-stopping-patience`'s val_balanced_accuracy.
    balanced_accuracy = float(np.mean([m["recall"] for m in per_class.values()]))

    print(f"Test set (n={len(true)}): accuracy={accuracy:.4f}  balanced_accuracy={balanced_accuracy:.4f}  macro-F1={macro_f1:.4f}")
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for name, m in per_class.items():
        print(f"{name:<12}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>10d}")
    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "".join(f"{name:>12}" for name in STATUS_VALUES)
    print(" " * 12 + header)
    for i, name in enumerate(STATUS_VALUES):
        print(f"{name:<12}" + "".join(f"{confusion[i, j]:>12d}" for j in range(num_classes)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for name, m in per_class.items():
            writer.writerow({"class": name, **m})
        writer.writerow({"class": "accuracy", "precision": "", "recall": "", "f1": accuracy, "support": len(true)})
        writer.writerow({"class": "balanced_accuracy", "precision": "", "recall": "", "f1": balanced_accuracy, "support": len(true)})
        writer.writerow({"class": "macro_avg_f1", "precision": "", "recall": "", "f1": macro_f1, "support": len(true)})
    print(f"\nMetrics written to {output_dir}/metrics.csv")

    with open(output_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\predicted"] + STATUS_VALUES)
        for i, name in enumerate(STATUS_VALUES):
            writer.writerow([name] + confusion[i, :].tolist())
    print(f"Confusion matrix written to {output_dir}/confusion_matrix.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        row_normalized = confusion / np.clip(confusion.sum(axis=1, keepdims=True), 1, None)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(row_normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(num_classes), STATUS_VALUES, rotation=45, ha="right")
        ax.set_yticks(range(num_classes), STATUS_VALUES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title("Confusion matrix (row-normalized)")
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, f"{confusion[i, j]}\n{row_normalized[i, j]:.1%}", ha="center", va="center",
                        color="white" if row_normalized[i, j] > 0.5 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="fraction of true class")
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
        print(f"Confusion matrix plot written to {output_dir}/confusion_matrix.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
