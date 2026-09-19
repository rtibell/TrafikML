# MachineLearning_MK-2 — Traffic Speed CNN Regressor

A PyTorch 1D-CNN model that predicts `speed` (km/h) for a single road-segment
measurement from its `SegmentSpeed` covariates (status, day-of-week, month,
minutes-since-daybreak, and holiday fields), exported by the TrafficML collector.

Unlike `MachineLearning_MK-1` (a per-segment sequence forecaster), this version treats
each `SegmentSpeed` row as an independent sample — no `sectionId`, no time ordering —
per the task's explicit feature list. See **[Solution.md](Solution.md)** for why that
matters and what it costs in accuracy.

- **[Solution.md](Solution.md)** — how the model is designed: architecture, data
  contract, feature engineering, the log-vs-raw-target experiment, train/test split,
  measured results, and suggested improvements.
- **[Operations.md](Operations.md)** — how to set up, get data, train, tune, and
  evaluate, plus a troubleshooting table.

```bash
cd MachineLearning_MK-2
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json
python train.py --data raw-trafic-data.json --epochs 30 --batch-size 512
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json
```
