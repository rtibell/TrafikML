# MachineLearning_MK-1 — Traffic Speed/Status Forecaster

A PyTorch encoder-decoder (GRU/LSTM) model that forecasts, per road segment, speed
(km/h) and congestion status minutes ahead, trained on the minute-by-minute
`SegmentSpeed` records exported by the TrafficML collector.

- **[Solution.md](Solution.md)** — how the model is designed: architecture, data
  contract, feature engineering, train/test split, and known limitations/future work
  (including why PyTorch Geometric/DGL weren't used, per the original task request).
- **[Operations.md](Operations.md)** — how to set up, get data, train, and evaluate,
  plus a troubleshooting table.

```bash
cd MachineLearning_MK-1
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json
python train.py --data raw-trafic-data.json --epochs 30 --batch-size 256
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json
```
