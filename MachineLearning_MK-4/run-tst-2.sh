python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json --limit-sections 100 > test-2a.log
python train.py --data raw-trafic-data.json --epochs 100 --num-layers 4 --batch-size 512 --activation leaky_relu --early-stopping-patience 0 --early-stopping-min-delta 1e-2 --momentum 0.95  --hidden-dim 256 >> test-2a.log
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json >> test-2a.log
