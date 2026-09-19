python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json --limit-sections 100 > test-6.log
python train.py --data raw-trafic-data.json --epochs 200 --num-layers 8 --hidden-dim 128 --batch-size 256 --activation leaky_relu --early-stopping-patience 20 --early-stopping-min-delta 1e-3 --momentum 0.90  >> test-6.log
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json >> test-6.log
