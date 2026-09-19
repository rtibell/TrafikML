python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json --limit-sections 100 > test-2.log
python train.py --data raw-trafic-data.json --epochs 30 --batch-size 1024 --device cpu >> test-2.log
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json >> test-2.log
python train.py --data raw-trafic-data.json --epochs 30 --batch-size 32 --device cpu >> test-2.log
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json >> test-2.log
