python export_raw_trafic_data.py --base-url http://localhost:8080 --out raw-trafic-data.json --limit-sections 100 > test-1.log
python train.py --data raw-trafic-data.json --epochs 30 --batch-size 512 --log-target --device cpu >> test-1.log
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json >> test-1.log
python train.py --data raw-trafic-data.json --epochs 30 --batch-size 512 --device cpu >> test-1.log
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json >> test-1.log
python train.py --data raw-trafic-data.json --epochs 30 --batch-size 64 --log-target --device cpu >> test-1.log
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json >> test-1.log
python train.py --data raw-trafic-data.json --epochs 30 --batch-size 64 --device cpu >> test-1.log
python evaluate.py --checkpoint checkpoints/best.pt --data raw-trafic-data.json >> test-1.log
