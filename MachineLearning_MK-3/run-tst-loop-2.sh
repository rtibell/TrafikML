python export_raw_trafic_data.py --base-url http://localhost:8080 \
	 --out raw-trafic-data.json \
	 --page-size-section 2000 \
	 --limit-sections 100 \
	 --page-size-speed 200000  > test-loop-2.log
for act in tanh softsign relu leaky_relu; do 
    python train.py --data raw-trafic-data.json \
	--epochs 200 --num-layers 16 \
	 --hidden-dim 256 --batch-size 512 \
	 --activation leaky_relu \
	 --early-stopping-patience 20 \
	 --early-stopping-min-delta 1e-2 \
	 --momentum 0.90 --activation "$act" \
	 --checkpoint-dir "checkpoints-$act"  >> test-loop-2.log
    python evaluate.py --checkpoint "checkpoints-$act/best.pt" --data raw-trafic-data.json  --output-dir "checkpoints-$act"  >> test-loop-2.log
done
