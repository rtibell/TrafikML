python export_raw_trafic_data.py --base-url http://localhost:8080 \
	 --out raw-trafic-data.json \
	 --page-size-section 2000 \
	 --limit-sections 100 \
	 --page-size-speed 200000  > test-loop-6.log
act=leaky_relu
hiddim=288
layers=24
for momentum in 0.80 0.85 0.90 0.95 0.97; do 
    python train.py --data raw-trafic-data.json \
	--epochs 200 --num-layers $layers \
	 --hidden-dim $hiddim --batch-size 512 \
	 --activation leaky_relu \
	 --early-stopping-patience 20 \
	 --device mps \
	 --early-stopping-min-delta 1e-2 \
	 --momentum $momentum --activation "$act" \
	 --checkpoint-dir "checkpoints-$hiddim-$layers-$momentum"  >> test-loop-6.log
    python evaluate.py --checkpoint "checkpoints-$hiddim-$layers-$momentum/best.pt" --data raw-trafic-data.json  --output-dir "checkpoints-$hiddim-$layers-$momentum" --device mps >> test-loop-6.log
    open checkpoints-$hiddim-$layers-$momentum/*.png
done
