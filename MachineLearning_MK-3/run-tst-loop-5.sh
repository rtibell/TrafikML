python export_raw_trafic_data.py --base-url http://localhost:8080 \
	 --out raw-trafic-data.json \
	 --page-size-section 2000 \
	 --limit-sections 100 \
	 --page-size-speed 200000  > test-loop-5.log
act=leaky_relu
for hiddim in 192 224 256 288 320; do 
    python train.py --data raw-trafic-data.json \
	--epochs 200 --num-layers 16 \
	 --hidden-dim $hiddim --batch-size 512 \
	 --activation leaky_relu \
	 --early-stopping-patience 20 \
	 --device mps \
	 --early-stopping-min-delta 1e-2 \
	 --momentum 0.95 --activation "$act" \
	 --checkpoint-dir "checkpoints-$hiddim"  >> test-loop-5.log
    python evaluate.py --checkpoint "checkpoints-$hiddim/best.pt" --data raw-trafic-data.json  --output-dir "checkpoints-$hiddim"  >> test-loop-5.log
    open checkpoints-$hiddim/confusion_matrix.png
done
