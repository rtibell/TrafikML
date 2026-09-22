python tune-ga.py --population-size 20 \
	--generations 20 \
	 --device mps \
	--mutation-percent-gene 15.0 \
	--trial-epochs 100 \
	--final-epochs 200 \
	--early-stopping-patience 20 \
	--data raw-trafic-data.json \
	--output-dir tuning-ga-compare > ga-run-2.log
python evaluate.py --checkpoint tuning-ga-compare/best_checkpoint/best.pt \
	 --device mps \
	--data raw-trafic-data.json \
	--output-dir tuning-ga-compare/eval >> ga-run-2.log
