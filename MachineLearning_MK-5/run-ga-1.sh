#python export_raw_trafic_data.py --base-url http://localhost:8080 \
#	 --out raw-trafic-data.json \
#	 --page-size-section 2000 \
#	 --limit-sections 1000 \
#	 --page-size-speed 200000  > ga-run-1.log

python tune-ga.py --population-size 10 \
	--generations 10 \
	 --device mps \
	--mutation-percent-gene 25.0 \
	--trial-epochs 25 \
	--final-epochs 100 \
	--early-stopping-patience 10 \
	--data raw-trafic-data.json \
	--output-dir tuning-ga-compare >> ga-run-1.log

python evaluate.py --checkpoint tuning-ga-compare/best_checkpoint/best.pt \
	 --device mps \
	--data raw-trafic-data.json \
	--output-dir tuning-ga-compare/eval >> ga-run-1.log
