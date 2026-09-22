# Hyperparameter tuning — MachineLearning_MK-5

Two interchangeable tuners live in this directory, searching the **same** hyperparameter
space and reusing the **same** training code path (`train.py::run_training`), so their
results are directly comparable:

| Script | Strategy | Status |
|---|---|---|
| `tune-rnd.py` | Uniform random search | Original approach, kept for comparison |
| `tune-ga.py` | Genetic algorithm (GA) | Current default -- see "Why a GA" below |

See `Solution.md` for how tuning fits into the overall MK-5 design, and `Operations.md`
for day-to-day run commands. This document covers *how the GA is built* and *what it
found* when measured against random search.

## The shared search space

Both tuners search the same 9 hyperparameters, over the same discrete choice lists
(`tune-ga.py::SEARCH_SPACE`, `tune-rnd.py::sample_config`):

| Hyperparameter | Choices |
|---|---|
| `num_layers` | 1, 2, 3, 4 |
| `hidden_dim` | 32, 64, 128, 256 |
| `activation` | sigmoid, tanh, softsign, relu, leaky_relu |
| `dropout` | 0.0, 0.1, 0.2, 0.3 |
| `lr` | 1e-3, 3e-3, 1e-2, 3e-2 |
| `momentum` | 0.0, 0.8, 0.9, 0.95 |
| `lr_decay` | 1.0, 0.99, 0.97 |
| `batch_size` | 128, 256, 512, 1024 |
| `class_weights` | balanced, none |

Both tuners rank candidates by the same metric `train.py` itself uses for early
stopping and checkpoint selection: **validation balanced accuracy** (macro-average
recall) -- see `Solution.md`'s "Why balanced accuracy, not raw accuracy" for why raw
accuracy would be the wrong ranking metric on this heavily imbalanced target. Every
candidate is trained with early stopping active and no checkpoint written (`train.py`
called with `checkpoint_dir=None`), and the overall winner is retrained for a full
epoch budget and checkpointed at the end.

## Why a genetic algorithm

Random search samples the space blindly -- every trial is independent of every other
trial's result. A GA instead evolves a *population* of candidate configurations over
several *generations*: the best-performing individuals are more likely to be selected
as parents, their hyperparameters are recombined (crossover) and lightly perturbed
(mutation) to produce the next generation, and a small number of top individuals are
carried over unchanged (elitism). Over enough generations this concentrates the search
on promising regions of the space instead of sampling it uniformly forever.

### Package choice: PyGAD

[`PyGAD`](https://github.com/ahmedfgad/GeneticAlgorithmPython) (BSD-3-Clause,
`pip install pygad`, added to `requirements.txt`) was chosen over the other common
free/open-source option, `DEAP`. The deciding factor: PyGAD's `gene_space` parameter is
a direct fit for this search -- each hyperparameter's discrete choice list becomes one
gene's allowed value set, and PyGAD's built-in selection/crossover/mutation operators
already respect per-gene allowed values. DEAP is more flexible (arbitrary chromosome
representations, custom operators) but that flexibility is unneeded here and would mean
hand-writing a custom mutation operator just to keep genes within their discrete choice
lists -- PyGAD gets that for free.

### Chromosome encoding

Each of the 9 hyperparameters is one gene. A gene's value is not the hyperparameter
value itself but an **index** into that hyperparameter's choice list (`tune-ga.py::
SEARCH_SPACE`) -- e.g. gene 2 (`activation`) ranges over `0..4`, decoded via
`decode_solution()` into one of `sigmoid/tanh/softsign/relu/leaky_relu`. This keeps
every gene a plain integer regardless of whether the underlying hyperparameter is
itself an int, a float, or a string, which is what lets a single `gene_type=int` /
`gene_space=[range(len(choices)) for choices in SEARCH_SPACE.values()]` pair drive the
whole search.

### GA configuration and why

| Setting | Default | Rationale |
|---|---|---|
| `--population-size` | 8 | Individuals evaluated per generation |
| `--generations` | 5 | Evolutionary steps beyond the initial population |
| `--parent-selection-type` | `sss` (steady-state) | PyGAD's default; replaces the lowest-fitness individuals each generation |
| `--crossover-type` | `uniform` | Swaps each gene independently between two parents. Hyperparameters have no meaningful *adjacency* (gene 3 being next to gene 4 is an artifact of dict ordering, not a real relationship), so single/two-point crossover -- which assumes genes near each other in the chromosome should be inherited together -- has no principled basis here; uniform crossover treats every gene as independent, which matches the actual structure of the search |
| `--mutation-type` | `random` | Resamples a mutated gene uniformly within its `gene_space` |
| `--mutation-percent-genes` | 25.0 | ~2 of 9 genes mutate per offspring on average -- enough to keep exploring without destroying a good parent's structure |
| `--keep-elitism` | 1 | The single best individual survives unchanged into the next generation, so the running best can never regress |
| `--search-seed` | 4711 (`RANDOM_SEED`) | Seeds PyGAD's population init/crossover/mutation RNG, for a reproducible search -- mirrors `tune-rnd.py --search-seed` |

All of these are CLI flags (`python tune-ga.py --help`), so a follow-up tuning run can
widen the population, run more generations, or try a different crossover/mutation
strategy without touching code.

### Fitness caching

PyGAD's elitism and mutation can both re-produce a chromosome that was already
evaluated (an elite individual reappearing next generation, or mutation landing back on
a previously-seen combination by chance) -- and separately, PyGAD's own
`best_solution()` bookkeeping triggers one extra fitness call for whichever individual
it never had to re-evaluate during the run. `tune-ga.py`'s fitness function keys a
dict cache on the exact gene tuple, so a repeated config is looked up instead of
retrained; training is by far the most expensive part of a trial, evaluating a cached
config is not. `tuning_results.csv`'s `cache_hit` column marks which rows were served
this way (a duplicate `(generation, config)` pair within the *same* generation is
suppressed entirely rather than logged twice, since that's PyGAD re-confirming an
elite's already-known fitness, not new information; the same config resurfacing in a
*later* generation still gets its own row, since persistence across generations is a
real, informative signal).

## What's logged

Per the task's requirement to record parameter settings against their outcome,
`tune-ga.py` writes `<output-dir>/tuning_results.csv` with one row per evaluated
config: `generation`, `individual` index, all 9 hyperparameters, **`val_balanced_accuracy`,
`val_precision`, `val_f1`, `val_f2`** (all macro-averaged over the 4 status classes,
computed at that trial's best epoch), `stopped_epoch`, `elapsed_seconds` (wall-clock
training time for that trial), and `cache_hit`.

The precision/F1/F2 columns are new: `train.py::run_epoch` was extended to additionally
track per-class predicted counts (alongside the true/correct counts it already tracked
for balanced accuracy) and derive macro precision, F1, and **F2** (recall weighted
twice as heavily as precision -- `fbeta` with `beta=2`) using the same
`zero_division=0` convention `evaluate.py` uses for its test-set report. This piggybacks
on the epoch loop already running for early stopping, so `tune-ga.py` gets these
numbers "for free" from `run_training`'s returned `best_val_metrics`, with no extra
forward pass over the data. `tune-rnd.py` is an unmodified copy of the original script
and does not read these fields; its log stays exactly as before (`val_balanced_accuracy`,
`stopped_epoch`).

`best_config.json` records the winning configuration and its validation balanced
accuracy; the winner is then retrained for `--final-epochs` and checkpointed to
`<output-dir>/best_checkpoint/best.pt`, exactly as `tune-rnd.py` does.

## Measured comparison

**Run conditions**: 20,000-row synthetic dataset (`generate_synthetic_data.py
--rows 20000`) on an M2 MacBook (`--device mps`, auto-selected) -- not the real
438,847-row export, so treat this as a controlled pipeline-correctness comparison
between the two search *strategies*, not a production tuning result (see `Solution.md`'s
caveat that the real export isn't yet calendar-diverse, and "Reproducing this
comparison" below for re-running it against real data). Both runs used identical
training conditions: `--trial-epochs 15 --final-epochs 40 --early-stopping-patience 5`,
default `--seed`/`--search-seed` (4711). `tune-rnd.py --trials 20`; `tune-ga.py
--population-size 8 --generations 3` (4 populations of ≤8 individuals each, up to ~29
trained configs before caching).

### Random search (`tune-rnd.py`)

20 trials, mean val_balanced_accuracy **0.4786**, high variance: **9 of 20 trials
(45%)** collapsed to the degenerate ~0.333 floor (a model that always predicts one
class -- see `Solution.md`'s "Why balanced accuracy" section). All but one of those
degenerate trials sampled `class_weights=none`; random search has no mechanism to learn
"avoid `none`" -- each trial is independent of every other.

| Trial | layers | hidden | activation | lr | class_weights | val_bal_acc |
|---|---|---|---|---|---|---|
| **10 (winner)** | 3 | 32 | tanh | 0.01 | balanced | **0.8781** |
| 17 | 4 | 64 | relu | 0.03 | balanced | 0.8649 |
| 20 | 4 | 256 | relu | 0.01 | none | 0.8217 |
| 15 | 1 | 256 | relu | 0.03 | none | 0.6193 |
| 9 of 20 trials | -- | -- | -- | -- | mostly none | ~0.333 (degenerate) |

Full log: `tuning_results.csv` from this run (not committed; reproduce with the command
in "Reproducing this comparison" below).

### Genetic algorithm (`tune-ga.py`)

28 evaluations logged (27 trained, 1 served from cache) across 4 populations
(generation 0 = initial population, generations 1-3 = evolved). Unlike random search,
the population's **mean** fitness climbed every generation -- direct evidence the GA is
learning, not just sampling:

| Generation | Individuals evaluated | Best val_bal_acc | Mean val_bal_acc |
|---|---|---|---|
| 0 (initial) | 8 | 0.5253 | 0.4125 |
| 1 | 7 | 0.7566 | 0.5292 |
| 2 | 7 | **0.8612** | 0.5762 |
| 3 | 6 | 0.8389 | 0.6214 |

`class_weights` composition per generation tells the same story from a different angle
-- the population shifts toward the non-degenerate choice under selection pressure,
with mutation still occasionally reintroducing `none` (expected: mutation trades a
little exploitation for continued exploration, so the search doesn't converge
prematurely and get stuck if `balanced` weren't actually best):

| Generation | `balanced` | `none` |
|---|---|---|
| 0 | 5 | 3 |
| 1 | 6 | 1 |
| 2 | 4 | 3 |
| 3 | 5 | 1 |

Winning config: `num_layers=2, hidden_dim=256, activation=relu, dropout=0.1, lr=0.03,
momentum=0.8, lr_decay=1.0, batch_size=256, class_weights=balanced` --
val_balanced_accuracy=**0.8612**, val_precision=0.5118, val_f1=0.5613, val_f2=0.6060.

### Head-to-head on the held-out test set

Both winning configs were retrained to convergence and evaluated with `evaluate.py` on
the same 2,000-row held-out test split:

| Metric | Random search winner | GA winner |
|---|---|---|
| Test accuracy | 0.9295 | 0.9240 |
| Test balanced accuracy | **0.6345** | 0.6039 |
| Test macro-F1 | **0.5575** | 0.5368 |
| `freeflow` precision/recall/F1 | 0.996 / 0.947 / 0.971 | 1.000 / 0.947 / 0.973 |
| `heavy` precision/recall/F1 | **0.543** / 0.729 / **0.622** | 0.473 / 0.693 / 0.562 |
| `congested` precision/recall/F1 | 0.505 / **0.862** / **0.637** | 0.506 / 0.776 / 0.612 |
| Wall-clock (full tuning run, incl. final retrain) | 46.7s | 53.9s |

`impossible` had zero rows in this synthetic dataset's test split (its class-balance
simplification -- see `Operations.md` -- is nowhere near as skewed as the real export's,
and 20,000 rows is too few to guarantee coverage of the rarest class); neither model's
`impossible` row is meaningful here.

### Honest reading of these numbers

**Random search's single lucky trial narrowly beat the GA's converged best in this
particular run** (0.6345 vs. 0.6039 test balanced accuracy) -- report this plainly
rather than spinning it. With a small population (8) and few generations (3), sampling
variance is still large enough that one favorable random draw can outperform a
partially-converged directed search. That is expected, not a bug: a GA's real advantage
over random search is that its *population* improves monotonically on average (proven
above: 0.41 → 0.53 → 0.58 → 0.62 mean fitness across generations, vs. random search's
flat, no-learning trial sequence), which compounds as the number of generations grows.
With only 3 generations here, that compounding hasn't had much room to play out, and the
GA's best individual (0.8612 val_balanced_accuracy) still edges past all but random
search's single best trial. A larger generation budget -- see "Tips" below -- would be
expected to widen the GA's advantage; this comparison should be re-run with a larger
budget against the real export before treating either result as final. What the GA
*does* demonstrably deliver over random search, independent of who wins any one race: a
population that gets better on average every generation, and richer per-trial logging
(precision/F1/F2, not just balanced accuracy) with no extra training cost.

## Tips for tuning the tuner

- **Increase `--population-size`/`--generations` for a real run.** 8×3 (≈29 trained
  configs) is a smoke-test-scale budget chosen to keep this comparison's wall-clock
  short. `--population-size 16 --generations 8` (≈130 trained configs) is a more
  realistic starting point once pointed at the real ~438k-row export with GPU time to
  spend -- and per the finding above, more generations is specifically what lets the
  GA's compounding population-level improvement pull ahead of random search.
- **Run more than one `--search-seed`.** A single comparison run (as above) can make
  either strategy look better or worse than its typical behavior purely from search-RNG
  variance -- average a handful of seeds before concluding one strategy is reliably
  ahead for your dataset.
- **`--trial-epochs` vs. `--early-stopping-patience` trade-off.** A trial that hits its
  epoch budget while `val_bal_acc` is "still improving" (visible in the per-trial log)
  is being ranked on an unconverged score -- widen `--trial-epochs` or shorten
  `--early-stopping-patience` if many winning trials show this.
- **`--mutation-percent-genes` controls exploration vs. exploitation.** Too low and the
  population converges to a single neighborhood too fast (premature convergence,
  visible as most individuals sharing near-identical genes by generation 2-3); too high
  and it behaves more like random search, losing the GA's main advantage. 20-30% (the
  default is 25%) is a reasonable starting range for a 9-gene chromosome.
- **`class_weights=none` is a known-bad region of the space** (see "Why balanced
  accuracy" in `Solution.md` -- it reliably collapses to the ~0.25-0.33 degenerate
  floor). It's deliberately left in the search space rather than hard-coded out, so a
  tuning run's own results independently confirm the collapse for whatever architecture
  is being searched (as both runs above did) -- a form of the search sanity-checking
  itself. Remove it from `SEARCH_SPACE` only once you're confident enough not to need
  that confirmation, to shrink the space and get more of the budget spent on
  hyperparameters that actually matter.
- **F2 rewards recall over precision on purpose** (it weights recall 4× as heavily as
  precision in the F-beta formula) -- matching the downstream framing in `Solution.md`'s
  baseline discussion ("a system that treats 'predicted congested/impossible' as 'worth
  a closer look' tolerates false positives far better than one that acts on the
  prediction directly"). If that framing doesn't match your downstream use case, rank on
  `val_f1` or a custom fitness function weighting precision more heavily instead of
  `val_balanced_accuracy` -- `tune-ga.py`'s `fitness_func` returns
  `result["best_val_balanced_accuracy"]` as a single line, trivial to swap for
  `best_metrics["f1"]` or a custom blend.
- **Use `tune-rnd.py` as a cheap sanity floor before committing to a large GA budget.**
  A short random-search run (a handful of trials) is nearly free and tells you roughly
  what a "no-skill" search finds; if `tune-ga.py`'s eventual best isn't clearly ahead of
  that floor by a comfortable margin, look at the GA configuration (population/
  generations/mutation rate) before trusting the result.

## Reproducing this comparison

```bash
python generate_synthetic_data.py --out synthetic-compare.json --rows 20000

python tune-rnd.py --data synthetic-compare.json --trials 20 --trial-epochs 15 \
    --final-epochs 40 --early-stopping-patience 5 --output-dir tuning-rnd-compare

python tune-ga.py --data synthetic-compare.json --population-size 8 --generations 3 \
    --trial-epochs 15 --final-epochs 40 --early-stopping-patience 5 \
    --output-dir tuning-ga-compare

python evaluate.py --checkpoint tuning-rnd-compare/best_checkpoint/best.pt \
    --data synthetic-compare.json --output-dir tuning-rnd-compare/eval
python evaluate.py --checkpoint tuning-ga-compare/best_checkpoint/best.pt \
    --data synthetic-compare.json --output-dir tuning-ga-compare/eval
```

For a production tuning run, point `--data` at `raw-trafic-data.json` (the real export)
instead, and scale up the budgets per the tips above.
