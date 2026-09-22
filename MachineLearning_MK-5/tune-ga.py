"""Hyperparameter tuning: genetic-algorithm (GA) search over the MLP's
architecture/optimization hyperparameters (layers, width, activation, dropout, learning
rate, momentum, LR decay, batch size, class-weighting) -- the exact same search space
`tune-rnd.py`'s random search used, so the two are directly comparable (see `Tuning.md`).

Each hyperparameter is one gene in a chromosome; a population of candidate
configurations evolves over a fixed number of generations via selection, crossover, and
mutation, using validation *balanced* accuracy (see train.py's module docstring for why
raw accuracy is the wrong ranking metric on this imbalanced target) as the fitness
function. Every candidate reuses train.py's `run_training` (short epoch budget, early
stopping still active, no checkpoint written) so ranking uses exactly the same training
code path as a standalone `train.py` run. The winning configuration across the whole run
is then retrained for a full epoch budget and checkpointed.

GA library: `PyGAD` (https://github.com/ahmedfgad/GeneticAlgorithmPython, BSD-3-Clause,
pure Python, `pip install pygad`). Chosen over DEAP (also free/open-source) because its
`gene_space` parameter is a direct fit for this search: each hyperparameter's discrete
choice list becomes one gene's allowed value set, so no custom chromosome
encoding/decoding, crossover, or mutation operators need to be hand-written -- PyGAD's
built-in operators already respect per-gene allowed values.

    python tune-ga.py --data raw-trafic-data.json --population-size 10 --generations 6 \\
        --trial-epochs 20 --final-epochs 100

See `Tuning.md` for a walkthrough of how the GA is configured (selection/crossover/
mutation strategy, population/generation sizing) and tips on tuning the tuner.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from torch import nn

import pygad

from dataset import RANDOM_SEED
from model import ACTIVATIONS
from train import run_training

GA_ACTIVATIONS: dict[str, type[nn.Module]] = {
#    "sigmoid": nn.Sigmoid,
#    "tanh": nn.Tanh,
#    "softsign": nn.Softsign,
#    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
}

ACTIVATION_OPTIONS = sorted(GA_ACTIVATIONS)

# Discrete choice list per hyperparameter -- identical to tune-rnd.py::sample_config's
# random-search space, so a GA run and a random-search run explore the same universe of
# configurations and their results are comparable. Dict order fixes gene order.
SEARCH_SPACE: dict[str, list] = {
    "num_layers": [1, 2, 3, 4],
    "hidden_dim": [32, 64, 128, 256],
    "activation": ACTIVATION_OPTIONS,
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "lr": [1e-3, 3e-3, 1e-2, 3e-2],
    "momentum": [0.0, 0.8, 0.9, 0.95],
    "lr_decay": [1.0, 0.99, 0.97],
    "batch_size": [128, 256, 512, 1024],
    "class_weights": ["balanced", "none"],
}
GENE_NAMES = list(SEARCH_SPACE.keys())


def decode_solution(solution) -> dict:
    """A GA "solution"/chromosome is a vector of integer gene values, one per
    `GENE_NAMES` entry, each an index into that hyperparameter's `SEARCH_SPACE` choice
    list (not the hyperparameter value itself) -- indices are what PyGAD's `gene_space`
    mutates/crosses over; decoding to real values happens only here, at evaluation time.
    """
    return {name: SEARCH_SPACE[name][int(gene)] for name, gene in zip(GENE_NAMES, solution)}


def build_args(base_args: argparse.Namespace, config: dict, epochs: int, checkpoint_dir: str | None, quiet: bool) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(base_args))
    for key, value in config.items():
        setattr(ns, key, value)
    ns.epochs = epochs
    ns.checkpoint_dir = checkpoint_dir
    ns.quiet = quiet
    return ns


def make_fitness_func(base_args: argparse.Namespace, trial_epochs: int, log_rows: list[dict], cache: dict[tuple, dict], logged_keys: set[tuple]):
    """Builds the PyGAD fitness function as a closure over the shared training args, the
    trial-epoch budget, and two pieces of run-level state:

    - `log_rows`: every evaluated config and its outcome (the task's "record a log of
      parameter settings and their outcome" requirement) -- balanced accuracy,
      precision, F1, F2, stopped epoch, and wall-clock training time -- appended here
      and written to `tuning_results.csv` once the GA run finishes.
    - `cache`: keyed by the exact gene tuple, so a config PyGAD re-evaluates (elitism
      carries the best individuals into the next generation unchanged, and mutation can
      by chance reproduce a previously-seen config) is looked up instead of retrained --
      training is by far the most expensive part of a trial, evaluating a chromosome is
      not.
    - `logged_keys`: PyGAD internally re-invokes the fitness function for elitism-kept
      individuals within the same generation (a bookkeeping detail of its selection
      step, not a second, independent evaluation); `(generation, gene tuple)` pairs
      already appended are skipped so `tuning_results.csv` gets one row per genuinely
      distinct (generation, config) pair rather than exact duplicates. A config that
      resurfaces in a *later* generation still gets its own row -- that repetition is
      real information about the search (e.g. an elite surviving many generations
      unchanged).
    """

    def fitness_func(ga_instance: pygad.GA, solution, solution_idx: int) -> float:
        config = decode_solution(solution)
        key = tuple(int(gene) for gene in solution)
        generation = ga_instance.generations_completed
        log_key = (generation, key)

        if key in cache:
            cached = cache[key]
            if log_key not in logged_keys:
                logged_keys.add(log_key)
                log_rows.append({
                    **config, "generation": generation, "individual": solution_idx,
                    "val_balanced_accuracy": cached["val_balanced_accuracy"],
                    "val_precision": cached["val_precision"],
                    "val_f1": cached["val_f1"],
                    "val_f2": cached["val_f2"],
                    "stopped_epoch": cached["stopped_epoch"],
                    "elapsed_seconds": 0.0,
                    "cache_hit": True,
                })
            return cached["val_balanced_accuracy"]

        trial_args = build_args(base_args, config, trial_epochs, checkpoint_dir=None, quiet=True)
        t0 = time.perf_counter()
        result = run_training(trial_args)
        elapsed = time.perf_counter() - t0

        val_bal_acc = result["best_val_balanced_accuracy"]
        best_metrics = result["best_val_metrics"] or {}
        val_precision = best_metrics.get("precision", float("nan"))
        val_f1 = best_metrics.get("f1", float("nan"))
        val_f2 = best_metrics.get("f2", float("nan"))
        stopped_epoch = result["stopped_epoch"] or trial_epochs

        cache[key] = {
            "val_balanced_accuracy": val_bal_acc,
            "val_precision": val_precision,
            "val_f1": val_f1,
            "val_f2": val_f2,
            "stopped_epoch": stopped_epoch,
        }
        logged_keys.add(log_key)
        log_rows.append({
            **config, "generation": generation, "individual": solution_idx,
            "val_balanced_accuracy": val_bal_acc,
            "val_precision": val_precision,
            "val_f1": val_f1,
            "val_f2": val_f2,
            "stopped_epoch": stopped_epoch,
            "elapsed_seconds": round(elapsed, 3),
            "cache_hit": False,
        })
        print(
            f"gen {generation:2d} ind {solution_idx:2d}  val_bal_acc={val_bal_acc:.4f} "
            f"precision={val_precision:.4f} f1={val_f1:.4f} f2={val_f2:.4f} "
            f"stopped@{stopped_epoch:3d}  [{elapsed:.1f}s]  {config}"
        )
        return val_bal_acc

    return fitness_func


def on_generation(ga_instance: pygad.GA) -> None:
    _, best_fitness, _ = ga_instance.best_solution()
    print(f"== generation {ga_instance.generations_completed}/{ga_instance.num_generations} -- best val_bal_acc so far: {best_fitness:.4f} ==")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="raw-trafic-data.json")

    # GA population / evolution controls
    p.add_argument("--population-size", type=int, default=8, help="individuals per generation")
    p.add_argument("--generations", type=int, default=5, help="number of GA generations to evolve")
    p.add_argument("--num-parents-mating", type=int, default=None, help="defaults to max(2, population-size // 2)")
    p.add_argument(
        "--parent-selection-type", default="sss",
        choices=["sss", "rws", "sus", "rank", "random", "tournament"],
        help="PyGAD parent-selection strategy (sss = steady-state selection, PyGAD's default)",
    )
    p.add_argument(
        "--crossover-type", default="uniform",
        choices=["single_point", "two_points", "uniform", "scattered"],
        help="uniform (default) swaps each gene independently -- a better fit than single/two-point crossover "
             "for hyperparameters that have no meaningful adjacency order",
    )
    p.add_argument("--mutation-type", default="random", choices=["random", "swap", "inversion", "scramble", "adaptive"])
    p.add_argument("--mutation-percent-genes", type=float, default=25.0, help="expected %% of genes mutated per offspring")
    p.add_argument("--keep-elitism", type=int, default=1, help="number of best individuals carried unchanged into the next generation")

    p.add_argument("--trial-epochs", type=int, default=20, help="max epoch budget used to rank candidate configs (early stopping may end a trial sooner)")
    p.add_argument("--final-epochs", type=int, default=100, help="max epoch budget for retraining the winning config")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--early-stopping-patience", type=int, default=5, help="applied to every trial and the final retrain")
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="seeds the data split/model init; held fixed across trials so they differ only in hyperparameters")
    p.add_argument("--search-seed", type=int, default=RANDOM_SEED, help="seeds PyGAD's population init/crossover/mutation RNG")
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="tuning-ga")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_args = argparse.Namespace(
        data=args.data,
        val_fraction=args.val_fraction,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )

    log_rows: list[dict] = []
    cache: dict[tuple, dict] = {}
    logged_keys: set[tuple] = set()
    fitness_func = make_fitness_func(base_args, args.trial_epochs, log_rows, cache, logged_keys)

    num_parents_mating = args.num_parents_mating or max(2, args.population_size // 2)
    gene_space = [list(range(len(values))) for values in SEARCH_SPACE.values()]

    ga_instance = pygad.GA(
        num_generations=args.generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=args.population_size,
        num_genes=len(GENE_NAMES),
        gene_space=gene_space,
        gene_type=int,
        parent_selection_type=args.parent_selection_type,
        crossover_type=args.crossover_type,
        mutation_type=args.mutation_type,
        mutation_percent_genes=args.mutation_percent_genes,
        keep_elitism=args.keep_elitism,
        random_seed=args.search_seed,
        on_generation=on_generation,
        suppress_warnings=True,
    )

    # PyGAD replaces (population_size - keep_elitism) individuals per generation with
    # fresh offspring (the elites are carried over without a new fitness_func call) --
    # this is an upper bound, since duplicate configs (elitism reappearing, or mutation
    # reproducing an already-seen chromosome) are served from `cache` instead of retrained.
    max_evals = args.population_size + max(args.population_size - args.keep_elitism, 0) * args.generations
    print(
        f"Running GA search: population={args.population_size} generations={args.generations} "
        f"parents_mating={num_parents_mating} (up to ~{max_evals} training runs, fewer with cache hits)"
    )
    t_start = time.perf_counter()
    ga_instance.run()
    total_elapsed = time.perf_counter() - t_start

    # best_solution() (below) can itself trigger one more fitness_func call -- for the
    # single elite individual PyGAD's own generation loop never re-invokes our function
    # for (it reuses its cached internal fitness directly) -- so it must run before the
    # log is written, or that row would be silently missing from tuning_results.csv.
    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_config = decode_solution(best_solution)

    with open(output_dir / "tuning_results.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "generation", "individual", *GENE_NAMES,
            "val_balanced_accuracy", "val_precision", "val_f1", "val_f2",
            "stopped_epoch", "elapsed_seconds", "cache_hit",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)
    n_evals = len(log_rows)
    n_cache_hits = sum(1 for row in log_rows if row["cache_hit"])
    print(
        f"\nGA search complete in {total_elapsed:.1f}s over {ga_instance.generations_completed} generations "
        f"({n_evals} evaluations logged, {n_cache_hits} served from cache, "
        f"{n_evals - n_cache_hits} configs actually trained)."
    )
    print(f"Best config (val_balanced_accuracy={best_fitness:.4f}): {best_config}")

    with open(output_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump({**best_config, "val_balanced_accuracy": float(best_fitness)}, f, indent=2)

    print(f"\nRetraining winning config for up to {args.final_epochs} epochs...")
    final_checkpoint_dir = output_dir / "best_checkpoint"
    final_args = build_args(base_args, best_config, args.final_epochs, checkpoint_dir=str(final_checkpoint_dir), quiet=False)
    final_result = run_training(final_args)
    final_metrics = final_result["best_val_metrics"] or {}
    print(
        f"Final model val balanced accuracy: {final_result['best_val_balanced_accuracy']:.4f} "
        f"(precision={final_metrics.get('precision', float('nan')):.4f} "
        f"f1={final_metrics.get('f1', float('nan')):.4f} "
        f"f2={final_metrics.get('f2', float('nan')):.4f}). "
        f"Checkpoint: {final_checkpoint_dir}/best.pt"
    )


if __name__ == "__main__":
    main()
