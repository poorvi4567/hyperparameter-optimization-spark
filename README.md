# Phase 1 — Setup & Baseline

## What this phase does

1. **Loads** the UCI Adult Income dataset (48K rows, 14 features)
2. **Preprocesses** it — encodes categoricals, assembles a feature vector
3. **Trains a baseline** GBTClassifier with a fixed default config
4. **Runs Random Search** across 10 randomly sampled configs
5. **Prints a benchmark summary** — these are the numbers ASHA will beat

## Files

| File | Purpose |
|------|---------|
| `data_loader.py` | Load dataset, preprocess, split train/test |
| `baseline_model.py` | Build, train, and evaluate a single GBT model |
| `random_search.py` | Define search space, sample configs, run naive HPO |
| `main.py` | End-to-end runner for all Phase 1 steps |
| `requirements.txt` | Python dependencies |

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run just data loading
python data_loader.py

# 3. Run just the baseline
python baseline_model.py

# 4. Run just random search
python random_search.py

# 5. Run everything (recommended)
python main.py
```

## Expected output

```
Total Phase 1 runtime: ~5-10 min (depends on machine)

Baseline AUC    : ~0.89
Best Random AUC : ~0.91
Random Search   : ~8-12 min for 10 configs (synchronous)
```

## Key design decisions

### Why GBTClassifier?
- Rich hyperparameter space (6 params)
- `maxIter` = number of trees = natural **budget** for SHA rungs
- Built into Spark MLlib — no extra dependencies

### Why Adult Income dataset?
- Clean, well-understood, binary classification
- Fast to train (~30s per config on a laptop)
- Good signal — HPO meaningfully improves results

### Hyperparameter search space

| Parameter | Values |
|-----------|--------|
| `maxIter` | 50, 100, 150, 200, 300 |
| `maxDepth` | 3, 4, 5, 6, 7, 8 |
| `stepSize` | 0.01, 0.05, 0.1, 0.15, 0.2 |
| `subsamplingRate` | 0.6, 0.7, 0.8, 0.9, 1.0 |
| `featureSubsetStrategy` | sqrt, log2, onethird |
| `minInstancesPerNode` | 1, 2, 5, 10 |

Total combinations: 5×6×5×5×3×4 = **9,000** possible configs.
Random search explores 10. ASHA will explore more, faster.

## What Phase 2 adds
Phase 2 implements **Synchronous SHA** — instead of training each config
to full budget, we use the halving schedule:
- Rung 1: 50 trees → keep top 1/3
- Rung 2: 150 trees → keep top 1/3
- Rung 3: 450 trees → 1 winner

# Phase 2 — Synchronous Successive Halving (SHA)

## What this phase does

1.  **Implements Synchronous SHA**: Trains `n` configurations in parallel for a small budget.
2.  **Synchronous Halving**: After each "rung", it waits for all configurations to complete, then eliminates the worst-performing `(1 - 1/eta)` configurations.
3.  **Increases Budget**: It increases the training budget for the surviving configurations and repeats the process.
4.  **Finds a Winner**: This continues until only one configuration remains, which has been trained on the largest budget.
5.  **Benchmarks against Phase 1**: Compares the total time and best AUC of SHA against the baseline and random search from Phase 1.

## Files

| File | Purpose |
| --- | --- |
| `phase2/main.py` | End-to-end runner for the Synchronous SHA experiment. |
| `phase2/sha_core.py` | Defines the SHA schedule and `RungTracker` to manage state. |
| `phase2/sha_runner.py` | Orchestrates the parallel training of configurations on Spark for each rung. The key logic with the **synchronization barrier** is here. |

## How to run

```bash
# Run the full Synchronous SHA experiment
python phase2/main.py
```

## Expected output

```
Total Phase 2 runtime: ~1-2 min

SHA Best AUC      : ~0.92
SHA Total Time    : ~50s
Speedup vs random : ~6x
```

The key takeaway is that SHA finds a model with comparable (or better) performance than random search in a fraction of the time. The synchronization barrier, however, means faster workers sit idle waiting for the slowest one in the rung to finish.

# Phase 3 — Asynchronous Successive Halving (ASHA)

## What this phase does

1.  **Implements Asynchronous SHA**: Removes the synchronization barrier from SHA.
2.  **No More Waiting**: When a configuration finishes training at a rung, it is immediately considered for promotion. If it's in the top `1/eta` of results *seen so far* for that rung, it gets promoted to the next rung without waiting for its peers to finish.
3.  **Continuous Worker Utilization**: This keeps all worker cores busy. As soon as a worker finishes a task, it picks up a new one—either a newly promoted configuration or a fresh one from the queue.
4.  **Benchmarks against Phase 1 & 2**: Compares ASHA's performance and speed against both Random Search and Synchronous SHA.

## Files

| File | Purpose |
| --- | --- |
| `phase3/main.py` | End-to-end runner for the ASHA experiment. |
| `phase3/asha_runner.py` | The core ASHA implementation. It uses a `ThreadPoolExecutor` to submit single-configuration Spark jobs and process results as they complete. |
| `phase3/asha_state.py` | A thread-safe state manager (`ASHARungState`) that tracks configuration progress and handles promotions asynchronously. |

## How to run

```bash
# Run the full ASHA experiment
python phase3/main.py
```

## Expected output

```
Total Phase 3 runtime: < 1 min

ASHA Best AUC          : ~0.92
ASHA Total Time        : ~30-40s
Speedup vs Random Search : ~8-10x
Speedup vs Sync SHA      : ~1.5x
```

ASHA delivers the same high-quality model as SHA but achieves even better speedup by eliminating idle worker time. This makes it a highly efficient and scalable HPO algorithm.
