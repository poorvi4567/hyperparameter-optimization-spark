"""
Phase 1 - Step 3: Random Search HPO
Randomly sample hyperparameter configs and evaluate them.
This is the naive HPO baseline we will compare ASHA against.
"""

import time
import random
import json
from pyspark.sql import SparkSession
from baseline_model import train_model, evaluate_model


# ──────────────────────────────────────────────
# Hyperparameter search space definition
# ──────────────────────────────────────────────
SEARCH_SPACE = {
    "maxIter":               [50, 100, 150, 200, 300],      # budget (n_estimators)
    "maxDepth":              [3, 4, 5, 6, 7, 8],            # tree depth
    "stepSize":              [0.01, 0.05, 0.1, 0.15, 0.2], # learning rate
    "subsamplingRate":       [0.6, 0.7, 0.8, 0.9, 1.0],    # row subsampling
    "featureSubsetStrategy": ["sqrt", "log2", "onethird"],  # feature sampling
    "minInstancesPerNode":   [1, 2, 5, 10],                 # min leaf size
}


def sample_config(seed=None):
    """Randomly sample one hyperparameter configuration."""
    rng = random.Random(seed)
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def sample_n_configs(n, base_seed=42):
    """Sample n distinct hyperparameter configs."""
    configs = []
    seen = set()
    attempt = 0
    while len(configs) < n:
        cfg = sample_config(seed=base_seed + attempt)
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
        attempt += 1
    return configs


def random_search(train_df, test_df, n_configs=10, spark=None):
    """
    Run random search: train n_configs models, evaluate each,
    track time, return sorted results.
    This runs SYNCHRONOUSLY — one config after another.
    This is the baseline we want to beat with ASHA.
    """
    configs = sample_n_configs(n_configs)

    print(f"\nStarting Random Search with {n_configs} configs...")
    print("=" * 60)

    results = []
    total_start = time.time()

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{n_configs}] Training config: maxIter={config['maxIter']}, "
              f"maxDepth={config['maxDepth']}, stepSize={config['stepSize']}")

        model, elapsed = train_model(train_df, config)
        metrics = evaluate_model(model, test_df)

        result = {
            "config_id": i + 1,
            "config": config,
            "metrics": metrics,
            "train_time_s": round(elapsed, 2),
            "wall_time_s": round(time.time() - total_start, 2),
        }
        results.append(result)

        print(f"  AUC={metrics['auc']}  Accuracy={metrics['accuracy']}  "
              f"Time={elapsed:.1f}s  (cumulative: {result['wall_time_s']}s)")

    total_time = round(time.time() - total_start, 2)

    # Sort by AUC descending
    results.sort(key=lambda r: r["metrics"]["auc"], reverse=True)

    print("\n" + "=" * 60)
    print("RANDOM SEARCH COMPLETE")
    print("=" * 60)
    print(f"Total wall-clock time: {total_time}s")
    print(f"\nTop 3 configs:")
    for rank, r in enumerate(results[:3], 1):
        cfg = r["config"]
        print(f"\n  Rank {rank}:")
        print(f"    AUC      : {r['metrics']['auc']}")
        print(f"    Accuracy : {r['metrics']['accuracy']}")
        print(f"    maxIter  : {cfg['maxIter']}")
        print(f"    maxDepth : {cfg['maxDepth']}")
        print(f"    stepSize : {cfg['stepSize']}")
        print(f"    Train time: {r['train_time_s']}s")

    best = results[0]
    print(f"\nBest AUC: {best['metrics']['auc']}")
    print(f"Best config: {best['config']}")

    return results, total_time


def summarize_search_space():
    """Print the full search space for reference."""
    print("\nHyperparameter Search Space:")
    print("-" * 40)
    total_combinations = 1
    for k, v in SEARCH_SPACE.items():
        print(f"  {k:30s}: {v}")
        total_combinations *= len(v)
    print(f"\nTotal possible combinations: {total_combinations}")
    print(f"Random search samples: small subset of these")


if __name__ == "__main__":
    from data_loader import create_spark_session, load_adult_dataset, preprocess, split_data

    spark = create_spark_session()

    summarize_search_space()

    df = load_adult_dataset(spark)
    processed_df, _, _ = preprocess(df)
    train_df, test_df = split_data(processed_df)

    # Run random search with 10 random configs
    results, total_time = random_search(train_df, test_df, n_configs=10, spark=spark)

    spark.stop()