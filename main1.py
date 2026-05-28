"""
Phase 1 - Main Runner
Runs the full Phase 1 pipeline:
  1. Load & preprocess data
  2. Train baseline (single fixed config)
  3. Run random search (naive HPO)
  4. Print benchmark summary

This establishes the baseline numbers that ASHA (Phase 3) will beat.
"""

import time
import sys
import os

# So we can import sibling modules
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import (
    create_spark_session,
    load_adult_dataset,
    preprocess,
    split_data
)
from baseline_model import run_baseline, BASELINE_CONFIG
from random_search import random_search, summarize_search_space, sample_n_configs


def print_banner(text):
    width = 60
    print("\n" + "█" * width)
    print(f"  {text}")
    print("█" * width)


def run_phase1(n_random_configs=10):
    """Run all Phase 1 steps end to end."""

    print_banner("PHASE 1: SETUP & BASELINE")
    overall_start = time.time()

    # ── Step 1: Spark + Data ──────────────────────────
    print_banner("Step 1: Loading Data")
    spark = create_spark_session("Phase1_HPO_Baseline")

    print("Loading Adult Income dataset from UCI...")
    raw_df = load_adult_dataset(spark)
    raw_count = raw_df.count()
    print(f"Raw dataset: {raw_count} rows, {len(raw_df.columns)} columns")

    print("\nPreprocessing (encoding, assembling features)...")
    processed_df, pipeline_model, feature_cols = preprocess(raw_df)
    processed_count = processed_df.count()
    print(f"Processed: {processed_count} rows (dropped {raw_count - processed_count} nulls)")

    print("\nSplitting into train (80%) / test (20%)...")
    train_df, test_df = split_data(processed_df)
    print(f"Train: {train_df.count()} rows | Test: {test_df.count()} rows")

    print("\nLabel distribution (train):")
    train_df.groupBy("label").count().orderBy("label").show()

    # ── Step 2: Baseline ──────────────────────────────
    print_banner("Step 2: Single Baseline Model")
    baseline_model, baseline_metrics, baseline_time = run_baseline(train_df, test_df)

    # ── Step 3: Random Search ─────────────────────────
    print_banner("Step 3: Random Search HPO (Synchronous Baseline)")
    summarize_search_space()

    random_results, random_total_time = random_search(
        train_df, test_df,
        n_configs=n_random_configs,
        spark=spark
    )
    best_random = random_results[0]

    # ── Step 4: Summary ───────────────────────────────
    print_banner("PHASE 1 BENCHMARK SUMMARY")
    overall_time = round(time.time() - overall_start, 2)

    print(f"""
  Dataset         : UCI Adult Income ({processed_count} rows)
  Features        : {len(feature_cols)} (after encoding)
  Model           : GBTClassifier (Spark MLlib)

  ┌─────────────────────────────────────────────┐
  │             BASELINE (fixed config)          │
  │  AUC      : {baseline_metrics['auc']}                         │
  │  Accuracy : {baseline_metrics['accuracy']}                         │
  │  Time     : {baseline_time:.1f}s                             │
  ├─────────────────────────────────────────────┤
  │          RANDOM SEARCH ({n_random_configs} configs)           │
  │  Best AUC : {best_random['metrics']['auc']}                         │
  │  Accuracy : {best_random['metrics']['accuracy']}                         │
  │  Time     : {random_total_time}s (total wall-clock)     │
  │  Speedup needed: ASHA should match this AUC │
  │  in significantly less time                 │
  └─────────────────────────────────────────────┘

  Total Phase 1 runtime: {overall_time}s

  ➤  These numbers are our BENCHMARK.
     ASHA (Phase 3) will target the same AUC
     in less total compute time by eliminating
     bad configs early.
    """)

    print("Configs evaluated in random search (sorted by AUC):")
    print(f"  {'Rank':<5} {'AUC':<8} {'Accuracy':<10} {'maxIter':<10} "
          f"{'maxDepth':<10} {'stepSize':<10} {'Time(s)'}")
    print("  " + "-" * 65)
    for rank, r in enumerate(random_results, 1):
        cfg = r["config"]
        m = r["metrics"]
        print(f"  {rank:<5} {m['auc']:<8} {m['accuracy']:<10} "
              f"{cfg['maxIter']:<10} {cfg['maxDepth']:<10} "
              f"{cfg['stepSize']:<10} {r['train_time_s']}")

    spark.stop()
    print("\nPhase 1 complete. SparkSession stopped.")
    return {
        "baseline_metrics": baseline_metrics,
        "baseline_time": baseline_time,
        "random_results": random_results,
        "random_total_time": random_total_time,
        "best_config": best_random["config"],
        "best_auc": best_random["metrics"]["auc"],
    }


if __name__ == "__main__":
    run_phase1(n_random_configs=10)