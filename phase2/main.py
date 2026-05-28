"""
Phase 2 - Main Runner
Runs full Synchronous SHA on Spark and compares against Phase 1 benchmark.

Usage:
    python main.py

Config:
    N_CONFIGS   = 9   (3^2 — ideal for eta=3, gives 3 rungs)
    MIN_BUDGET  = 50  (same as Phase 1 smallest maxIter)
    ETA         = 3   (keep top 1/3 at each rung)

SHA Schedule produced:
    Rung 1: 9 configs x  50 trees  = 450  compute units
    Rung 2: 3 configs x 150 trees  = 450  compute units
    Rung 3: 1 config  x 450 trees  = 450  compute units
    Total : 1350 compute units

    Phase 1 random search (10 configs, avg ~150 trees): ~1500 units
    SHA saves compute by eliminating bad configs at rung 1.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loader import (
    create_spark_session,
    load_adult_dataset,
    preprocess,
    split_data,
)
from sha_runner import run_synchronous_sha, print_sha_results


# ── Config ────────────────────────────────────────────────────
N_CONFIGS  = 9    # must be a power of ETA for clean rungs
MIN_BUDGET = 50   # trees at rung 1
ETA        = 3    # halving rate


def print_banner(text):
    width = 60
    print("\n" + "█" * width)
    print(f"  {text}")
    print("█" * width)


def main():
    print_banner("PHASE 2: SYNCHRONOUS SUCCESSIVE HALVING (SHA)")

    # ── Data ──────────────────────────────────────────────────
    print_banner("Loading & Preprocessing Data")
    spark = create_spark_session("Phase2_SHA")

    df = load_adult_dataset(spark)
    processed_df, _, _ = preprocess(df)
    train_df, test_df = split_data(processed_df)

    print(f"Train: {train_df.count()} rows | Test: {test_df.count()} rows")

    # ── Run SHA ───────────────────────────────────────────────
    print_banner("Running Synchronous SHA")
    print(f"  n_configs  = {N_CONFIGS}")
    print(f"  min_budget = {MIN_BUDGET} trees")
    print(f"  eta        = {ETA}  (keep top 1/{ETA} per rung)")

    result = run_synchronous_sha(
        spark=spark,
        train_df=train_df,
        test_df=test_df,
        n_configs=N_CONFIGS,
        min_budget=MIN_BUDGET,
        eta=ETA,
    )

    # ── Results ───────────────────────────────────────────────
    print_banner("Results & Comparison")
    print_sha_results(result)

    spark.stop()
    print("\nPhase 2 complete. SparkSession stopped.")


if __name__ == "__main__":
    main()