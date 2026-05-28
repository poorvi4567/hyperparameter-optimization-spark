"""
Phase 3 - Main Runner: Asynchronous SHA (ASHA)

Benchmarks baked in from previous phases:
  Phase 1 Random Search : AUC=0.9163  Time=314.28s
  Phase 2 Sync SHA      : AUC=0.9194  Time=52.01s  (6.04x speedup)

ASHA target: match SHA's AUC with additional speedup
by removing the synchronization barrier between rungs.

Usage:
    python main.py

Tune MAX_WORKERS to match your CPU core count for best results.
"""

import sys
import os
import multiprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loader import create_spark_session, load_adult_dataset, preprocess, split_data
from asha_runner import run_asha, print_asha_results


# ── Config ────────────────────────────────────────────────────
N_CONFIGS   = 9
MIN_BUDGET  = 50
ETA         = 3
MAX_WORKERS = min(multiprocessing.cpu_count(), 9)  # use all available cores


def print_banner(text):
    print("\n" + "█" * 60)
    print(f"  {text}")
    print("█" * 60)


def main():
    print_banner("PHASE 3: ASYNCHRONOUS SUCCESSIVE HALVING (ASHA)")
    print(f"""
  Key difference from Phase 2 (Sync SHA):
  ✗ SHA  — waits for ALL configs in a rung before promoting
  ✓ ASHA — promotes each config THE MOMENT it finishes,
            workers never idle-wait for peers

  Config:
    n_configs   = {N_CONFIGS}
    min_budget  = {MIN_BUDGET} trees
    eta         = {ETA}
    max_workers = {MAX_WORKERS} (concurrent Spark jobs)
    """)

    # ── Data ──────────────────────────────────────────────────
    print_banner("Loading & Preprocessing Data")
    spark = create_spark_session("Phase3_ASHA")

    df = load_adult_dataset(spark)
    processed_df, _, _ = preprocess(df)
    train_df, test_df = split_data(processed_df)
    print(f"Train: {train_df.count()} rows | Test: {test_df.count()} rows")

    # ── Run ASHA ──────────────────────────────────────────────
    print_banner("Running ASHA")
    result = run_asha(
        spark=spark,
        train_df=train_df,
        test_df=test_df,
        n_configs=N_CONFIGS,
        min_budget=MIN_BUDGET,
        eta=ETA,
        max_workers=MAX_WORKERS,
    )

    # ── Results ───────────────────────────────────────────────
    print_banner("Results & Full Comparison")
    print_asha_results(result)

    spark.stop()
    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()