"""
Phase 2 - Synchronous SHA Runner on Spark

Each rung runs as a Spark parallel job:
  - All configs in a rung are trained in parallel across partitions
  - Driver WAITS for every config in the rung to finish (sync barrier)
  - Top 1/eta are promoted; rest are eliminated
  - Next rung starts only after the previous rung fully completes

This sync barrier is exactly what ASHA removes in Phase 3.
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyspark.sql import SparkSession

from sha_core import (
    build_sha_schedule,
    print_schedule,
    compute_total_budget,
    RungTracker,
)
from random_search import sample_n_configs, SEARCH_SPACE
from data_loader import create_spark_session, load_adult_dataset, preprocess, split_data


# ──────────────────────────────────────────────────────────────
# Per-worker training function
# ──────────────────────────────────────────────────────────────

def train_config_at_budget(args):
    """
    Runs on a Spark executor (worker).
    Uses sklearn GradientBoostingClassifier — workers are plain Python
    processes and cannot create a SparkSession (Spark forbids it).
    Spark's role here is purely parallelism: it fans out tasks across
    cores and collects results. The actual ML runs in sklearn.

    Returns (config_id, auc_score, train_time_s).
    """
    import time
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    config_id, config, budget, train_rows, test_rows = args

    # Rows are collected Spark Row objects — extract numpy arrays
    X_train = np.array([row.features.toArray() for row in train_rows])
    y_train = np.array([row.label for row in train_rows])
    X_test  = np.array([row.features.toArray() for row in test_rows])
    y_test  = np.array([row.label for row in test_rows])

    # Map Spark MLlib param names to sklearn equivalents
    feat_map = {"sqrt": "sqrt", "log2": "log2", "onethird": None}
    max_features = feat_map.get(str(config["featureSubsetStrategy"]), "sqrt")

    clf = GradientBoostingClassifier(
        n_estimators=int(budget),
        max_depth=int(config["maxDepth"]),
        learning_rate=float(config["stepSize"]),
        subsample=float(config["subsamplingRate"]),
        max_features=max_features,
        min_samples_leaf=int(config["minInstancesPerNode"]),
        random_state=42,
    )

    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start

    proba = clf.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, proba)

    return (config_id, round(score, 4), round(elapsed, 2))


# ──────────────────────────────────────────────────────────────
# Synchronous SHA Runner
# ──────────────────────────────────────────────────────────────

def run_synchronous_sha(
    spark,
    train_df,
    test_df,
    n_configs: int = 9,
    min_budget: int = 50,
    eta: int = 3,
):
    """
    Full synchronous SHA run on Spark.

    For each rung:
      1. Collect current rung's pending configs
      2. Build Spark RDD of (config_id, config, budget, data) tuples
      3. sc.parallelize() + .map(train_config_at_budget) — parallel training
      4. .collect() ← SYNCHRONIZATION BARRIER — driver waits for ALL workers
      5. Record results, promote top 1/eta, eliminate rest
      6. Repeat for next rung

    The .collect() in step 4 is the sync barrier.
    Fast workers sit idle waiting for slow ones before rung can advance.
    This is the bottleneck ASHA eliminates.
    """
    sc = spark.sparkContext
    schedule = build_sha_schedule(n_configs, min_budget, eta)
    print_schedule(schedule, eta)

    # Sample configs
    configs = sample_n_configs(n_configs, base_seed=99)
    tracker = RungTracker(configs, schedule)

    # Pre-collect train/test data to pass to workers
    # (workers can't access the driver's distributed DataFrame directly)
    print("\nCollecting train/test data for broadcast to workers...")
    train_rows = train_df.collect()
    test_rows = test_df.collect()
    print(f"  Train rows: {len(train_rows)} | Test rows: {len(test_rows)}")

    overall_start = time.time()
    rung_logs = []

    for rung_info in schedule:
        rung = rung_info["rung"]
        budget = rung_info["budget"]

        pending = tracker.get_pending(rung)
        if not pending:
            continue

        print(f"\n{'─'*55}")
        print(f"RUNG {rung}  |  budget={budget} trees  |  {len(pending)} configs")
        print(f"{'─'*55}")

        # Build task list for Spark
        tasks = [
            (
                t["config_id"],
                t["config"],
                budget,
                train_rows,
                test_rows,
            )
            for t in pending
        ]

        rung_start = time.time()

        # ── SPARK PARALLEL TRAINING ──────────────────────────
        rdd = sc.parallelize(tasks, numSlices=len(tasks))
        results = rdd.map(train_config_at_budget).collect()
        # ↑ .collect() is the SYNC BARRIER
        # All configs in this rung must finish before we proceed

        rung_time = round(time.time() - rung_start, 2)

        # Record results
        for config_id, score, train_time in results:
            tracker.record_result(config_id, score, train_time)
            trial = tracker.trials[config_id]
            print(f"  Config {config_id:2d}  AUC={score}  "
                  f"time={train_time}s  "
                  f"[maxDepth={trial['config']['maxDepth']}, "
                  f"stepSize={trial['config']['stepSize']}]")

        # Promote top 1/eta
        promoted, eliminated = tracker.promote_top_k(rung, eta)

        print(f"\n  Rung {rung} wall time : {rung_time}s")
        print(f"  Promoted  ({len(promoted)}) : {promoted}")
        print(f"  Eliminated({len(eliminated)}): {eliminated}")

        # Scores at this rung
        rung_scores = sorted(
            [(t["config_id"], t["rung_scores"].get(rung, 0))
             for t in tracker.trials.values()
             if rung in t.get("rung_scores", {})],
            key=lambda x: x[1], reverse=True
        )

        rung_logs.append({
            "rung": rung,
            "budget": budget,
            "n_configs": len(pending),
            "wall_time_s": rung_time,
            "promoted": promoted,
            "eliminated": eliminated,
            "scores": rung_scores,
        })

    total_time = round(time.time() - overall_start, 2)

    # ── Final winner ─────────────────────────────────────────
    winner_trial = tracker.get_winner()

    return {
        "winner": winner_trial,
        "tracker": tracker,
        "schedule": schedule,
        "rung_logs": rung_logs,
        "total_time_s": total_time,
    }


# ──────────────────────────────────────────────────────────────
# Results printer
# ──────────────────────────────────────────────────────────────

def print_sha_results(result: dict):
    """Print full SHA results and comparison against Phase 1."""
    winner = result["winner"]
    total_time = result["total_time_s"]
    rung_logs = result["rung_logs"]

    PHASE1_RANDOM_TIME = 314.28
    PHASE1_BEST_AUC = 0.9163
    PHASE1_BASELINE_AUC = 0.9115

    print("\n" + "=" * 60)
    print("SYNCHRONOUS SHA RESULTS")
    print("=" * 60)

    print("\nRung-by-rung summary:")
    print(f"  {'Rung':<6} {'Budget':>8} {'Configs':>9} {'WallTime':>10} {'TopAUC':>10}")
    print("  " + "-" * 48)
    for log in rung_logs:
        top_score = log["scores"][0][1] if log["scores"] else 0
        print(f"  {log['rung']:<6} {log['budget']:>8} {log['n_configs']:>9} "
              f"{log['wall_time_s']:>9}s {top_score:>10}")

    if winner:
        w_auc = winner["rung_scores"].get(
            max(winner["rung_scores"].keys()), 0
        )
        print(f"""
  ┌──────────────────────────────────────────────────┐
  │                   WINNER CONFIG                  │
  │  Config ID : {winner['config_id']:<36} │
  │  Final AUC : {w_auc:<36} │
  │  maxIter   : {winner['config']['maxIter']:<36} │
  │  maxDepth  : {winner['config']['maxDepth']:<36} │
  │  stepSize  : {winner['config']['stepSize']:<36} │
  └──────────────────────────────────────────────────┘""")

    auc_gap = round(w_auc - PHASE1_BEST_AUC, 4) if winner else "N/A"
    speedup = round(PHASE1_RANDOM_TIME / total_time, 2) if total_time > 0 else "N/A"

    print(f"""
  ┌──────────────────────────────────────────────────┐
  │           COMPARISON vs PHASE 1                  │
  │                                                  │
  │  Phase 1 Baseline AUC  : {PHASE1_BASELINE_AUC:<23} │
  │  Phase 1 Random AUC    : {PHASE1_BEST_AUC:<23} │
  │  Phase 1 Random Time   : {PHASE1_RANDOM_TIME:<22}s │
  │                                                  │
  │  SHA Best AUC          : {w_auc if winner else 'N/A':<23} │
  │  SHA Total Time        : {total_time:<22}s │
  │  AUC diff vs random    : {auc_gap:<23} │
  │  Speedup vs random     : {speedup:<21}x   │
  │                                                  │
  │  ➤ Phase 3 (ASHA) target: same AUC,            │
  │    even faster by removing sync barrier          │
  └──────────────────────────────────────────────────┘""")