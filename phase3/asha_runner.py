"""
Phase 3 - Asynchronous SHA (ASHA) Runner on Spark

How this differs from Phase 2 (synchronous SHA):

  SHA:  parallelize(rung_configs) → map(train) → collect()
        ↑ .collect() blocks. All configs must finish before anyone is promoted.

  ASHA: A task queue feeds workers continuously.
        Each worker picks one (config, rung, budget) task.
        The moment it finishes, the driver:
          - Records the result
          - Immediately promotes if score qualifies (no waiting for peers)
          - Puts the promoted config back in the queue at next rung
          - Dispatches a fresh config to the now-free worker
        Workers are NEVER idle waiting for stragglers.

Implementation on Spark:
  - Driver maintains a task queue (list of pending tasks)
  - Each Spark job is ONE task (one config, one rung)
  - spark.sparkContext.parallelize([task]).map(train).collect()
    fires a single-task Spark job per config per rung
  - Jobs are submitted in a loop; the driver doesn't wait for a
    full rung — it submits the next job as soon as any result arrives
  - Python's concurrent.futures.ThreadPoolExecutor manages concurrent
    Spark job submissions so multiple workers run truly in parallel
"""

import time
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyspark.sql import SparkSession

from asha_state import ASHARungState
from phase2.sha_core import build_sha_schedule, print_schedule
from random_search import sample_n_configs
from data_loader import create_spark_session, load_adult_dataset, preprocess, split_data


# ──────────────────────────────────────────────────────────────
# Worker function (same sklearn approach as Phase 2)
# ──────────────────────────────────────────────────────────────

def train_one_config(args):
    """
    Spark worker function — trains ONE config at ONE rung budget.
    Returns (config_id, rung, score, train_time).
    Identical to Phase 2's worker but returns rung too.
    """
    import time
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    config_id, config, rung, budget, train_rows, test_rows = args

    X_train = np.array([row.features.toArray() for row in train_rows])
    y_train = np.array([row.label for row in train_rows])
    X_test  = np.array([row.features.toArray() for row in test_rows])
    y_test  = np.array([row.label for row in test_rows])

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

    return (config_id, rung, round(score, 4), round(elapsed, 2))


# ──────────────────────────────────────────────────────────────
# ASHA Runner
# ──────────────────────────────────────────────────────────────

def submit_task(sc, task, train_rows, test_rows):
    """
    Submit one (config_id, config, rung, budget) task as a Spark job.
    Returns the result tuple when done.
    Each call is a single-task Spark job — no sync barrier.
    """
    config_id, config, rung, budget = task
    args = (config_id, config, rung, budget, train_rows, test_rows)
    result = sc.parallelize([args], numSlices=1) \
               .map(train_one_config) \
               .collect()[0]
    return result


def run_asha(
    spark,
    train_df,
    test_df,
    n_configs: int = 9,
    min_budget: int = 50,
    eta: int = 3,
    max_workers: int = 3,
):
    """
    Full ASHA run.

    max_workers controls how many Spark jobs run concurrently.
    Set to number of available CPU cores for maximum parallelism.

    The async loop:
      1. Seed the queue with all n_configs at rung 1
      2. Submit up to max_workers jobs concurrently via ThreadPoolExecutor
      3. As each job completes:
           - record result in ASHARungState
           - if promoted: add (config_id, next_rung) to queue
           - if queue has pending tasks AND a worker is free: submit next job
      4. Continue until queue is empty AND all submitted jobs are done
    """
    sc = spark.sparkContext

    # Build schedule
    schedule = build_sha_schedule(n_configs, min_budget, eta)
    print_schedule(schedule, eta)
    schedule_map = {r["rung"]: r for r in schedule}

    # Sample configs
    configs = sample_n_configs(n_configs, base_seed=77)
    config_map = {i: cfg for i, cfg in enumerate(configs)}

    # ASHA state manager
    state = ASHARungState(schedule, eta)
    for cid in config_map:
        state.register_config(cid)

    # Pre-collect data for workers
    print("\nCollecting train/test data for workers...")
    train_rows = train_df.collect()
    test_rows  = test_df.collect()
    print(f"  Train: {len(train_rows)} rows | Test: {len(test_rows)} rows")

    # Initial task queue — all configs at rung 1
    task_queue = [
        (cid, config_map[cid], 1, schedule_map[1]["budget"])
        for cid in config_map
    ]

    overall_start = time.time()
    queue_lock = threading.Lock()
    completed_count = 0
    total_tasks = 0

    print(f"\nStarting ASHA with {max_workers} concurrent workers...")
    print(f"{'─'*60}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        def submit_next():
            """Pop next task from queue and submit to executor."""
            with queue_lock:
                if task_queue:
                    task = task_queue.pop(0)
                    future = executor.submit(
                        submit_task, sc, task, train_rows, test_rows
                    )
                    futures[future] = task
                    return True
            return False

        # Seed initial workers
        for _ in range(min(max_workers, len(task_queue))):
            submit_next()

        # Process results as they arrive
        while futures:
            # Wait for ANY future to complete
            done_futures = []
            for future in list(futures.keys()):
                if future.done():
                    done_futures.append(future)

            if not done_futures:
                time.sleep(0.05)
                continue

            for future in done_futures:
                task = futures.pop(future)
                config_id, config, rung, budget = task

                config_id_r, rung_r, score, train_time = future.result()
                wall_time = time.time() - overall_start
                completed_count += 1

                # Immediately record + attempt promotion (no waiting!)
                promoted, next_rung = state.record_and_promote(
                    config_id_r, rung_r, score, train_time, wall_time
                )

                action = f"→ rung {next_rung}" if promoted else "✗ eliminated"
                print(f"  [{wall_time:6.1f}s] Config {config_id_r:2d} "
                      f"rung {rung_r} | AUC={score} | "
                      f"train={train_time}s | {action}")

                # If promoted, queue next rung task immediately
                if promoted and next_rung is not None:
                    next_budget = schedule_map[next_rung]["budget"]
                    with queue_lock:
                        task_queue.append((
                            config_id_r,
                            config_map[config_id_r],
                            next_rung,
                            next_budget,
                        ))

                # Submit next queued task to now-free worker
                submit_next()

    total_time = round(time.time() - overall_start, 2)

    winner_id, winner_score = state.get_winner()
    winner_config = config_map[winner_id]

    return {
        "state": state,
        "schedule": schedule,
        "configs": config_map,
        "winner_id": winner_id,
        "winner_score": winner_score,
        "winner_config": winner_config,
        "total_time_s": total_time,
        "timeline": state.timeline,
    }


# ──────────────────────────────────────────────────────────────
# Results printer
# ──────────────────────────────────────────────────────────────

def print_asha_results(result: dict):
    PHASE1_RANDOM_TIME = 314.28
    PHASE1_BEST_AUC    = 0.9163
    PHASE2_SHA_TIME    = 52.01
    PHASE2_SHA_AUC     = 0.9194

    w_score    = result["winner_score"]
    w_config   = result["winner_config"]
    w_id       = result["winner_id"]
    total_time = result["total_time_s"]
    state      = result["state"]

    speedup_vs_random = round(PHASE1_RANDOM_TIME / total_time, 2)
    speedup_vs_sha    = round(PHASE2_SHA_TIME / total_time, 2)

    print("\n" + "=" * 60)
    print("ASYNCHRONOUS SHA (ASHA) RESULTS")
    print("=" * 60)

    # Rung summary
    snap = state.snapshot()
    print("\nConfigs evaluated per rung:")
    for rung, count in sorted(snap["rung_counts"].items()):
        scores = [s for _, s in state.rung_results[rung]]
        print(f"  Rung {rung}: {count:2d} configs evaluated  "
              f"| best={max(scores):.4f}  avg={sum(scores)/len(scores):.4f}")

    # Status breakdown
    statuses = snap["statuses"]
    eliminated = sum(1 for s in statuses.values() if s == "eliminated")
    promoted   = sum(1 for s in statuses.values() if s in ("running", "winner"))
    print(f"\n  Eliminated early : {eliminated} configs (saved their full budget)")
    print(f"  Reached final    : {promoted} config(s)")

    print(f"""
  ┌──────────────────────────────────────────────────┐
  │                   WINNER CONFIG                  │
  │  Config ID : {w_id:<36}│
  │  Best AUC  : {w_score:<36}│
  │  maxDepth  : {w_config['maxDepth']:<36}│
  │  stepSize  : {w_config['stepSize']:<36}│
  │  subsample : {w_config['subsamplingRate']:<36}│
  └──────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────┐
  │            FULL COMPARISON                       │
  │                                                  │
  │  Phase 1 Random Search                           │
  │    AUC  : {PHASE1_BEST_AUC:<40}│
  │    Time : {PHASE1_RANDOM_TIME:<39}s│
  │                                                  │
  │  Phase 2 Synchronous SHA                         │
  │    AUC  : {PHASE2_SHA_AUC:<40}│
  │    Time : {PHASE2_SHA_TIME:<39}s│
  │                                                  │
  │  Phase 3 ASHA (this run)                         │
  │    AUC  : {w_score:<40}│
  │    Time : {total_time:<39}s│
  │                                                  │
  │  Speedup vs Random Search : {speedup_vs_random:<21}x│
  │  Speedup vs Sync SHA      : {speedup_vs_sha:<21}x│
  └──────────────────────────────────────────────────┘""")

    # Timeline
    print("\nASHA event timeline (async promotions — no barriers):")
    print(f"  {'Time':>7}  {'Config':>7}  {'Rung':>5}  {'AUC':>7}  Action")
    print("  " + "-" * 48)
    for event in sorted(result["timeline"], key=lambda e: e["wall_time"]):
        print(f"  {event['wall_time']:>6.1f}s  "
              f"cfg {event['config_id']:>2}   "
              f"rung {event['rung']}  "
              f"{event['score']:>7.4f}  "
              f"{event['action']}")