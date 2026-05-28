"""
Phase 2 - Synchronous Successive Halving (SHA)
Core algorithm logic: rungs, halving, promotion.

KEY IDEA:
  - Start with N configs, each given a small budget (min_budget trees)
  - After each rung, keep only top 1/eta configs
  - Double (x eta) the budget and repeat
  - Only 1 config survives to the final rung

Phase 1 benchmark to beat:
  Random Search: 314.28s for 10 configs, best AUC = 0.9163
"""

import math


# ──────────────────────────────────────────────────────────────
# SHA Schedule
# ──────────────────────────────────────────────────────────────

def build_sha_schedule(
    n_configs: int = 27,
    min_budget: int = 50,
    eta: int = 3,
) -> list[dict]:
    """
    Build the SHA rung schedule.

    Parameters
    ----------
    n_configs   : total configs to start with (ideally a power of eta)
    min_budget  : smallest training budget (n_estimators at rung 1)
    eta         : halving rate — keep top 1/eta at each rung

    Returns list of rung dicts:
        [
          {"rung": 1, "budget": 50,  "n_configs": 27},
          {"rung": 2, "budget": 150, "n_configs": 9},
          {"rung": 3, "budget": 450, "n_configs": 3},
          {"rung": 4, "budget": 1350,"n_configs": 1},
        ]

    Total compute = sum(budget * n_configs per rung)
    Compare to random search: 10 * ~avg_budget
    """
    n_rungs = math.floor(math.log(n_configs, eta)) + 1
    schedule = []

    for rung in range(1, n_rungs + 1):
        n = math.ceil(n_configs / (eta ** (rung - 1)))
        budget = min_budget * (eta ** (rung - 1))
        schedule.append({
            "rung": rung,
            "budget": budget,
            "n_configs": n,
        })

    return schedule


def compute_total_budget(schedule: list[dict]) -> int:
    """Total tree-training units across all rungs."""
    return sum(r["budget"] * r["n_configs"] for r in schedule)


def print_schedule(schedule: list[dict], eta: int = 3):
    """Print the SHA schedule in a readable table."""
    total = compute_total_budget(schedule)
    phase1_total = 10 * 150  # approx avg budget in random search

    print("\n" + "=" * 55)
    print("SHA SCHEDULE")
    print("=" * 55)
    print(f"  {'Rung':<6} {'Budget':>8} {'Configs':>10} {'Compute':>10}")
    print("  " + "-" * 40)
    for r in schedule:
        compute = r["budget"] * r["n_configs"]
        print(f"  {r['rung']:<6} {r['budget']:>8} {r['n_configs']:>10} {compute:>10}")
    print("  " + "-" * 40)
    print(f"  {'TOTAL':<6} {'':>8} {'':>10} {total:>10}")
    print(f"\n  Phase 1 random search total compute : ~{phase1_total}")
    print(f"  SHA total compute                   :  {total}")
    print(f"  Compute ratio                       :  {total/phase1_total:.2f}x")
    print("=" * 55)


# ──────────────────────────────────────────────────────────────
# Rung State Tracker
# ──────────────────────────────────────────────────────────────

class RungTracker:
    """
    Tracks which configs are alive at each rung.
    In synchronous SHA, we complete a full rung before promoting.

    State per config:
        config_id  : int
        config     : dict of hyperparams
        rung       : current rung number
        budget     : trees trained so far
        score      : validation AUC at current rung (None = not yet evaluated)
        status     : "pending" | "evaluated" | "promoted" | "eliminated"
    """

    def __init__(self, configs: list[dict], schedule: list[dict]):
        self.schedule = schedule
        self.schedule_map = {r["rung"]: r for r in schedule}
        self.max_rung = max(r["rung"] for r in schedule)

        # Initialize all configs at rung 1
        self.trials = {
            i: {
                "config_id": i,
                "config": cfg,
                "rung": 1,
                "budget": schedule[0]["budget"],
                "score": None,
                "status": "pending",
                "rung_scores": {},   # rung -> score history
                "train_times": {},   # rung -> time taken
            }
            for i, cfg in enumerate(configs)
        }

    def get_pending(self, rung: int) -> list[dict]:
        """Get all trials pending evaluation at a given rung."""
        return [
            t for t in self.trials.values()
            if t["rung"] == rung and t["status"] == "pending"
        ]

    def record_result(self, config_id: int, score: float, train_time: float):
        """Record evaluation result for a trial."""
        trial = self.trials[config_id]
        trial["score"] = score
        trial["status"] = "evaluated"
        trial["rung_scores"][trial["rung"]] = score
        trial["train_times"][trial["rung"]] = train_time

    def promote_top_k(self, rung: int, eta: int = 3) -> tuple[list[int], list[int]]:
        """
        After all configs in a rung are evaluated:
        - Keep top 1/eta by score -> promote to next rung
        - Eliminate the rest
        Returns (promoted_ids, eliminated_ids)
        """
        evaluated = [
            t for t in self.trials.values()
            if t["rung"] == rung and t["status"] == "evaluated"
        ]

        if not evaluated:
            return [], []

        # Sort by score descending
        evaluated.sort(key=lambda t: t["score"], reverse=True)

        next_rung_info = self.schedule_map.get(rung + 1)
        if next_rung_info is None:
            # Final rung — mark all as promoted (winner) or eliminated
            promoted_ids = [evaluated[0]["config_id"]]
            eliminated_ids = [t["config_id"] for t in evaluated[1:]]
        else:
            k = next_rung_info["n_configs"]
            promoted_ids = [t["config_id"] for t in evaluated[:k]]
            eliminated_ids = [t["config_id"] for t in evaluated[k:]]

        next_budget = next_rung_info["budget"] if next_rung_info else None

        for cid in promoted_ids:
            t = self.trials[cid]
            t["status"] = "promoted"
            if next_rung_info:
                t["rung"] = rung + 1
                t["budget"] = next_budget
                t["score"] = None
                t["status"] = "pending"

        for cid in eliminated_ids:
            self.trials[cid]["status"] = "eliminated"

        return promoted_ids, eliminated_ids

    def is_rung_complete(self, rung: int) -> bool:
        """True if all configs at this rung have been evaluated."""
        pending = self.get_pending(rung)
        in_progress = [
            t for t in self.trials.values()
            if t["rung"] == rung and t["status"] not in ("evaluated", "promoted", "eliminated")
        ]
        return len(pending) == 0 and len(in_progress) == 0

    def get_winner(self) -> dict | None:
        """Return the surviving config after all rungs, or None."""
        final = [
            t for t in self.trials.values()
            if t["status"] == "promoted" and t["rung"] == self.max_rung
        ]
        if not final:
            # Check evaluated at final rung
            final = [
                t for t in self.trials.values()
                if t["rung"] == self.max_rung and t["status"] == "evaluated"
            ]
        return max(final, key=lambda t: t["score"]) if final else None

    def summary(self) -> dict:
        """Return a summary of all trials."""
        by_status = {}
        for t in self.trials.values():
            by_status.setdefault(t["status"], []).append(t)
        return by_status