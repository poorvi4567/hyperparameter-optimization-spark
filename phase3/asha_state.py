"""
Phase 3 - ASHA Rung State Manager (driver-side)

In synchronous SHA, the driver waits for ALL configs in a rung to finish
before promoting anyone. ASHA removes this barrier:

  - Each config is promoted THE MOMENT it finishes a rung,
    if its score beats the current rung threshold.
  - Workers never wait for peers. As soon as a worker finishes,
    it either gets promoted to the next rung OR a new config is
    dispatched to it immediately.

This class manages all rung state on the driver.
Workers report results back; the driver promotes instantly.
"""

import threading
import time
from collections import defaultdict


class ASHARungState:
    """
    Thread-safe rung state manager for ASHA.

    Key difference from SHA's RungTracker:
      SHA  — promote_top_k() is called ONCE after entire rung completes.
      ASHA — try_promote()   is called after EACH individual result arrives.
             Promotion happens immediately if the config beats current threshold.

    Rung promotion rule (ASHA paper, Li et al. 2018):
      A config at rung r is promoted if:
        score >= the 1/eta quantile of ALL scores seen at rung r so far.
      i.e. it's in the top 1/eta of everything evaluated at this rung,
      even if most configs haven't reached this rung yet.
    """

    def __init__(self, schedule: list[dict], eta: int = 3):
        self.schedule = schedule
        self.eta = eta
        self.schedule_map = {r["rung"]: r for r in schedule}
        self.max_rung = max(r["rung"] for r in schedule)

        # rung -> list of (config_id, score) seen so far
        self.rung_results: dict[int, list[tuple]] = defaultdict(list)

        # config_id -> current rung
        self.config_rung: dict[int, int] = {}

        # config_id -> status: "running" | "promoted" | "eliminated" | "winner"
        self.config_status: dict[int, str] = {}

        # config_id -> full score history {rung: score}
        self.config_scores: dict[int, dict] = defaultdict(dict)

        # config_id -> train times {rung: seconds}
        self.config_times: dict[int, dict] = defaultdict(dict)

        # Thread lock — results arrive concurrently from Spark workers
        self._lock = threading.Lock()

        # Timeline log for Gantt chart later
        self.timeline: list[dict] = []

    def register_config(self, config_id: int):
        """Called when a config starts rung 1."""
        with self._lock:
            self.config_rung[config_id] = 1
            self.config_status[config_id] = "running"

    def record_and_promote(
        self,
        config_id: int,
        rung: int,
        score: float,
        train_time: float,
        wall_time: float,
    ) -> tuple[bool, int | None]:
        """
        Called when a worker finishes evaluating config_id at rung.
        Immediately decides: promote or eliminate.

        Returns (promoted: bool, next_rung: int | None)

        ASHA promotion rule:
          Promoted if score >= 1/eta quantile of all rung scores seen so far.
          Equivalently: at least 1 - 1/eta fraction of configs scored LOWER.
        """
        with self._lock:
            self.config_scores[config_id][rung] = round(score, 4)
            self.config_times[config_id][rung] = round(train_time, 2)
            self.rung_results[rung].append((config_id, score))

            self.timeline.append({
                "config_id": config_id,
                "rung": rung,
                "score": round(score, 4),
                "train_time": round(train_time, 2),
                "wall_time": round(wall_time, 2),
                "action": None,  # filled below
            })

            # Can't promote beyond max rung
            if rung >= self.max_rung:
                self.config_status[config_id] = "winner"
                self.timeline[-1]["action"] = "winner"
                return False, None

            # ASHA promotion rule:
            # Among all scores seen at this rung, keep top 1/eta
            all_scores = [s for _, s in self.rung_results[rung]]
            all_scores_sorted = sorted(all_scores, reverse=True)

            # Threshold = score at position ceil(n/eta) - 1
            n = len(all_scores_sorted)
            k = max(1, n // self.eta)  # top k get promoted
            threshold = all_scores_sorted[k - 1]

            promoted = score >= threshold

            if promoted:
                next_rung = rung + 1
                self.config_rung[config_id] = next_rung
                self.config_status[config_id] = "running"
                self.timeline[-1]["action"] = f"promoted→rung{next_rung}"
            else:
                self.config_status[config_id] = "eliminated"
                self.timeline[-1]["action"] = "eliminated"

            return promoted, (rung + 1 if promoted else None)

    def get_winner(self) -> tuple[int, float] | None:
        """Return (config_id, best_score) of the best config seen."""
        with self._lock:
            all_scores = []
            for cid, scores in self.config_scores.items():
                if scores:
                    best = max(scores.values())
                    all_scores.append((cid, best))
            if not all_scores:
                return None
            return max(all_scores, key=lambda x: x[1])

    def snapshot(self) -> dict:
        """Thread-safe snapshot of current state for logging."""
        with self._lock:
            return {
                "rung_counts": {r: len(v) for r, v in self.rung_results.items()},
                "statuses": dict(self.config_status),
                "scores": {
                    cid: dict(scores)
                    for cid, scores in self.config_scores.items()
                },
            }