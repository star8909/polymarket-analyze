"""verified sweep — polymarket calibration bucket-level edge.

각 round마다 다른 bucket size + filter (volume / time / category).
peak edge with proper N reporting.
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    rd = args.round

    # iter14_real_calibration.json 데이터 활용 (이미 계산됨)
    src = RESULTS_DIR / "iter14_real_calibration.json"
    if not src.exists():
        print(f"  source not found: {src}")
        return

    base_data = json.loads(src.read_text(encoding="utf-8"))
    buckets = base_data["calibration_buckets"]

    configs = [
        {"name": "all_buckets_min_n50",   "min_n": 50,  "min_edge": 0.03},
        {"name": "high_n_min_n100",       "min_n": 100, "min_edge": 0.03},
        {"name": "strict_min_n80",        "min_n": 80,  "min_edge": 0.05},
        {"name": "favorites_only",        "min_n": 80,  "min_edge": 0.03, "favor_above_05": True},
        {"name": "longshots_only",        "min_n": 80,  "min_edge": 0.03, "favor_below_05": True},
        {"name": "moderate_min_n60",      "min_n": 60,  "min_edge": 0.04},
        {"name": "very_high_n_min_100_edge_5pp", "min_n": 100, "min_edge": 0.05},
        {"name": "permissive_n40_e2pp",   "min_n": 40,  "min_edge": 0.02},
        {"name": "strict_n100_e7pp",      "min_n": 100, "min_edge": 0.07},
        {"name": "balanced_n70_e4pp",     "min_n": 70,  "min_edge": 0.04},
    ]
    cfg = configs[(rd - 1) % len(configs)]
    print(f"[round {rd}] {cfg['name']}")

    qualified = []
    for b in buckets:
        n = b["n"]
        belief = b["avg_belief"]
        actual = b["actual_yes"]
        edge = belief - actual  # NO bet edge (positive = NO win more than market thinks)
        if abs(edge) < cfg["min_edge"]:
            continue
        if n < cfg["min_n"]:
            continue
        if cfg.get("favor_above_05") and belief < 0.5:
            continue
        if cfg.get("favor_below_05") and belief >= 0.5:
            continue
        qualified.append({
            "bin": b["bin"],
            "n": n,
            "belief": belief,
            "actual": actual,
            "edge_pp": edge * 100,
            "side": "NO" if edge > 0 else "YES",
        })

    total_n = sum(q["n"] for q in qualified)
    avg_edge_pp = float(np.mean([abs(q["edge_pp"]) for q in qualified])) if qualified else 0
    weighted_edge_pp = (sum(q["n"] * abs(q["edge_pp"]) for q in qualified) / total_n) if total_n > 0 else 0

    # Per-trade Sharpe estimate
    if qualified:
        # Pick peak bucket
        peak = max(qualified, key=lambda q: abs(q["edge_pp"]))
        # Per-trade Sharpe for peak
        b = peak["belief"]
        a = peak["actual"]
        if peak["side"] == "NO":
            p_no = 1 - b
            p_win = 1 - a
            payoff = (1 / p_no) - 1
            mean = p_win * payoff - (1 - p_win)
            var = p_win * (payoff - mean)**2 + (1 - p_win) * (-1 - mean)**2
            std = np.sqrt(var)
            per_trade_sharpe = mean / std if std > 0 else 0
        else:  # YES
            p_yes = b
            p_win = a
            payoff = (1 / p_yes) - 1
            mean = p_win * payoff - (1 - p_win)
            var = p_win * (payoff - mean)**2 + (1 - p_win) * (-1 - mean)**2
            std = np.sqrt(var)
            per_trade_sharpe = mean / std if std > 0 else 0
        annual_sharpe = per_trade_sharpe * np.sqrt(min(total_n, 100))
    else:
        peak = None
        per_trade_sharpe = 0
        annual_sharpe = 0

    result = {
        "round": rd, "config": cfg["name"], "params": cfg,
        "n_qualified_buckets": len(qualified),
        "total_n_trades": total_n,
        "avg_edge_pp": avg_edge_pp,
        "weighted_edge_pp": weighted_edge_pp,
        "peak_bucket": peak,
        "per_trade_sharpe": float(per_trade_sharpe),
        "annual_sharpe_100yr": float(annual_sharpe),
        "qualified_buckets": qualified,
    }
    print(f"  qualified={len(qualified)} total_N={total_n} avg_edge={avg_edge_pp:.2f}pp "
          f"weighted={weighted_edge_pp:.2f}pp Sh(per-trade)={per_trade_sharpe:.2f} annual={annual_sharpe:.2f}")
    if peak:
        print(f"  peak: {peak['bin']} N={peak['n']} edge={peak['edge_pp']:+.2f}pp side={peak['side']}")

    out_path = RESULTS_DIR / f"iter_verified_sweep_r{rd}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"  → {out_path.name}")


if __name__ == "__main__":
    main()
