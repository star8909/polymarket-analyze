"""verified sweep — 50 configs (polymarket calibration + MDD MC)."""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.mdd_sim import simulate_mdd_realistic


def load_buckets():
    src = RESULTS_DIR / "iter14_real_calibration.json"
    if not src.exists():
        return []
    return json.loads(src.read_text(encoding="utf-8"))["calibration_buckets"]


def per_trade_sharpe(belief, actual, side):
    if side == "NO":
        p_no = 1 - belief
        p_win = 1 - actual
        if p_no <= 0 or p_no >= 1: return 0
        payoff = 1.0 / p_no - 1
    else:
        p_yes = belief
        p_win = actual
        if p_yes <= 0 or p_yes >= 1: return 0
        payoff = 1.0 / p_yes - 1
    mean = p_win * payoff - (1 - p_win)
    var = p_win * (payoff - mean) ** 2 + (1 - p_win) * (-1 - mean) ** 2
    std = np.sqrt(var) if var > 0 else 1
    return mean / std if std > 0 else 0


def filter_buckets(buckets, edge_min=0.03, n_min=50, side=None,
                   belief_min=0.0, belief_max=1.0):
    out = []
    for b in buckets:
        belief = b["avg_belief"]; actual = b["actual_yes"]
        edge = belief - actual
        if abs(edge) < edge_min or b["n"] < n_min:
            continue
        if not (belief_min <= belief <= belief_max):
            continue
        s = "NO" if edge > 0 else "YES"
        if side and s != side: continue
        out.append({"belief": belief, "actual": actual, "n": b["n"], "side": s,
                    "edge_pp": edge * 100,
                    "per_trade_sh": per_trade_sharpe(belief, actual, s)})
    return out


def simulate_mdd(qualified, kelly=0.25, n_sims=500, seed=42):
    return simulate_mdd_realistic(qualified, kelly=kelly, n_sims=n_sims, seed=seed)


# 50 configs: filter combinations × Kelly fraction
CONFIGS = []
filters = [
    ("longshots_e5", {"edge_min": 0.05, "n_min": 80, "belief_max": 0.5}),
    ("longshots_e7", {"edge_min": 0.07, "n_min": 80, "belief_max": 0.5}),
    ("favorites_e5", {"edge_min": 0.05, "n_min": 80, "belief_min": 0.5}),
    ("favorites_e7", {"edge_min": 0.07, "n_min": 80, "belief_min": 0.5}),
    ("balanced_e4", {"edge_min": 0.04, "n_min": 70}),
    ("balanced_e5", {"edge_min": 0.05, "n_min": 70}),
    ("strict_e7_n100", {"edge_min": 0.07, "n_min": 100}),
    ("strict_e8_n100", {"edge_min": 0.08, "n_min": 100}),
    ("permissive_e3_n40", {"edge_min": 0.03, "n_min": 40}),
    ("very_strict_n200", {"edge_min": 0.05, "n_min": 200}),
]
kellys = [0.05, 0.10, 0.15, 0.25, 0.50]
for f_name, f in filters:
    for k in kellys:
        CONFIGS.append({"name": f"{f_name}_k{int(k*100)}", "filt": f, "kelly": k})

assert len(CONFIGS) == 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    rd = args.round

    cfg = CONFIGS[(rd - 1) % len(CONFIGS)]
    print(f"[round {rd}] {cfg['name']}")

    buckets = load_buckets()
    if not buckets:
        return

    qualified = filter_buckets(buckets, **cfg["filt"])
    if not qualified:
        result = {"round": rd, "strategy": cfg["name"], "skipped": True, "params": cfg}
    else:
        total_n = sum(q["n"] for q in qualified)
        weighted_edge = sum(q["n"] * abs(q["edge_pp"]) for q in qualified) / total_n
        weighted_sh = sum(q["n"] * q["per_trade_sh"] for q in qualified) / total_n
        mc = simulate_mdd(qualified, kelly=cfg["kelly"])
        result = {
            "round": rd, "strategy": cfg["name"], "params": cfg,
            "n_qualified_buckets": len(qualified),
            "total_n_trades": total_n,
            "weighted_edge_pp": weighted_edge,
            "per_trade_sharpe": weighted_sh,
            "annual_sharpe_100yr": weighted_sh * np.sqrt(min(total_n, 100)),
            "mc_mdd": mc,
        }
        print(f"  bkts={len(qualified)} N={total_n} edge={weighted_edge:.2f}pp "
              f"per-trade Sh={weighted_sh:.3f}")
        if mc:
            print(f"  MC MDD: mean={mc['mdd_mean_pct']:.1f}% p5={mc['mdd_p5_pct']:.1f}% "
                  f"worst={mc['mdd_worst_pct']:.1f}%")
            print(f"  MC Final: mean={mc['final_mean_pct']:+.0f}% win={mc['win_pct']:.0f}%")

    out_path = RESULTS_DIR / f"iter_verified_sweep_r{rd}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
