"""Polymarket 다양한 전략 — calibration 외 다른 alpha 시도.

20가지 전략 (10개 핵심 필터 + 10개 cross-section/asymmetric/extreme):
1-10. 핵심 longshot/favorite/edge/N 조합
11-20. 극단 longshot, ultra-conservative, sweet-spot, 비대칭 edge 등
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


def load_calibration_buckets():
    src = RESULTS_DIR / "iter14_real_calibration.json"
    if not src.exists():
        return []
    return json.loads(src.read_text(encoding="utf-8"))["calibration_buckets"]


def per_trade_sharpe(belief, actual, side):
    """Binary bet Sharpe: NO at price (1-belief) wins (1-actual). YES at belief wins actual."""
    if side == "NO":
        p_no = 1 - belief
        p_win = 1 - actual
        if p_no <= 0 or p_no >= 1:
            return 0
        payoff = 1.0 / p_no - 1
    else:
        p_yes = belief
        p_win = actual
        if p_yes <= 0 or p_yes >= 1:
            return 0
        payoff = 1.0 / p_yes - 1
    mean = p_win * payoff - (1 - p_win)
    var = p_win * (payoff - mean)**2 + (1 - p_win) * (-1 - mean)**2
    std = np.sqrt(var) if var > 0 else 1
    return mean / std if std > 0 else 0


def evaluate(buckets, edge_min=0.03, n_min=50, side_filter=None,
             belief_min=0.0, belief_max=1.0):
    """Filter buckets, return summary."""
    qualified = []
    for b in buckets:
        belief = b["avg_belief"]
        actual = b["actual_yes"]
        edge = belief - actual
        if abs(edge) < edge_min:
            continue
        if b["n"] < n_min:
            continue
        if not (belief_min <= belief <= belief_max):
            continue
        side = "NO" if edge > 0 else "YES"
        if side_filter and side != side_filter:
            continue
        sh = per_trade_sharpe(belief, actual, side)
        qualified.append({
            "bin": b["bin"], "n": b["n"], "belief": belief, "actual": actual,
            "edge_pp": edge * 100, "side": side, "per_trade_sh": sh,
        })
    if not qualified:
        return None
    total_n = sum(q["n"] for q in qualified)
    weighted_edge = sum(q["n"] * abs(q["edge_pp"]) for q in qualified) / total_n
    weighted_sh = sum(q["n"] * q["per_trade_sh"] for q in qualified) / total_n
    return {
        "n_buckets": len(qualified),
        "total_n": total_n,
        "weighted_edge_pp": weighted_edge,
        "weighted_per_trade_sh": weighted_sh,
        "annual_sh_100yr": weighted_sh * np.sqrt(min(total_n, 100)),
        "top_bucket": max(qualified, key=lambda q: abs(q["edge_pp"])),
    }


STRATEGIES = [
    # 1-10 기본 longshot/favorite/edge 조합
    ("longshots_only_strict",   {"edge_min": 0.05, "n_min": 80, "belief_max": 0.5}),
    ("favorites_only_strict",   {"edge_min": 0.05, "n_min": 80, "belief_min": 0.5}),
    ("tail_buckets_no_only",    {"edge_min": 0.03, "n_min": 50, "belief_max": 0.4, "side_filter": "NO"}),
    ("mid_range_avoid",         {"edge_min": 0.05, "n_min": 60, "belief_min": 0.0, "belief_max": 0.45}),
    ("very_strict_n200_e8",     {"edge_min": 0.08, "n_min": 200}),
    ("permissive_n30_e2",       {"edge_min": 0.02, "n_min": 30}),
    ("yes_side_only",           {"edge_min": 0.04, "n_min": 50, "side_filter": "YES"}),
    ("no_side_only",            {"edge_min": 0.04, "n_min": 50, "side_filter": "NO"}),
    ("balanced_med_n100",       {"edge_min": 0.04, "n_min": 100, "belief_min": 0.15, "belief_max": 0.85}),
    ("extreme_edges_n40",       {"edge_min": 0.10, "n_min": 40}),
    # 11-20 새 전략 (극단/비대칭/sweet-spot/ultra-conservative)
    ("ultra_longshot_p015",     {"edge_min": 0.04, "n_min": 60, "belief_max": 0.15}),  # 15센트 이하 극단
    ("ultra_favorite_p085",     {"edge_min": 0.04, "n_min": 60, "belief_min": 0.85}),  # 85센트 이상
    ("sweet_spot_25_45",        {"edge_min": 0.06, "n_min": 80, "belief_min": 0.25, "belief_max": 0.45}),
    ("sweet_spot_55_75",        {"edge_min": 0.06, "n_min": 80, "belief_min": 0.55, "belief_max": 0.75}),
    ("ultra_conservative_n500", {"edge_min": 0.10, "n_min": 500}),  # 가장 robust
    ("longshot_strong_e12",     {"edge_min": 0.12, "n_min": 50, "belief_max": 0.35}),  # 12pp+ 큰 edge
    ("favorite_strong_e12",     {"edge_min": 0.12, "n_min": 50, "belief_min": 0.65}),
    ("asymmetric_no_aggressive",{"edge_min": 0.03, "n_min": 40, "belief_max": 0.5, "side_filter": "NO"}),
    ("asymmetric_yes_aggressive",{"edge_min": 0.03, "n_min": 40, "belief_min": 0.5, "side_filter": "YES"}),
    ("max_edge_nominal_n20",    {"edge_min": 0.15, "n_min": 20}),  # 가장 큰 edge만, N 적어도 OK
    # Tier-S 알파메일 (longshot 깊이별 + N strict)
    ("deep_longshot_p010_e8",   {"edge_min": 0.08, "n_min": 100, "belief_max": 0.10}),  # 10센트 이하 강한 edge
    ("longshot_p020_n300",      {"edge_min": 0.06, "n_min": 300, "belief_max": 0.20}),  # n 풍부 + 깊은 longshot
    ("favorite_p080_n300",      {"edge_min": 0.06, "n_min": 300, "belief_min": 0.80}),  # 80센트+ favorites
    ("balanced_e8_n400",        {"edge_min": 0.08, "n_min": 400, "belief_min": 0.10, "belief_max": 0.90}),  # robust
    ("triple_tier_strict",      {"edge_min": 0.07, "n_min": 250, "side_filter": "NO"}),  # NO만 + strict
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    rd = args.round

    name, params = STRATEGIES[(rd - 1) % len(STRATEGIES)]
    print(f"[diverse round {rd}] {name}")

    buckets = load_calibration_buckets()
    if not buckets:
        print("  no calibration source")
        return

    r = evaluate(buckets, **params)
    if r is None:
        print("  no qualified buckets")
        result = {"round": rd, "strategy": name, "skipped": True, "params": params}
    else:
        print(f"  buckets={r['n_buckets']} N={r['total_n']} edge={r['weighted_edge_pp']:+.2f}pp "
              f"per-trade Sh={r['weighted_per_trade_sh']:.3f} annual={r['annual_sh_100yr']:.2f}")
        print(f"  peak: {r['top_bucket']['bin']} N={r['top_bucket']['n']} "
              f"edge={r['top_bucket']['edge_pp']:+.2f}pp ({r['top_bucket']['side']})")
        result = {"round": rd, "strategy": name, "params": params, **r}

    out_path = RESULTS_DIR / f"iter_diverse_strategies_r{rd}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"  → {out_path.name}")


if __name__ == "__main__":
    main()
