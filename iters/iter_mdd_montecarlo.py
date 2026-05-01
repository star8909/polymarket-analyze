"""Polymarket MDD Monte Carlo — calibration bet sequence 시뮬레이션.

Closed markets 데이터는 시간순 없음 → random 순서로 N번 베팅 시뮬레이션 → MDD 분포.

각 round마다 다른 strategy + Kelly fraction 조합:
1. iter14 peak (0.30-0.40 NO) Kelly 0.25
2. iter14 peak Kelly 0.5
3. iter14 peak full Kelly
4. longshots_only 0.25
5. balanced 모든 buckets 0.25
6. favorites_only 0.25
7. high-edge only (5pp+) 0.25
8. peak iter17 (0.35-0.40 NO N=44) 0.25
9. mixed N=200+ 0.25
10. Conservative N=300+ 0.10
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


def load_buckets():
    src = RESULTS_DIR / "iter14_real_calibration.json"
    if not src.exists():
        return []
    return json.loads(src.read_text(encoding="utf-8"))["calibration_buckets"]


def load_buckets_05():
    src = RESULTS_DIR / "iter17_precise_buckets.json"
    if not src.exists():
        return []
    return json.loads(src.read_text(encoding="utf-8")).get("buckets_005", [])


def filter_buckets(buckets, edge_min=0.03, n_min=50, side=None,
                   belief_min=0.0, belief_max=1.0):
    out = []
    for b in buckets:
        belief = b.get("avg_belief", b.get("belief"))
        actual = b.get("actual_yes", b.get("actual"))
        if belief is None or actual is None:
            continue
        edge = belief - actual
        if abs(edge) < edge_min:
            continue
        if b["n"] < n_min:
            continue
        if not (belief_min <= belief <= belief_max):
            continue
        s = "NO" if edge > 0 else "YES"
        if side and s != side:
            continue
        out.append({"belief": belief, "actual": actual, "n": b["n"], "side": s})
    return out


def simulate_mdd(buckets, kelly_frac=0.25, n_sims=1000, seed=42):
    """Monte Carlo: random 순서 베팅 → equity curve → MDD."""
    if not buckets:
        return None
    rng = np.random.RandomState(seed)
    # 베팅 한번씩: 각 bucket의 N만큼 베팅 발생 가정
    bets = []
    for b in buckets:
        belief, actual, side = b["belief"], b["actual"], b["side"]
        # Kelly fraction
        if side == "NO":
            p_no = 1 - belief
            p_win = 1 - actual
            payoff = 1.0 / p_no - 1
        else:
            p_yes = belief
            p_win = actual
            payoff = 1.0 / p_yes - 1
        # Outcomes for this bucket: n bets, each with p_win win probability
        for _ in range(b["n"]):
            bets.append((p_win, payoff, kelly_frac))

    if not bets:
        return None
    n_bets = len(bets)

    mdds = []
    final_returns = []
    for s in range(n_sims):
        rng2 = np.random.RandomState(seed + s)
        order = rng2.permutation(n_bets)
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for idx in order:
            p_win, payoff, k = bets[idx]
            outcome = rng2.random() < p_win
            if outcome:
                equity *= (1 + k * payoff)
            else:
                equity *= (1 - k)
            peak = max(peak, equity)
            dd = equity / peak - 1
            max_dd = min(max_dd, dd)
        mdds.append(max_dd)
        final_returns.append(equity - 1)
    mdds = np.array(mdds)
    finals = np.array(final_returns)
    return {
        "n_bets": n_bets,
        "n_sims": n_sims,
        "kelly_frac": kelly_frac,
        "mdd_mean_pct": float(mdds.mean() * 100),
        "mdd_p5_pct": float(np.percentile(mdds, 5) * 100),
        "mdd_p50_pct": float(np.percentile(mdds, 50) * 100),
        "mdd_p95_pct": float(np.percentile(mdds, 95) * 100),
        "mdd_worst_pct": float(mdds.min() * 100),
        "final_return_mean_pct": float(finals.mean() * 100),
        "final_return_p5_pct": float(np.percentile(finals, 5) * 100),
        "final_return_p50_pct": float(np.percentile(finals, 50) * 100),
        "final_return_p95_pct": float(np.percentile(finals, 95) * 100),
        "win_pct": float((finals > 0).mean() * 100),
    }


CONFIGS = [
    ("iter14_peak_NO_k025", {"buckets": "iter14", "filt": {"edge_min": 0.10, "n_min": 80, "side": "NO", "belief_min": 0.25, "belief_max": 0.45}, "kelly": 0.25}),
    ("iter14_peak_NO_k050", {"buckets": "iter14", "filt": {"edge_min": 0.10, "n_min": 80, "side": "NO", "belief_min": 0.25, "belief_max": 0.45}, "kelly": 0.50}),
    ("iter14_peak_NO_k100", {"buckets": "iter14", "filt": {"edge_min": 0.10, "n_min": 80, "side": "NO", "belief_min": 0.25, "belief_max": 0.45}, "kelly": 1.00}),
    ("longshots_only_k025", {"buckets": "iter14", "filt": {"edge_min": 0.05, "n_min": 80, "belief_max": 0.5}, "kelly": 0.25}),
    ("balanced_all_k025",   {"buckets": "iter14", "filt": {"edge_min": 0.04, "n_min": 70}, "kelly": 0.25}),
    ("favorites_k025",      {"buckets": "iter14", "filt": {"edge_min": 0.05, "n_min": 80, "belief_min": 0.5}, "kelly": 0.25}),
    ("high_edge_5pp_k025",  {"buckets": "iter14", "filt": {"edge_min": 0.05, "n_min": 80}, "kelly": 0.25}),
    ("iter17_peak_NO_k025", {"buckets": "iter17", "filt": {"edge_min": 0.10, "n_min": 30, "side": "NO"}, "kelly": 0.25}),
    ("mixed_n200_k025",     {"buckets": "iter14", "filt": {"edge_min": 0.04, "n_min": 200}, "kelly": 0.25}),
    ("conservative_n300_k010", {"buckets": "iter14", "filt": {"edge_min": 0.03, "n_min": 300}, "kelly": 0.10}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    rd = args.round

    name, cfg = CONFIGS[(rd - 1) % len(CONFIGS)]
    print(f"[mdd_mc round {rd}] {name}")

    if cfg["buckets"] == "iter17":
        all_buckets = load_buckets_05()
        # iter17 has different field names
        for b in all_buckets:
            if "edge" in b and "n" in b and "belief" in b and "actual" in b:
                b["avg_belief"] = b["belief"]
                b["actual_yes"] = b["actual"]
    else:
        all_buckets = load_buckets()

    if not all_buckets:
        print("  no buckets")
        return

    qualified = filter_buckets(all_buckets, **cfg["filt"])
    if not qualified:
        print("  no qualified buckets")
        result = {"round": rd, "strategy": name, "skipped": True}
    else:
        print(f"  qualified={len(qualified)} total_n={sum(b['n'] for b in qualified)}")
        mc = simulate_mdd(qualified, kelly_frac=cfg["kelly"], n_sims=1000)
        if mc:
            print(f"  N={mc['n_bets']} kelly={mc['kelly_frac']:.2f}")
            print(f"  MDD: mean={mc['mdd_mean_pct']:.1f}% p5={mc['mdd_p5_pct']:.1f}% p95={mc['mdd_p95_pct']:.1f}% worst={mc['mdd_worst_pct']:.1f}%")
            print(f"  Final: mean={mc['final_return_mean_pct']:+.1f}% p5={mc['final_return_p5_pct']:+.1f}% p95={mc['final_return_p95_pct']:+.1f}% win={mc['win_pct']:.0f}%")
            result = {"round": rd, "strategy": name, "qualified_buckets": qualified, **mc}
        else:
            result = {"round": rd, "strategy": name, "skipped": True}

    out_path = RESULTS_DIR / f"iter_mdd_montecarlo_r{rd}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"  → {out_path.name}")


if __name__ == "__main__":
    main()
