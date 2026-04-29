"""iter16: iter14 calibration 기반 Kelly sizing 1년 시뮬레이션.

iter14/15에서 확인:
- 0.30~0.40 가격 마켓: No 베팅 시 평균 edge -11.9%
- 0.60~0.70 가격 마켓: Yes 베팅 시 평균 edge +7.1%

iter16 시뮬:
- 모든 closing belief 기반 베팅 → Kelly 0.25x로 사이즈
- $10k 자본으로 1년 시뮬
- 누적 PnL + 승률 + Sharpe
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR


def kelly_size(p: float, b: float, fraction: float = 0.25) -> float:
    """Kelly criterion. p = 진짜 확률, b = decimal odds (= (1-price)/price for Yes 매수)."""
    if b <= 0:
        return 0
    f = (p * b - (1 - p)) / b
    return max(0, min(f * fraction, 0.10))  # max 10% per bet


def main():
    print("[iter16] iter14 calibration 기반 Kelly 시뮬레이션")

    # iter14 calibration 곡선 (price → actual_yes 비율)
    calibration_curve = {
        # (price_min, price_max): observed_actual_rate
        (0.0, 0.05): 0.005,   # 롱샷 (시장 정확)
        (0.05, 0.10): 0.020,
        (0.10, 0.20): 0.074,  # iter14: 0.146 belief → 0.074 actual = -7.1% edge
        (0.20, 0.30): 0.150,  # 추정
        (0.30, 0.40): 0.227,  # iter14: 0.346 → 0.227 = -11.9% edge ⭐
        (0.40, 0.50): 0.381,  # iter14: 0.457 → 0.381 = -7.7%
        (0.50, 0.60): 0.600,  # iter14: 0.549 → 0.600 = +5.1%
        (0.60, 0.70): 0.718,  # iter14: 0.646 → 0.718 = +7.1%
        (0.70, 0.80): 0.800,  # 추정
        (0.80, 0.90): 0.900,  # 추정
        (0.90, 1.0): 0.977,   # iter14
    }

    def true_prob(price: float) -> float:
        for (low, high), actual in calibration_curve.items():
            if low <= price < high:
                return actual
        if price >= 1.0:
            return 1.0
        return price

    # 시뮬레이션 (가상 1년 = 365일 × ~3 베팅)
    np.random.seed(42)
    INITIAL = 10000
    cap = INITIAL
    n_days = 365
    bets_per_day = 3  # 평균 매일 3 베팅

    pnl_list = []
    bet_log = []

    for day in range(n_days):
        for _ in range(bets_per_day):
            # 시장에서 가격 선택 (가격 분포: closing belief 분포 가정)
            # 실제 polymarket의 종가 분포는 0.0 또는 1.0에 몰림 (대부분 settle 확정 마켓)
            # 중간 가격 (0.10~0.90)에서 베팅
            r = np.random.random()
            if r < 0.4:
                market_price = np.random.uniform(0.10, 0.50)  # 40% 베팅이 0.1-0.5 영역
            elif r < 0.7:
                market_price = np.random.uniform(0.50, 0.80)  # 30%
            else:
                market_price = np.random.uniform(0.10, 0.90)  # 30% 랜덤

            # 진짜 확률 (calibration 기반)
            p_true = true_prob(market_price)
            edge = p_true - market_price

            # 의사결정: edge > 2% (수수료 + slippage)일 때만 베팅
            if abs(edge) < 0.02:
                continue

            if edge < 0:
                # 시장 과대평가 → No 매수
                no_price = 1 - market_price
                p_no_true = 1 - p_true
                b = market_price / no_price  # decimal odds for No
                f = kelly_size(p_no_true, b, fraction=0.25)
                bet_size = cap * f
                if bet_size < 1:
                    continue
                # 결과 시뮬: actual is Yes with prob p_true
                actual_yes = np.random.random() < p_true
                if actual_yes:
                    pnl = -bet_size  # No 매수 손실
                else:
                    # No 매수 페이오프: bet_size * (1/no_price - 1)
                    pnl = bet_size * (1 - no_price) / no_price
                cap += pnl
                pnl_list.append(pnl)
                bet_log.append({'side': 'No', 'price': market_price, 'edge': edge, 'pnl': pnl, 'cap': cap})
            else:
                # 시장 과소평가 → Yes 매수
                b = (1 - market_price) / market_price
                f = kelly_size(p_true, b, fraction=0.25)
                bet_size = cap * f
                if bet_size < 1:
                    continue
                actual_yes = np.random.random() < p_true
                if actual_yes:
                    pnl = bet_size * (1 - market_price) / market_price
                else:
                    pnl = -bet_size
                cap += pnl
                pnl_list.append(pnl)
                bet_log.append({'side': 'Yes', 'price': market_price, 'edge': edge, 'pnl': pnl, 'cap': cap})

    print(f"\n=== Kelly 0.25x 시뮬 (1년, 평균 일 3 베팅) ===")
    print(f"  초기 자본: ${INITIAL:,}")
    print(f"  최종 자본: ${cap:,.0f}")
    print(f"  총 수익: ${cap - INITIAL:,.0f} ({(cap/INITIAL - 1)*100:.1f}%)")
    print(f"  총 베팅: {len(bet_log)}")
    if bet_log:
        wins = sum(1 for b in bet_log if b['pnl'] > 0)
        win_rate = wins / len(bet_log) * 100
        print(f"  승률: {win_rate:.1f}%")
        avg_pnl = np.mean(pnl_list)
        std_pnl = np.std(pnl_list)
        sharpe = avg_pnl / std_pnl * np.sqrt(252 * bets_per_day) if std_pnl > 0 else 0
        print(f"  평균 PnL/베팅: ${avg_pnl:.2f} (std={std_pnl:.2f}, sharpe={sharpe:.2f})")

    # Kelly fraction sweep
    print(f"\n=== Kelly fraction sweep ===")
    for k in [0.10, 0.25, 0.50, 1.0]:
        np.random.seed(42)
        cap_k = INITIAL
        for day in range(n_days):
            for _ in range(bets_per_day):
                r = np.random.random()
                if r < 0.4:
                    market_price = np.random.uniform(0.10, 0.50)
                elif r < 0.7:
                    market_price = np.random.uniform(0.50, 0.80)
                else:
                    market_price = np.random.uniform(0.10, 0.90)
                p_true = true_prob(market_price)
                edge = p_true - market_price
                if abs(edge) < 0.02:
                    continue
                if edge < 0:
                    no_price = 1 - market_price
                    p_no_true = 1 - p_true
                    b = market_price / no_price
                    f = kelly_size(p_no_true, b, fraction=k)
                    bet_size = cap_k * f
                    if bet_size < 1:
                        continue
                    actual_yes = np.random.random() < p_true
                    pnl = -bet_size if actual_yes else bet_size * (1 - no_price) / no_price
                else:
                    b = (1 - market_price) / market_price
                    f = kelly_size(p_true, b, fraction=k)
                    bet_size = cap_k * f
                    if bet_size < 1:
                        continue
                    actual_yes = np.random.random() < p_true
                    pnl = bet_size * (1 - market_price) / market_price if actual_yes else -bet_size
                cap_k += pnl
        print(f"  Kelly {k:.2f}x: ${INITIAL:,} → ${cap_k:,.0f} ({(cap_k/INITIAL-1)*100:+.1f}%)")

    out = {
        "initial_capital": INITIAL,
        "final_capital": float(cap),
        "total_return_pct": float((cap / INITIAL - 1) * 100),
        "n_bets": len(bet_log),
        "win_rate_pct": float(win_rate) if bet_log else 0,
    }
    out_path = RESULTS_DIR / "iter16_kelly_simulate.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
