"""iter13: iter05 + iter09 메인 전략 PnL 시뮬레이션.

iter05 (152 negRisk arb) + iter09 (101 stuck markets) 가 작동하는지
종료 마켓 데이터로 backtest.

방법:
1. closed markets 중 거래량 큰 마켓 sample
2. 각 마켓이 종료 7일 전 어떤 가격이었는지 추정 (oneWeekPriceChange 활용)
3. 그 가격 + Kelly 0.25x 베팅 시 누적 PnL 계산

가설: stuck markets 매일 5~10 베팅 + 평균 edge 2.5% × Kelly 0.25 →
   연 수익률 30~70% (실거래 슬리피지 포함)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, RESULTS_DIR


def parse_settle(row):
    try:
        prices_str = row.get('outcomePrices', '[]')
        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
        if not isinstance(prices, list) or len(prices) != 2:
            return None
        prices_f = [float(p) for p in prices]
        if prices_f[0] > 0.99:
            return 1
        if prices_f[1] > 0.99:
            return 0
        return None
    except Exception:
        return None


def kelly_size(p, b, fraction=0.25):
    """Kelly criterion. f* = (p × b - (1-p)) / b. fraction = Kelly의 1/4 등."""
    if b <= 0:
        return 0
    f = (p * b - (1 - p)) / b
    return max(0, f * fraction)


def simulate_stuck_strategy(df, capital=10000, edge_threshold=0.01, kelly_fraction=0.25):
    """iter09 stuck markets 시뮬레이션.

    각 마켓 종료 7일 전 가격이 0.95+ 또는 0.05-였으면 베팅.
    승: payoff 1.0 - bet_price (Yes 매수) or bet_price (No 매수)
    패: -bet_price (Yes 매수) or -(1-bet_price) (No 매수)
    """
    pnl_per_market = []
    bets = []
    cap = capital

    for _, row in df.iterrows():
        last = float(row.get('lastTradePrice', 0)) if pd.notna(row.get('lastTradePrice')) else 0
        change_w = row.get('oneWeekPriceChange', None)
        if pd.isna(change_w):
            continue
        change_w = float(change_w)
        if (1 + change_w) < 0.01:
            continue
        # 1주일 전 가격 추정
        prev_price = last / (1 + change_w)
        if prev_price < 0 or prev_price > 1:
            continue

        actual = row.get('actual_yes')
        if pd.isna(actual):
            continue
        actual = int(actual)

        # iter09 logic: stuck (0.95+ Yes 매수 또는 0.05- No 매수)
        if prev_price >= 0.95:
            # Yes 매수
            edge = (1 - prev_price) - 0.005  # 슬리피지 0.5% 가정
            if edge < edge_threshold:
                continue
            payout = 1 - prev_price if actual == 1 else -prev_price
            implied_prob = 0.99
            decimal_odds = 1 / prev_price - 1
            f = kelly_size(implied_prob, decimal_odds, kelly_fraction)
            bet_size = cap * f
            pnl = bet_size * payout / prev_price
            cap += pnl
            pnl_per_market.append(pnl)
            bets.append({'price': prev_price, 'side': 'Yes', 'actual': actual, 'pnl': pnl, 'cap': cap})
        elif prev_price <= 0.05:
            # No 매수
            edge = prev_price - 0.005
            if edge < edge_threshold:
                continue
            payout = -(1 - prev_price) if actual == 1 else prev_price
            implied_prob = 0.99
            decimal_odds = 1 / (1 - prev_price) - 1
            f = kelly_size(implied_prob, decimal_odds, kelly_fraction)
            bet_size = cap * f
            pnl = bet_size * payout / (1 - prev_price)
            cap += pnl
            pnl_per_market.append(pnl)
            bets.append({'price': prev_price, 'side': 'No', 'actual': actual, 'pnl': pnl, 'cap': cap})

    return pnl_per_market, bets, cap


def main():
    print("[iter13] iter09 stuck markets 시뮬레이션 (Kelly 0.25x)")
    cache_path = CACHE_DIR / "closed_all.parquet"
    if not cache_path.exists():
        print(f"  ❌ {cache_path} 없음.")
        return

    df = pd.read_parquet(cache_path)
    print(f"  종료 마켓: {len(df)}")

    df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df = df[df['volume_num'] >= 50000].copy()
    print(f"  vol 50k+: {len(df)}")

    df['actual_yes'] = df.apply(parse_settle, axis=1)
    df = df.dropna(subset=['actual_yes'])
    df['actual_yes'] = df['actual_yes'].astype(int)
    df['lastTradePrice'] = pd.to_numeric(df['lastTradePrice'], errors='coerce')
    df['oneWeekPriceChange'] = pd.to_numeric(df['oneWeekPriceChange'], errors='coerce')

    print(f"  settle 확정: {len(df)}")

    # 시뮬레이션
    INITIAL = 10000
    pnl_list, bets, final_cap = simulate_stuck_strategy(df, capital=INITIAL, kelly_fraction=0.25)
    print(f"\n=== 시뮬레이션 결과 ===")
    print(f"  초기자본: ${INITIAL:,}")
    print(f"  최종자본: ${final_cap:,.0f}")
    print(f"  총 수익: ${final_cap - INITIAL:,.0f} ({(final_cap/INITIAL - 1)*100:.1f}%)")
    print(f"  총 베팅: {len(bets)}")
    if bets:
        wins = sum(1 for b in bets if b['pnl'] > 0)
        losses = sum(1 for b in bets if b['pnl'] < 0)
        win_rate = wins / len(bets) * 100
        print(f"  승: {wins}, 패: {losses}, 승률: {win_rate:.1f}%")
        avg_pnl = np.mean(pnl_list)
        print(f"  평균 PnL/베팅: ${avg_pnl:.2f}")

    # Kelly sweep
    print(f"\n=== Kelly fraction sweep ===")
    for k in [0.10, 0.25, 0.50, 1.0]:
        _, _, cap_k = simulate_stuck_strategy(df, capital=INITIAL, kelly_fraction=k)
        print(f"  Kelly {k:.2f}x: ${INITIAL:,} → ${cap_k:,.0f} ({(cap_k/INITIAL-1)*100:+.1f}%)")

    out = {
        "initial_capital": INITIAL,
        "final_capital": float(final_cap),
        "total_return_pct": float((final_cap / INITIAL - 1) * 100),
        "n_bets": len(bets),
        "win_rate_pct": float(win_rate) if bets else 0,
        "avg_pnl_per_bet": float(np.mean(pnl_list)) if pnl_list else 0,
    }
    out_path = RESULTS_DIR / "iter13_simulate.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
