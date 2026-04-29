"""iter10: Liquidity / spread 차익.

가설: bid-ask spread가 큰 마켓은 시장 가격 미스프라이싱 가능성.
- 거래량 큰데 spread 5%+ → 시장 메이커 부족 → 호가 비대칭
- spread 큰 마켓에서 bid 매수 + ask 매도로 사실상 차익

근데 polymarket은 limit order로 직접 매수/매도 가능 →
- ask 가격에 매수 → bid 가격에 매도 = 음수 수익 (정상 trader)
- 하지만 spread가 비대칭(예: bid=0.30, ask=0.50, mid=0.40)인데
  실제 진짜 가치가 0.45면 → bid 가까이 limit buy 후 wait

이건 market making 전략. 자동화 봇 + 24h 운영 필요.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import fetch_active_markets


def main():
    print("[iter10] Spread / liquidity 차익")

    df = fetch_active_markets(min_volume=10000.0, max_pages=20)
    print(f"  활성 마켓 (vol 10k+): {len(df)}")

    if df.empty:
        print("  ❌ 데이터 없음.")
        return

    df['last_price'] = pd.to_numeric(df.get('lastTradePrice', None), errors='coerce')
    df['bid'] = pd.to_numeric(df.get('bestBid', None), errors='coerce')
    df['ask'] = pd.to_numeric(df.get('bestAsk', None), errors='coerce')
    df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

    df = df.dropna(subset=['last_price', 'bid', 'ask'])
    df = df[(df['bid'] > 0) & (df['ask'] < 1) & (df['ask'] > df['bid'])]
    df['spread'] = df['ask'] - df['bid']
    df['spread_pct'] = df['spread'] / ((df['ask'] + df['bid']) / 2) * 100
    df['mid'] = (df['ask'] + df['bid']) / 2
    df['last_vs_mid'] = df['last_price'] - df['mid']

    print(f"  유효 bid/ask: {len(df)}")
    print(f"\n=== Spread 분포 ===")
    print(f"  median spread: {df['spread'].median():.4f}")
    print(f"  90th: {df['spread'].quantile(0.9):.4f}")
    print(f"  99th: {df['spread'].quantile(0.99):.4f}")

    # 큰 spread + 큰 거래량 = market making 후보
    big_spread = df[(df['spread'] > 0.05) & (df['volume_num'] >= 50000)].copy()
    print(f"\n  큰 spread (≥5%) + vol 50k+: {len(big_spread)}")

    # last vs mid 차이가 큰 마켓 = stale 가격
    stale = df[df['last_vs_mid'].abs() > 0.05].copy()
    print(f"  Stale price (|last - mid| > 5%): {len(stale)}")

    print(f"\n=== Top 큰 spread (market making 후보) ===")
    if not big_spread.empty:
        for _, r in big_spread.sort_values('spread', ascending=False).head(15).iterrows():
            print(f"  bid={r['bid']:.3f} ask={r['ask']:.3f} spread={r['spread']:.3f} ({r['spread_pct']:.1f}%) | "
                  f"last={r['last_price']:.3f} | vol=${r['volume_num']:,.0f}")
            print(f"     {str(r.get('question', '?'))[:80]}")

    print(f"\n=== Top stale (last 가격이 mid에서 5%+ 떨어짐) ===")
    if not stale.empty:
        for _, r in stale.sort_values('last_vs_mid', key=abs, ascending=False).head(10).iterrows():
            direction = "↑" if r['last_vs_mid'] > 0 else "↓"
            print(f"  last={r['last_price']:.3f} mid={r['mid']:.3f} {direction} {abs(r['last_vs_mid'])*100:.1f}% | "
                  f"vol=${r['volume_num']:,.0f}")
            print(f"     {str(r.get('question', '?'))[:80]}")

    out = {
        "n_total": int(len(df)),
        "n_big_spread": int(len(big_spread)),
        "n_stale": int(len(stale)),
        "median_spread": float(df['spread'].median()),
        "top_big_spread": [
            {
                'question': str(r.get('question', ''))[:80],
                'bid': float(r['bid']),
                'ask': float(r['ask']),
                'spread': float(r['spread']),
                'spread_pct': float(r['spread_pct']),
                'volume': float(r['volume_num']),
            }
            for _, r in big_spread.sort_values('spread', ascending=False).head(15).iterrows()
        ],
    }
    out_path = RESULTS_DIR / "iter10_spread_arb.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
