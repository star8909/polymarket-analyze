"""iter23: Volume-weighted calibration on active markets.

iter17/20 baseline: bucket 0.30-0.40 → -16.7% edge (전체 closed)
iter23: 활성 마켓 중 volume_24h 가중하여 institutional flow 가까운 calibration.
       소량 거래는 noise → 큰 거래만 신호로 간주.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import fetch_active_markets


def main():
    print("[iter23] Volume-weighted calibration")
    df = fetch_active_markets(min_volume=10000.0, max_pages=20)
    if df.empty:
        print("  ❌ 데이터 없음")
        return

    df['last_price'] = pd.to_numeric(df.get('lastTradePrice', None), errors='coerce')
    df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df['oneDayPriceChange'] = pd.to_numeric(df.get('oneDayPriceChange', None), errors='coerce')
    df['volume_24h'] = pd.to_numeric(df.get('volume24hr', df.get('volume_24h', 0)), errors='coerce').fillna(0)
    df = df.dropna(subset=['last_price', 'volume_num'])
    df = df[(df['last_price'] > 0.01) & (df['last_price'] < 0.99)]
    print(f"  활성 마켓: {len(df)}")

    print(f"\n=== Volume_24h 분포 ===")
    print(f"  vol24h mean: ${df['volume_24h'].mean():,.0f}")
    print(f"  vol24h median: ${df['volume_24h'].median():,.0f}")
    print(f"  vol24h 90%: ${df['volume_24h'].quantile(0.9):,.0f}")
    print(f"  vol24h max: ${df['volume_24h'].max():,.0f}")

    # 큰 vol 마켓만 집중 분석
    print(f"\n=== High-vol 마켓 (vol_24h > $50k) calibration ===")
    high = df[df['volume_24h'] > 50000].copy()
    print(f"  N={len(high)}")
    if len(high) >= 50:
        bins = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
        high['bucket'] = pd.cut(high['last_price'], bins=bins)
        for b, sub in high.groupby('bucket'):
            if len(sub) > 0:
                m = sub['last_price'].mean()
                v = sub['volume_24h'].sum() / 1e6
                print(f"  {str(b):20s} N={len(sub):>4} mean={m:.3f} sum_vol_24h=${v:>7.2f}M")

    # Volume vs price 변화 상관
    print(f"\n=== Vol vs 1d price change ===")
    valid = df.dropna(subset=['volume_24h', 'oneDayPriceChange'])
    if len(valid) > 100:
        corr = valid['volume_24h'].corr(valid['oneDayPriceChange'].abs())
        print(f"  |Δprice| vs vol_24h corr: {corr:.3f}")
        # 큰 거래량 + 가격 stable = 진짜 가격 (institutional)
        stable_high_vol = valid[(valid['volume_24h'] > 50000) & (valid['oneDayPriceChange'].abs() < 0.05)]
        print(f"  Institutional gauge (high vol + stable price): N={len(stable_high_vol)}")
        # mid-priced (0.30-0.50) 만 보자
        mid = stable_high_vol[(stable_high_vol['last_price'] >= 0.30) & (stable_high_vol['last_price'] <= 0.50)]
        if len(mid) > 0:
            print(f"  Mid-priced (0.30-0.50) high-vol stable: N={len(mid)}")
            print(f"    avg price: {mid['last_price'].mean():.3f}")
            for _, r in mid.head(15).iterrows():
                print(f"     price={r['last_price']:.3f} vol_24h=${r['volume_24h']:>10,.0f}  {str(r.get('question', ''))[:60]}")

    # 0.45-0.55 범위 (가장 불확실, 가장 큰 alpha 가능)
    print(f"\n=== Coin-flip 영역 (0.45-0.55) high vol ===")
    flip = df[(df['last_price'] >= 0.45) & (df['last_price'] <= 0.55) & (df['volume_24h'] > 30000)]
    print(f"  N={len(flip)}")
    for _, r in flip.head(15).iterrows():
        print(f"    price={r['last_price']:.3f} 1d Δ={r.get('oneDayPriceChange', 0)*100:+.1f}% vol_24h=${r['volume_24h']:>10,.0f}")
        print(f"      {str(r.get('question', ''))[:80]}")

    out_path = RESULTS_DIR / "iter23_volume_weighted.json"
    out_path.write_text(json.dumps({"n": int(len(df)), "high_vol_n": int(len(df[df['volume_24h'] > 50000]))}, indent=2), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
