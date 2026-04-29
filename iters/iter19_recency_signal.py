"""iter19: 1시간 가격 변화 mean reversion (활성 마켓).

가설: 1시간 가격 +20%/-20% 극단 변화 후 평균회귀.
iter07 mean reversion (1일 ±2σ)보다 더 짧은 노이즈 필터.

분석:
- oneHourPriceChange 분포
- 극단값 (±10%, ±20%) 후 다음 N시간 회귀 가능성
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
    print("[iter19] 1시간 가격 변화 mean reversion")
    df = fetch_active_markets(min_volume=10000.0, max_pages=20)
    if df.empty:
        print("  ❌ 데이터 없음")
        return

    df['last_price'] = pd.to_numeric(df.get('lastTradePrice', None), errors='coerce')
    df['oneHourPriceChange'] = pd.to_numeric(df.get('oneHourPriceChange', None), errors='coerce')
    df['oneDayPriceChange'] = pd.to_numeric(df.get('oneDayPriceChange', None), errors='coerce')
    df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df = df.dropna(subset=['last_price', 'oneHourPriceChange'])
    df = df[(df['last_price'] > 0.01) & (df['last_price'] < 0.99)]
    print(f"  활성 마켓 (의미 가격): {len(df)}")

    print(f"\n=== 1시간 가격 변화 분포 ===")
    print(f"  Mean: {df['oneHourPriceChange'].mean()*100:+.3f}%")
    print(f"  Std: {df['oneHourPriceChange'].std()*100:.3f}%")
    print(f"  Skew: {df['oneHourPriceChange'].skew():.2f}")
    print(f"  Min: {df['oneHourPriceChange'].min()*100:+.1f}%")
    print(f"  Max: {df['oneHourPriceChange'].max()*100:+.1f}%")
    print(f"  +99% percentile: {df['oneHourPriceChange'].quantile(0.99)*100:+.2f}%")
    print(f"  -99% percentile: {df['oneHourPriceChange'].quantile(0.01)*100:+.2f}%")

    # ±2σ 극단
    mu, sigma = df['oneHourPriceChange'].mean(), df['oneHourPriceChange'].std()
    df['z_1h'] = (df['oneHourPriceChange'] - mu) / sigma

    extreme_pos = df[df['z_1h'] > 2].sort_values('z_1h', ascending=False)
    extreme_neg = df[df['z_1h'] < -2].sort_values('z_1h')

    print(f"\n=== Top 1시간 +2σ overshooting (NO 베팅 후보) ===")
    print(f"  발견: {len(extreme_pos)}개")
    for _, r in extreme_pos.head(10).iterrows():
        print(f"    Δ={r['oneHourPriceChange']*100:+5.1f}% (z={r['z_1h']:+.2f}) | "
              f"price={r['last_price']:.3f} | vol=${r['volume_num']:,.0f}")
        print(f"      {str(r.get('question', ''))[:70]}")

    print(f"\n=== Top 1시간 -2σ overshooting (YES 베팅 후보) ===")
    print(f"  발견: {len(extreme_neg)}개")
    for _, r in extreme_neg.head(10).iterrows():
        print(f"    Δ={r['oneHourPriceChange']*100:+5.1f}% (z={r['z_1h']:+.2f}) | "
              f"price={r['last_price']:.3f} | vol=${r['volume_num']:,.0f}")
        print(f"      {str(r.get('question', ''))[:70]}")

    # 1h vs 1d 일관성
    print(f"\n=== 1h vs 1d 변화 상관 ===")
    valid_corr = df.dropna(subset=['oneHourPriceChange', 'oneDayPriceChange'])
    if len(valid_corr) > 100:
        corr = valid_corr['oneHourPriceChange'].corr(valid_corr['oneDayPriceChange'])
        print(f"  Correlation: {corr:.3f}")
        # 같은 방향 비율
        same_dir = ((valid_corr['oneHourPriceChange'] > 0) == (valid_corr['oneDayPriceChange'] > 0)).sum() / len(valid_corr) * 100
        print(f"  같은 방향 비율: {same_dir:.1f}%")

    out = {
        "n_markets": int(len(df)),
        "extreme_pos_2sigma": int(len(extreme_pos)),
        "extreme_neg_2sigma": int(len(extreme_neg)),
        "top_extreme_pos": [
            {
                'question': str(r.get('question', ''))[:80],
                'price': float(r['last_price']),
                'change_1h_pct': float(r['oneHourPriceChange'] * 100),
                'volume': float(r['volume_num']),
            }
            for _, r in extreme_pos.head(15).iterrows()
        ],
    }
    out_path = RESULTS_DIR / "iter19_recency_signal.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
