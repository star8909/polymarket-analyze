"""iter07: 활성 마켓 time-series 분포 (Beta/logit) → mean reversion 신호.

iter06 한계: closed market의 가격 시계열 못 가져옴 (CLOB API).
iter07 우회: 활성 마켓의 oneDayPriceChange/oneWeekPriceChange로
   가격 변화율 분포 분석 → 극단 마켓 (±2σ) 발견.

가설:
- oneDayPriceChange가 +20%/-20% 같은 극단 = 뉴스 발화 후 과잉반응
- 마켓 종가는 평균회귀 → 극단 직후 반대 베팅 +EV
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
    print("[iter07] 활성 마켓 가격 변화율 분포 → mean reversion 신호")

    df = fetch_active_markets(min_volume=10000.0, max_pages=20)
    print(f"  활성 마켓 (거래량 10k+): {len(df)}")

    if df.empty:
        print("  ❌ 데이터 없음.")
        return

    # 가격 변화율 컬럼
    for col in ['oneDayPriceChange', 'oneHourPriceChange', 'oneWeekPriceChange']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # lastTradePrice (활성 마켓은 의미있음)
    df['last_price'] = pd.to_numeric(df.get('lastTradePrice', None), errors='coerce')
    df = df.dropna(subset=['last_price'])
    print(f"  last_price 있음: {len(df)}")

    # 1) 1일 가격 변화율 분포 (극단 발견)
    if 'oneDayPriceChange' in df.columns:
        valid = df.dropna(subset=['oneDayPriceChange']).copy()
        changes = valid['oneDayPriceChange'].values
        print(f"\n=== 1일 가격 변화율 분포 ===")
        print(f"  N={len(changes)}, mean={changes.mean()*100:.2f}%, std={changes.std()*100:.2f}%")
        print(f"  +95%: {np.percentile(changes, 95)*100:+.2f}%")
        print(f"  -95%: {np.percentile(changes, 5)*100:+.2f}%")
        print(f"  +99%: {np.percentile(changes, 99)*100:+.2f}%")
        print(f"  -99%: {np.percentile(changes, 1)*100:+.2f}%")

        # 극단 마켓 (±2σ)
        mu, sigma = changes.mean(), changes.std()
        z_scores = (changes - mu) / sigma
        valid['z_1d'] = z_scores
        extreme_pos = valid[valid['z_1d'] > 2].sort_values('z_1d', ascending=False)
        extreme_neg = valid[valid['z_1d'] < -2].sort_values('z_1d')

        print(f"\n=== Top 1일 가격 +2σ (overshooting) — Mean reversion 후보 (NO 베팅) ===")
        print(f"  발견: {len(extreme_pos)}개")
        for _, r in extreme_pos.head(15).iterrows():
            print(f"    Δ={r['oneDayPriceChange']*100:+6.1f}% (z={r['z_1d']:+.2f}) | "
                  f"price={r['last_price']:.3f} | "
                  f"vol={pd.to_numeric(r.get('volume', 0)):,.0f} | "
                  f"{str(r.get('question', '?'))[:60]}")

        print(f"\n=== Top 1일 가격 -2σ (overshooting down) — Mean reversion 후보 (YES 베팅) ===")
        print(f"  발견: {len(extreme_neg)}개")
        for _, r in extreme_neg.head(15).iterrows():
            print(f"    Δ={r['oneDayPriceChange']*100:+6.1f}% (z={r['z_1d']:+.2f}) | "
                  f"price={r['last_price']:.3f} | "
                  f"vol={pd.to_numeric(r.get('volume', 0)):,.0f} | "
                  f"{str(r.get('question', '?'))[:60]}")

    # 2) Logit 변환 후 정규성 체크 (Beta 분포 fit)
    valid_p = df[(df['last_price'] > 0.02) & (df['last_price'] < 0.98)].copy()
    if len(valid_p) >= 50:
        valid_p['logit'] = np.log(valid_p['last_price'] / (1 - valid_p['last_price']))
        print(f"\n=== Logit 변환 분포 (last_price 2~98%) ===")
        print(f"  N={len(valid_p)}, mean logit={valid_p['logit'].mean():.3f}, std={valid_p['logit'].std():.3f}")
        # Skewness, kurtosis
        skew = valid_p['logit'].skew()
        kurt = valid_p['logit'].kurt()
        print(f"  skew={skew:.2f}, kurtosis={kurt:.2f}")
        if abs(skew) > 0.5:
            direction = "Yes 쪽 편향" if skew > 0 else "No 쪽 편향"
            print(f"  → {direction} (시장 평균 가격 ≠ 0.5)")

    # 결과 저장
    out = {
        "n_markets": len(df),
        "n_extreme_pos_2sigma": int(len(extreme_pos)) if 'oneDayPriceChange' in df.columns else 0,
        "n_extreme_neg_2sigma": int(len(extreme_neg)) if 'oneDayPriceChange' in df.columns else 0,
        "top_extreme_pos": [
            {
                'question': str(r.get('question', ''))[:80],
                'price': float(r['last_price']),
                'change_1d_pct': float(r['oneDayPriceChange'] * 100),
                'z': float(r['z_1d']),
                'volume': float(pd.to_numeric(r.get('volume', 0))),
            }
            for _, r in extreme_pos.head(10).iterrows()
        ] if 'oneDayPriceChange' in df.columns else [],
        "top_extreme_neg": [
            {
                'question': str(r.get('question', ''))[:80],
                'price': float(r['last_price']),
                'change_1d_pct': float(r['oneDayPriceChange'] * 100),
                'z': float(r['z_1d']),
                'volume': float(pd.to_numeric(r.get('volume', 0))),
            }
            for _, r in extreme_neg.head(10).iterrows()
        ] if 'oneDayPriceChange' in df.columns else [],
    }
    out_path = RESULTS_DIR / "iter07_active_distribution.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
