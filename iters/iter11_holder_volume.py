"""iter11: Holder 수 vs 거래량 비대칭 (manipulation 의심).

가설: 거래량 큰데 holder 수 적은 마켓 = 소수 whales 매매로 가격 manipulation.
이런 마켓은:
- 가격이 진짜 확률 반영 X
- 큰 손이 빠지면 가격 폭락
- contrarian 베팅 (whale 반대) +EV 가능성

또는 반대: holder 많고 거래량 적은 마켓 = retail dumb money. 시장 가격 신뢰 가능.

Polymarket의 마켓 메타에 holder count 있을지 확인 필요.
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
    print("[iter11] Holder/volume 비대칭 분석")

    df = fetch_active_markets(min_volume=50000.0, max_pages=20)
    print(f"  활성 마켓 (vol 50k+): {len(df)}")

    if df.empty:
        print("  ❌ 데이터 없음.")
        return

    # 컬럼 확인
    holder_cols = [c for c in df.columns if 'holder' in c.lower() or 'trader' in c.lower() or 'count' in c.lower()]
    print(f"  Holder 관련 컬럼: {holder_cols}")

    # 거래량 vs liquidity 비율
    df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df['liquidity_num'] = pd.to_numeric(df.get('liquidity', 0), errors='coerce').fillna(0)
    df['vol_per_liq'] = df['volume_num'] / df['liquidity_num'].replace(0, 1)
    df['last_price'] = pd.to_numeric(df.get('lastTradePrice', None), errors='coerce')
    df = df.dropna(subset=['last_price'])

    print(f"\n=== Volume / Liquidity 비율 분포 ===")
    print(f"  median ratio: {df['vol_per_liq'].median():.2f}")
    print(f"  90th: {df['vol_per_liq'].quantile(0.9):.2f}")
    print(f"  99th: {df['vol_per_liq'].quantile(0.99):.2f}")

    # 비정상적으로 vol/liquidity 높음 = wash trading 또는 whales
    suspicious = df[df['vol_per_liq'] > df['vol_per_liq'].quantile(0.95)].copy()
    print(f"\n  의심 마켓 (vol/liq > 95th percentile): {len(suspicious)}")

    # 24h 거래량 vs 누적 거래량 비율
    if 'volume24hr' in df.columns:
        df['vol_24hr'] = pd.to_numeric(df['volume24hr'], errors='coerce').fillna(0)
        df['vol_24h_pct'] = df['vol_24hr'] / df['volume_num'].replace(0, 1) * 100
        recent_active = df[df['vol_24h_pct'] > 50].copy()  # 누적의 50%+가 24h 안에
        print(f"\n  최근 활발 (24h vol > 50% of total): {len(recent_active)}")

        if not recent_active.empty:
            print(f"\n=== Top 최근 활발 마켓 (whales 진입 가능성) ===")
            for _, r in recent_active.sort_values('vol_24hr', ascending=False).head(10).iterrows():
                print(f"  vol_24h={r['vol_24hr']:>10,.0f} ({r['vol_24h_pct']:.0f}%) | "
                      f"price={r['last_price']:.3f} | total=${r['volume_num']:>10,.0f}")
                print(f"     {str(r.get('question', '?'))[:80]}")

    # Liquidity 적은데 거래량 큰 마켓 (slippage 위험)
    high_vol_low_liq = df[(df['volume_num'] > 100000) & (df['liquidity_num'] < 5000)].copy()
    print(f"\n  거래량 100k+ but liquidity < 5k: {len(high_vol_low_liq)}")

    out = {
        "n_total": int(len(df)),
        "median_vol_per_liq": float(df['vol_per_liq'].median()),
        "n_suspicious": int(len(suspicious)),
        "n_high_vol_low_liq": int(len(high_vol_low_liq)),
        "n_recent_active": int(len(recent_active)) if 'volume24hr' in df.columns else 0,
        "top_suspicious": [
            {
                'question': str(r.get('question', ''))[:80],
                'volume': float(r['volume_num']),
                'liquidity': float(r['liquidity_num']),
                'vol_per_liq': float(r['vol_per_liq']),
                'price': float(r['last_price']),
            }
            for _, r in suspicious.sort_values('vol_per_liq', ascending=False).head(10).iterrows()
        ],
    }
    out_path = RESULTS_DIR / "iter11_holder_volume.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
