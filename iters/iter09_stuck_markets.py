"""iter09: 만기 임박 stuck markets (확정 직전 가격 갭).

가설: 만기 D-1 ~ D-7일 마켓 중 가격 0.95+ 또는 0.05- 인 마켓은
   거의 settle 결과 확정. 시장이 0.97 매기는데 진짜 확률이 0.99+면
   미세 차익 (~2~3%) 가능.

또는: 0.05- 인데 실제 0.01 확률 → No 매수 +EV.

검증:
1. 활성 마켓 중 endDate ≤ 7일 마켓
2. lastTradePrice > 0.95 또는 < 0.05
3. 거래량 충분 (10k+)
4. spread 작음 (bestAsk - bestBid < 0.05)
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
    print("[iter09] 만기 임박 stuck markets")

    df = fetch_active_markets(min_volume=10000.0, max_pages=20)
    print(f"  활성 마켓 (vol 10k+): {len(df)}")

    if df.empty:
        print("  ❌ 데이터 없음.")
        return

    # 만기 정보
    df['endDate_dt'] = pd.to_datetime(df['endDate'], errors='coerce', utc=True)
    now = pd.Timestamp.now(tz='UTC')
    df['days_to_end'] = (df['endDate_dt'] - now).dt.days

    # 만기 ≤ 7일
    near = df[(df['days_to_end'] >= 0) & (df['days_to_end'] <= 7)].copy()
    print(f"  만기 D≤7일: {len(near)}")

    if near.empty:
        print("  ❌ 만기 임박 마켓 없음.")
        return

    near['last_price'] = pd.to_numeric(near.get('lastTradePrice', None), errors='coerce')
    near['bid'] = pd.to_numeric(near.get('bestBid', None), errors='coerce')
    near['ask'] = pd.to_numeric(near.get('bestAsk', None), errors='coerce')
    near['spread'] = near['ask'] - near['bid']
    near['volume_num'] = pd.to_numeric(near['volume'], errors='coerce').fillna(0)

    near = near.dropna(subset=['last_price', 'bid', 'ask'])
    print(f"  bid/ask 있음: {len(near)}")

    # Stuck price 후보
    stuck_high = near[(near['last_price'] >= 0.95) & (near['spread'] < 0.05)].copy()
    stuck_low = near[(near['last_price'] <= 0.05) & (near['spread'] < 0.05)].copy()
    print(f"  Stuck 0.95+ + spread<5%: {len(stuck_high)}")
    print(f"  Stuck 0.05- + spread<5%: {len(stuck_low)}")

    # Edge 추정: 가격 갭 (0.99 - 가격) for high, (가격 - 0.01) for low
    stuck_high['edge_pct'] = (0.99 - stuck_high['last_price']) * 100
    stuck_low['edge_pct'] = (stuck_low['last_price'] - 0.01) * 100

    print(f"\n=== Stuck High (0.95+) — Yes 매수 후보 ===")
    if not stuck_high.empty:
        for i, r in stuck_high.sort_values('edge_pct', ascending=False).head(10).iterrows():
            print(f"  Δd={r['days_to_end']:>2.0f} | price={r['last_price']:.3f} | "
                  f"spread={r['spread']:.3f} | edge={r['edge_pct']:>4.1f}% | "
                  f"vol=${r['volume_num']:,.0f}")
            print(f"     {str(r.get('question', '?'))[:80]}")
    else:
        print(f"  ❌ 없음")

    print(f"\n=== Stuck Low (0.05-) — No 매수 후보 ===")
    if not stuck_low.empty:
        for i, r in stuck_low.sort_values('edge_pct', ascending=False).head(10).iterrows():
            print(f"  Δd={r['days_to_end']:>2.0f} | price={r['last_price']:.3f} | "
                  f"spread={r['spread']:.3f} | edge={r['edge_pct']:>4.1f}% | "
                  f"vol=${r['volume_num']:,.0f}")
            print(f"     {str(r.get('question', '?'))[:80]}")
    else:
        print(f"  ❌ 없음")

    # 종합
    total_safe = len(stuck_high) + len(stuck_low)
    avg_edge = 0
    if total_safe > 0:
        all_edges = list(stuck_high['edge_pct']) + list(stuck_low['edge_pct'])
        avg_edge = sum(all_edges) / len(all_edges)

    print(f"\n=== iter09 종합 ===")
    print(f"  총 stuck 마켓: {total_safe} (평균 edge {avg_edge:.2f}%)")
    if avg_edge > 1:
        print(f"  ✅ 평균 edge {avg_edge:.2f}% — 미세 차익 매일 가능")
    else:
        print(f"  ⚠️ 평균 edge {avg_edge:.2f}% — 너무 작음")

    out = {
        "n_near_expiry_7d": int(len(near)),
        "n_stuck_high": int(len(stuck_high)),
        "n_stuck_low": int(len(stuck_low)),
        "avg_edge_pct": round(avg_edge, 2),
        "stuck_high_top10": [
            {
                'question': str(r.get('question', ''))[:80],
                'price': float(r['last_price']),
                'spread': float(r['spread']),
                'days_to_end': int(r['days_to_end']),
                'volume': float(r['volume_num']),
                'edge_pct': float(r['edge_pct']),
            }
            for _, r in stuck_high.sort_values('edge_pct', ascending=False).head(10).iterrows()
        ],
        "stuck_low_top10": [
            {
                'question': str(r.get('question', ''))[:80],
                'price': float(r['last_price']),
                'spread': float(r['spread']),
                'days_to_end': int(r['days_to_end']),
                'volume': float(r['volume_num']),
                'edge_pct': float(r['edge_pct']),
            }
            for _, r in stuck_low.sort_values('edge_pct', ascending=False).head(10).iterrows()
        ],
    }
    out_path = RESULTS_DIR / "iter09_stuck_markets.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
