"""iter05: 진짜 안전 차익 필터 (deviation 작음 + 거래량 큼).

iter04 발견:
- 큰 위반 (sum=6.2 등)은 거래 비대칭 risk + 거래량 부족
- 진짜 차익 후보 = 작은 deviation (2~5%) + 모든 마켓 거래량 충분

iter05 필터:
1. negRisk=True
2. sum 편차 < 20% (큰 위반은 무시 — 가짜 신호)
3. 모든 마켓 거래량 ≥ $5k
4. n_markets ≤ 10 (작은 그룹만 — 동시 거래 가능)
5. n_markets ≥ 2

목표: deviation 2~10%인 차익만 추출. Kelly 0.25x 사이즈로 시도.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import defaultdict

import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import fetch_active_markets, parse_market_outcome


def main():
    print("[iter05] 안전 negRisk 차익 필터 (deviation 작음 + 거래량 큼)")

    df = fetch_active_markets(min_volume=5000.0, max_pages=20)  # 거래량 5k+ 필터
    print(f"  활성 마켓 (거래량 5k+): {len(df)}")

    if df.empty:
        print("  ❌ 데이터 없음.")
        return

    # negRisk만
    if df['negRisk'].dtype == bool:
        neg = df[df['negRisk'] == True].copy()
    else:
        neg = df[df['negRisk'].astype(str).str.lower() == 'true'].copy()
    print(f"  negRisk=True 마켓: {len(neg)}")

    if 'negRiskMarketID' not in neg.columns:
        print("  ❌ negRiskMarketID 컬럼 없음.")
        return

    groups = neg.groupby('negRiskMarketID')

    safe_arbs = []   # deviation 2~20%
    big_devs = []    # deviation 20%+ (가짜 신호 추정)

    for grp_id, grp_df in groups:
        n = len(grp_df)
        if n < 2 or n > 15:  # 너무 큰 그룹 (Eurovision 35명) 제외
            continue

        yes_prices = []
        slugs = []
        markets_meta = []
        all_volumes_ok = True
        for _, row in grp_df.iterrows():
            parsed = parse_market_outcome(row.to_dict())
            if not parsed.get('prices') or len(parsed['prices']) < 2:
                continue
            yes_p = parsed['prices'][0]
            vol = float(pd.to_numeric(row.get('volume', 0), errors='coerce') or 0)
            if vol < 5000:
                all_volumes_ok = False
            yes_prices.append(yes_p)
            slugs.append(row.get('slug', '???'))
            markets_meta.append({
                'slug': row.get('slug'),
                'question': str(row.get('question', ''))[:80],
                'yes_price': round(yes_p, 4),
                'volume': vol,
            })

        if len(yes_prices) < 2 or not all_volumes_ok:
            continue

        total = sum(yes_prices)
        deviation = abs(total - 1.0)
        deviation_pct = deviation * 100

        if deviation < 0.02:
            continue

        record = {
            'negRiskMarketID': grp_id,
            'n_markets': len(yes_prices),
            'sum_Yes': round(total, 4),
            'deviation': round(deviation, 4),
            'deviation_pct': round(deviation_pct, 2),
            'direction': 'BUY_ALL_YES' if total < 1.0 else 'BUY_ALL_NO',
            'expected_edge_pct': round(deviation_pct, 2),
            'min_volume': round(min(m['volume'] for m in markets_meta), 0),
            'total_volume': round(sum(m['volume'] for m in markets_meta), 0),
            'markets': markets_meta,
        }

        if deviation < 0.20:
            safe_arbs.append(record)
        else:
            big_devs.append(record)

    safe_arbs.sort(key=lambda r: r['deviation_pct'], reverse=True)

    print(f"\n=== 🛡️ 안전 차익 (deviation 2~20%, 모든 마켓 vol 5k+, n_markets 2~15) ===")
    if not safe_arbs:
        print(f"  ❌ 안전 차익 없음. 시장 효율적 또는 거래량 부족.")
    else:
        print(f"  발견: {len(safe_arbs)}개")
        print(f"\n  Top 15:")
        for i, arb in enumerate(safe_arbs[:15], 1):
            print(f"\n  [{i}] sum_Yes={arb['sum_Yes']:.4f} edge={arb['deviation_pct']:.2f}% direction={arb['direction']}")
            print(f"      n_markets={arb['n_markets']}, min_vol=${arb['min_volume']:,.0f}, total_vol=${arb['total_volume']:,.0f}")
            for m in arb['markets'][:3]:
                print(f"        - {m['question'][:60]} | Yes={m['yes_price']:.3f} | vol=${m['volume']:,.0f}")

    print(f"\n=== ⚠️ 큰 deviation (20%+, 거래 비대칭 위험) ===")
    print(f"  무시: {len(big_devs)}개 (Eurovision 등 multi-winner / stale price)")

    out = {
        "total_negrisk_groups": int(neg['negRiskMarketID'].nunique()),
        "safe_arbs_count": len(safe_arbs),
        "big_devs_count": len(big_devs),
        "safe_arbs_top15": safe_arbs[:15],
    }
    out_path = RESULTS_DIR / "iter05_safe_arb.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
