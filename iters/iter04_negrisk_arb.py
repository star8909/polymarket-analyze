"""iter04: negRisk (exclusive) 마켓에서 진짜 sum-to-1 차익 스캐너.

iter03 false alarm 원인:
- Eurovision Top 10 같은 "Top N" 마켓은 sum=N이 정상 (sum=10이 아니라 11이면 ~10% 위반)
- "Independent" 마켓 (BTC 10만/12만 돌파)은 sum과 무관

진짜 차익:
- **negRisk = True** 마켓 = Polymarket "exclusive" 그룹 (단 하나만 Yes)
- 같은 negRiskMarketID 공유 → 모든 Yes 합 = 1.0이어야 함

negRisk 마켓 (예):
- "Who will win 2024 Election" 후보들
- "Which team wins Super Bowl" 팀들
- 가격 합 위반 = 무위험 차익
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
    print("[iter04] negRisk (exclusive) 마켓 sum-to-1 진짜 차익 스캐너")

    df = fetch_active_markets(min_volume=500.0, max_pages=20)
    print(f"  활성 마켓 (거래량 500+): {len(df)}")

    if df.empty or 'negRisk' not in df.columns:
        print("  ❌ negRisk 컬럼 없음.")
        return

    # negRisk = True인 마켓만 필터
    if df['negRisk'].dtype == bool:
        neg = df[df['negRisk'] == True].copy()
    else:
        neg = df[df['negRisk'].astype(str).str.lower() == 'true'].copy()
    print(f"  negRisk=True 마켓: {len(neg)}")

    if 'negRiskMarketID' not in neg.columns:
        print("  ❌ negRiskMarketID 컬럼 없음.")
        return

    # 같은 negRiskMarketID 그룹화
    groups = neg.groupby('negRiskMarketID')
    arbs = []
    for grp_id, grp_df in groups:
        if len(grp_df) < 2:
            continue
        yes_prices = []
        slugs = []
        markets_meta = []
        for _, row in grp_df.iterrows():
            parsed = parse_market_outcome(row.to_dict())
            if not parsed.get('prices') or len(parsed['prices']) < 2:
                continue
            yes_p = parsed['prices'][0]
            yes_prices.append(yes_p)
            slugs.append(row.get('slug', '???'))
            markets_meta.append({
                'slug': row.get('slug'),
                'question': row.get('question', '')[:80],
                'yes_price': yes_p,
                'volume': float(pd.to_numeric(row.get('volume', 0), errors='coerce') or 0),
            })

        if len(yes_prices) < 2:
            continue
        total = sum(yes_prices)
        deviation = abs(total - 1.0)
        if deviation > 0.02:
            arbs.append({
                'negRiskMarketID': grp_id,
                'n_markets': len(yes_prices),
                'sum_Yes': round(total, 4),
                'deviation': round(deviation, 4),
                'expected_edge_pct': round(deviation * 100, 2),
                'direction': 'BUY_ALL_YES' if total < 1.0 else 'BUY_ALL_NO',
                'markets': markets_meta[:10],
            })

    arbs.sort(key=lambda r: r['expected_edge_pct'], reverse=True)

    print(f"\n=== 진짜 negRisk Sum-to-1 위반 (편차 2%+) ===")
    if not arbs:
        print(f"  ❌ 위반 없음. 시장 effiency 매우 높음.")
    else:
        print(f"  발견: {len(arbs)}개 (5%+ deviation: {sum(1 for a in arbs if a['deviation'] > 0.05)}개)")
        print(f"\n  Top 20:")
        for i, arb in enumerate(arbs[:20], 1):
            print(f"\n  [{i}] negRiskID {arb['negRiskMarketID']}")
            print(f"      n_markets={arb['n_markets']}, sum_Yes={arb['sum_Yes']:.4f}, edge={arb['expected_edge_pct']:.2f}%")
            print(f"      direction: {arb['direction']}")
            for m in arb['markets'][:3]:
                print(f"        - {m['question'][:60]} | Yes={m['yes_price']:.3f} | vol=${m['volume']:,.0f}")

    out = {
        "total_negrisk_markets": int(len(neg)),
        "total_groups": int(neg['negRiskMarketID'].nunique()),
        "violations_2pct": len(arbs),
        "violations_5pct": sum(1 for a in arbs if a['deviation'] > 0.05),
        "arbs_top20": arbs[:20],
    }
    out_path = RESULTS_DIR / "iter04_negrisk_arb.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
