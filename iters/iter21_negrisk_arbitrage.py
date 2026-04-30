"""iter21: NegRisk arbitrage — sum-to-1 deviation 마켓 발견.

negRisk 마켓 = "정확히 하나만 발생" (예: 대선 우승자, 1위 후보).
모든 outcome 가격 합 = 1.0이어야 정상.
sum > 1.05 → 단기 차익 (모두 short)
sum < 0.95 → 단기 차익 (모두 long)

데이터: fetch_active_markets, market group 추출.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import defaultdict
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import fetch_active_markets


def main():
    print("[iter21] negRisk arbitrage — sum-to-1 deviation")
    df = fetch_active_markets(min_volume=5000.0, max_pages=20)
    if df.empty:
        print("  ❌ 데이터 없음")
        return

    df['last_price'] = pd.to_numeric(df.get('lastTradePrice', None), errors='coerce')
    df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df = df.dropna(subset=['last_price'])

    # negRisk flag
    if 'negRisk' in df.columns:
        df['negRisk_bool'] = df['negRisk'].fillna(False)
    else:
        df['negRisk_bool'] = False

    # Group by event_id or eventTitle
    group_col = None
    for cand in ['eventId', 'event_id', 'event', 'eventTitle']:
        if cand in df.columns:
            group_col = cand
            break

    if group_col is None:
        print("  ⚠️ event 그룹 컬럼 없음 → ticker prefix로 추정")
        df['group_key'] = df.get('slug', df.get('ticker', '')).astype(str).str[:30]
        group_col = 'group_key'

    print(f"  활성 마켓: {len(df)}, group 컬럼: {group_col}")

    # Group sum
    grouped = df.groupby(group_col).agg(
        n_outcomes=('last_price', 'count'),
        price_sum=('last_price', 'sum'),
        total_volume=('volume_num', 'sum'),
        any_negrisk=('negRisk_bool', 'any')
    ).reset_index()

    # multi-outcome groups (>=2)
    multi = grouped[grouped['n_outcomes'] >= 2]
    print(f"\n  Multi-outcome groups: {len(multi)}")
    if len(multi) == 0:
        return

    print(f"  price_sum 평균: {multi['price_sum'].mean():.3f}")
    print(f"  price_sum std: {multi['price_sum'].std():.3f}")

    # Top arbitrage candidates (sum > 1.10)
    over = multi[multi['price_sum'] > 1.05].sort_values('price_sum', ascending=False)
    under = multi[(multi['price_sum'] < 0.95) & (multi['n_outcomes'] >= 3)].sort_values('price_sum')

    print(f"\n=== Top sum > 1.05 (short all - arbitrage) ===")
    print(f"  발견: {len(over)}개")
    for _, r in over.head(15).iterrows():
        ng = "negRisk" if r['any_negrisk'] else ""
        print(f"    sum={r['price_sum']:.3f} ({r['n_outcomes']} outcomes) vol=${r['total_volume']:>10,.0f}  {ng}")
        print(f"      {str(r[group_col])[:80]}")

    print(f"\n=== Top sum < 0.95 with 3+ outcomes (long all - arbitrage) ===")
    print(f"  발견: {len(under)}개")
    for _, r in under.head(15).iterrows():
        ng = "negRisk" if r['any_negrisk'] else ""
        print(f"    sum={r['price_sum']:.3f} ({r['n_outcomes']} outcomes) vol=${r['total_volume']:>10,.0f}  {ng}")
        print(f"      {str(r[group_col])[:80]}")

    # 추정 edge
    if len(over) > 0:
        avg_over = over['price_sum'].mean()
        edge = (avg_over - 1.0) / avg_over * 100
        print(f"\n  Sum > 1.05 평균 edge: {edge:+.2f}% (short)")

    out = {
        'n_groups': int(len(multi)),
        'sum_mean': float(multi['price_sum'].mean()),
        'n_arb_over': int(len(over)),
        'n_arb_under': int(len(under)),
        'top_over': [
            {
                'group': str(r[group_col])[:80],
                'sum': float(r['price_sum']),
                'n_outcomes': int(r['n_outcomes']),
                'volume': float(r['total_volume']),
            }
            for _, r in over.head(20).iterrows()
        ],
    }
    out_path = RESULTS_DIR / "iter21_negrisk_arbitrage.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
