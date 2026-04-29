"""iter03: Sum-to-1 arbitrage 스캐너.

활성 마켓 중 multi-outcome 이벤트 (예: "2026 대선 후보") 찾아서
모든 Yes 가격 합 측정.

- 합 < 1 → 전부 Yes 매수 (수익 보장)
- 합 > 1 → 전부 No 매수 (수익 보장)

매일 수십 개씩 발생.
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
    print("[iter03] Sum-to-1 arbitrage 스캐너")
    print("  활성 multi-outcome 마켓에서 가격 합 ≠ 1.0 위반 찾기")

    # 활성 마켓 수집 (거래량 1k+ 필터)
    df = fetch_active_markets(min_volume=1000.0)
    print(f"  활성 마켓 (거래량 1k+): {len(df)}")

    if df.empty:
        print("  ❌ 활성 마켓 없음.")
        return

    # 같은 이벤트 (eventId 또는 group)에 속한 마켓 그룹화
    event_groups: dict = defaultdict(list)
    if 'events' in df.columns:
        for _, row in df.iterrows():
            try:
                events_str = row.get('events', '[]')
                if isinstance(events_str, str):
                    events = json.loads(events_str)
                else:
                    events = events_str
                if isinstance(events, list) and events:
                    for e in events:
                        if isinstance(e, dict):
                            event_id = e.get('id') or e.get('slug')
                            if event_id:
                                event_groups[event_id].append(row.to_dict())
                                break
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue

    print(f"  이벤트 그룹: {len(event_groups)}")

    # multi-outcome 이벤트 (2개 이상의 마켓이 같은 이벤트에 속한)
    multi = {eid: ms for eid, ms in event_groups.items() if len(ms) >= 2}
    print(f"  multi-outcome 이벤트 (≥2 마켓): {len(multi)}")

    arbs = []
    for eid, markets in multi.items():
        # 각 마켓의 Yes 가격 (binary 마켓 가정 — Yes 가격 = prices[0])
        yes_prices = []
        slugs = []
        for m in markets:
            parsed = parse_market_outcome(m)
            if not parsed.get('prices') or len(parsed['prices']) < 2:
                continue
            yes_p = parsed['prices'][0]
            yes_prices.append(yes_p)
            slugs.append(m.get('slug', m.get('question', '???')))
        if len(yes_prices) < 2:
            continue
        total = sum(yes_prices)
        deviation = abs(total - 1.0)
        if deviation > 0.05:  # 5%+ 위반만 관심
            arbs.append({
                'event_id': eid,
                'n_markets': len(yes_prices),
                'sum_yes': round(total, 4),
                'deviation': round(deviation, 4),
                'direction': 'BUY_ALL_YES' if total < 1.0 else 'BUY_ALL_NO',
                'expected_edge': round(abs(1.0 - total), 4),
                'markets': slugs[:5],  # 첫 5개만 미리보기
            })

    arbs.sort(key=lambda r: r['deviation'], reverse=True)

    print(f"\n=== Sum-to-1 위반 (편차 5%+) ===")
    if not arbs:
        print(f"  ❌ 5%+ 위반 없음. 시장 effiency 높음 또는 multi-outcome 그룹 부족.")
    else:
        print(f"  발견: {len(arbs)}개")
        for i, arb in enumerate(arbs[:20], 1):
            print(f"\n  [{i}] event {arb['event_id']}")
            print(f"      n_markets={arb['n_markets']}, sum_Yes={arb['sum_yes']:.3f}, edge={arb['expected_edge']:.3f}")
            print(f"      direction: {arb['direction']}")
            print(f"      markets: {', '.join(arb['markets'][:3])}")

    out = {
        "total_arbs": len(arbs),
        "arbs_top20": arbs[:20],
    }
    out_path = RESULTS_DIR / "iter03_sum_to_one.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
