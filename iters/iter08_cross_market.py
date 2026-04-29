"""iter08: Cross-market consistency (조건부 확률 위반 차익).

가설: Polymarket의 여러 마켓이 조건부 관계 가져야 함.
P(트럼프 대통령) ≤ P(트럼프 공화당 후보) — 후자가 전자의 superset.
P(BTC 12만 12/31) ≤ P(BTC 10만 12/31) — 12만은 10만의 subset.

위반:
- P(A) > P(B) 인데 A ⊂ B → 차익 (P(A) 매도, P(B) 매수)

찾기:
- 같은 키워드 (Trump, BTC, ETH 등) + 같은 endDate
- 강도 비교 가능한 페어
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import re
from collections import defaultdict
from itertools import combinations

import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import fetch_active_markets, parse_market_outcome


def extract_threshold(question: str) -> tuple[str | None, float | None]:
    """질문에서 키워드 + 임계값 추출.

    예: "Will BTC reach $120k by Dec 31?" → ('BTC', 120000)
        "Will Trump win the 2024 election?" → ('Trump-election', None)
    """
    q = question.lower()
    # $XXk, $XXM 패턴
    m = re.search(r'\$([\d,]+)\s*(k|m|b)?', q)
    if m:
        n = float(m.group(1).replace(',', ''))
        unit = m.group(2)
        if unit == 'k':
            n *= 1000
        elif unit == 'm':
            n *= 1_000_000
        elif unit == 'b':
            n *= 1_000_000_000
        return q, n
    return None, None


def main():
    print("[iter08] Cross-market consistency (조건부 확률 위반)")

    df = fetch_active_markets(min_volume=10000.0, max_pages=20)
    print(f"  활성 마켓 (vol 10k+): {len(df)}")

    if df.empty:
        print("  ❌ 데이터 없음.")
        return

    # 가격 추출
    df['last_price'] = pd.to_numeric(df.get('lastTradePrice', None), errors='coerce')
    df = df.dropna(subset=['last_price'])
    df = df[(df['last_price'] > 0.01) & (df['last_price'] < 0.99)]
    print(f"  의미있는 가격 (0.01~0.99): {len(df)}")

    if 'endDate' not in df.columns or 'question' not in df.columns:
        print("  ❌ endDate/question 없음.")
        return

    # 같은 endDate + 키워드 매칭으로 페어 찾기
    df['endDate_str'] = df['endDate'].astype(str).str[:10]  # 날짜 단위
    df['q_lower'] = df['question'].astype(str).str.lower()

    # 키워드별 그룹화 (간단한 매칭)
    keywords = ['trump', 'btc', 'bitcoin', 'eth', 'ethereum', 'sol', 'biden',
                'recession', 'rate cut', 'rate hike', 'fed', 'cpi', 'inflation',
                'deepseek', 'gpt', 'tesla', 'apple', 'nvidia']

    violations = []
    for kw in keywords:
        sub = df[df['q_lower'].str.contains(kw, na=False)].copy()
        if len(sub) < 2:
            continue

        # 같은 endDate끼리 비교
        for date_grp, dg in sub.groupby('endDate_str'):
            if len(dg) < 2:
                continue

            # 임계값 비교 (BTC 100k vs 120k 같은)
            for (i, r1), (j, r2) in combinations(enumerate(dg.itertuples()), 2):
                p1, p2 = r1.last_price, r2.last_price
                q1, q2 = r1.q_lower, r2.q_lower
                # 임계값 추출
                _, n1 = extract_threshold(q1)
                _, n2 = extract_threshold(q2)
                if n1 is None or n2 is None or n1 == n2:
                    continue
                # n1 < n2 이면 P(reach n2) ≤ P(reach n1) 이어야 함
                # 즉 가격(n2) > 가격(n1)이면 위반
                if (n1 < n2 and p2 > p1 + 0.02) or (n2 < n1 and p1 > p2 + 0.02):
                    edge = abs(p1 - p2) - 0.02
                    if edge > 0.03:
                        violations.append({
                            'keyword': kw,
                            'endDate': date_grp,
                            'q1': r1.question[:80],
                            'q2': r2.question[:80],
                            'p1': round(p1, 4),
                            'p2': round(p2, 4),
                            'n1': n1,
                            'n2': n2,
                            'edge_pct': round(edge * 100, 2),
                            'vol1': float(pd.to_numeric(getattr(r1, 'volume', 0), errors='coerce') or 0),
                            'vol2': float(pd.to_numeric(getattr(r2, 'volume', 0), errors='coerce') or 0),
                        })

    violations.sort(key=lambda r: r['edge_pct'], reverse=True)

    print(f"\n=== Cross-market 조건부 위반 (수치 임계값 기반) ===")
    if not violations:
        print(f"  ❌ 위반 없음. 시장 효율적이거나 매칭 어려움.")
    else:
        print(f"  발견: {len(violations)}개")
        for i, v in enumerate(violations[:15], 1):
            print(f"\n  [{i}] {v['keyword'].upper()} | endDate={v['endDate']}")
            print(f"      Q1: {v['q1'][:70]} | n={v['n1']:,.0f} | price={v['p1']:.3f}")
            print(f"      Q2: {v['q2'][:70]} | n={v['n2']:,.0f} | price={v['p2']:.3f}")
            print(f"      edge={v['edge_pct']:.2f}% (낮은 임계값 가격이 더 비싸야 정상)")

    out = {
        "n_keywords": len(keywords),
        "violations_count": len(violations),
        "top_15": violations[:15],
    }
    out_path = RESULTS_DIR / "iter08_cross_market.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
