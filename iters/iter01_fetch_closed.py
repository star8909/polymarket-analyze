"""iter01: Polymarket 종료 마켓 데이터 수집 (Gamma API).

calibration 곡선 분석의 ground truth.
가격 vs 실제 결과를 알기 위해 종료된 모든 마켓 필요.

목표:
- 카테고리별 분리 수집
- parquet 캐시 저장
- 통계 출력 (총 마켓 수, 카테고리 분포, 평균 거래량)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import Counter

import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import fetch_all_closed_markets


def main():
    print("[iter01] Polymarket 종료 마켓 수집 (Gamma API)")
    print("  warning: 첫 실행 시 수만 마켓 페이지네이션 → 5~30분 소요")
    print()

    # 전체 종료 마켓 fetch (카테고리 무관)
    df = fetch_all_closed_markets(category=None, max_markets=20000)
    print(f"\n총 수집: {len(df)} markets")

    if df.empty:
        print("  ❌ 데이터 없음. API 응답 확인 필요.")
        return

    # 기본 통계
    print(f"\n=== 통계 ===")
    print(f"  컬럼 수: {len(df.columns)}")
    print(f"  주요 컬럼: {list(df.columns)[:15]}")

    # 카테고리 분포 (tags 컬럼)
    if 'tags' in df.columns:
        all_tags: list[str] = []
        for tag_str in df['tags'].dropna():
            try:
                if isinstance(tag_str, str):
                    tags = json.loads(tag_str)
                    if isinstance(tags, list):
                        all_tags.extend([t.get('slug', '') if isinstance(t, dict) else str(t) for t in tags])
                elif isinstance(tag_str, list):
                    all_tags.extend([t.get('slug', '') if isinstance(t, dict) else str(t) for t in tag_str])
            except (json.JSONDecodeError, AttributeError):
                continue
        tag_counts = Counter(all_tags)
        print(f"\n=== Top 20 카테고리 ===")
        for tag, count in tag_counts.most_common(20):
            print(f"  {tag:30s}: {count:5d}")

    # 거래량 분포
    if 'volume' in df.columns:
        df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        print(f"\n=== 거래량 ===")
        print(f"  평균: ${df['volume_num'].mean():,.0f}")
        print(f"  중간값: ${df['volume_num'].median():,.0f}")
        print(f"  거래량 1k+ 마켓: {(df['volume_num'] >= 1000).sum():,}")
        print(f"  거래량 10k+ 마켓: {(df['volume_num'] >= 10000).sum():,}")
        print(f"  거래량 100k+ 마켓: {(df['volume_num'] >= 100000).sum():,}")

    # outcomes/prices 파싱 시도
    if 'outcomes' in df.columns and 'outcomePrices' in df.columns:
        valid_outcome = 0
        for _, row in df.head(100).iterrows():
            try:
                if isinstance(row['outcomes'], str):
                    o = json.loads(row['outcomes'])
                    if isinstance(o, list) and len(o) >= 2:
                        valid_outcome += 1
            except (json.JSONDecodeError, AttributeError):
                pass
        print(f"\n  outcome 파싱 가능: {valid_outcome}/100 (100 샘플 기준)")

    # JSON 요약 저장
    summary = {
        "total_markets": len(df),
        "columns": list(df.columns),
        "top_categories": dict(tag_counts.most_common(20)) if 'tags' in df.columns else {},
        "volume_stats": {
            "mean": float(df['volume_num'].mean()) if 'volume_num' in df.columns else None,
            "median": float(df['volume_num'].median()) if 'volume_num' in df.columns else None,
            "ge_1k": int((df['volume_num'] >= 1000).sum()) if 'volume_num' in df.columns else None,
            "ge_10k": int((df['volume_num'] >= 10000).sum()) if 'volume_num' in df.columns else None,
        },
    }
    out = RESULTS_DIR / "iter01_fetch_closed.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\n  → {out}")
    print(f"  → 캐시: data/cache/closed_all.parquet")


if __name__ == "__main__":
    main()
