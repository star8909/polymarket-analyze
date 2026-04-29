"""iter02: Calibration 곡선 분석.

질문: "0.05~0.10 가격에 거래된 마켓 중 실제 Yes로 끝난 비율은?"

종료 마켓 데이터로 calibration buckets 분석:
- 가격 0.0~1.0을 20 bin으로 나누고
- 각 bin의 종가(closing price) 마켓들이 실제로 Yes로 끝난 비율 측정
- 시장 가격 ≠ 진짜 확률인 영역 찾음

기대 패턴 (학술 정설):
- 5센트 이하 롱샷: 시장 과대평가 (실제 1~3%만 Yes)
- 95센트 이상 favorite: 시장 약간 과소평가 (실제 92~95% Yes)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, RESULTS_DIR


def parse_outcome(row: pd.Series) -> tuple[float | None, int | None]:
    """단일 마켓 row에서 (close_price, actual_yes) 추출.

    actual_yes: 1 if Yes로 끝남, 0 if No, None if 불명
    close_price: 종가 (마지막 거래)
    """
    try:
        outcomes_str = row.get('outcomes', '[]')
        prices_str = row.get('outcomePrices', '[]')
        if isinstance(outcomes_str, str):
            outcomes = json.loads(outcomes_str)
        else:
            outcomes = outcomes_str
        if isinstance(prices_str, str):
            prices = json.loads(prices_str)
        else:
            prices = prices_str

        if not isinstance(outcomes, list) or not isinstance(prices, list):
            return None, None
        if len(outcomes) != 2 or len(prices) != 2:
            return None, None  # binary only

        prices_f = [float(p) for p in prices]
        # binary 마켓: 보통 [Yes, No]
        # 종가 1.0/0.0 → 결과 확정
        if prices_f[0] > 0.99:
            return prices_f[0], 1  # Yes 측 가격, Yes로 끝남
        if prices_f[1] > 0.99:
            return prices_f[0], 0  # Yes 측 가격은 ~0, No로 끝남
        # 미확정 (가격 중간)
        return prices_f[0], None
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return None, None


def main():
    print("[iter02] Calibration 곡선 분석")
    cache_path = CACHE_DIR / "closed_all.parquet"
    if not cache_path.exists():
        print(f"  ❌ {cache_path} 없음. iter01 먼저 실행 필요.")
        return

    df = pd.read_parquet(cache_path)
    print(f"  종료 마켓 로드: {len(df)} rows")

    # 거래량 필터 (소량 마켓 노이즈 제거)
    if 'volume' in df.columns:
        df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        before = len(df)
        df = df[df['volume_num'] >= 1000]
        print(f"  거래량 1k+ 필터 후: {len(df)} ({before - len(df)} 제거)")

    # outcomes/prices 파싱
    print(f"\n  outcomes 파싱 중...")
    parsed = df.apply(parse_outcome, axis=1)
    df['close_price'] = [p[0] for p in parsed]
    df['actual_yes'] = [p[1] for p in parsed]

    # 결과 확정된 마켓만
    valid = df.dropna(subset=['close_price', 'actual_yes']).copy()
    valid['actual_yes'] = valid['actual_yes'].astype(int)
    print(f"  결과 확정 마켓: {len(valid)}")

    if len(valid) < 100:
        print(f"  ❌ 결과 확정 마켓 너무 적음. 데이터 더 필요.")
        return

    # ─── 메인 calibration ─────────────────────────────────
    print(f"\n=== Calibration 곡선 (가격 vs 실제 Yes 비율) ===")
    bins = np.arange(0.0, 1.05, 0.05)
    valid['bin'] = pd.cut(valid['close_price'], bins=bins, include_lowest=True)
    grouped = valid.groupby('bin').agg(
        n=('close_price', 'size'),
        avg_price=('close_price', 'mean'),
        actual_yes_rate=('actual_yes', 'mean'),
    ).reset_index()
    grouped = grouped[grouped['n'] >= 5]  # 최소 5개

    print(f"  {'Price bin':25s} {'N':>6} {'Avg Price':>10} {'Actual Yes':>10}  Edge")
    print(f"  {'-'*70}")
    for _, row in grouped.iterrows():
        edge = row['actual_yes_rate'] - row['avg_price']
        edge_str = f"{edge:+.3f}"
        marker = "🚀 LONG" if edge > 0.05 else "⚠️ SHORT" if edge < -0.05 else ""
        print(f"  {str(row['bin']):25s} {row['n']:>6.0f} {row['avg_price']:>10.3f} {row['actual_yes_rate']:>10.3f}  {edge_str}  {marker}")

    # ─── 롱샷 편향 체크 ──────────────────────────────────
    print(f"\n=== 롱샷 편향 체크 (5센트 이하) ===")
    longshot = valid[valid['close_price'] <= 0.10]
    if len(longshot) >= 20:
        actual = longshot['actual_yes'].mean()
        avg_p = longshot['close_price'].mean()
        edge = actual - avg_p
        print(f"  N={len(longshot)}, Avg Price={avg_p:.3f}, Actual Yes={actual:.3f}, Edge={edge:+.3f}")
        if edge < -0.03:
            print(f"  🎯 롱샷 편향 확인: 5센트 이하 마켓 → 실제 확률 낮음 → No 베팅 +EV")
        elif edge > 0.03:
            print(f"  🔄 역 롱샷: 5센트 이하 마켓 → 실제 확률 높음 → Yes 베팅 +EV")
        else:
            print(f"  ➡️ 롱샷 편향 없음 ({edge:+.3f}). calibration 잘 됨.")

    print(f"\n=== Favorite 편향 체크 (95센트 이상) ===")
    favorite = valid[valid['close_price'] >= 0.90]
    if len(favorite) >= 20:
        actual = favorite['actual_yes'].mean()
        avg_p = favorite['close_price'].mean()
        edge = actual - avg_p
        print(f"  N={len(favorite)}, Avg Price={avg_p:.3f}, Actual Yes={actual:.3f}, Edge={edge:+.3f}")
        if edge > 0.03:
            print(f"  🎯 Favorite 과소평가: 95센트 이상 → 실제 확률 더 높음 → Yes 베팅 +EV")
        elif edge < -0.03:
            print(f"  🔄 Favorite 과대평가: 95센트 이상 → 실제 확률 낮음 → No 베팅 +EV")
        else:
            print(f"  ➡️ Favorite 편향 없음 ({edge:+.3f}).")

    # ─── 결과 저장 ──────────────────────────────────────
    out = {
        "total_valid_markets": len(valid),
        "calibration_buckets": grouped.to_dict(orient="records"),
        "longshot_edge": float(longshot['actual_yes'].mean() - longshot['close_price'].mean()) if len(longshot) >= 20 else None,
        "favorite_edge": float(favorite['actual_yes'].mean() - favorite['close_price'].mean()) if len(favorite) >= 20 else None,
    }
    out_path = RESULTS_DIR / "iter02_calibration.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
