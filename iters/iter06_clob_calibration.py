"""iter06: CLOB API로 진짜 closing belief 기반 calibration.

iter02 실패 원인: closed market의 lastTradePrice는 settle 후 reset → 가짜 0.

iter06 해결:
1. clobTokenIds[0] (Yes 토큰) 추출
2. CLOB /prices-history로 가격 시계열 fetch
3. 종료 7일 전 평균 가격 = 진짜 "closing belief"
4. 이 closing belief vs actual outcome (settle 0/1)로 calibration 곡선

거래량 100k+ 마켓 (~3,289개)만 sample — 진짜 의미있는 calibration.
샘플 ~500개 (API 호출 감안). 분당 ~3 req → 500 req = 3분.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, RESULTS_DIR
from src.data_loader import fetch_price_history


def parse_settle(row: pd.Series) -> int | None:
    """outcomePrices에서 settle 결과 (0/1) 추출."""
    try:
        prices_str = row.get('outcomePrices', '[]')
        if isinstance(prices_str, str):
            prices = json.loads(prices_str)
        else:
            prices = prices_str
        if not isinstance(prices, list) or len(prices) != 2:
            return None
        prices_f = [float(p) for p in prices]
        if prices_f[0] > 0.99:
            return 1
        if prices_f[1] > 0.99:
            return 0
        return None
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return None


def get_yes_token_id(row: pd.Series) -> str | None:
    """clobTokenIds[0] (Yes 토큰) 추출."""
    try:
        ids_str = row.get('clobTokenIds', '[]')
        if isinstance(ids_str, str):
            ids = json.loads(ids_str)
        else:
            ids = ids_str
        if isinstance(ids, list) and len(ids) >= 1:
            return str(ids[0])
        return None
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return None


def fetch_closing_belief(token_id: str, end_date: str, days_before: int = 7) -> float | None:
    """종료 N일 전 평균 가격 = closing belief."""
    try:
        df = fetch_price_history(token_id, interval="1h", fidelity=60)
        if df.empty:
            return None
        # 종료일 N일 전 ~ 종료일 1일 전 사이
        end_ts = pd.Timestamp(end_date)
        start_ts = end_ts - pd.Timedelta(days=days_before + 1)
        end_cutoff = end_ts - pd.Timedelta(days=1)
        window = df[(df.index >= start_ts) & (df.index <= end_cutoff)]
        if window.empty:
            return None
        return float(window['price'].mean())
    except Exception:
        return None


def main():
    print("[iter06] CLOB prices-history → 진짜 calibration")
    cache_path = CACHE_DIR / "closed_all.parquet"
    if not cache_path.exists():
        print(f"  ❌ {cache_path} 없음. iter01 먼저.")
        return

    df = pd.read_parquet(cache_path)
    print(f"  종료 마켓 로드: {len(df)} rows")

    # 거래량 100k+ 필터 (진짜 의미있는 마켓만)
    df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df = df[df['volume_num'] >= 100000].copy()
    print(f"  거래량 100k+ 필터: {len(df)}")

    # settle 결과 파싱
    df['actual_yes'] = df.apply(parse_settle, axis=1)
    df = df.dropna(subset=['actual_yes'])
    df['actual_yes'] = df['actual_yes'].astype(int)
    print(f"  settle 확정: {len(df)}")

    # Yes 토큰 ID
    df['yes_token'] = df.apply(get_yes_token_id, axis=1)
    df = df.dropna(subset=['yes_token'])
    print(f"  토큰 ID 있음: {len(df)}")

    if 'endDate' not in df.columns:
        print("  ❌ endDate 컬럼 없음.")
        return

    # 샘플 추출 (거래량 큰 순으로 500개)
    df = df.sort_values('volume_num', ascending=False).head(500)
    print(f"  샘플: {len(df)} (거래량 큰 순)")

    # 각 마켓의 closing belief fetch
    print(f"\n  CLOB API로 가격 히스토리 fetch (예상 5~10분)...")
    closing_beliefs = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i > 0 and i % 50 == 0:
            print(f"    progress: {i}/{len(df)}")
        try:
            belief = fetch_closing_belief(row['yes_token'], row['endDate'], days_before=7)
            closing_beliefs.append(belief)
        except Exception:
            closing_beliefs.append(None)
        time.sleep(0.3)  # rate limit

    df['closing_belief'] = closing_beliefs
    valid = df.dropna(subset=['closing_belief']).copy()
    print(f"\n  closing belief 추출 성공: {len(valid)}/{len(df)}")

    if len(valid) < 50:
        print(f"  ❌ 샘플 너무 적음. CLOB API 응답 문제 가능성.")
        return

    # ─── Calibration 곡선 ──────────────────────────────
    print(f"\n=== 진짜 Calibration 곡선 (CLOB closing belief vs actual outcome) ===")
    bins = np.arange(0.0, 1.05, 0.05)
    valid['bin'] = pd.cut(valid['closing_belief'], bins=bins, include_lowest=True)
    grouped = valid.groupby('bin', observed=True).agg(
        n=('closing_belief', 'size'),
        avg_belief=('closing_belief', 'mean'),
        actual_yes_rate=('actual_yes', 'mean'),
    ).reset_index()
    grouped = grouped[grouped['n'] >= 5]

    print(f"  {'Price bin':25s} {'N':>6} {'Belief':>8} {'Actual':>8}  Edge   Signal")
    print(f"  {'-'*72}")
    for _, row in grouped.iterrows():
        edge = row['actual_yes_rate'] - row['avg_belief']
        edge_str = f"{edge:+.3f}"
        marker = "🚀 LONG Yes" if edge > 0.05 else "⚠️ SHORT Yes" if edge < -0.05 else ""
        print(f"  {str(row['bin']):25s} {row['n']:>6.0f} {row['avg_belief']:>8.3f} {row['actual_yes_rate']:>8.3f}  {edge_str}  {marker}")

    # 롱샷/Favorite 편향
    print(f"\n=== 롱샷 (10센트 이하) ===")
    longshot = valid[valid['closing_belief'] <= 0.10]
    if len(longshot) >= 10:
        avg_b = longshot['closing_belief'].mean()
        actual = longshot['actual_yes'].mean()
        edge = actual - avg_b
        print(f"  N={len(longshot)}, avg belief={avg_b:.3f}, actual={actual:.3f}, edge={edge:+.3f}")
        if edge < -0.02:
            print(f"  🎯 롱샷 편향 확인: 시장 과대평가 → No 베팅 +EV")

    print(f"\n=== Favorite (90센트 이상) ===")
    favorite = valid[valid['closing_belief'] >= 0.90]
    if len(favorite) >= 10:
        avg_b = favorite['closing_belief'].mean()
        actual = favorite['actual_yes'].mean()
        edge = actual - avg_b
        print(f"  N={len(favorite)}, avg belief={avg_b:.3f}, actual={actual:.3f}, edge={edge:+.3f}")
        if edge > 0.02:
            print(f"  🎯 Favorite 과소평가 → Yes 베팅 +EV")

    out = {
        "n_total_samples": len(df),
        "n_valid": len(valid),
        "calibration": grouped.to_dict(orient="records"),
        "longshot_edge": float(longshot['actual_yes'].mean() - longshot['closing_belief'].mean()) if len(longshot) >= 10 else None,
        "favorite_edge": float(favorite['actual_yes'].mean() - favorite['closing_belief'].mean()) if len(favorite) >= 10 else None,
    }
    out_path = RESULTS_DIR / "iter06_clob_calibration.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
