"""iter12: 카테고리별 calibration (정치 / 스포츠 / 크립토 / macro 분리).

가설: 카테고리마다 시장 효율성 다름.
- 정치 (대선, 총선): 효율적 (많은 사람 정보 보유)
- 스포츠: 약간 비효율 (감정 베팅)
- 크립토: 변동성 크고 비합리 (FOMO/FUD)
- macro (Fed, CPI): 가장 효율적 (전문가 dominated)

각 카테고리에 별도 calibration 곡선 → 베팅 전략 차별화.

iter02 한계 (lastTradePrice settle reset)는 그대로지만,
oneWeekPriceChange로 종료 1주일 전 가격 역산 시도:
  closing_belief ≈ lastTradePrice / (1 + oneWeekPriceChange)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import Counter

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, RESULTS_DIR


def parse_settle(row: pd.Series) -> int | None:
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


def categorize(question: str) -> str:
    q = question.lower()
    political_kws = ['election', 'president', 'vote', 'senate', 'governor', 'congress',
                     'trump', 'biden', 'harris', 'putin', 'primary', 'mayor']
    sports_kws = ['nfl', 'nba', 'mlb', 'nhl', 'ufc', 'soccer', 'football', 'basketball',
                  'tennis', 'golf', 'world cup', 'super bowl', 'champion', 'olympic',
                  'wimbledon', 'masters', 'race', 'tournament']
    crypto_kws = ['bitcoin', 'btc', 'eth', 'ethereum', 'solana', 'sol', 'crypto',
                  'coin', 'token', 'matic', 'avax', 'memecoin', 'altcoin']
    macro_kws = ['fed', 'fomc', 'cpi', 'inflation', 'rate cut', 'rate hike', 'gdp',
                 'unemployment', 'jobs report', 'pce', 'recession', 'yield']
    tech_kws = ['ai', 'gpt', 'openai', 'anthropic', 'claude', 'gemini', 'tesla',
                'apple', 'microsoft', 'nvidia', 'google', 'meta']

    for kw in political_kws:
        if kw in q:
            return 'politics'
    for kw in sports_kws:
        if kw in q:
            return 'sports'
    for kw in crypto_kws:
        if kw in q:
            return 'crypto'
    for kw in macro_kws:
        if kw in q:
            return 'macro'
    for kw in tech_kws:
        if kw in q:
            return 'tech'
    return 'other'


def main():
    print("[iter12] 카테고리별 calibration 분석")
    cache_path = CACHE_DIR / "closed_all.parquet"
    if not cache_path.exists():
        print(f"  ❌ {cache_path} 없음.")
        return

    df = pd.read_parquet(cache_path)
    print(f"  종료 마켓 로드: {len(df)}")

    # 거래량 100k+
    df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df = df[df['volume_num'] >= 100000].copy()
    print(f"  거래량 100k+: {len(df)}")

    df['actual_yes'] = df.apply(parse_settle, axis=1)
    df = df.dropna(subset=['actual_yes'])
    df['actual_yes'] = df['actual_yes'].astype(int)
    print(f"  settle 확정: {len(df)}")

    # closing belief 역산 (oneWeekPriceChange로)
    df['lastTradePrice'] = pd.to_numeric(df.get('lastTradePrice', 0), errors='coerce')
    df['oneWeekPriceChange'] = pd.to_numeric(df.get('oneWeekPriceChange', None), errors='coerce')

    # 1주일 전 가격 = last / (1 + change) — change가 percentage (0.X) 가정
    df['closing_belief'] = df.apply(
        lambda r: r['lastTradePrice'] / (1 + r['oneWeekPriceChange']) if pd.notna(r['oneWeekPriceChange']) and (1 + r['oneWeekPriceChange']) > 0.01 else None,
        axis=1
    )
    df = df.dropna(subset=['closing_belief'])
    df = df[(df['closing_belief'] > 0) & (df['closing_belief'] < 1)]
    print(f"  closing belief 역산: {len(df)}")

    # 카테고리 분류
    df['category'] = df['question'].astype(str).apply(categorize)
    cat_counts = Counter(df['category'])
    print(f"\n  카테고리 분포: {dict(cat_counts)}")

    # 카테고리별 calibration
    print(f"\n=== 카테고리별 Calibration (저가 0~10% bucket) ===")
    out_data = {"categories": {}}
    for cat in ['politics', 'sports', 'crypto', 'macro', 'tech', 'other']:
        sub = df[df['category'] == cat]
        if len(sub) < 30:
            continue
        # 5센트 이하
        low = sub[sub['closing_belief'] <= 0.10]
        if len(low) >= 10:
            avg_b = low['closing_belief'].mean()
            actual = low['actual_yes'].mean()
            edge = actual - avg_b
            print(f"  {cat:12s} | N={len(low):>4} | avg belief={avg_b:.3f} | actual={actual:.3f} | edge={edge:+.3f}")
            out_data["categories"][cat] = {
                'n_low': int(len(low)),
                'avg_belief_low': float(avg_b),
                'actual_yes_low': float(actual),
                'edge_low': float(edge),
            }

    print(f"\n=== 카테고리별 Calibration (고가 90~100% bucket) ===")
    for cat in ['politics', 'sports', 'crypto', 'macro', 'tech', 'other']:
        sub = df[df['category'] == cat]
        if len(sub) < 30:
            continue
        high = sub[sub['closing_belief'] >= 0.90]
        if len(high) >= 10:
            avg_b = high['closing_belief'].mean()
            actual = high['actual_yes'].mean()
            edge = actual - avg_b
            print(f"  {cat:12s} | N={len(high):>4} | avg belief={avg_b:.3f} | actual={actual:.3f} | edge={edge:+.3f}")
            if cat in out_data["categories"]:
                out_data["categories"][cat]['n_high'] = int(len(high))
                out_data["categories"][cat]['avg_belief_high'] = float(avg_b)
                out_data["categories"][cat]['actual_yes_high'] = float(actual)
                out_data["categories"][cat]['edge_high'] = float(edge)

    print(f"\n=== 전체 calibration (모든 카테고리 합쳐) ===")
    bins = np.arange(0.0, 1.05, 0.10)
    df['bin'] = pd.cut(df['closing_belief'], bins=bins, include_lowest=True)
    grouped = df.groupby('bin', observed=True).agg(
        n=('closing_belief', 'size'),
        avg_belief=('closing_belief', 'mean'),
        actual_yes=('actual_yes', 'mean'),
    ).reset_index()
    grouped = grouped[grouped['n'] >= 20]
    for _, row in grouped.iterrows():
        edge = row['actual_yes'] - row['avg_belief']
        marker = "🚀 LONG" if edge > 0.05 else "⚠️ SHORT" if edge < -0.05 else ""
        print(f"  {str(row['bin']):20s} N={row['n']:>4} | belief={row['avg_belief']:.3f} | "
              f"actual={row['actual_yes']:.3f} | edge={edge:+.3f}  {marker}")

    out_data["total_n"] = int(len(df))
    out_data["calibration_buckets_10"] = grouped.to_dict(orient="records")
    out_path = RESULTS_DIR / "iter12_category_calib.json"
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
