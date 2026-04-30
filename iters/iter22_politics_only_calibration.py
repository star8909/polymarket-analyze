"""iter22: Politics-only calibration on warproxxx 데이터.

iter17 (전체 closed): 0.35-0.40 bucket -16.7% edge
iter22: 정치 마켓만 (slug에 'will-trump'/'election'/'president' etc) → 더 정교 calibration.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import defaultdict
import numpy as np
import pandas as pd

from src.config import CACHE_DIR, RESULTS_DIR

POLY_CSV = Path(__file__).resolve().parent.parent / "data" / "poly_data" / "orderFilled_complete.csv"

POLITICS_KEYWORDS = ['election', 'president', 'trump', 'biden', 'harris', 'congress',
                     'senate', 'governor', 'mayor', 'primary', 'vote', 'candidate', 'democra',
                     'republican', 'gop', 'parliament', 'pm-', 'minister', 'win-the']


def main():
    if not POLY_CSV.exists():
        print(f"  ❌ {POLY_CSV} 없음")
        return
    print(f"[iter22] Politics-only calibration on {POLY_CSV}")
    print(f"  CSV size: {POLY_CSV.stat().st_size / 1024**3:.1f} GB")

    chunks = []
    chunk_count = 0
    politics_count = 0
    rows_seen = 0
    try:
        for chunk in pd.read_csv(POLY_CSV, chunksize=500_000, encoding='latin-1', on_bad_lines='skip',
                                 dtype=str, low_memory=False):
            rows_seen += len(chunk)
            slug = chunk.get('slug', pd.Series(['' ] * len(chunk))).fillna('').str.lower()
            mask = pd.Series(False, index=chunk.index)
            for kw in POLITICS_KEYWORDS:
                mask = mask | slug.str.contains(kw, na=False)
            sub = chunk[mask].copy()
            if len(sub) > 0:
                chunks.append(sub)
                politics_count += len(sub)
            chunk_count += 1
            if chunk_count % 20 == 0:
                print(f"    progress: {rows_seen:,} rows, politics so far: {politics_count:,}")
            if rows_seen > 50_000_000:
                break
    except Exception as e:
        print(f"  ⚠️ chunk error: {e}")
    if not chunks:
        print("  ❌ politics 데이터 없음")
        return

    df = pd.concat(chunks, ignore_index=True)
    print(f"\n  Politics 마켓 trades: {len(df):,}")

    # parse settle
    for col in ['makerAmountFilled', 'takerAmountFilled', 'price', 'maker_amount_filled', 'taker_amount_filled']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # closing_belief = price 마지막 체결 평균
    if 'price' not in df.columns:
        print("  ❌ 'price' column missing")
        print(f"  available: {list(df.columns)[:30]}")
        return

    df = df.dropna(subset=['price'])
    df = df[(df['price'] > 0) & (df['price'] < 1)]
    print(f"  Valid trades (price 0-1): {len(df):,}")
    if len(df) < 1000:
        return

    # market identifier
    market_col = None
    for c in ['conditionId', 'condition_id', 'marketId', 'market_id', 'tokenId', 'token_id', 'slug']:
        if c in df.columns:
            market_col = c
            break
    if not market_col:
        print(f"  ❌ market column missing")
        return
    print(f"  Market column: {market_col}")

    # 각 market별 closing_belief = 마지막 5% 거래의 평균 가격
    df_sorted = df.sort_values(market_col)
    grouped = df.groupby(market_col).agg(
        n_trades=('price', 'count'),
        median_price=('price', 'median'),
        last_price=('price', 'last'),
        first_price=('price', 'first'),
    ).reset_index()
    grouped = grouped[grouped['n_trades'] >= 10]
    print(f"  Markets with 10+ trades: {len(grouped):,}")

    # use last_price as closing belief
    grouped['close_belief'] = grouped['last_price']

    # 결과는 last_price로 추정 (정확한 settle 데이터 부재 시)
    # closing_belief의 분포만 보고 calibration 근사 측정
    print(f"\n=== Politics 마켓 closing_belief 분포 ===")
    print(f"  n_markets: {len(grouped)}")
    print(f"  belief mean: {grouped['close_belief'].mean():.3f}")
    print(f"  belief 25%: {grouped['close_belief'].quantile(0.25):.3f}")
    print(f"  belief 50%: {grouped['close_belief'].quantile(0.5):.3f}")
    print(f"  belief 75%: {grouped['close_belief'].quantile(0.75):.3f}")

    print(f"\n=== Bucket별 분포 ===")
    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
    grouped['bucket'] = pd.cut(grouped['close_belief'], bins=bins)
    for b, sub in grouped.groupby('bucket'):
        if len(sub) > 0:
            print(f"  {str(b):20s} N={len(sub):>5} mean={sub['close_belief'].mean():.3f}")

    out = {
        'n_politics_trades': int(len(df)),
        'n_markets': int(len(grouped)),
        'belief_mean': float(grouped['close_belief'].mean()),
        'belief_median': float(grouped['close_belief'].quantile(0.5)),
    }
    out_path = RESULTS_DIR / "iter22_politics_only.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
