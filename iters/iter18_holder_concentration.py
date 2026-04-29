"""iter18: Holder concentration (gini coefficient) → market efficiency.

가설: 소수 holder가 차지하는 비율 큰 마켓 = manipulation risk.
- 이런 마켓에서 calibration mispricing 더 클 가능성 (or 반대)
- holder distribution analysis

데이터: warproxxx orderFilled에서 maker/taker address별 거래량.
각 마켓의 unique trader 수 + Gini coefficient 계산.
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


def gini(values):
    """Gini coefficient (0 = equal, 1 = max inequality)."""
    if len(values) == 0:
        return 0
    sorted_v = sorted(values)
    n = len(sorted_v)
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    total = sum(sorted_v)
    if total == 0:
        return 0
    return (2 * cumsum) / (n * total) - (n + 1) / n


def main():
    print("[iter18] Holder concentration (Gini coefficient)")
    cache_path = CACHE_DIR / "closed_all.parquet"
    df = pd.read_parquet(cache_path)
    df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df = df[df['volume_num'] >= 100000].copy()
    print(f"  vol 100k+: {len(df)}")

    # outcomes 파싱
    def parse_settle(row):
        try:
            prices_str = row.get('outcomePrices', '[]')
            prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
            if not isinstance(prices, list) or len(prices) != 2:
                return None
            prices_f = [float(p) for p in prices]
            if prices_f[0] > 0.99:
                return 1
            if prices_f[1] > 0.99:
                return 0
            return None
        except Exception:
            return None

    def get_yes_token(row):
        try:
            ids_str = row.get('clobTokenIds', '[]')
            ids = json.loads(ids_str) if isinstance(ids_str, str) else ids_str
            if isinstance(ids, list) and len(ids) >= 1:
                return str(ids[0])
            return None
        except Exception:
            return None

    df['actual_yes'] = df.apply(parse_settle, axis=1)
    df = df.dropna(subset=['actual_yes'])
    df['actual_yes'] = df['actual_yes'].astype(int)
    df['yes_token'] = df.apply(get_yes_token, axis=1)
    df = df.dropna(subset=['yes_token'])
    print(f"  settle + token: {len(df)}")

    target_tokens = set(df['yes_token'].tolist())

    # Streaming: maker/taker address별 거래량 집계
    print(f"\n  Streaming CSV (trader concentration 측정)...")
    market_traders = defaultdict(lambda: defaultdict(float))  # {token: {address: volume}}

    chunks = 0
    csv_iter = pd.read_csv(POLY_CSV, chunksize=1_000_000, dtype='string',
                           encoding='latin-1', on_bad_lines='skip')
    while True:
        try:
            chunk = next(csv_iter)
        except StopIteration:
            break
        except Exception:
            continue
        chunks += 1
        chunk['makerAmountFilled'] = pd.to_numeric(chunk['makerAmountFilled'], errors='coerce')
        chunk['takerAmountFilled'] = pd.to_numeric(chunk['takerAmountFilled'], errors='coerce')
        chunk = chunk.dropna(subset=['makerAmountFilled', 'takerAmountFilled'])
        if chunk.empty:
            continue

        # 둘 중 하나가 token (target)
        for col_token, col_addr, col_amt in [
            ('takerAssetId', 'taker', 'takerAmountFilled'),
            ('makerAssetId', 'maker', 'makerAmountFilled'),
        ]:
            mask = chunk[col_token].isin(target_tokens)
            sub = chunk[mask]
            if sub.empty:
                continue
            for _, r in sub.iterrows():
                tok = r[col_token]
                addr = r[col_addr]
                amt = r[col_amt]
                market_traders[tok][addr] += amt

        if chunks % 30 == 0:
            print(f"    chunks={chunks}, markets={len(market_traders)}")

    print(f"\n  Markets with trader data: {len(market_traders)}")

    # Gini 계산
    records = []
    for tok, traders in market_traders.items():
        if len(traders) < 5:
            continue
        volumes = list(traders.values())
        g = gini(volumes)
        n_traders = len(traders)
        # actual outcome 매칭
        match = df[df['yes_token'] == tok]
        if match.empty:
            continue
        actual = int(match['actual_yes'].iloc[0])
        records.append({
            'token': tok,
            'n_traders': n_traders,
            'gini': g,
            'actual_yes': actual,
            'total_vol': sum(volumes),
        })

    rec_df = pd.DataFrame(records)
    print(f"  분석 가능 마켓: {len(rec_df)}")

    if len(rec_df) > 50:
        print(f"\n=== Trader concentration 분포 ===")
        print(f"  Gini 평균: {rec_df['gini'].mean():.3f}")
        print(f"  Gini median: {rec_df['gini'].median():.3f}")
        print(f"  Trader 수 평균: {rec_df['n_traders'].mean():.0f}")
        print(f"  Trader 수 median: {rec_df['n_traders'].median():.0f}")

        # Gini high vs low
        print(f"\n=== Gini bucket vs actual outcome ===")
        for low, high, label in [(0, 0.5, "Low Gini (분산)"), (0.5, 0.8, "Mid Gini"), (0.8, 1.0, "High Gini (집중)")]:
            sub = rec_df[(rec_df['gini'] >= low) & (rec_df['gini'] < high)]
            if len(sub) >= 10:
                yes_pct = sub['actual_yes'].mean() * 100
                print(f"  {label} ({low:.1f}~{high:.1f}): N={len(sub)}, Yes 비율 {yes_pct:.1f}%")

        # Trader 수별
        print(f"\n=== Trader 수 bucket vs Yes 비율 ===")
        for low, high, label in [(5, 50, "소수 (5-50)"), (50, 200, "중간 (50-200)"), (200, 100000, "대중 (200+)")]:
            sub = rec_df[(rec_df['n_traders'] >= low) & (rec_df['n_traders'] < high)]
            if len(sub) >= 10:
                yes_pct = sub['actual_yes'].mean() * 100
                print(f"  {label}: N={len(sub)}, Yes 비율 {yes_pct:.1f}%")

    out = {
        "n_markets_analyzed": int(len(rec_df)),
        "gini_mean": float(rec_df['gini'].mean()) if len(rec_df) > 0 else None,
    }
    out_path = RESULTS_DIR / "iter18_holder_concentration.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
