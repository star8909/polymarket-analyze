"""iter17: 0.05 단위 정밀 calibration buckets.

iter14 baseline: 0.10 단위 buckets → 0.30~0.40에서 -11.9% edge 발견
iter17: 0.05 단위로 더 정밀 → sweet spot 좁히기.

특히 0.20~0.50, 0.50~0.80 영역 정밀 분석.
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


def main():
    print("[iter17] 0.05 단위 정밀 calibration buckets")
    cache_path = CACHE_DIR / "closed_all.parquet"
    df = pd.read_parquet(cache_path)
    df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df = df[df['volume_num'] >= 100000].copy()
    df['actual_yes'] = df.apply(parse_settle, axis=1)
    df = df.dropna(subset=['actual_yes'])
    df['actual_yes'] = df['actual_yes'].astype(int)
    df['yes_token'] = df.apply(get_yes_token, axis=1)
    df = df.dropna(subset=['yes_token'])
    df['endDate_dt'] = pd.to_datetime(df['endDate'], errors='coerce', utc=True)
    df = df.dropna(subset=['endDate_dt'])
    df['end_ts'] = df['endDate_dt'].astype('int64') // 10**9
    print(f"  종료 마켓: {len(df)}")

    target_tokens = set(df['yes_token'].tolist())
    token_meta = {r['yes_token']: {'end_ts': int(r['end_ts']), 'actual_yes': int(r['actual_yes'])} for _, r in df.iterrows()}

    print(f"\n  Streaming CSV...")
    market_trades = defaultdict(list)
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
        chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
        chunk['makerAmountFilled'] = pd.to_numeric(chunk['makerAmountFilled'], errors='coerce')
        chunk['takerAmountFilled'] = pd.to_numeric(chunk['takerAmountFilled'], errors='coerce')
        chunk = chunk.dropna(subset=['timestamp', 'makerAmountFilled', 'takerAmountFilled'])
        if chunk.empty:
            continue

        m_zero = chunk['makerAssetId'] == '0'
        t_zero = chunk['takerAssetId'] == '0'

        buy = chunk[m_zero & ~t_zero & chunk['takerAssetId'].isin(target_tokens)]
        if not buy.empty:
            buy_v = buy[buy['takerAmountFilled'] > 0].copy()
            buy_v['price'] = buy_v['makerAmountFilled'] / buy_v['takerAmountFilled']
            buy_v = buy_v[(buy_v['price'] > 0) & (buy_v['price'] < 1)]
            for _, r in buy_v.iterrows():
                market_trades[r['takerAssetId']].append((int(r['timestamp']), float(r['price'])))

        sell = chunk[~m_zero & t_zero & chunk['makerAssetId'].isin(target_tokens)]
        if not sell.empty:
            sell_v = sell[sell['makerAmountFilled'] > 0].copy()
            sell_v['price'] = sell_v['takerAmountFilled'] / sell_v['makerAmountFilled']
            sell_v = sell_v[(sell_v['price'] > 0) & (sell_v['price'] < 1)]
            for _, r in sell_v.iterrows():
                market_trades[r['makerAssetId']].append((int(r['timestamp']), float(r['price'])))

        if chunks % 20 == 0:
            print(f"    chunks={chunks}")

    cb_records = []
    for tok, trades in market_trades.items():
        info = token_meta[tok]
        end_ts = info['end_ts']
        ws, we = end_ts - 7*86400, end_ts - 86400
        in_window = [p for ts, p in trades if ws <= ts <= we]
        if not in_window:
            continue
        cb_records.append({
            'closing_belief': sum(in_window) / len(in_window),
            'actual_yes': info['actual_yes'],
        })

    cb_df = pd.DataFrame(cb_records)
    print(f"\n  Closing belief: {len(cb_df)}")

    print(f"\n=== 0.05 단위 Calibration buckets ===")
    bins = np.arange(0.0, 1.05, 0.05)
    cb_df['bin'] = pd.cut(cb_df['closing_belief'], bins=bins, include_lowest=True)
    grouped = cb_df.groupby('bin', observed=True).agg(
        n=('closing_belief', 'size'),
        avg_belief=('closing_belief', 'mean'),
        actual_yes=('actual_yes', 'mean'),
    ).reset_index()
    grouped = grouped[grouped['n'] >= 5]

    print(f"  {'Price bin':25s} {'N':>5} {'Belief':>8} {'Actual':>8}  Edge   Signal")
    print(f"  {'-'*70}")
    out_data = []
    for _, row in grouped.iterrows():
        edge = row['actual_yes'] - row['avg_belief']
        marker = "🚀 LONG Yes" if edge > 0.05 else \
                 "🔥 SHORT Yes (No 매수)" if edge < -0.05 else ""
        print(f"  {str(row['bin']):25s} {row['n']:>5.0f} {row['avg_belief']:>8.3f} {row['actual_yes']:>8.3f}  {edge:+.3f}  {marker}")
        out_data.append({
            'bin': str(row['bin']),
            'n': int(row['n']),
            'belief': float(row['avg_belief']),
            'actual': float(row['actual_yes']),
            'edge': float(edge),
        })

    # Sweet spot 자동 발견
    print(f"\n=== Sweet spots (|edge| > 5%, N≥10) ===")
    strong_buckets = [d for d in out_data if abs(d['edge']) > 0.05 and d['n'] >= 10]
    for d in strong_buckets:
        side = "No" if d['edge'] < 0 else "Yes"
        print(f"  {d['bin']:25s} N={d['n']} edge={d['edge']:+.3f} → {side} 매수 +EV")

    # 통합 strategy
    print(f"\n=== 통합 베팅 strategy ===")
    short_zones = [d for d in out_data if d['edge'] < -0.05 and d['n'] >= 10]
    long_zones = [d for d in out_data if d['edge'] > 0.05 and d['n'] >= 10]
    if short_zones:
        avg_short_edge = sum(d['edge'] * d['n'] for d in short_zones) / sum(d['n'] for d in short_zones)
        total_n = sum(d['n'] for d in short_zones)
        print(f"  No 매수 영역: {len(short_zones)} buckets, total N={total_n}, weighted avg edge {avg_short_edge:+.3f}")
    if long_zones:
        avg_long_edge = sum(d['edge'] * d['n'] for d in long_zones) / sum(d['n'] for d in long_zones)
        total_n = sum(d['n'] for d in long_zones)
        print(f"  Yes 매수 영역: {len(long_zones)} buckets, total N={total_n}, weighted avg edge {avg_long_edge:+.3f}")

    out = {
        "n_calibrated": len(cb_df),
        "buckets_005": out_data,
        "sweet_spots": strong_buckets,
    }
    out_path = RESULTS_DIR / "iter17_precise_buckets.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
