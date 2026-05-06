"""iter14b: calibration 재실행 — 갱신된 closed_all (683k 마켓) + 원본 CSV + parallel windows.

데이터 소스:
1. closed_all.parquet (683k markets, 2026-04-30까지)
2. orderFilled_complete.csv (37GB, ~2025-10-07까지)
3. goldsky_parallel/window_00..07.csv (41GB partial, 10/07~04/11 spotty)
"""
from __future__ import annotations

import sys
import json
import glob
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, RESULTS_DIR


_BASE = Path("c:/Projects/Quant/polymarket-analyze/data/poly_data")

# .csv 또는 .csv.xz 모두 지원 (pandas auto-detect)
def _pick_main():
    for name in ["orderFilled_complete.csv", "orderFilled_complete.csv.xz"]:
        p = _BASE / name
        if p.exists():
            return p
    return None

CSV_FILES = ([_pick_main()] if _pick_main() else []) + \
    sorted((_BASE / "goldsky_parallel").glob("window_*.csv")) + \
    sorted((_BASE / "goldsky_parallel").glob("window_*.csv.xz"))


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
    print("[iter14b] calibration v2 — 새 데이터 + 부분 windows")
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

    print(f"  종료+settle 마켓 (vol 100k+): {len(df)}")

    target_tokens = set(df['yes_token'].tolist())
    token_to_market = {r['yes_token']: {
        'end_ts': int(r['end_ts']),
        'actual_yes': int(r['actual_yes']),
        'q': str(r.get('question', ''))[:80],
        'category': str(r.get('category', '')) if pd.notna(r.get('category', None)) else '',
        'volume': float(r.get('volumeNum', 0)) if pd.notna(r.get('volumeNum', 0)) else 0.0,
    } for _, r in df.iterrows()}
    print(f"  Target tokens: {len(target_tokens)}")

    market_trades = defaultdict(list)
    rows_seen_total = 0
    rows_matched_total = 0

    for csv_path in CSV_FILES:
        if not csv_path.exists():
            print(f"  ⚠️  skip (없음): {csv_path}")
            continue
        size_gb = csv_path.stat().st_size / 1e9
        print(f"\n  Reading {csv_path.name} ({size_gb:.1f}GB)...")
        chunks_in_file = 0
        rows_seen_file = 0
        rows_matched_file = 0

        try:
            csv_iter = pd.read_csv(csv_path, chunksize=1_000_000, dtype='string',
                                    encoding='latin-1', on_bad_lines='skip')
        except Exception as e:
            print(f"    ❌ open error: {e}")
            continue

        while True:
            try:
                chunk = next(csv_iter)
            except StopIteration:
                break
            except Exception as e:
                print(f"    chunk read error: {e}, skipping")
                continue

            chunks_in_file += 1
            rows_seen_file += len(chunk)

            chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
            chunk['makerAmountFilled'] = pd.to_numeric(chunk['makerAmountFilled'], errors='coerce')
            chunk['takerAmountFilled'] = pd.to_numeric(chunk['takerAmountFilled'], errors='coerce')
            chunk = chunk.dropna(subset=['timestamp', 'makerAmountFilled', 'takerAmountFilled'])
            if chunk.empty:
                continue

            m_zero = chunk['makerAssetId'] == '0'
            t_zero = chunk['takerAssetId'] == '0'

            buy_mask = m_zero & ~t_zero & chunk['takerAssetId'].isin(target_tokens)
            buy = chunk[buy_mask]
            if not buy.empty:
                buy_valid = buy[buy['takerAmountFilled'] > 0].copy()
                buy_valid['price'] = buy_valid['makerAmountFilled'] / buy_valid['takerAmountFilled']
                buy_valid = buy_valid[(buy_valid['price'] > 0) & (buy_valid['price'] < 1)]
                for _, r in buy_valid.iterrows():
                    market_trades[r['takerAssetId']].append((int(r['timestamp']), float(r['price'])))
                    rows_matched_file += 1

            sell_mask = ~m_zero & t_zero & chunk['makerAssetId'].isin(target_tokens)
            sell = chunk[sell_mask]
            if not sell.empty:
                sell_valid = sell[sell['makerAmountFilled'] > 0].copy()
                sell_valid['price'] = sell_valid['takerAmountFilled'] / sell_valid['makerAmountFilled']
                sell_valid = sell_valid[(sell_valid['price'] > 0) & (sell_valid['price'] < 1)]
                for _, r in sell_valid.iterrows():
                    market_trades[r['makerAssetId']].append((int(r['timestamp']), float(r['price'])))
                    rows_matched_file += 1

            if chunks_in_file % 10 == 0:
                print(f"    chunks={chunks_in_file}, rows={rows_seen_file:,}, "
                      f"matched={rows_matched_file:,}, total markets={len(market_trades)}")

        rows_seen_total += rows_seen_file
        rows_matched_total += rows_matched_file
        print(f"    DONE: chunks={chunks_in_file}, rows={rows_seen_file:,}, matched={rows_matched_file:,}")

    print(f"\n  Total: rows={rows_seen_total:,}, matched={rows_matched_total:,}")
    print(f"  Markets with trades: {len(market_trades)}")

    # closing belief 계산
    closing_beliefs = []
    for tok, trades in market_trades.items():
        info = token_to_market[tok]
        end_ts = info['end_ts']
        ws = end_ts - 7 * 86400
        we = end_ts - 1 * 86400
        in_window = [p for ts, p in trades if ws <= ts <= we]
        if not in_window:
            continue
        avg_price = sum(in_window) / len(in_window)
        closing_beliefs.append({
            'token': tok,
            'closing_belief': avg_price,
            'actual_yes': info['actual_yes'],
            'n_trades': len(in_window),
            'q': info['q'],
            'end_ts': info['end_ts'],
            'category': info['category'],
            'volume': info['volume'],
        })

    print(f"\n  Closing belief 추출: {len(closing_beliefs)} markets")
    if closing_beliefs:
        cb_full = pd.DataFrame(closing_beliefs)
        cb_path = CACHE_DIR / "closing_beliefs_v2.parquet"
        cb_full.to_parquet(cb_path, index=False)
        print(f"  → {cb_path}")

    if len(closing_beliefs) < 50:
        print(f"  ❌ 샘플 너무 적음")
        return

    cb_df = pd.DataFrame(closing_beliefs)

    print(f"\n=== 🏆 v2 Calibration 곡선 ===")
    bins = np.arange(0.0, 1.05, 0.10)
    cb_df['bin'] = pd.cut(cb_df['closing_belief'], bins=bins, include_lowest=True)
    grouped = cb_df.groupby('bin', observed=True).agg(
        n=('closing_belief', 'size'),
        avg_belief=('closing_belief', 'mean'),
        actual_yes=('actual_yes', 'mean'),
    ).reset_index()
    grouped = grouped[grouped['n'] >= 5]

    print(f"\n  {'Price bin':25s} {'N':>5} {'Belief':>8} {'Actual':>8}  Edge   Side")
    print(f"  {'-'*70}")
    for _, row in grouped.iterrows():
        edge = row['actual_yes'] - row['avg_belief']
        side = "🚀 NO" if edge < -0.05 else "🟢 YES" if edge > 0.05 else "(정확)"
        print(f"  {str(row['bin']):25s} {row['n']:>5.0f} {row['avg_belief']:>8.3f} "
              f"{row['actual_yes']:>8.3f}  {edge:+.3f}  {side}")

    out = {
        "version": "v2",
        "n_calibrated": len(closing_beliefs),
        "n_markets_metadata": len(target_tokens),
        "calibration_buckets": grouped.to_dict(orient="records"),
        "data_sources": [str(p) for p in CSV_FILES if p.exists()],
        "rows_processed": rows_seen_total,
        "rows_matched": rows_matched_total,
    }
    out_path = RESULTS_DIR / "iter14b_real_calibration_v2.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
