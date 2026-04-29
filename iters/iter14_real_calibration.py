"""iter14: 진짜 calibration with warproxxx onchain orderFilled (file-based).

데이터: 25.4GB CSV decompressed from 5.92GB .xz
86.5M onchain trades (Polygon, Goldsky).

가격 추출:
- makerAssetId == 0 (USDC) → takerAssetId가 token. price = maker/taker
- takerAssetId == 0 (USDC) → makerAssetId가 token. price = taker/maker

closing belief = 종료 [end-7d, end-1d] 거래 평균.
calibration: closing_belief vs actual_yes (settle 0/1).
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
    print("[iter14] 진짜 calibration with full CSV (25.4GB)")
    if not POLY_CSV.exists():
        print(f"  ❌ {POLY_CSV} 없음")
        return

    cache_path = CACHE_DIR / "closed_all.parquet"
    if not cache_path.exists():
        print(f"  ❌ closed_all.parquet 없음")
        return

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

    print(f"  종료 마켓 (vol 100k+, settle, token): {len(df)}")

    target_tokens = set(df['yes_token'].tolist())
    token_to_market = {r['yes_token']: {'end_ts': int(r['end_ts']), 'actual_yes': int(r['actual_yes']), 'q': str(r.get('question', ''))[:80]} for _, r in df.iterrows()}
    print(f"  Target tokens: {len(target_tokens)}")

    # File-based CSV 처리 (압축 해제됨)
    print(f"\n  Reading {POLY_CSV.name}... (25.4GB, ~5분 소요)")
    market_trades = defaultdict(list)

    chunks_processed = 0
    rows_seen = 0
    rows_matched = 0

    csv_iter = pd.read_csv(POLY_CSV, chunksize=1_000_000, dtype='string',
                            encoding='latin-1', on_bad_lines='skip')
    while True:
        try:
            chunk = next(csv_iter)
        except StopIteration:
            break
        except Exception as e:
            print(f"    chunk read error: {e}, skipping")
            continue

        chunks_processed += 1
        rows_seen += len(chunk)

        # 손상 row 제거 — 숫자 변환 안 되는 것
        chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
        chunk['makerAmountFilled'] = pd.to_numeric(chunk['makerAmountFilled'], errors='coerce')
        chunk['takerAmountFilled'] = pd.to_numeric(chunk['takerAmountFilled'], errors='coerce')
        chunk = chunk.dropna(subset=['timestamp', 'makerAmountFilled', 'takerAmountFilled'])
        if chunk.empty:
            continue

        m_zero = chunk['makerAssetId'] == '0'
        t_zero = chunk['takerAssetId'] == '0'

        # USDC (maker=0) → token (taker) — vectorized
        buy_mask = m_zero & ~t_zero & chunk['takerAssetId'].isin(target_tokens)
        buy = chunk[buy_mask]
        if not buy.empty:
            buy_valid = buy[buy['takerAmountFilled'] > 0].copy()
            buy_valid['price'] = buy_valid['makerAmountFilled'] / buy_valid['takerAmountFilled']
            buy_valid = buy_valid[(buy_valid['price'] > 0) & (buy_valid['price'] < 1)]
            for _, r in buy_valid.iterrows():
                market_trades[r['takerAssetId']].append((int(r['timestamp']), float(r['price'])))
                rows_matched += 1

        # token (maker) → USDC (taker=0)
        sell_mask = ~m_zero & t_zero & chunk['makerAssetId'].isin(target_tokens)
        sell = chunk[sell_mask]
        if not sell.empty:
            sell_valid = sell[sell['makerAmountFilled'] > 0].copy()
            sell_valid['price'] = sell_valid['takerAmountFilled'] / sell_valid['makerAmountFilled']
            sell_valid = sell_valid[(sell_valid['price'] > 0) & (sell_valid['price'] < 1)]
            for _, r in sell_valid.iterrows():
                market_trades[r['makerAssetId']].append((int(r['timestamp']), float(r['price'])))
                rows_matched += 1

        if chunks_processed % 5 == 0:
            print(f"    chunks={chunks_processed}, rows={rows_seen:,}, matched={rows_matched:,}, markets={len(market_trades)}")

    print(f"\n  Total: chunks={chunks_processed}, rows={rows_seen:,}, matched={rows_matched:,}")
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
        })

    print(f"\n  Closing belief 추출: {len(closing_beliefs)}")

    if len(closing_beliefs) < 50:
        print(f"  ❌ 샘플 너무 적음")
        return

    cb_df = pd.DataFrame(closing_beliefs)

    print(f"\n=== 🏆 진짜 Calibration 곡선 ===")
    bins = np.arange(0.0, 1.05, 0.10)
    cb_df['bin'] = pd.cut(cb_df['closing_belief'], bins=bins, include_lowest=True)
    grouped = cb_df.groupby('bin', observed=True).agg(
        n=('closing_belief', 'size'),
        avg_belief=('closing_belief', 'mean'),
        actual_yes=('actual_yes', 'mean'),
    ).reset_index()
    grouped = grouped[grouped['n'] >= 5]

    print(f"  {'Price bin':25s} {'N':>5} {'Belief':>8} {'Actual':>8}  Edge   Signal")
    print(f"  {'-'*70}")
    for _, row in grouped.iterrows():
        edge = row['actual_yes'] - row['avg_belief']
        marker = "🚀 LONG Yes" if edge > 0.05 else "⚠️ SHORT Yes" if edge < -0.05 else ""
        print(f"  {str(row['bin']):25s} {row['n']:>5.0f} {row['avg_belief']:>8.3f} {row['actual_yes']:>8.3f}  {edge:+.3f}  {marker}")

    print(f"\n=== 롱샷 (10센트 이하) ===")
    longshot = cb_df[cb_df['closing_belief'] <= 0.10]
    if len(longshot) >= 20:
        avg_b = longshot['closing_belief'].mean()
        actual = longshot['actual_yes'].mean()
        edge = actual - avg_b
        print(f"  N={len(longshot)}, avg belief={avg_b:.3f}, actual={actual:.3f}, edge={edge:+.3f}")
        if edge < -0.02:
            print(f"  🎯 롱샷 편향 — No 베팅 +EV ({-edge*100:.1f}%)")

    print(f"\n=== Favorite (90센트 이상) ===")
    favorite = cb_df[cb_df['closing_belief'] >= 0.90]
    if len(favorite) >= 20:
        avg_b = favorite['closing_belief'].mean()
        actual = favorite['actual_yes'].mean()
        edge = actual - avg_b
        print(f"  N={len(favorite)}, avg belief={avg_b:.3f}, actual={actual:.3f}, edge={edge:+.3f}")
        if edge > 0.02:
            print(f"  🎯 Favorite 과소평가 — Yes 베팅 +EV ({edge*100:.1f}%)")

    out = {
        "n_calibrated": len(closing_beliefs),
        "calibration_buckets": grouped.to_dict(orient="records"),
        "longshot_edge": float(longshot['actual_yes'].mean() - longshot['closing_belief'].mean()) if len(longshot) >= 20 else None,
        "favorite_edge": float(favorite['actual_yes'].mean() - favorite['closing_belief'].mean()) if len(favorite) >= 20 else None,
    }
    out_path = RESULTS_DIR / "iter14_real_calibration.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
