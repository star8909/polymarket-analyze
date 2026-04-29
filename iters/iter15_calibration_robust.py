"""iter15: iter14 calibration 카테고리별 분리 robust 검증.

iter14 발견:
- 0.30~0.40 가격: 시장 0.346 vs 실제 0.227 → No 베팅 +EV (Edge -11.9%)
- 0.60~0.70: Yes 베팅 +EV (Edge +7.1%)

iter15 검증:
- 카테고리별 (정치/스포츠/크립토/macro)로 패턴 일관성
- 시간대별 (2022/2023/2024) drift 확인
- 거래량 100k+ vs 1M+ 차이
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


def categorize(question: str) -> str:
    q = str(question).lower()
    political_kws = ['election', 'president', 'vote', 'senate', 'governor', 'congress',
                     'trump', 'biden', 'harris', 'putin', 'primary', 'mayor']
    sports_kws = ['nfl', 'nba', 'mlb', 'nhl', 'ufc', 'soccer', 'football', 'basketball',
                  'tennis', 'golf', 'world cup', 'super bowl', 'champion', 'olympic',
                  'wimbledon', 'masters', 'race', 'tournament', 'eurovision']
    crypto_kws = ['bitcoin', 'btc', 'eth', 'ethereum', 'solana', 'sol', 'crypto',
                  'coin', 'token', 'matic', 'avax', 'memecoin', 'altcoin', 'binance']
    macro_kws = ['fed', 'fomc', 'cpi', 'inflation', 'rate cut', 'rate hike', 'gdp',
                 'unemployment', 'jobs report', 'pce', 'recession', 'yield', 'gold']
    tech_kws = ['ai', 'gpt', 'openai', 'anthropic', 'claude', 'gemini', 'tesla',
                'apple', 'microsoft', 'nvidia', 'google', 'meta', 'spacex', 'starlink']

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
    print("[iter15] iter14 calibration 카테고리별 robust 검증")
    if not POLY_CSV.exists():
        print(f"  ❌ {POLY_CSV} 없음")
        return

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
    df['category'] = df['question'].apply(categorize)
    print(f"  종료 마켓: {len(df)}")

    target_tokens = set(df['yes_token'].tolist())
    token_meta = {r['yes_token']: {
        'end_ts': int(r['end_ts']),
        'actual_yes': int(r['actual_yes']),
        'category': str(r['category']),
        'volume': float(r['volume_num']),
        'year': int(r['endDate_dt'].year),
    } for _, r in df.iterrows()}

    # CSV 처리 (이전 iter14 처리 결과 재사용 — 빠른 path)
    print(f"\n  Reading CSV (chunked)...")
    market_trades = defaultdict(list)
    chunks_processed = 0

    csv_iter = pd.read_csv(POLY_CSV, chunksize=1_000_000, dtype='string',
                            encoding='latin-1', on_bad_lines='skip')
    while True:
        try:
            chunk = next(csv_iter)
        except StopIteration:
            break
        except Exception:
            continue
        chunks_processed += 1

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
            buy_v = buy[buy['takerAmountFilled'] > 0].copy()
            buy_v['price'] = buy_v['makerAmountFilled'] / buy_v['takerAmountFilled']
            buy_v = buy_v[(buy_v['price'] > 0) & (buy_v['price'] < 1)]
            for _, r in buy_v.iterrows():
                market_trades[r['takerAssetId']].append((int(r['timestamp']), float(r['price'])))

        sell_mask = ~m_zero & t_zero & chunk['makerAssetId'].isin(target_tokens)
        sell = chunk[sell_mask]
        if not sell.empty:
            sell_v = sell[sell['makerAmountFilled'] > 0].copy()
            sell_v['price'] = sell_v['takerAmountFilled'] / sell_v['makerAmountFilled']
            sell_v = sell_v[(sell_v['price'] > 0) & (sell_v['price'] < 1)]
            for _, r in sell_v.iterrows():
                market_trades[r['makerAssetId']].append((int(r['timestamp']), float(r['price'])))

        if chunks_processed % 10 == 0:
            print(f"    chunks={chunks_processed}, markets={len(market_trades)}")

    # closing belief 계산 + 메타 결합
    print(f"\n  closing belief 계산...")
    cb_records = []
    for tok, trades in market_trades.items():
        info = token_meta[tok]
        end_ts = info['end_ts']
        ws, we = end_ts - 7*86400, end_ts - 86400
        in_window = [p for ts, p in trades if ws <= ts <= we]
        if not in_window:
            continue
        cb_records.append({
            'token': tok,
            'closing_belief': sum(in_window) / len(in_window),
            'actual_yes': info['actual_yes'],
            'category': info['category'],
            'volume': info['volume'],
            'year': info['year'],
            'n_trades': len(in_window),
        })

    cb_df = pd.DataFrame(cb_records)
    print(f"  Closing belief: {len(cb_df)}")

    # ─── 카테고리별 calibration ────────────────────────
    print(f"\n=== 카테고리별 Calibration (0.30~0.40 bucket — iter14 핵심 발견) ===")
    cat_results = {}
    for cat in ['politics', 'sports', 'crypto', 'macro', 'tech', 'other']:
        sub = cb_df[cb_df['category'] == cat]
        if len(sub) < 30:
            print(f"  {cat:12s}: N={len(sub)} (너무 적음)")
            continue

        # 0.30~0.40 bucket
        mid = sub[(sub['closing_belief'] > 0.30) & (sub['closing_belief'] <= 0.40)]
        if len(mid) >= 5:
            avg_b = mid['closing_belief'].mean()
            actual = mid['actual_yes'].mean()
            edge = actual - avg_b
            color = "🚀" if edge < -0.05 else "✅" if edge < 0 else "⚠️"
            print(f"  {color} {cat:12s} | N={len(mid):>4} | belief={avg_b:.3f} | actual={actual:.3f} | edge={edge:+.3f}")
            cat_results[cat] = {"n_30_40": len(mid), "edge_30_40": float(edge)}

    print(f"\n=== 카테고리별 Calibration (0.60~0.70 bucket) ===")
    for cat in ['politics', 'sports', 'crypto', 'macro', 'tech', 'other']:
        sub = cb_df[cb_df['category'] == cat]
        if len(sub) < 30:
            continue
        mid = sub[(sub['closing_belief'] > 0.60) & (sub['closing_belief'] <= 0.70)]
        if len(mid) >= 5:
            avg_b = mid['closing_belief'].mean()
            actual = mid['actual_yes'].mean()
            edge = actual - avg_b
            color = "🚀" if edge > 0.05 else "✅" if edge > 0 else "⚠️"
            print(f"  {color} {cat:12s} | N={len(mid):>4} | belief={avg_b:.3f} | actual={actual:.3f} | edge={edge:+.3f}")
            if cat in cat_results:
                cat_results[cat]['n_60_70'] = len(mid)
                cat_results[cat]['edge_60_70'] = float(edge)

    # ─── 시간대별 drift ────────────────────────────────
    print(f"\n=== 시간대별 (year) calibration drift (0.30~0.40 bucket) ===")
    for year in sorted(cb_df['year'].unique()):
        sub = cb_df[cb_df['year'] == year]
        mid = sub[(sub['closing_belief'] > 0.30) & (sub['closing_belief'] <= 0.40)]
        if len(mid) >= 5:
            avg_b = mid['closing_belief'].mean()
            actual = mid['actual_yes'].mean()
            edge = actual - avg_b
            print(f"  {year}: N={len(mid):>4} | belief={avg_b:.3f} | actual={actual:.3f} | edge={edge:+.3f}")

    # ─── 거래량 분리 ───────────────────────────────────
    print(f"\n=== 거래량별 calibration (0.30~0.40 bucket) ===")
    high_vol = cb_df[cb_df['volume'] >= 1000000]
    mid_vol = cb_df[(cb_df['volume'] >= 100000) & (cb_df['volume'] < 1000000)]
    for label, sub in [('vol 100k-1M', mid_vol), ('vol 1M+', high_vol)]:
        bucket = sub[(sub['closing_belief'] > 0.30) & (sub['closing_belief'] <= 0.40)]
        if len(bucket) >= 5:
            avg_b = bucket['closing_belief'].mean()
            actual = bucket['actual_yes'].mean()
            edge = actual - avg_b
            print(f"  {label:15s} | N={len(bucket):>4} | belief={avg_b:.3f} | actual={actual:.3f} | edge={edge:+.3f}")

    # 종합
    print(f"\n=== iter15 종합 ===")
    consistent_neg = sum(1 for v in cat_results.values() if v.get('edge_30_40', 0) < -0.05)
    print(f"  카테고리 중 0.30~0.40 edge < -5% (No 베팅 strong): {consistent_neg}/{len(cat_results)}")
    if consistent_neg >= len(cat_results) * 0.5:
        print(f"  🏆 iter14 calibration robust 통과! 카테고리별 일관성 확인")
    else:
        print(f"  ⚠️ 카테고리별 차이 큼 — 일부 카테고리에서만 alpha")

    out = {
        "n_calibrated": len(cb_df),
        "category_results": cat_results,
    }
    out_path = RESULTS_DIR / "iter15_calibration_robust.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
