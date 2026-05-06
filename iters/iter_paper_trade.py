"""iter_paper_trade.py — 시간순 walk-forward paper trading sim.

벤치마크 monte carlo (iter_verified_sweep) 와의 차이:
- 가짜 random permutation X → end_ts 시간순 진행
- 캘리브레이션 walk-forward (과거 데이터만으로 학습) → in-sample bias 제거
- bankroll 추적, 마켓당 max $100 cap, 카테고리 추적
- 비용 2% 차감 (스프레드+슬리피지)
- equity curve 월간 resample → realistic Sharpe

usage:
  PYTHONIOENCODING=utf-8 python iters/iter_paper_trade.py
  PYTHONIOENCODING=utf-8 python iters/iter_paper_trade.py --belief-max 0.5 --kelly 0.05
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, RESULTS_DIR

CB_PATH = CACHE_DIR / "closing_beliefs.parquet"


def calibrate(df_train: pd.DataFrame, n_min: int = 50) -> dict:
    """0.1-width bucket calibration from training set."""
    bins = np.arange(0.0, 1.05, 0.10)
    df = df_train[(df_train['closing_belief'] >= 0) & (df_train['closing_belief'] <= 1)].copy()
    df['bin'] = pd.cut(df['closing_belief'], bins=bins, include_lowest=True)
    grouped = df.groupby('bin', observed=True).agg(
        n=('closing_belief', 'size'),
        avg_belief=('closing_belief', 'mean'),
        actual_yes=('actual_yes', 'mean'),
    ).reset_index()
    out = {}
    for _, row in grouped.iterrows():
        if row['n'] < n_min:
            continue
        idx = int(row['avg_belief'] * 10)
        if idx >= 10:
            idx = 9
        out[idx] = {
            'n': int(row['n']),
            'avg_belief': float(row['avg_belief']),
            'actual_yes': float(row['actual_yes']),
        }
    return out


def paper_trade(df: pd.DataFrame, **params):
    initial_capital = params.get('initial_capital', 10000.0)
    max_per_market = params.get('max_per_market', 100.0)
    kelly_frac = params.get('kelly_frac', 0.05)
    cost = params.get('cost', 0.02)
    edge_min = params.get('edge_min', 0.05)
    n_min_bucket = params.get('n_min_bucket', 50)
    train_min_markets = params.get('train_min_markets', 300)
    recalibrate_every_days = params.get('recalibrate_every_days', 30)
    belief_min = params.get('belief_min', 0.0)
    belief_max = params.get('belief_max', 0.5)
    pos_cap_pct = params.get('pos_cap_pct', 0.05)

    df = df.sort_values('end_ts').reset_index(drop=True).copy()
    bankroll = initial_capital
    equity_curve = []
    bets_log = []
    last_recal_ts = 0
    cal = None

    for i, row in df.iterrows():
        ts = int(row['end_ts'])
        if i >= train_min_markets and (cal is None or ts - last_recal_ts > recalibrate_every_days * 86400):
            cal = calibrate(df.iloc[:i], n_min=n_min_bucket)
            last_recal_ts = ts

        equity_curve.append((ts, bankroll))

        if cal is None:
            continue

        belief = float(row['closing_belief'])
        if not (belief_min <= belief <= belief_max):
            continue

        bin_idx = min(int(belief * 10), 9)
        bucket = cal.get(bin_idx)
        if bucket is None:
            continue

        edge = bucket['avg_belief'] - bucket['actual_yes']
        if abs(edge) < edge_min:
            continue

        if edge > 0:
            side = 'NO'
            entry = (1 - belief) + cost
            win = (int(row['actual_yes']) == 0)
        else:
            side = 'YES'
            entry = belief + cost
            win = (int(row['actual_yes']) == 1)

        if entry <= 0 or entry >= 1:
            continue
        payoff = 1.0 / entry - 1

        bet_size = min(max_per_market, kelly_frac * bankroll, pos_cap_pct * bankroll)
        if bet_size < 1.0:
            continue

        pnl = bet_size * payoff if win else -bet_size
        bankroll += pnl
        bets_log.append({
            'end_ts': ts,
            'belief': belief,
            'side': side,
            'size': bet_size,
            'pnl': pnl,
            'win': bool(win),
            'category': str(row.get('category', '')),
        })
        equity_curve[-1] = (ts, bankroll)

        if bankroll <= 0:
            print(f"  RUIN @ i={i} ts={ts}")
            break

    return equity_curve, bets_log, bankroll


def metrics_from_curve(equity_curve, initial_capital):
    if len(equity_curve) < 2:
        return None
    arr = pd.DataFrame(equity_curve, columns=['ts', 'eq']).drop_duplicates('ts', keep='last')
    arr['date'] = pd.to_datetime(arr['ts'], unit='s')
    arr = arr.set_index('date').sort_index()

    daily = arr['eq'].resample('D').last().ffill()
    monthly = arr['eq'].resample('M').last().ffill()
    if len(monthly) < 6:
        return None

    monthly_returns = monthly.pct_change().dropna()
    sharpe = (monthly_returns.mean() / monthly_returns.std() * np.sqrt(12)) if monthly_returns.std() > 0 else 0.0

    peak = daily.cummax()
    dd = daily / peak - 1
    mdd = dd.min()

    final = daily.iloc[-1]
    days = (daily.index[-1] - daily.index[0]).days
    years = max(days / 365.25, 1e-6)
    cagr = (final / initial_capital) ** (1 / years) - 1

    return {
        'final': float(final),
        'total_return_pct': float((final / initial_capital - 1) * 100),
        'cagr_pct': float(cagr * 100),
        'mdd_pct': float(mdd * 100),
        'monthly_sharpe': float(sharpe),
        'n_months': int(len(monthly)),
        'years': float(years),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--belief-max', type=float, default=0.5)
    ap.add_argument('--belief-min', type=float, default=0.0)
    ap.add_argument('--edge-min', type=float, default=0.05)
    ap.add_argument('--kelly', type=float, default=0.05)
    ap.add_argument('--max-per-market', type=float, default=100.0)
    ap.add_argument('--pos-cap-pct', type=float, default=0.05)
    ap.add_argument('--initial-capital', type=float, default=10000.0)
    ap.add_argument('--cost', type=float, default=0.02)
    ap.add_argument('--n-min-bucket', type=int, default=50)
    ap.add_argument('--train-min-markets', type=int, default=300)
    args = ap.parse_args()

    if not CB_PATH.exists():
        print(f"❌ {CB_PATH} 없음. iter14_real_calibration.py 먼저 실행")
        return

    df = pd.read_parquet(CB_PATH)
    print(f"Markets: {len(df)}")
    print(f"Date: {pd.to_datetime(df['end_ts'].min(), unit='s'):%Y-%m-%d} → "
          f"{pd.to_datetime(df['end_ts'].max(), unit='s'):%Y-%m-%d}")
    if 'category' in df.columns:
        cats = df['category'].value_counts().head(8)
        print(f"Top cats: {dict(cats)}")

    eq, bets, final = paper_trade(
        df,
        belief_min=args.belief_min, belief_max=args.belief_max,
        edge_min=args.edge_min, kelly_frac=args.kelly,
        max_per_market=args.max_per_market, pos_cap_pct=args.pos_cap_pct,
        initial_capital=args.initial_capital, cost=args.cost,
        n_min_bucket=args.n_min_bucket, train_min_markets=args.train_min_markets,
    )

    print(f"\nBets: {len(bets)}")
    print(f"Final: ${final:.0f}")

    m = metrics_from_curve(eq, args.initial_capital)
    if m:
        print(f"\n=== Realistic Metrics ===")
        print(f"  Final:        ${m['final']:.0f}")
        print(f"  Total return: {m['total_return_pct']:+.1f}%")
        print(f"  CAGR:         {m['cagr_pct']:+.1f}%")
        print(f"  MDD:          {m['mdd_pct']:.1f}%")
        print(f"  Sharpe (M):   {m['monthly_sharpe']:.2f}")
        print(f"  Months:       {m['n_months']}  Years: {m['years']:.1f}")

    if bets:
        wins = sum(1 for b in bets if b['win'])
        print(f"  Win rate:     {wins/len(bets)*100:.1f}% ({wins}/{len(bets)})")
        no_n = sum(1 for b in bets if b['side'] == 'NO')
        no_pnl = sum(b['pnl'] for b in bets if b['side'] == 'NO')
        yes_n = sum(1 for b in bets if b['side'] == 'YES')
        yes_pnl = sum(b['pnl'] for b in bets if b['side'] == 'YES')
        print(f"  NO:           ${no_pnl:+.0f}  ({no_n} bets)")
        print(f"  YES:          ${yes_pnl:+.0f}  ({yes_n} bets)")

        cat_pnl = {}
        for b in bets:
            cat_pnl.setdefault(b['category'] or '∅', [0, 0])
            cat_pnl[b['category'] or '∅'][0] += b['pnl']
            cat_pnl[b['category'] or '∅'][1] += 1
        print(f"  Top categories by PnL:")
        for c, (p, n) in sorted(cat_pnl.items(), key=lambda x: -x[1][0])[:6]:
            print(f"    {c[:30]:30s} ${p:+.0f}  ({n} bets)")

    out = {
        'params': vars(args),
        'n_markets_universe': int(len(df)),
        'n_bets': int(len(bets)),
        'final_bankroll': float(final),
        'metrics': m,
    }
    out_path = RESULTS_DIR / 'iter_paper_trade.json'
    out_path.write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding='utf-8')
    print(f"\n→ {out_path}")


if __name__ == '__main__':
    main()
