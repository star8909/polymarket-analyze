"""Paper trading 시뮬레이터 — 실제 주문 X, 결정만 기록.

매시간 활성 마켓 스캔 → calibration 의사결정 → 가상 포지션 추가.
종료 시 settle 결과로 PnL 계산.

사용:
    python live/paper_trade.py --capital 1000 --duration-days 28
"""
from __future__ import annotations

import sys
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polymarket_client import (
    load_env, make_client, get_midpoint, get_spread, get_order_book,
)
from decision_engine import load_calibration, decide
from kill_switch import is_kill_active

import requests


GAMMA_API = "https://gamma-api.polymarket.com"


def fetch_active_markets(min_volume: float, min_days_to_settle: int) -> list[dict]:
    """활성 마켓 중 필터 통과한 것만."""
    now = datetime.utcnow()
    cutoff = now + timedelta(days=min_days_to_settle)
    markets = []
    offset = 0
    while True:
        r = requests.get(
            f"{GAMMA_API}/markets",
            params={"closed": "false", "active": "true",
                    "limit": 500, "offset": offset, "order": "volume", "ascending": "false"},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json() if isinstance(r.json(), list) else r.json().get("data", [])
        if not batch:
            break
        for m in batch:
            try:
                vol = float(m.get("volume", 0) or 0)
                if vol < min_volume:
                    continue
                end_str = m.get("endDate")
                if not end_str:
                    continue
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                if end_dt.replace(tzinfo=None) < cutoff:
                    continue
                tokens = m.get("clobTokenIds")
                if not tokens:
                    continue
                if isinstance(tokens, str):
                    tokens = json.loads(tokens)
                if not isinstance(tokens, list) or len(tokens) < 2:
                    continue
                markets.append({
                    "id": m.get("id"),
                    "question": (m.get("question") or "")[:80],
                    "yes_token": str(tokens[0]),
                    "volume": vol,
                    "end_date": end_str,
                })
            except Exception:
                continue
        if len(batch) < 500:
            break
        offset += 500
        if offset > 5000:  # 상위 5000개만
            break
    return markets


def simulate_run(capital: float, duration_days: int, env: dict, log_path: Path):
    buckets = load_calibration()
    print(f"[paper] 자본 ${capital:.0f}, 기간 {duration_days}일, edge≥{env['edge_threshold_pp']}pp")
    print(f"  calibration buckets: {len(buckets)}")

    ro = make_client(read_only=True)
    positions = []  # {market_id, token, side, size_usd, entry_price, end_date}
    realized_pnl = 0.0
    start = datetime.now()
    end_time = start + timedelta(days=duration_days)

    log = open(log_path, "w", encoding="utf-8")
    log.write("ts,event,market_id,question,token,side,size,price,actual_yes,edge_pp,reason\n")

    iteration = 0
    while datetime.now() < end_time:
        iteration += 1
        if is_kill_active(env["kill_switch_file"]):
            print(f"\n  🛑 KILL 감지 — 정지")
            break

        print(f"\n[iter {iteration}] {datetime.now().isoformat()}")

        # 1. 활성 마켓 스캔
        try:
            markets = fetch_active_markets(env["min_volume_24h"], env["min_days_to_settle"])
            print(f"  활성 마켓: {len(markets)}")
        except Exception as e:
            print(f"  ❌ fetch error: {e}")
            time.sleep(60)
            continue

        # 2. 각 마켓 의사결정
        new_bets = 0
        for m in markets[:200]:  # 상위 200개만 (API rate 보호)
            try:
                spread = get_spread(ro, m["yes_token"])
                if spread > 0.05:  # spread 5pp 이상은 비싸서 skip
                    continue
                mid = get_midpoint(ro, m["yes_token"])
                d = decide(
                    market_price=mid,
                    buckets=buckets,
                    capital_usd=capital - sum(p["size_usd"] for p in positions) - 0,
                    edge_threshold_pp=env["edge_threshold_pp"],
                    kelly_fraction_arg=env["kelly_fraction"],
                    max_bet_usd=env["max_bet_usd"],
                )
                if not d.bet:
                    continue
                # 중복 베팅 방지 (같은 마켓 이미 보유)
                if any(p["market_id"] == m["id"] for p in positions):
                    continue
                pos = {
                    "market_id": m["id"],
                    "question": m["question"],
                    "token": m["yes_token"],
                    "side": d.side,
                    "size_usd": d.size_usd,
                    "entry_price": mid if d.side == "YES" else 1 - mid,
                    "end_date": m["end_date"],
                    "edge_pp": d.edge_pp,
                }
                positions.append(pos)
                new_bets += 1
                line = f"{datetime.now().isoformat()},BET,{m['id']},{m['question'][:40]},{m['yes_token'][:16]}...,{d.side},{d.size_usd:.2f},{pos['entry_price']:.4f},{d.actual_rate:.4f},{d.edge_pp:.2f},{d.reason}\n"
                log.write(line)
                log.flush()
                print(f"  🎯 BET {d.side} ${d.size_usd:.2f} @ {pos['entry_price']:.3f} | {m['question'][:50]}")
                if new_bets >= 10:  # 시간당 최대 10 새 베팅
                    break
            except Exception as e:
                continue

        print(f"  새 베팅: {new_bets}, 총 포지션: {len(positions)}, 사용 자본: ${sum(p['size_usd'] for p in positions):.2f}")

        # 3. 시간 대기 (1시간 간격)
        time.sleep(3600)

    log.close()
    print(f"\n✅ 시뮬 종료. 포지션 {len(positions)}, 로그 → {log_path}")
    print(f"  주의: 실제 settle은 마켓 endDate 도달 시점이라 별도 settle 처리 필요.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=1000)
    ap.add_argument("--duration-days", type=int, default=28)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    env = load_env()
    log_path = Path(args.log) if args.log else Path(__file__).parent / f"paper_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

    simulate_run(args.capital, args.duration_days, env, log_path)


if __name__ == "__main__":
    main()
