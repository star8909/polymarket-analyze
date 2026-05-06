"""실거래 봇 메인 루프.

페이퍼 4주 통과 후만 사용. drawdown -10% 자동 정지.

사용:
    python live/bot.py --capital 100
"""
from __future__ import annotations

import sys
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from polymarket_client import (
    load_env, make_client, get_balance, get_midpoint, get_spread,
    place_market, get_open_orders, cancel_all,
)
from decision_engine import load_calibration, decide
from kill_switch import is_kill_active, emergency_shutdown, trigger_kill
from paper_trade import fetch_active_markets


def get_usdc_balance(client) -> float:
    """USDC.e 잔고를 USD로 반환. 실제 응답 형식에 따라 추출."""
    try:
        bal = get_balance(client)
        # py-clob-client 반환 형식 다양 — 실제 보고 패치
        if isinstance(bal, dict):
            # 예: {"balance": "100.5", "allowance": "..."}
            b = bal.get("balance") or bal.get("usdcBalance") or "0"
            return float(b)
        return 0.0
    except Exception as e:
        print(f"  잔고 조회 실패: {e}")
        return 0.0


def run_bot(capital: float, env: dict):
    """메인 루프 — 매 시간 의사결정 + 주문."""
    buckets = load_calibration()
    print(f"[bot] 시작 자본 ${capital:.2f}, edge≥{env['edge_threshold_pp']}pp")

    if is_kill_active(env["kill_switch_file"]):
        print("  🛑 시작 시 KILL 활성 — 종료. 수동 삭제 후 재시작.")
        return

    client = make_client(read_only=False)
    initial_balance = get_usdc_balance(client)
    print(f"  실 USDC 잔고: ${initial_balance:.2f}")
    if initial_balance < capital * 0.95:
        print(f"  ⚠️ 잔고 부족 (목표 ${capital}, 실제 ${initial_balance:.2f})")
        return

    peak_equity = initial_balance
    daily_loss = 0.0
    last_day = datetime.now().date()
    log_path = Path(__file__).parent / f"live_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    log = open(log_path, "w", encoding="utf-8")
    log.write("ts,event,market_id,token,side,size,price,response\n")

    iteration = 0
    while True:
        iteration += 1
        now = datetime.now()

        # ─── KILL 체크 ───
        if is_kill_active(env["kill_switch_file"]):
            emergency_shutdown(client, env["kill_switch_file"], "KILL detected")
            break

        # ─── 일일 리셋 ───
        if now.date() != last_day:
            print(f"  [reset] 일일 손실 카운터 리셋")
            daily_loss = 0.0
            last_day = now.date()

        # ─── DD/일일 손실 체크 ───
        cur_balance = get_usdc_balance(client)
        peak_equity = max(peak_equity, cur_balance)
        dd = (cur_balance - peak_equity) / peak_equity if peak_equity > 0 else 0
        if dd < -0.10:
            emergency_shutdown(client, env["kill_switch_file"],
                               f"Drawdown {dd*100:.1f}% 초과 -10%")
            break
        if daily_loss > env["daily_loss_limit"]:
            emergency_shutdown(client, env["kill_switch_file"],
                               f"일일 손실 ${daily_loss:.2f} > ${env['daily_loss_limit']}")
            break

        print(f"\n[{now.strftime('%H:%M')}] balance=${cur_balance:.2f}, peak=${peak_equity:.2f}, dd={dd*100:+.1f}%")

        # ─── 활성 마켓 스캔 ───
        try:
            markets = fetch_active_markets(env["min_volume_24h"], env["min_days_to_settle"])
            print(f"  활성 마켓 후보: {len(markets)}")
        except Exception as e:
            print(f"  fetch error: {e}")
            time.sleep(300)
            continue

        # ─── 의사결정 + 주문 ───
        ro = make_client(read_only=True)
        new_orders = 0
        for m in markets[:100]:
            try:
                spread = get_spread(ro, m["yes_token"])
                if spread > 0.05:
                    continue
                mid = get_midpoint(ro, m["yes_token"])
                d = decide(
                    market_price=mid,
                    buckets=buckets,
                    capital_usd=cur_balance,
                    edge_threshold_pp=env["edge_threshold_pp"],
                    kelly_fraction_arg=env["kelly_fraction"],
                    max_bet_usd=env["max_bet_usd"],
                )
                if not d.bet:
                    continue

                # NO 베팅이면 No 토큰 가격으로 매수 = (1-mid)
                # token이 yes이므로 no를 사려면 별도 token (보통 tokens[1])
                # 단순화: yes_token만 사용. side="BUY"는 yes 매수, "SELL"은 yes 매도(=no 매수)
                if d.side == "YES":
                    side_arg = "BUY"
                    target_price = mid
                else:
                    side_arg = "SELL"  # yes 매도 = no long 효과
                    target_price = 1 - mid

                # 안전장치: 한 번에 한 마켓만 새로 진입
                resp = place_market(client, m["yes_token"], d.size_usd, side=side_arg)
                line = f"{now.isoformat()},BET,{m['id']},{m['yes_token'][:16]}...,{d.side},{d.size_usd:.2f},{target_price:.4f},{json.dumps(resp)[:100]}\n"
                log.write(line)
                log.flush()
                print(f"  🎯 {d.side} ${d.size_usd:.2f} @ {target_price:.3f} | {m['question'][:50]}")
                new_orders += 1
                if new_orders >= 5:  # 시간당 max 5 새 주문
                    break
                time.sleep(2)  # 주문 간 간격
            except Exception as e:
                print(f"  주문 에러: {e}")
                continue

        print(f"  새 주문: {new_orders}")
        time.sleep(3600)  # 1시간 대기

    log.close()
    print(f"  로그: {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=100)
    args = ap.parse_args()

    env = load_env()
    print("⚠️  실거래 봇 — paper 4주 통과한 경우에만 사용")
    print(f"   자본: ${args.capital}, MAX 베팅/마켓: ${env['max_bet_usd']}, 일 손실 한도: ${env['daily_loss_limit']}")
    print()

    confirm = input("계속하려면 'YES' 입력: ")
    if confirm != "YES":
        print("취소")
        return

    run_bot(args.capital, env)


if __name__ == "__main__":
    main()
