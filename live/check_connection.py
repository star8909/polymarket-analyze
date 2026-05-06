"""연결 + 인증 + 잔고 + 가격 1점 조회 sanity check.

처음 .env 작성 후 이거 먼저 돌려서 OK 확인.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from polymarket_client import (
    load_env, make_client, get_balance, get_midpoint, get_open_orders,
)


def main():
    print("=" * 60)
    print(" Polymarket 연결 sanity check")
    print("=" * 60)

    env = load_env()
    print(f"\n[env]")
    print(f"  host: {env['host']}")
    print(f"  chain_id: {env['chain_id']}")
    print(f"  signature_type: {env['signature_type']}")
    print(f"  funder: {env['funder']}")

    # ─── 1. Read-only 연결 ───
    print(f"\n[read-only]")
    ro = make_client(read_only=True)
    print(f"  ✅ 연결 OK")

    # ─── 2. 가격 조회 (예시 토큰) ───
    # 임의 활성 마켓의 token_id (예시 — 실제 봇은 활성 마켓 fetch해서 token 결정)
    sample_token = "21742633143463906290569050155826241533067272736897614950488156847949938836455"
    try:
        mid = get_midpoint(ro, sample_token)
        print(f"  sample token midpoint: {mid:.4f}")
    except Exception as e:
        print(f"  ⚠️ midpoint 에러 (토큰 만료 가능): {e}")

    # ─── 3. 인증 연결 ───
    if not env["private_key"] or env["private_key"].startswith("0x_여기"):
        print(f"\n[auth] PRIVATE_KEY 미설정 → 인증 테스트 skip")
        return

    print(f"\n[authenticated]")
    try:
        c = make_client(read_only=False)
        print(f"  ✅ 인증 OK")
    except Exception as e:
        print(f"  ❌ 인증 실패: {e}")
        return

    # ─── 4. 잔고 ───
    try:
        bal = get_balance(c)
        print(f"  잔고: {bal}")
    except Exception as e:
        print(f"  ⚠️ 잔고 조회 에러: {e}")

    # ─── 5. 미체결 주문 ───
    try:
        orders = get_open_orders(c)
        print(f"  미체결 주문: {len(orders)}개")
    except Exception as e:
        print(f"  ⚠️ 주문 조회 에러: {e}")

    print("\n  → 모두 OK 면 paper_trade.py 진행 가능")


if __name__ == "__main__":
    main()
