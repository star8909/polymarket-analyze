"""Polymarket CLOB API 래퍼.

read-only 함수 + 인증 함수 분리. 봇은 둘 다 씀.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    OrderArgs, MarketOrderArgs, OrderType, OpenOrderParams,
)
from py_clob_client.order_builder.constants import BUY, SELL


def load_env() -> dict[str, Any]:
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    return {
        "private_key": os.getenv("PRIVATE_KEY"),
        "funder": os.getenv("FUNDER_ADDRESS"),
        "signature_type": int(os.getenv("SIGNATURE_TYPE", "1")),
        "host": os.getenv("CLOB_HOST", "https://clob.polymarket.com"),
        "chain_id": int(os.getenv("CHAIN_ID", "137")),
        "max_bet_usd": float(os.getenv("MAX_BET_USD", "20")),
        "daily_loss_limit": float(os.getenv("DAILY_LOSS_LIMIT_USD", "100")),
        "edge_threshold_pp": float(os.getenv("EDGE_THRESHOLD_PP", "5")),
        "kelly_fraction": float(os.getenv("KELLY_FRACTION", "0.25")),
        "kill_switch_file": os.getenv("KILL_SWITCH_FILE", "KILL"),
        "min_volume_24h": float(os.getenv("MIN_VOLUME_24H_USD", "5000")),
        "min_days_to_settle": int(os.getenv("MIN_DAYS_TO_SETTLE", "7")),
    }


def make_client(read_only: bool = False) -> ClobClient:
    """ClobClient 생성. read_only=True면 인증 스킵 (가격 조회만)."""
    env = load_env()
    if read_only:
        return ClobClient(env["host"])
    if not env["private_key"] or not env["funder"]:
        raise ValueError("PRIVATE_KEY와 FUNDER_ADDRESS 필수 (.env 작성 필요)")
    client = ClobClient(
        env["host"],
        key=env["private_key"],
        chain_id=env["chain_id"],
        signature_type=env["signature_type"],
        funder=env["funder"],
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    return client


# ─── Read-only 헬퍼 ───────────────────────────────────────

def get_midpoint(client: ClobClient, token_id: str) -> float:
    return float(client.get_midpoint(token_id))


def get_order_book(client: ClobClient, token_id: str) -> dict:
    return client.get_order_book(token_id)


def get_spread(client: ClobClient, token_id: str) -> float:
    """Bid-ask spread in pp."""
    book = get_order_book(client, token_id)
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids or not asks:
        return 1.0  # 호가 없으면 큰 spread
    best_bid = float(bids[0]["price"]) if bids else 0
    best_ask = float(asks[0]["price"]) if asks else 1
    return best_ask - best_bid


# ─── Authenticated 헬퍼 ───────────────────────────────────

def get_balance(client: ClobClient) -> dict:
    """USDC + 토큰 잔고."""
    return client.get_balance_allowance(params={})


def place_limit(client: ClobClient, token_id: str, price: float,
                size: float, side: str = BUY) -> dict:
    """GTC 지정가 주문."""
    args = OrderArgs(token_id=token_id, price=price, size=size, side=side)
    signed = client.create_order(args)
    return client.post_order(signed, OrderType.GTC)


def place_market(client: ClobClient, token_id: str, amount_usd: float,
                  side: str = BUY) -> dict:
    """FAK 시장가 (즉시 체결, 부분 체결 OK)."""
    args = MarketOrderArgs(
        token_id=token_id, amount=amount_usd, side=side, order_type=OrderType.FAK,
    )
    signed = client.create_market_order(args)
    return client.post_order(signed, OrderType.FAK)


def cancel_one(client: ClobClient, order_id: str) -> dict:
    return client.cancel(order_id)


def cancel_all(client: ClobClient) -> dict:
    """비상 정지: 모든 주문 취소."""
    return client.cancel_all()


def get_open_orders(client: ClobClient) -> list:
    return client.get_orders(OpenOrderParams())


def get_my_trades(client: ClobClient) -> list:
    return client.get_trades()
