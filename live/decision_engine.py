"""Calibration 기반 베팅 의사결정.

iter14_real_calibration.json 의 bucket 표 사용:
- 가격 → bucket lookup → historical actual_yes
- edge = actual_yes - market_belief
- |edge| >= threshold 면 베팅, side=NO if edge<0 else YES
- size = fractional Kelly × max_bet_usd 캡
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


CALIBRATION_PATH = (
    Path(__file__).resolve().parent.parent / "results" / "iter14_real_calibration.json"
)


@dataclass
class Decision:
    bet: bool
    side: str = ""        # "YES" or "NO"
    size_usd: float = 0.0
    reason: str = ""
    bucket: str = ""
    edge_pp: float = 0.0
    market_price: float = 0.0
    actual_rate: float = 0.0


def load_calibration() -> list[dict]:
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"calibration 결과 없음: {CALIBRATION_PATH}")
    data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    return data["calibration_buckets"]


def lookup_bucket(price: float, buckets: list[dict]) -> dict | None:
    """price가 들어가는 bucket 반환. 없으면 None."""
    for b in buckets:
        # bin: "(0.1, 0.2]" 형태
        s = b["bin"].strip()
        # parse 양 끝 숫자
        s_clean = s.replace("(", "").replace("[", "").replace("]", "").replace(")", "")
        lo_str, hi_str = s_clean.split(",")
        lo, hi = float(lo_str.strip()), float(hi_str.strip())
        if lo < price <= hi:
            return b
    return None


def kelly_fraction(my_p: float, market_p: float) -> float:
    """Full Kelly fraction. clip [0, 1]."""
    if market_p <= 0 or market_p >= 1:
        return 0.0
    odds_b = (1 - market_p) / market_p
    f = (my_p * odds_b - (1 - my_p)) / odds_b
    return max(0.0, min(1.0, f))


def decide(
    market_price: float,
    buckets: list[dict],
    capital_usd: float,
    edge_threshold_pp: float = 5.0,
    kelly_fraction_arg: float = 0.25,
    max_bet_usd: float = 20.0,
) -> Decision:
    """단일 마켓 베팅 의사결정.

    Args:
        market_price: 현재 Yes 토큰 mid price (0~1)
        buckets: calibration 결과
        capital_usd: 전체 가용 자본
        edge_threshold_pp: edge 기준 (5pp 미만은 skip)
        kelly_fraction_arg: fractional Kelly 배수 (0.25 권장)
        max_bet_usd: 마켓당 최대 베팅
    """
    bucket = lookup_bucket(market_price, buckets)
    if not bucket:
        return Decision(bet=False, reason="bucket 매칭 실패", market_price=market_price)

    actual_yes = bucket["actual_yes"]
    edge_pp = (actual_yes - market_price) * 100  # +면 YES, -면 NO

    if abs(edge_pp) < edge_threshold_pp:
        return Decision(
            bet=False,
            reason=f"edge {edge_pp:+.1f}pp < {edge_threshold_pp}pp",
            bucket=bucket["bin"],
            edge_pp=edge_pp,
            market_price=market_price,
            actual_rate=actual_yes,
        )

    if edge_pp < 0:
        side = "NO"
        my_p = 1 - actual_yes
        market_p = 1 - market_price
    else:
        side = "YES"
        my_p = actual_yes
        market_p = market_price

    f_full = kelly_fraction(my_p, market_p)
    f_kelly = f_full * kelly_fraction_arg
    notional = capital_usd * f_kelly
    notional = min(notional, max_bet_usd)

    if notional < 1.0:  # 최소 베팅 1달러
        return Decision(
            bet=False,
            reason=f"Kelly 결과 너무 작음 (${notional:.2f})",
            side=side,
            bucket=bucket["bin"],
            edge_pp=edge_pp,
            market_price=market_price,
            actual_rate=actual_yes,
        )

    return Decision(
        bet=True,
        side=side,
        size_usd=notional,
        reason=f"edge {edge_pp:+.1f}pp, Kelly {f_full*100:.1f}% × {kelly_fraction_arg}x = ${notional:.2f}",
        bucket=bucket["bin"],
        edge_pp=edge_pp,
        market_price=market_price,
        actual_rate=actual_yes,
    )
