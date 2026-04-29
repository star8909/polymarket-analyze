"""Polymarket Gamma + CLOB API 데이터 로더.

Gamma API: 마켓 메타데이터, 종료 마켓, 활성 마켓
CLOB API: 가격 시계열, 오더북

캐시: parquet 형식, gitignored.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import (
    GAMMA_API, CLOB_API, CACHE_DIR,
    DEFAULT_TIMEOUT, GAMMA_RATE_LIMIT
)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
def _get(url: str, params: dict | None = None) -> dict:
    """공통 GET (재시도 + rate limit 처리)."""
    r = requests.get(url, params=params or {}, timeout=DEFAULT_TIMEOUT)
    if r.status_code == 429:
        time.sleep(5)
        raise Exception("Rate limited")
    r.raise_for_status()
    return r.json()


# ─── Gamma API: 마켓 데이터 ───────────────────────────────

def fetch_markets(closed: bool = True, limit: int = 500, offset: int = 0,
                  active: bool | None = None,
                  category: str | None = None) -> list[dict]:
    """마켓 리스트. closed=True면 종료된 마켓 (calibration용)."""
    params = {
        "limit": limit,
        "offset": offset,
    }
    if closed:
        params["closed"] = "true"
    else:
        params["closed"] = "false"
        if active is not None:
            params["active"] = "true" if active else "false"
    if category:
        params["tag_slug"] = category
    data = _get(f"{GAMMA_API}/markets", params)
    if isinstance(data, list):
        return data
    return data.get("data", []) or []


def fetch_all_closed_markets(category: str | None = None,
                              max_markets: int | None = None,
                              save_cache: bool = True) -> pd.DataFrame:
    """모든 종료 마켓 페이지네이션 수집."""
    cache_key = f"closed_{category or 'all'}.parquet"
    cache_path = CACHE_DIR / cache_key

    all_markets: list[dict] = []
    offset = 0
    limit = 500
    while True:
        time.sleep(1.0 / GAMMA_RATE_LIMIT)
        batch = fetch_markets(closed=True, limit=limit, offset=offset, category=category)
        if not batch:
            break
        all_markets.extend(batch)
        print(f"  [fetch_closed] offset={offset}, +{len(batch)}, total={len(all_markets)}")
        offset += limit
        if len(batch) < limit:
            break
        if max_markets and len(all_markets) >= max_markets:
            break

    df = pd.DataFrame(all_markets)
    if save_cache and not df.empty:
        # JSON 컬럼은 string으로 변환 (parquet 저장용)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )
        df.to_parquet(cache_path, index=False)
        print(f"  [cache] saved → {cache_path} ({len(df)} rows)")
    return df


def fetch_active_markets(category: str | None = None,
                         min_volume: float = 1000.0,
                         max_pages: int = 20) -> pd.DataFrame:
    """활성 마켓 (sum-to-1 스캐너용). max_pages로 제한해 timeout 방지."""
    all_m: list[dict] = []
    offset = 0
    limit = 500
    page = 0
    while page < max_pages:
        time.sleep(1.0 / GAMMA_RATE_LIMIT)
        try:
            batch = fetch_markets(closed=False, active=True, limit=limit,
                                  offset=offset, category=category)
        except Exception as e:
            print(f"  [fetch_active] page {page} error: {e}, stopping")
            break
        if not batch:
            break
        all_m.extend(batch)
        print(f"  [fetch_active] page {page}, +{len(batch)}, total={len(all_m)}")
        offset += limit
        page += 1
        if len(batch) < limit:
            break

    df = pd.DataFrame(all_m)
    if df.empty:
        return df
    # 거래량 필터
    if 'volume' in df.columns:
        df['volume_num'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        df = df[df['volume_num'] >= min_volume]
    return df


# ─── CLOB API: 가격 시계열 ────────────────────────────────

def fetch_price_history(token_id: str, interval: str = "1h",
                         fidelity: int = 60) -> pd.DataFrame:
    """단일 토큰의 가격 히스토리. interval = '1m'/'1h'/'1d'.

    반환: DataFrame index=timestamp, columns=['price']
    """
    cache_path = CACHE_DIR / f"prices_{token_id}_{interval}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    params = {
        "market": token_id,
        "interval": interval,
        "fidelity": fidelity,
    }
    try:
        data = _get(f"{CLOB_API}/prices-history", params)
    except Exception as e:
        print(f"  [price_history] error for {token_id}: {e}")
        return pd.DataFrame()

    history = data.get("history", []) if isinstance(data, dict) else []
    if not history:
        return pd.DataFrame()

    df = pd.DataFrame(history)
    if 't' in df.columns:
        df['timestamp'] = pd.to_datetime(df['t'], unit='s')
        df = df.rename(columns={'p': 'price'})
        df = df[['timestamp', 'price']].set_index('timestamp')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.to_parquet(cache_path)
    return df


# ─── 헬퍼: outcomes/prices 파싱 ───────────────────────────

def parse_market_outcome(market_row: dict | pd.Series) -> dict:
    """Gamma API 마켓 row에서 outcome/price 추출.

    반환: {'outcomes': ['Yes','No'], 'prices': [0.4, 0.6], 'token_ids': [...]}
    """
    out = {}
    for key in ('outcomes', 'outcomePrices', 'clobTokenIds'):
        v = market_row.get(key) if isinstance(market_row, dict) else market_row.get(key, None)
        if isinstance(v, str):
            try:
                out[key] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                out[key] = []
        elif isinstance(v, (list, tuple)):
            out[key] = list(v)
        else:
            out[key] = []
    return {
        'outcomes': out.get('outcomes', []),
        'prices': [float(p) for p in out.get('outcomePrices', []) if p],
        'token_ids': out.get('clobTokenIds', []),
    }
