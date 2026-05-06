"""iter01b: 2025-06-23 이후 누락된 종료 마켓 fetch (월별 윈도우).

Gamma offset > 100k에서 422 에러 → 월별로 쪼개서 안전하게 페이지네이션.
"""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import CACHE_DIR, GAMMA_API, DEFAULT_TIMEOUT, GAMMA_RATE_LIMIT


def _get(url, params, max_retries=3):
    """GET with simple retry (no retry on 400/422)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if r.status_code == 429:
                time.sleep(5)
                last_err = Exception("Rate limited")
                continue
            if r.status_code in (400, 422):
                # 클라이언트 에러 — 재시도 의미 없음
                r.raise_for_status()
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (400, 422):
                raise
            last_err = e
            time.sleep(2 ** attempt)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err


def fetch_window(date_min: str, date_max: str) -> list[dict]:
    """Fetch closed markets in [date_min, date_max). 422 hit → return what we have."""
    all_m = []
    offset = 0
    limit = 500
    while True:
        time.sleep(1.0 / GAMMA_RATE_LIMIT)
        params = {
            "closed": "true", "limit": limit, "offset": offset,
            "end_date_min": date_min, "end_date_max": date_max,
            "order": "endDate", "ascending": "true",
        }
        try:
            data = _get(f"{GAMMA_API}/markets", params)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (400, 422):
                print(f"    {date_min}~{date_max}: 422 at offset={offset}, stopping window")
                break
            raise
        except Exception as e:
            print(f"    error: {e}, stopping window")
            break
        batch = data if isinstance(data, list) else data.get("data", []) or []
        if not batch:
            break
        all_m.extend(batch)
        offset += limit
        if len(batch) < limit:
            break
    print(f"    {date_min}~{date_max}: {len(all_m)} markets")
    return all_m


def main():
    print("[iter01b] 누락 마켓 fetch (월별 windowing)")
    # 시작: 2025-06-23 (기존 캐시 끝)
    # 종료: 2026-05-04 (오늘)
    start = datetime(2025, 6, 23)
    end = datetime(2026, 5, 5)

    all_new = []
    cur = start
    while cur < end:
        nxt = cur + timedelta(days=30)
        if nxt > end:
            nxt = end
        markets = fetch_window(cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d"))
        all_new.extend(markets)
        cur = nxt

    print(f"\n  신규 fetch: {len(all_new)}")
    if not all_new:
        return

    new_df = pd.DataFrame(all_new)
    for col in new_df.columns:
        if new_df[col].dtype == 'object':
            new_df[col] = new_df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )

    cache_path = CACHE_DIR / "closed_all.parquet"
    if cache_path.exists():
        old_df = pd.read_parquet(cache_path)
        print(f"  old cache: {len(old_df)}")
        all_cols = list(set(old_df.columns) | set(new_df.columns))
        for c in all_cols:
            if c not in old_df.columns:
                old_df[c] = None
            if c not in new_df.columns:
                new_df[c] = None
        old_df = old_df[all_cols]
        new_df = new_df[all_cols]
        merged = pd.concat([old_df, new_df], ignore_index=True)
        if "id" in merged.columns:
            merged = merged.drop_duplicates(subset=["id"], keep="last")
        elif "conditionId" in merged.columns:
            merged = merged.drop_duplicates(subset=["conditionId"], keep="last")
        print(f"  merged total: {len(merged)}")
    else:
        merged = new_df

    merged.to_parquet(cache_path, index=False)
    merged["endDate_dt"] = pd.to_datetime(merged["endDate"], errors='coerce', utc=True)
    print(f"  최신 endDate: {merged['endDate_dt'].max()}")
    print(f"  → {cache_path}")


if __name__ == "__main__":
    main()
