"""프로젝트 디렉토리 + API 상수."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SNAPSHOT_DIR = DATA_DIR / "snapshot"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in (DATA_DIR, CACHE_DIR, SNAPSHOT_DIR, RESULTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# Polymarket API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

# Rate limits (보수적)
GAMMA_RATE_LIMIT = 5      # req/sec
CLOB_RATE_LIMIT = 3       # req/sec
DEFAULT_TIMEOUT = 30      # 초

# 베팅 안전 한도
MAX_BET_PER_MARKET = 100  # USD per market
MAX_DAILY_BET = 1000      # USD per day
KELLY_FRACTION = 0.25     # Fractional Kelly
