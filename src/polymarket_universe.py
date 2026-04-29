"""Polymarket 카테고리 universe.

Polymarket의 주요 카테고리 (Gamma API tag_slug 기준).
카테고리별로 calibration 곡선이 다름 → 분리 학습 필요.
"""
from __future__ import annotations


# Gamma API tag_slug (URL slug)
CATEGORIES = {
    "politics": "politics",
    "elections": "elections",
    "trump": "trump",          # 자주 등장하는 sub-tag
    "crypto": "crypto",
    "sports": "sports",
    "nba": "nba",
    "nfl": "nfl",
    "soccer": "soccer",
    "football": "football",
    "tennis": "tennis",
    "ufc": "ufc",
    "world": "world",
    "tech": "tech",
    "ai": "ai",
    "economy": "economy",
    "fed": "fed",
    "covid": "covid",
    "climate": "climate",
    "entertainment": "entertainment",
    "movies": "movies",
    "music": "music",
}


# Calibration 분석 시 분리 학습할 그룹
CALIBRATION_GROUPS = {
    "politics_us": ["politics", "elections", "trump"],
    "sports_team": ["nba", "nfl", "soccer", "football"],
    "sports_individual": ["tennis", "ufc"],
    "crypto": ["crypto"],
    "macro": ["economy", "fed"],
    "tech": ["tech", "ai"],
    "world_events": ["world", "covid", "climate"],
    "entertainment": ["entertainment", "movies", "music"],
}


def all_categories() -> list[str]:
    """모든 카테고리 slug 1차원 리스트."""
    return list(CATEGORIES.values())


def calibration_group_for(tag_slug: str) -> str | None:
    """주어진 tag_slug가 속한 calibration group 반환."""
    for group, tags in CALIBRATION_GROUPS.items():
        if tag_slug in tags:
            return group
    return None
