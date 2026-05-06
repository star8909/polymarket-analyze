"""비상 정지 메커니즘.

KILL 파일 존재 → 즉시 모든 미체결 주문 취소 + 봇 종료.

사용:
1. cron/터미널 어디서든 `touch live/KILL` → 봇이 다음 루프에서 정지
2. 봇 자체에서도 daily loss / mismatch 감지 시 자동 KILL 생성
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def is_kill_active(kill_path: str) -> bool:
    return os.path.exists(kill_path)


def trigger_kill(kill_path: str, reason: str) -> None:
    Path(kill_path).write_text(
        json.dumps({"triggered_at": datetime.now().isoformat(), "reason": reason}, indent=2),
        encoding="utf-8"
    )


def clear_kill(kill_path: str) -> None:
    if os.path.exists(kill_path):
        os.remove(kill_path)


def emergency_shutdown(client, kill_path: str, reason: str) -> None:
    """비상 절차: cancel_all + KILL 파일 생성 + 로그."""
    print(f"\n🛑 EMERGENCY SHUTDOWN: {reason}")
    try:
        result = client.cancel_all()
        print(f"  cancel_all 응답: {result}")
    except Exception as e:
        print(f"  cancel_all 에러: {e}")
    trigger_kill(kill_path, reason)
    print(f"  KILL 파일 생성: {kill_path}")
    print("  봇 종료. KILL 파일 수동 삭제 전엔 재시작 안 됨.")
