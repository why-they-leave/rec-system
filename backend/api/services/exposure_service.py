"""
노출 이력 추적 서비스 — Twiddler Rule 2(노출 이력 패널티)용 상태 저장소.

프로세스 메모리에만 존재한다(재시작 시 초기화, DB 없음) — 데모 범위의 알려진 한계.
추후 recommend.db(SQLite) 전환 시 이 모듈의 내부 저장소만 교체하면 되도록
get/record 두 함수 뒤로 상태를 캡슐화해 둔다.
"""

from __future__ import annotations

_exposure_store: dict[tuple[int, str], dict[int, float]] = {}


def get_recent_exposure(user_id: int, context: str) -> dict[int, float]:
    return dict(_exposure_store.get((user_id, context), {}))


def reset(user_id: int, context: str) -> None:
    """특정 유저/컨텍스트의 노출 이력을 초기화한다 — 새로고침 시뮬레이션을 매번 깨끗한 상태(라운드 0)에서 시작하기 위함."""
    _exposure_store.pop((user_id, context), None)


def record_exposure(user_id: int, context: str, shown_item_ids: list[int], decay: float) -> None:
    """decay는 유저 개인화 값(persona_service.get_user_decay)을 그대로 받는다."""
    key = (user_id, context)
    counts = _exposure_store.setdefault(key, {})

    # 기존 노출 카운트는 한 번 감쇠시켜 오래된 노출이 자연히 약해지게 함
    for item_id in list(counts.keys()):
        decayed = counts[item_id] * decay
        if decayed < 0.01:
            del counts[item_id]
        else:
            counts[item_id] = decayed

    for item_id in shown_item_ids:
        counts[item_id] = counts.get(item_id, 0.0) + 1.0
