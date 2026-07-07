"""
Twiddler(페르소나 가중치 재랭킹) 서비스.

user_id로 페르소나를 조회하고(persona_service), 카테고리(catalog_service)와 노출 이력
(exposure_service)을 주입해 src/modeling/twiddler/rerank.py의 순수 재랭킹 함수를 호출한다.
ALS 메인 추천(id_key="item_id", context="main")과 보완재 상세 추천
(id_key="rec_item_id", context="detail") 양쪽에서 동일하게 재사용된다.
"""

from __future__ import annotations

from src.modeling.twiddler import rerank as rerank_mod
from backend.api.services import catalog_service, complementary_service, exposure_service, persona_service


def apply_twiddler(
    items: list[dict],
    phase: str,
    user_id: int,
    *,
    id_key: str = "item_id",
    context: str = "main",
    top_k: int,
) -> tuple[list[dict], str, str | None]:
    """반환: (items, status, message). status는 "ok" 또는 "not_implemented"."""
    if phase == "before":
        return items[:top_k], "ok", None

    persona_label = persona_service.get_persona(user_id)
    if persona_label is None:
        return items[:top_k], "not_implemented", "이 유저의 페르소나 정보를 찾을 수 없어 Before 결과로 대체합니다."

    category_map = catalog_service.get_category_map()
    affinity = persona_service.get_segment_affinity(persona_label)
    alpha = persona_service.get_segment_alpha(persona_label)
    exposure_counts = exposure_service.get_recent_exposure(user_id, context)
    low_exposure_ids = (
        complementary_service.get_low_exposure_items() if context == "detail" else None
    )

    reranked = rerank_mod.rerank(
        items,
        id_key=id_key,
        category_map=category_map,
        affinity=affinity,
        alpha=alpha,
        exposure_counts=exposure_counts,
        low_exposure_ids=low_exposure_ids,
        top_k=top_k,
    )
    exposure_service.record_exposure(user_id, context, [item[id_key] for item in reranked])
    return reranked, "ok", None
