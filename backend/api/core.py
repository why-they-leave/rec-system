"""
추천 오케스트레이션 핵심 로직 — FastAPI 라우터(backend/api/routers/*)와 Streamlit(app/)의
in-process 직접 호출(app/utils/data_loader.py) 양쪽에서 동일하게 재사용한다.
FastAPI 전용 타입(Query, 응답 스키마)에 의존하지 않는 순수 함수로 유지해,
호출 경로(HTTP든 직접 함수 호출이든)와 무관하게 로직이 하나로 유지되도록 한다.
"""

from __future__ import annotations

from backend.api.services import als_service, complementary_service, lightgcn_service, twiddler_service
from src.modeling.twiddler.rerank import POOL_MULTIPLIER


def get_main_recommendation_items(
    user_id: int,
    model_type: str,
    graph_type: str = "tripartite",
    twiddler: str = "before",
    top_n: int = 10,
) -> tuple[list[dict], str, str | None, str, str]:
    """
    반환: (items, status, message, response_model_type, response_twiddler).
    model_type="LightGCN"일 때만 graph_type이 의미를 가진다("bipartite" | "tripartite").
    response_model_type은 LightGCN인 경우 "LightGCN-<graph_type>"으로 반환해 어떤 변형인지 구분한다.
    response_twiddler는 LightGCN인 경우 항상 "before"로 강제된다(Twiddler 미적용).
    """
    if model_type == "ALS":
        fetch_n = top_n * POOL_MULTIPLIER if twiddler == "after" else top_n
        items, status, message = als_service.get_recommendations(user_id, fetch_n)
        if status == "ok":
            items, status, message = twiddler_service.apply_twiddler(
                items, twiddler, user_id, id_key="item_id", context="main", top_k=top_n
            )
        response_model_type = model_type
        response_twiddler = twiddler
    else:
        items, status, message = lightgcn_service.get_recommendations(user_id, top_n, graph_type)
        response_model_type = f"LightGCN-{graph_type}"
        response_twiddler = "before"

    return items, status, message, response_model_type, response_twiddler


def get_detail_recommendation_items(
    item_id: int,
    top_n: int = 8,
    user_id: int | None = None,
    twiddler: str = "before",
) -> tuple[list[dict], str, str | None, str]:
    """반환: (items, status, message, response_twiddler)."""
    apply_after = twiddler == "after" and user_id is not None
    fetch_n = top_n * POOL_MULTIPLIER if apply_after else top_n
    items, status, message = complementary_service.get_recommendations(item_id, fetch_n)
    if status == "ok" and user_id is not None:
        items, status, message = twiddler_service.apply_twiddler(
            items, twiddler, user_id, id_key="rec_item_id", context="detail", top_k=top_n
        )
    else:
        twiddler = "before"

    return items, status, message, twiddler
