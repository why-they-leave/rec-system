from fastapi import APIRouter, Query

from backend.api import schemas
from backend.api.core import get_main_recommendation_items

router = APIRouter()


@router.get("/recommend/main", response_model=schemas.RecommendMainResponse)
def recommend_main(
    user_id: int = Query(...),
    model_type: str = Query(..., pattern="^(ALS|LightGCN)$"),
    graph_type: str = Query("tripartite", pattern="^(bipartite|tripartite)$"),
    twiddler: str = Query("before", pattern="^(before|after)$"),
    top_n: int = Query(10, ge=1, le=50),
) -> schemas.RecommendMainResponse:
    """
    model_type="LightGCN"일 때만 graph_type이 의미를 가진다:
    "bipartite"(유저-아이템, 페르소나 미포함) 또는 "tripartite"(유저-아이템-페르소나).
    응답의 model_type은 LightGCN인 경우 "LightGCN-<graph_type>"으로 반환해 어떤 변형인지 구분한다.
    LightGCN-bipartite는 ALS와 동일하게 twiddler 파라미터가 적용된다(그 외 LightGCN 변형은 미적용).

    실제 오케스트레이션 로직은 backend/api/core.py에 있다 — app/utils/data_loader.py가
    Streamlit 프로세스 안에서 이 HTTP 계층을 거치지 않고 그 함수를 직접 호출한다.
    """
    items, status, message, response_model_type, response_twiddler = get_main_recommendation_items(
        user_id, model_type, graph_type, twiddler, top_n
    )

    return schemas.RecommendMainResponse(
        status=status,
        message=message,
        items=[
            schemas.MainRecommendItem(
                user_id=user_id,
                item_id=item["item_id"],
                score=item["score"],
                rank=item["rank"],
                model_type=response_model_type,
                twiddler=response_twiddler,
                user_type=item["user_type"],
            )
            for item in items
        ],
    )
