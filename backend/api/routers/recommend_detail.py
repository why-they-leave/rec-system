from fastapi import APIRouter, Query

from backend.api import schemas
from backend.api.core import get_detail_recommendation_items

router = APIRouter()


@router.get("/recommend/detail", response_model=schemas.RecommendDetailResponse)
def recommend_detail(
    item_id: int = Query(...),
    user_id: int | None = Query(None),
    twiddler: str = Query("before", pattern="^(before|after)$"),
    top_n: int = Query(8, ge=1, le=50),
) -> schemas.RecommendDetailResponse:
    """
    실제 오케스트레이션 로직은 backend/api/core.py에 있다 — app/utils/data_loader.py가
    Streamlit 프로세스 안에서 이 HTTP 계층을 거치지 않고 그 함수를 직접 호출한다.
    """
    items, status, message, response_twiddler = get_detail_recommendation_items(
        item_id, top_n, user_id, twiddler
    )

    return schemas.RecommendDetailResponse(
        status=status,
        message=message,
        items=[
            schemas.DetailRecommendItem(
                item_id=item_id,
                rec_item_id=item["rec_item_id"],
                score=item["score"],
                rank=item["rank"],
                twiddler=response_twiddler,
            )
            for item in items
        ],
    )
