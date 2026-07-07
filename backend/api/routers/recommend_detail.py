from fastapi import APIRouter, Query

from backend.api import schemas
from backend.api.services import complementary_service, twiddler_service
from src.modeling.twiddler.rerank import POOL_MULTIPLIER

router = APIRouter()


@router.get("/recommend/detail", response_model=schemas.RecommendDetailResponse)
def recommend_detail(
    item_id: int = Query(...),
    user_id: int | None = Query(None),
    twiddler: str = Query("before", pattern="^(before|after)$"),
    top_n: int = Query(8, ge=1, le=50),
) -> schemas.RecommendDetailResponse:
    apply_after = twiddler == "after" and user_id is not None
    fetch_n = top_n * POOL_MULTIPLIER if apply_after else top_n
    items, status, message = complementary_service.get_recommendations(item_id, fetch_n)
    if status == "ok" and user_id is not None:
        items, status, message = twiddler_service.apply_twiddler(
            items, twiddler, user_id, id_key="rec_item_id", context="detail", top_k=top_n
        )
    else:
        twiddler = "before"

    return schemas.RecommendDetailResponse(
        status=status,
        message=message,
        items=[
            schemas.DetailRecommendItem(
                item_id=item_id,
                rec_item_id=item["rec_item_id"],
                score=item["score"],
                rank=item["rank"],
                twiddler=twiddler,
            )
            for item in items
        ],
    )
