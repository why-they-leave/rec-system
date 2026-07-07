from fastapi import APIRouter, Query

from backend.api import schemas
from backend.api.services import als_service, lightgcn_service, twiddler_service
from src.modeling.twiddler.rerank import POOL_MULTIPLIER

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
    """
    if model_type == "ALS":
        fetch_n = top_n * POOL_MULTIPLIER if twiddler == "after" else top_n
        items, status, message = als_service.get_recommendations(user_id, fetch_n)
        if status == "ok":
            items, status, message = twiddler_service.apply_twiddler(
                items, twiddler, user_id, id_key="item_id", context="main", top_k=top_n
            )
        response_model_type = model_type
    else:
        items, status, message = lightgcn_service.get_recommendations(user_id, top_n, graph_type)
        twiddler = "before"
        response_model_type = f"LightGCN-{graph_type}"

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
                twiddler=twiddler,
                user_type=item["user_type"],
            )
            for item in items
        ],
    )
