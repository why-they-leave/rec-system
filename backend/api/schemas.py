"""
API 응답 스키마 — data/dashboard/PRED_MAIN_RECOMMEND.csv / PRED_DETAIL_RECOMMEND.csv와
동일한 필드 구성을 유지해, app/utils/data_loader.py가 기존 DataFrame 스키마 그대로
API 응답을 소비할 수 있게 한다.
"""

from pydantic import BaseModel


class MainRecommendItem(BaseModel):
    user_id: int
    item_id: int
    score: float
    rank: int
    model_type: str
    twiddler: str
    user_type: str


class RecommendMainResponse(BaseModel):
    status: str = "ok"  # "ok" | "not_implemented"
    message: str | None = None
    items: list[MainRecommendItem] = []


class DetailRecommendItem(BaseModel):
    item_id: int
    rec_item_id: int
    score: float
    rank: int
    twiddler: str = "before"


class RecommendDetailResponse(BaseModel):
    status: str = "ok"  # "ok" | "not_implemented"
    message: str | None = None
    items: list[DetailRecommendItem] = []
