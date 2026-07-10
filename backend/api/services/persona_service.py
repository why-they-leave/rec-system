"""
페르소나(유저 단위 행동 성향) 조회 서비스.

data/processed/customer_segments_labeled_train_only.csv(customer_id -> segment_id/segment_name,
그리고 top_view_category/top_purchase_category/view_purchase_category_match/
dominant_purchase_category_ratio/category_diversity_purchase/dominant_view_category_ratio 등
유저 단위 구조화 컬럼)를 기동 시 1회 로드해 메모리에 캐시한다.

Rule 1/2는 세그먼트 평균이 아니라 유저 단위 연속값을 쓴다(세그먼트 평균으로 뭉개면 같은
세그먼트 유저가 전부 동일한 배율/감쇠를 받아 "유동적인 소비 선호도"를 포착하지 못하고,
개인화 alpha에 상한이 없으면 로열티가 매우 높은 유저가 필터버블에 갇힌다 — 둘 다 검증됨,
notebooks/20260708_ML_twiddler_final_design.ipynb 참고):

- get_persona(user_id): user_id -> persona_label(=segment_name) 조회. 페르소나 데이터 존재 여부
  게이트로만 쓰인다(twiddler_service.apply_twiddler의 "not_implemented" 분기).
- get_user_affinity(user_id): 유저 본인의 top_view/purchase_category 기반 카테고리별 선호도 편차
- get_user_alpha(user_id): 유저 개인의 category_loyalty x (1-exploration_tendency)로 계산한
  Rule 1 가중치 강도 (배율 자체의 상한/하한은 src/modeling/twiddler/rerank.py가 적용)
- get_user_decay(user_id): exploration_tendency의 percentile rank 기반 Rule 2 노출 감쇠율
  (population 평균이 EXPOSURE_DECAY 근방에 고정되도록 원값이 아니라 rank를 사용)

exploration_tendency(탐색 성향, 0~1): 구매 이력이 있는 유저는 category_diversity_purchase(구매
카테고리 다양성)와 (1-view_purchase_category_match)(조회-구매 카테고리 불일치)를 절반씩 결합한다.
구매 이력이 없는 유저는 dominant_view_category_ratio의 역수로 근사한다(구매 데이터가 없어
데이터 없이 임의로 가정하지 않기 위한 분기).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.api.services import catalog_service
from src.modeling.twiddler.rerank import (
    BASE_ALPHA,
    EXPLORATION_DECAY_MAX,
    EXPLORATION_DECAY_MIN,
    EXPOSURE_DECAY,
    NUM_CATEGORIES,
)

_TABLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "customer_segments_labeled_train_only.csv"
)

_UNIFORM_BASELINE = 1.0 / NUM_CATEGORIES

_feat_df: pd.DataFrame | None = None
_persona_by_user: dict[int, str] | None = None
_all_categories: list[str] | None = None


def _build_features() -> None:
    global _feat_df, _persona_by_user, _all_categories
    if _feat_df is not None:
        return

    df = pd.read_csv(_TABLE_PATH)
    _persona_by_user = dict(zip(df["customer_id"], df["segment_name"]))
    _all_categories = sorted(set(catalog_service.get_category_map().values()) - {None})

    diversity_norm = (df["category_diversity_purchase"] - 1) / (NUM_CATEGORIES - 1)
    purchase_based = 0.5 * diversity_norm + 0.5 * (1 - df["view_purchase_category_match"])
    view_based = 1 - df["dominant_view_category_ratio"]

    has_purchase = df["dominant_purchase_category_ratio"].notna()
    df["exploration_tendency"] = np.where(has_purchase, purchase_based, view_based)
    df["exploration_tendency"] = df["exploration_tendency"].fillna(0.5).clip(0, 1)
    df["exploration_pct_rank"] = df["exploration_tendency"].rank(pct=True)

    df["category_loyalty"] = df["dominant_purchase_category_ratio"].fillna(0.0)
    df["w_purchase"] = np.where(df["view_purchase_category_match"] == 1, 1.0, 0.5)
    df["w_view"] = 1 - df["w_purchase"]

    _feat_df = df.set_index("customer_id")


def get_persona(user_id: int) -> str | None:
    _build_features()
    return _persona_by_user.get(user_id)


def get_user_affinity(user_id: int) -> dict[str, float]:
    _build_features()
    if user_id not in _feat_df.index:
        return {}
    row = _feat_df.loc[user_id]
    return {
        category: (
            (row["w_purchase"] if category == row["top_purchase_category"] else 0.0)
            + (row["w_view"] if category == row["top_view_category"] else 0.0)
            - _UNIFORM_BASELINE
        )
        for category in _all_categories
    }


def get_user_alpha(user_id: int) -> float:
    _build_features()
    if user_id not in _feat_df.index:
        return 0.0
    row = _feat_df.loc[user_id]
    return BASE_ALPHA * row["category_loyalty"] * (1 - row["exploration_tendency"])


def get_user_decay(user_id: int) -> float:
    _build_features()
    if user_id not in _feat_df.index:
        return EXPOSURE_DECAY
    row = _feat_df.loc[user_id]
    return EXPLORATION_DECAY_MAX - (EXPLORATION_DECAY_MAX - EXPLORATION_DECAY_MIN) * row["exploration_pct_rank"]
