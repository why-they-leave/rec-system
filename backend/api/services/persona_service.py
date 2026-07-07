"""
페르소나(세그먼트) 조회 서비스.

data/processed/customer_segments_labeled_train_only.csv(customer_id -> segment_id/segment_name,
그리고 top_view_category/top_purchase_category/view_purchase_category_match/
dominant_purchase_category_ratio 등 구조화 컬럼)를 기동 시 1회 로드해 메모리에 캐시한다.

- get_persona(user_id): user_id -> persona_label(=segment_name) 조회
- get_segment_affinity(persona_label): 해당 세그먼트의 카테고리별 선호도 편차(affinity) 반환
- get_segment_alpha(persona_label): Rule 1의 가중치 강도(alpha) 반환
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.modeling.twiddler.rerank import BASE_ALPHA, NUM_CATEGORIES

_TABLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "customer_segments_labeled_train_only.csv"
)

_df: pd.DataFrame | None = None
_persona_by_user: dict[int, str] | None = None
_affinity_by_persona: dict[str, dict[str, float]] | None = None
_alpha_by_persona: dict[str, float] | None = None


def _load() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(_TABLE_PATH)
    return _df


def _build_lookups() -> None:
    global _persona_by_user, _affinity_by_persona, _alpha_by_persona
    if _persona_by_user is not None:
        return

    df = _load()
    _persona_by_user = dict(zip(df["customer_id"], df["segment_name"]))

    uniform_baseline = 1.0 / NUM_CATEGORIES
    _affinity_by_persona = {}
    _alpha_by_persona = {}

    for persona_label, seg_df in df.groupby("segment_name"):
        match_rate = seg_df["view_purchase_category_match"].mean()
        match_rate = match_rate if pd.notna(match_rate) else 0.0
        w_purchase = 0.5 + 0.5 * match_rate
        w_view = 1 - w_purchase

        purchase_share = seg_df["top_purchase_category"].value_counts(normalize=True)
        view_share = seg_df["top_view_category"].value_counts(normalize=True)
        categories = set(purchase_share.index) | set(view_share.index)

        affinity = {
            category: (
                w_purchase * purchase_share.get(category, 0.0)
                + w_view * view_share.get(category, 0.0)
                - uniform_baseline
            )
            for category in categories
        }
        _affinity_by_persona[persona_label] = affinity

        # 비구매(browsing-only) 세그먼트는 dominant_purchase_category_ratio가 전부
        # 결측일 수 있음(구매 자체가 없으므로) -> 이 경우 alpha=0으로 Rule 1을 사실상 비활성화.
        dominant_ratio_mean = seg_df["dominant_purchase_category_ratio"].mean()
        _alpha_by_persona[persona_label] = (
            BASE_ALPHA * dominant_ratio_mean if pd.notna(dominant_ratio_mean) else 0.0
        )


def get_persona(user_id: int) -> str | None:
    _build_lookups()
    return _persona_by_user.get(user_id)


def get_segment_affinity(persona_label: str) -> dict[str, float]:
    _build_lookups()
    return _affinity_by_persona.get(persona_label, {})


def get_segment_alpha(persona_label: str) -> float:
    _build_lookups()
    return _alpha_by_persona.get(persona_label, 0.0)
