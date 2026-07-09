"""
LightGCN 추천 서비스.

유저-아이템 이분(bipartite) 그래프와 유저-아이템-페르소나 삼분(tripartite) 그래프
두 버전을 비교 전시할 예정이라 graph_type으로 구분해 둔다(src/modeling/lightgcn/model.py 참고).

"bipartite"는 retail-clickstream-analysis에서 학습해 내보낸 전체 유저 추천 결과
(data/outputs/LightGCN/PRED_MAIN_RECOMMEND.csv, 유저당 top-100)를 als_service.py와
동일한 패턴(아티팩트를 한 번만 로드해 메모리에 캐싱 + 조회)으로 서빙한다 — 이 모델은
아직 이 레포 안에서 직접 학습/추론하지 않고, 미리 계산된 CSV를 조회만 한다.
"tripartite"는 아직 학습 산출물이 없어 기존과 동일하게 항상 not_implemented를 반환한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.modeling.lightgcn.model import GRAPH_TYPES

_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "outputs" / "LightGCN" / "PRED_MAIN_RECOMMEND.csv"
)

_MESSAGES = {
    "bipartite": "LightGCN(이분그래프 · 페르소나 미포함) 모델이 아직 준비되지 않았습니다.",
    "tripartite": "LightGCN(삼분그래프 · 페르소나 포함) 모델이 아직 준비되지 않았습니다.",
}

_recs_by_user: dict[int, list[dict]] | None = None


def _load_bipartite_recs() -> dict[int, list[dict]]:
    global _recs_by_user
    if _recs_by_user is None:
        df = pd.read_csv(_ARTIFACT_PATH)
        _recs_by_user = {
            int(user_id): [
                {"item_id": int(row.item_id), "score": float(row.score), "rank": int(row.rank)}
                for row in g.sort_values("rank").itertuples(index=False)
            ]
            for user_id, g in df.groupby("user_id")
        }
    return _recs_by_user


def get_recommendations(
    user_id: int, top_n: int = 10, graph_type: str = "tripartite"
) -> tuple[list[dict], str, str | None]:
    """반환: (items, status, message). status는 "ok" 또는 "not_implemented".

    graph_type: "bipartite" 또는 "tripartite" — src.modeling.lightgcn.model.GRAPH_TYPES 참고.
    "bipartite"만 실제 데이터를 서빙하고("user_type"은 cold/heavy 구분이 없어 "all" 고정값),
    "tripartite"는 아직 학습 산출물이 없어 항상 not_implemented다.
    """
    if graph_type not in GRAPH_TYPES:
        graph_type = "tripartite"
    if graph_type != "bipartite":
        return [], "not_implemented", _MESSAGES[graph_type]

    recs_by_user = _load_bipartite_recs()
    user_items = recs_by_user.get(user_id)
    if user_items is None:
        return [], "not_implemented", "이 유저는 LightGCN(이분그래프) 학습 데이터에 없습니다."

    items = [{**it, "user_type": "all"} for it in user_items[:top_n]]
    return items, "ok", None
