"""
LightGCN 추천 서비스.

유저-아이템 이분(bipartite) 그래프와 유저-아이템-페르소나 삼분(tripartite) 그래프
두 버전을 비교 전시할 예정이라 graph_type으로 구분한다(src/modeling/lightgcn/model.py 참고).

bipartite: retail-clickstream-analysis#34에서 학습한 결과(PRED_MAIN_RECOMMEND.csv, top-100)를
data/outputs/LightGCN/에 가져와 als_service.py와 동일한 패턴(아티팩트 로드 + 조회)으로 서빙한다
(실시간 재추론이 아니라 사전 계산된 CSV 조회 — LightGCN은 학습 비용이 커서 als_service.py처럼
매 요청마다 model.recommend()를 부르지 않는다).
tripartite: 아직 이 레포로 안 가져옴 — not_implemented 유지.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.modeling.lightgcn.model import GRAPH_TYPES

_ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "data" / "outputs" / "LightGCN"
_ARTIFACT_PATHS = {
    "bipartite": _ARTIFACT_DIR / "PRED_MAIN_RECOMMEND.csv",
}
_MESSAGES = {
    "tripartite": "LightGCN(삼분그래프 · 페르소나 포함) 모델이 아직 준비되지 않았습니다.",
}

_artifact_cache: dict[str, pd.DataFrame] = {}


def _load_artifact(graph_type: str) -> pd.DataFrame:
    if graph_type not in _artifact_cache:
        _artifact_cache[graph_type] = pd.read_csv(_ARTIFACT_PATHS[graph_type])
    return _artifact_cache[graph_type]


def _recommendations_from_df(df: pd.DataFrame, user_id: int, top_n: int) -> list[dict]:
    """저장된 추천 결과에서 유저의 top_n개를 rank 순으로 뽑아 dict 리스트로 변환한다."""
    user_recs = df[df["user_id"] == user_id].sort_values("rank").head(top_n)
    return [
        {"item_id": int(row.item_id), "score": float(row.score), "rank": int(row.rank)}
        for row in user_recs.itertuples(index=False)
    ]


def get_recommendations(
    user_id: int, top_n: int = 10, graph_type: str = "tripartite"
) -> tuple[list[dict], str, str | None]:
    """반환: (items, status, message). status는 "ok" 또는 "not_implemented".

    graph_type: "bipartite" 또는 "tripartite" — src.modeling.lightgcn.model.GRAPH_TYPES 참고.
    tripartite는 아직 미구현이라 항상 not_implemented를 반환한다.
    """
    if graph_type not in GRAPH_TYPES:
        graph_type = "tripartite"
    if graph_type not in _ARTIFACT_PATHS:
        return [], "not_implemented", _MESSAGES[graph_type]

    df = _load_artifact(graph_type)
    items = _recommendations_from_df(df, user_id, top_n)
    if not items:
        return [], "not_implemented", "이 유저의 LightGCN 추천 결과를 찾을 수 없습니다."
    return items, "ok", None
