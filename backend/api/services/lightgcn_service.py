"""
LightGCN 추천 서비스.

유저-아이템 이분(bipartite) 그래프와 유저-아이템-페르소나 삼분(tripartite) 그래프
두 버전을 비교 전시한다(src/modeling/lightgcn/model.py의 GRAPH_TYPES 참고).
retail-clickstream-analysis #34 결론: 페르소나 결합 효과는 하이퍼파라미터에 좌우되는
약하고 불안정한 효과.

두 그래프의 추천 결과는 retail-clickstream-analysis에서 학습해 내보낸 CSV를 이 레포
data/outputs/LightGCN/에 복사해 서빙한다(als_service.py와 동일하게 아티팩트를 한 번만
로드해 메모리에 캐싱) — 이 레포 안에서 직접 학습/추론하지 않는다.

retail-clickstream-analysis #41에서 파일명을 graph_mode별로 분리했다
(PRED_MAIN_RECOMMEND_bipartite.csv / _tripartite.csv). 이 레포의 bipartite 데이터는
그 이전에 고정 파일명(PRED_MAIN_RECOMMEND.csv)으로 이미 복사돼 GDrive 배포 번들
(app/utils/data_bootstrap.py REQUIRED_FILES)에도 그 이름으로 들어가 있으므로, 새
이름을 우선 찾고 없으면 예전 이름으로 폴백한다 — 기존 배포를 깨지 않기 위함이다.
tripartite는 예전 이름 자체가 존재한 적이 없어 폴백이 없다. 두 그래프 모두 아직
데이터가 없으면(파일 없음) not_implemented를 반환한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.modeling.lightgcn.model import GRAPH_TYPES

_ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "data" / "outputs" / "LightGCN"

_ARTIFACT_FILENAME_CANDIDATES = {
    "bipartite": ["PRED_MAIN_RECOMMEND_bipartite.csv", "PRED_MAIN_RECOMMEND.csv"],
    "tripartite": ["PRED_MAIN_RECOMMEND_tripartite.csv"],
}

_MESSAGES = {
    "bipartite": "LightGCN(이분그래프 · 페르소나 미포함) 모델이 아직 준비되지 않았습니다.",
    "tripartite": "LightGCN(삼분그래프 · 페르소나 포함) 모델이 아직 준비되지 않았습니다.",
}

_recs_cache: dict[str, dict[int, list[dict]]] = {}


def resolve_artifact_path(graph_type: str) -> Path | None:
    """graph_type에 맞는 CSV 경로를 찾는다 — 후보 목록 순서대로 먼저 존재하는 파일 사용."""
    for filename in _ARTIFACT_FILENAME_CANDIDATES[graph_type]:
        candidate = _ARTIFACT_DIR / filename
        if candidate.exists():
            return candidate
    return None


def _load_recs(graph_type: str) -> dict[int, list[dict]] | None:
    """graph_type에 해당하는 CSV를 로드해 캐싱한다. 파일이 없으면 None."""
    if graph_type in _recs_cache:
        return _recs_cache[graph_type]

    artifact_path = resolve_artifact_path(graph_type)
    if artifact_path is None:
        return None

    df = pd.read_csv(artifact_path)
    recs_by_user = {
        int(user_id): [
            {"item_id": int(row.item_id), "score": float(row.score), "rank": int(row.rank)}
            for row in g.sort_values("rank").itertuples(index=False)
        ]
        for user_id, g in df.groupby("user_id")
    }
    _recs_cache[graph_type] = recs_by_user
    return recs_by_user


def get_recommendations(
    user_id: int, top_n: int = 10, graph_type: str = "tripartite"
) -> tuple[list[dict], str, str | None]:
    """반환: (items, status, message). status는 "ok" 또는 "not_implemented".

    graph_type: "bipartite" 또는 "tripartite" — src.modeling.lightgcn.model.GRAPH_TYPES 참고.
    두 그래프 모두 "user_type"은 cold/heavy 구분이 없어 "all" 고정값이다. 아직 CSV가
    준비되지 않은 graph_type은 not_implemented를 반환한다.
    """
    if graph_type not in GRAPH_TYPES:
        graph_type = "tripartite"

    recs_by_user = _load_recs(graph_type)
    if recs_by_user is None:
        return [], "not_implemented", _MESSAGES[graph_type]

    user_items = recs_by_user.get(user_id)
    if user_items is None:
        return [], "not_implemented", f"이 유저는 LightGCN({graph_type}) 학습 데이터에 없습니다."

    items = [{**it, "user_type": "all"} for it in user_items[:top_n]]
    return items, "ok", None
