"""
LightGCN 추천 서비스 — 모델 미구현 스텁.

유저-아이템 이분(bipartite) 그래프와 유저-아이템-페르소나 삼분(tripartite) 그래프
두 버전을 비교 전시할 예정이라 graph_type으로 구분해 둔다(src/modeling/lightgcn/model.py 참고).
그 모듈이 채워지고 아티팩트가 생기면, als_service.py와 동일한 패턴(아티팩트 로드 + recommend
조회)으로 graph_type별 아티팩트 경로만 나눠 구현하면 된다. 그 전까지는 항상 not_implemented를
반환해 app/main.py의 LightGCN 두 섹션(이분/삼분)이 각각 "모델 준비 중" 안내로 표시되게 한다.
"""

from __future__ import annotations

from src.modeling.lightgcn.model import GRAPH_TYPES

_MESSAGES = {
    "bipartite": "LightGCN(이분그래프 · 페르소나 미포함) 모델이 아직 준비되지 않았습니다.",
    "tripartite": "LightGCN(삼분그래프 · 페르소나 포함) 모델이 아직 준비되지 않았습니다.",
}


def get_recommendations(
    user_id: int, top_n: int = 10, graph_type: str = "tripartite"
) -> tuple[list[dict], str, str | None]:
    """반환: (items, status, message). status는 항상 "not_implemented"(현재 미구현).

    graph_type: "bipartite" 또는 "tripartite" — src.modeling.lightgcn.model.GRAPH_TYPES 참고.
    """
    if graph_type not in GRAPH_TYPES:
        graph_type = "tripartite"
    return [], "not_implemented", _MESSAGES[graph_type]
