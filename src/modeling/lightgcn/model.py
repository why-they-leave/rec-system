"""
LightGCN — 인터페이스 스텁.

유저-아이템 이분(bipartite) 그래프와 유저-아이템-페르소나 삼분(tripartite) 그래프,
두 버전을 각각 학습해 비교 전시할 예정이다(reports/STREAMLIT_UI_DESIGN.md 참고).
graph_type으로 구분:
    "bipartite"  — 유저-아이템 그래프만 사용(페르소나 미포함, 순수 협업 필터링 baseline)
    "tripartite" — 유저-아이템-페르소나 그래프(페르소나 포함)

학습/추론 로직이 정해지면 채운다 — backend/api/services/lightgcn_service.py가
train()으로 만든 아티팩트를 graph_type별로 로드해 recommend()를 호출하도록 연결하면 된다
(als_service.py와 동일한 패턴).
"""

GRAPH_TYPES = ("bipartite", "tripartite")


def train(interactions_path: str, params: dict, graph_type: str = "tripartite") -> None:
    """유저-아이템(-페르소나) 상호작용 데이터로 LightGCN을 학습한다 (미구현).

    graph_type: "bipartite"(유저-아이템) 또는 "tripartite"(유저-아이템-페르소나).
    """
    raise NotImplementedError(f"LightGCN({graph_type}) 학습 로직이 아직 구현되지 않았습니다.")


def recommend(user_id: int, top_n: int = 10, graph_type: str = "tripartite") -> list[dict]:
    """학습된 LightGCN({graph_type}) 아티팩트로 유저 추천을 생성한다 (미구현)."""
    raise NotImplementedError(f"LightGCN({graph_type}) 추론 로직이 아직 구현되지 않았습니다.")
