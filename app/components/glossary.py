"""용어 해석 페이지 — 데모를 보기 전에 읽는 용어 참고 자료.

내용은 팀 handoff 문서(2026-07-09) 기준. 논문식 설명 대신 2~3문장 이내 짧은
설명 위주로 구성하고, "페르소나 = 실제 고객 정체성"으로 오해하지 않도록
주의 문구를 포함한다. 카드형 그리드로 렌더링한다(요청 반영: "이쁘게" 개선).
"""

import re

import streamlit as st

_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "데이터 / 피처",
        [
            (
                "Clickstream",
                "사용자가 웹사이트에서 남긴 행동 로그입니다.\n\n예: 상품 조회, 장바구니 담기, 체크아웃, 구매",
            ),
            (
                "Session",
                "사용자가 한 번 방문해서 남긴 행동 묶음입니다. 하나의 세션 안에는 여러 이벤트가 포함될 수 있습니다.",
            ),
            (
                "Event",
                "사용자가 수행한 개별 행동입니다. 본 프로젝트에서는 page_view, add_to_cart, "
                "checkout, purchase 등이 사용됩니다.",
            ),
            (
                "Customer Feature",
                "이벤트·세션·주문 로그를 고객 단위로 집계한 피처입니다.\n\n예: 조회 수, 장바구니 비율, "
                "주문 수, 최근 방문일, 주요 구매 카테고리",
            ),
            (
                "Derived Feature",
                "단순 집계값을 조합해 만든 파생 피처입니다.\n\n예: atc_rate, purchase_per_session, total_spend_log",
            ),
        ],
    ),
    (
        "세그먼트 / 페르소나",
        [
            (
                "Segment",
                "비슷한 행동 패턴을 가진 고객군입니다. 본 프로젝트에서는 KMeans 클러스터링으로 6개의 행동 기반 세그먼트를 생성했습니다.",
            ),
            (
                "Heavy / Cold 유저",
                "학습 기간(train) 동안 남긴 이벤트 수를 기준으로 나눈 활동량 구분입니다. "
                "10건 이상이면 Heavy, 미만이면 Cold입니다.\n\n"
                "Cold 유저는 행동 데이터가 부족해 Twiddler 재랭킹을 적용하지 않고 "
                "인기도 기반 추천으로 대체합니다.",
            ),
            (
                "Persona",
                "세그먼트를 사람이 이해하기 쉽게 표현한 이름과 설명입니다. **주의**: 페르소나는 "
                "실제 고객의 정체성이나 인구통계 정답 라벨이 아니라 행동 패턴을 요약한 해석용 "
                "라벨입니다.",
            ),
            (
                "Segment Labeling",
                "각 segment_id에 이름과 설명을 붙이는 과정입니다. LLM은 고객을 직접 분류하지 않고, 이미 생성된 세그먼트의 요약 통계를 보고 이름만 부여합니다.",
            ),
            (
                "Train-only Segment",
                "추천 모델 평가에서 데이터 누수를 막기 위해, 평가 기간 이후 데이터를 제외하고 "
                "train 기간 데이터만으로 다시 계산한 세그먼트입니다.",
            ),
        ],
    ),
    (
        "추천 모델",
        [
            (
                "ALS",
                "Alternating Least Squares의 약자입니다. user-item 상호작용 행렬을 사용자 벡터와 "
                "상품 벡터로 분해해 아직 보지 않은 상품의 선호 점수를 예측하는 협업 필터링 모델입니다.",
            ),
            (
                "LightGCN",
                "유저와 상품을 그래프의 노드로 보고, 연결된 이웃 노드의 정보를 전파해 추천 임베딩을 학습하는 그래프 기반 추천 모델입니다.",
            ),
            (
                "Bipartite Graph",
                "두 종류의 노드로 구성된 그래프입니다. 본 프로젝트에서는 User와 Product만 연결한 유저-상품 그래프를 의미합니다.\n\n`User ─ Product`",
            ),
            (
                "Tri-graph",
                "세 종류의 노드로 구성된 그래프입니다. 본 프로젝트에서는 User, Product, Segment를 "
                "함께 연결한 그래프를 의미합니다.\n\n`User ─ Product` / `Product ─ Segment`",
            ),
            (
                "Graph Embedding",
                "그래프 구조를 학습해 유저와 상품을 벡터로 표현한 것입니다. 추천 점수는 보통 유저 벡터와 상품 벡터의 유사도로 계산됩니다.",
            ),
        ],
    ),
    (
        "Twiddler",
        [
            (
                "Twiddler",
                "기본 추천 모델이 만든 후보 상품의 순서를 후처리로 조정하는 재랭킹 모듈입니다. "
                "모델을 다시 학습하지 않고, 페르소나 선호와 노출 이력을 반영해 추천 순위를 바꿉니다.",
            ),
            (
                "Persona Preference",
                "각 페르소나가 선호하는 카테고리 정보를 의미합니다. Twiddler는 유저의 페르소나와 잘 맞는 카테고리 상품에 가중치를 줄 수 있습니다.",
            ),
            (
                "Exposure Penalty",
                "이미 여러 번 노출된 상품의 점수를 낮추는 규칙입니다. 같은 상품이 반복적으로 추천되는 문제를 줄이고, 추천 다양성을 높이기 위해 사용합니다.",
            ),
            (
                "Reranking",
                "추천 후보 자체를 새로 만드는 것이 아니라, 이미 생성된 후보들의 순서만 다시 정렬하는 과정입니다.",
            ),
        ],
    ),
    (
        "평가 지표",
        [
            (
                "Hit Rate@K",
                "추천된 상위 K개 상품 안에 실제 구매 상품이 포함되었는지를 보는 지표입니다. 값이 높을수록 추천이 실제 구매와 더 잘 맞았다는 의미입니다.",
            ),
            ("Recall@K", "실제 구매한 상품 중 추천 상위 K개에 포함된 비율입니다."),
            (
                "NDCG@K",
                "정답 상품이 추천 리스트의 몇 번째에 등장했는지를 반영하는 지표입니다. 상위 순위에 정답이 있을수록 더 높은 점수를 받습니다.",
            ),
            (
                "Repetition Rate",
                "반복 새로고침이나 재방문 상황에서 같은 상품이 얼마나 반복 노출되는지를 나타냅니다. 값이 낮을수록 추천 다양성이 높습니다.",
            ),
            (
                "Category Diversity",
                "추천 결과에 포함된 상품 카테고리의 다양성을 나타냅니다. 더 많은 카테고리가 포함될수록 탐색 폭이 넓은 추천이라고 볼 수 있습니다.",
            ),
        ],
    ),
]

_SECTION_ICONS: dict[str, str] = {
    "데이터 / 피처": "📊",
    "세그먼트 / 페르소나": "🧬",
    "추천 모델": "🤖",
    "Twiddler": "🎛️",
    "평가 지표": "📈",
}


def _desc_html(desc: str) -> str:
    """용어 설명에 쓰는 최소한의 마크다운(굵게/코드/문단나눔)만 HTML로 변환한다."""
    html = desc.replace("\n\n", "<br><br>")
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
    return html


def render_glossary() -> None:
    st.title("용어 해석")
    st.caption("데모에서 사용되는 추천 시스템, 그래프, 페르소나 관련 주요 용어를 정리합니다.")

    for section_title, terms in _SECTIONS:
        icon = _SECTION_ICONS.get(section_title, "")
        st.markdown(
            f'<div class="glossary-section-title">{icon} {section_title}</div>',
            unsafe_allow_html=True,
        )
        cards_html = "".join(
            f'<div class="glossary-term-card">'
            f'<div class="glossary-term-name">{term}</div>'
            f'<div class="glossary-term-desc">{_desc_html(desc)}</div>'
            f"</div>"
            for term, desc in terms
        )
        st.markdown(f'<div class="glossary-grid">{cards_html}</div>', unsafe_allow_html=True)
