"""프로젝트 소개 페이지.

설명 문서가 아니라 데모사이트의 체험 진입 화면처럼 보이도록 구성한다.
"""

from textwrap import dedent

import streamlit as st
from utils.product_icons import icon_url

_PREVIEW_PRODUCTS = [
    ("headphones", "Noise Cancelling Headphones", "Electronics", "매칭 96%"),
    ("coffee_maker", "Compact Coffee Maker", "Home & Kitchen", "매칭 91%"),
    ("sneakers", "Daily Running Sneakers", "Fashion", "매칭 87%"),
]

_EXPERIENCE_POINTS = [
    ("01", "유저를 바꾼다", "페르소나와 활동량이 다른 유저를 고르면 추천 후보가 달라집니다."),
    (
        "02",
        "Twiddler를 켠다",
        "같은 후보라도 선호 카테고리와 노출 이력에 따라 순위가 재정렬됩니다.",
    ),
    ("03", "상품을 눌러본다", "메인 추천에서 보완재 추천으로 이어지는 쇼핑 흐름을 확인합니다."),
]

_WATCH_ITEMS = [
    ("순위 변화", "Before / After에서 어떤 상품이 올라오고 내려가는지"),
    ("반복 노출", "새로고침 시뮬레이션에서 같은 상품 반복이 줄어드는지"),
    ("해석 가능성", "페르소나와 그래프가 추천 결과를 설명하는지"),
]


def _render_preview_products() -> str:
    items = []
    for rank, (slug, name, category, match) in enumerate(_PREVIEW_PRODUCTS, start=1):
        items.append(
            dedent(
                f"""
            <div class="intro-preview-product">
                <div class="intro-preview-rank">{rank}</div>
                <img src="{icon_url(slug)}" alt="" />
                <div class="intro-preview-copy">
                    <strong>{name}</strong>
                    <span>{category}</span>
                </div>
                <em>{match}</em>
            </div>
            """
            ).strip()
        )
    return "".join(items)


def render_project_intro() -> None:
    st.title("프로젝트 소개")
    st.caption(
        "추천 결과를 직접 체험하며 재랭킹, 보완재 추천, 페르소나 해석을 확인하는 데모사이트입니다."
    )

    st.markdown(
        dedent(
            f"""
        <section class="intro-demo-stage">
            <div class="intro-demo-copy">
                <h2>추천 모델을 설명하는 페이지가 아니라, 추천 결과를 직접 바꿔보는 화면입니다.</h2>
                <p>
                    이 데모는 유저 선택, Twiddler 재랭킹, 보완재 추천, 페르소나 해석을
                    하나의 쇼핑 경험처럼 이어서 보여줍니다.
                </p>
                <div class="intro-demo-tags">
                    <span>ALS / LightGCN</span>
                    <span>Twiddler</span>
                    <span>Complementary</span>
                    <span>Persona</span>
                </div>
            </div>
            <div class="intro-preview-panel">
                <div class="intro-preview-toolbar">
                    <span>User 00259</span>
                    <div><b>Before</b><strong>After</strong></div>
                </div>
                <div class="intro-preview-persona">
                    <span>Frequent Browser</span>
                    <small>선호 카테고리와 반복 노출 패널티 반영</small>
                </div>
                {_render_preview_products()}
            </div>
        </section>
        """,
        ).strip(),
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='intro-divider'>", unsafe_allow_html=True)

    step_cols = st.columns(3)
    for col, (num, title, desc) in zip(step_cols, _EXPERIENCE_POINTS):
        with col:
            st.markdown(
                dedent(
                    f"""
                <div class="intro-action">
                    <span>{num}</span>
                    <strong>{title}</strong>
                    <p>{desc}</p>
                </div>
                """,
                ).strip(),
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='intro-divider'>", unsafe_allow_html=True)

    st.markdown(
        dedent(
            """
        <div class="intro-flow-strip">
            <div><span>행동 로그</span><strong>조회 · 장바구니 · 구매</strong></div>
            <i></i>
            <div><span>후보 생성</span><strong>ALS · LightGCN · 보완재</strong></div>
            <i></i>
            <div><span>재랭킹</span><strong>페르소나 · 노출 이력</strong></div>
            <i></i>
            <div><span>데모 화면</span><strong>카드 · 순위 · 근거</strong></div>
        </div>
        """,
        ).strip(),
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='intro-divider'>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.05, 1], gap="large")
    with col_left:
        st.markdown(
            '<div class="intro-section-label">데모에서 볼 포인트</div>', unsafe_allow_html=True
        )
        for title, desc in _WATCH_ITEMS:
            st.markdown(
                dedent(
                    f"""
                <div class="intro-watch-row">
                    <strong>{title}</strong>
                    <span>{desc}</span>
                </div>
                """,
                ).strip(),
                unsafe_allow_html=True,
            )

    with col_right:
        st.markdown(
            dedent(
                """
            <div class="intro-final-note">
                <span>최종 방향</span>
                <strong>안정적인 후보 생성 + 해석 가능한 후처리 재랭킹</strong>
                <p>
                    tri-graph 성능 개선은 하이퍼파라미터에 따라 불안정했습니다.
                    그래서 최종 데모는 추천 후보를 안정적으로 만든 뒤, Twiddler로
                    페르소나와 노출 이력을 반영하는 흐름을 보여줍니다.
                </p>
            </div>
            """,
            ).strip(),
            unsafe_allow_html=True,
        )
