"""데모 안내 페이지 — 프로젝트가 왜/어떻게 만들어졌는지가 아니라, 이 데모를 어떻게
체험하면 되는지에만 집중한다(요청 반영: 프로젝트 소개와 데모 안내를 별도 페이지로
분리). 프로젝트 동기·모델 실험·최종 설계는 components.project_about으로 옮겼다.

페이지 순서: 체험 유도(히어로+사용 방법) → 결과 비교 예시 → 시스템 파이프라인 →
체험 포인트.
"""

from textwrap import dedent

import streamlit as st
from utils.product_icons import icon_url

_COMPARE_PROFILE = [
    ("Persona", "Frequent Browser"),
    ("Activity Level", "Heavy"),
    ("Preferred Category", "Electronics"),
    ("Purchase Tendency", "Low Conversion"),
]

_BEFORE_ITEMS = [
    ("headphones", "Noise Cancelling Headphones", "Electronics", "0.84"),
    ("coffee_maker", "Compact Coffee Maker", "Home & Kitchen", "0.81"),
    ("sneakers", "Daily Running Sneakers", "Fashion", "0.79"),
]

# (slug, 상품명, 카테고리, 순위변화 방향, 변동 폭, 재랭킹 근거) — Before 대비 순위가
# 바뀐 이유까지 같이 보여준다(요청 반영: "왜 순위가 바뀌었는지를 같이 보여주는 것이
# 이 프로젝트의 강점").
_AFTER_ITEMS = [
    ("sneakers", "Daily Running Sneakers", "Fashion", "up", 2, "선호 카테고리"),
    ("headphones", "Noise Cancelling Headphones", "Electronics", "down", 1, "최근 노출 1회"),
    ("coffee_maker", "Compact Coffee Maker", "Home & Kitchen", "down", 1, "카테고리 가중치 낮음"),
]

_USAGE_STEPS = [
    ("01", "유저 선택", "활동량과 구매 성향이 다른 유저를 선택합니다."),
    ("02", "재랭킹 비교", "Twiddler를 켜고 추천 순위가 어떻게 달라지는지 확인합니다."),
    ("03", "상품 탐색", "상품을 선택해 함께 구매할 가능성이 높은 보완재까지 살펴봅니다."),
]

# 상단 nav 탭(추천 비교/효과 해석/유저 목록)이 아니라 "이 데모에서 체험할 수 있는
# 행동" 기준으로 묶는다(요청 반영: 탭·실험·UI 기능이 섞여 있던 기존 목록의 추상화
# 수준을 통일).
_WATCH_ITEMS = [
    ("순위 변화", "재랭킹 전후에 어떤 상품이 올라가고 내려가는지 확인합니다."),
    ("반복 노출 감소", "반복 방문 시 이미 본 상품의 노출 순위가 낮아지는지 확인합니다."),
    ("추천 근거", "페르소나와 노출 이력이 각 상품의 순위에 미친 영향을 확인합니다."),
    ("보완재 연결", "메인 추천 상품에서 관련 보완재 추천으로 이어지는 흐름을 체험합니다."),
]


def _go_to_user_select() -> None:
    """유저 선택 화면으로 이동(요청 반영: 히어로/사용법 CTA 버튼이 실제로 데모를 시작하게)."""
    st.session_state["main_tab"] = "rerank"
    st.session_state["view"] = "main"
    st.session_state["rerank_page"] = "user"


def _render_before_rows() -> str:
    rows = []
    for rank, (slug, name, category, score) in enumerate(_BEFORE_ITEMS, start=1):
        rows.append(
            dedent(
                f"""
            <div class="rank-summary-row">
                <div class="rank-summary-rank">{rank}</div>
                <img src="{icon_url(slug)}" alt="" style="width:26px;height:26px;
                    object-fit:contain;display:block;" />
                <div class="rank-summary-copy">
                    <strong>{name}</strong>
                    <span>{category}</span>
                </div>
                <em>{score}</em>
            </div>
            """
            ).strip()
        )
    return "".join(rows)


def _render_after_rows() -> str:
    rows = []
    for rank, (slug, name, category, direction, steps, reason) in enumerate(_AFTER_ITEMS, start=1):
        arrow = "▲" if direction == "up" else "▼"
        rows.append(
            dedent(
                f"""
            <div class="rank-summary-row">
                <div class="rank-summary-rank">{rank}</div>
                <img src="{icon_url(slug)}" alt="" style="width:26px;height:26px;
                    object-fit:contain;display:block;" />
                <div class="rank-summary-copy">
                    <strong>{name}</strong>
                    <span>{category}</span>
                </div>
                <span class="rank-summary-delta rank-summary-delta-{direction}">
                    {arrow}{steps}단계</span>
                <span class="intro-compare-reason">{reason}</span>
            </div>
            """
            ).strip()
        )
    return "".join(rows)


def _render_compare_profile() -> str:
    items = "".join(
        f'<div class="intro-compare-profile-item"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in _COMPARE_PROFILE
    )
    return f'<div class="intro-compare-profile">{items}</div>'


def render_project_intro() -> None:
    st.title("데모 안내")

    # ── Hero + 사용 방법: 기술명(ALS/LightGCN/Twiddler 등)보다 "무엇을 조작하고
    # 무엇이 바뀌는지"부터 보여준다(요청 반영: "첫 화면은 기술 설명보다 행동 중심으로").
    # 사용 방법(01/02/03)을 히어로 바로 아래로 올려 하나의 블록으로 묶었고(요청 반영:
    # "이것도 차라리 맨 위로 올리던가"), CTA 버튼은 제목과 같은 줄에 배치한다(요청
    # 반영: "보라색 버튼을 차라리 제목 옆에 넣는게 나을 것 같은데"). ──
    col_title, col_button = st.columns([3, 1.1], vertical_alignment="center")
    with col_title:
        st.markdown(
            '<h2 class="intro-hero-title">같은 추천 후보도, 유저에 따라 순위는 ' "달라집니다</h2>",
            unsafe_allow_html=True,
        )
    with col_button:
        if st.button("데모 시작하기", key="intro_hero_start_btn", type="primary"):
            _go_to_user_select()
            st.rerun()
    st.markdown(
        '<p class="intro-hero-text">ALS와 LightGCN이 생성한 추천 후보에 유저의 '
        "페르소나와 최근 노출 이력을 반영해 순위를 다시 조정합니다.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="intro-hero-note">이 데모사이트에서는 추천 모델 자체보다, 생성된 '
        "추천 결과를 어떻게 조정하고 설명하는지를 확인할 수 있습니다.</p>",
        unsafe_allow_html=True,
    )
    step_cols = st.columns(3)
    for col, (num, title, desc) in zip(step_cols, _USAGE_STEPS):
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

    # ── 추천 결과 비교: Before/After를 실제 예시 수치로 보여주고, 순위가 왜
    # 바뀌었는지 근거까지 같이 노출한다(요청 반영: "매칭 96%"처럼 오해 소지가 있는
    # 표현 대신 재랭킹 점수·순위변화·근거로 표기) ──
    st.markdown('<div class="intro-section-label">추천 결과 비교</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="intro-compare-note">각 상품 카드에서 기존 순위, 변경된 순위, 최종 '
        "재랭킹 점수, 순위가 변경된 근거를 확인할 수 있습니다.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(_render_compare_profile(), unsafe_allow_html=True)
    col_before, col_after = st.columns(2, gap="medium")
    with col_before:
        st.markdown(
            '<div class="intro-compare-col-title">Twiddler 적용 전</div>'
            '<div class="rank-summary-panel">' + _render_before_rows() + "</div>",
            unsafe_allow_html=True,
        )
    with col_after:
        st.markdown(
            '<div class="intro-compare-col-title">Twiddler 적용 후</div>'
            '<div class="rank-summary-panel">' + _render_after_rows() + "</div>",
            unsafe_allow_html=True,
        )

    # ── 추천 결과가 만들어지는 과정: 보완재 추천을 후보 생성과 같은 단계로 묶으면
    # 역할이 모호해진다는 피드백(요청 반영) — 재랭킹 다음 단계로 분리한 5단계 파이프라인 ──
    st.markdown(
        '<div class="intro-section-label">추천 결과가 만들어지는 과정</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        dedent(
            """
        <div class="intro-flow-strip">
            <div><span>행동 로그</span><strong>조회 · 장바구니 · 구매</strong></div>
            <i></i>
            <div><span>추천 후보 생성</span><strong>ALS · LightGCN</strong></div>
            <i></i>
            <div><span>Twiddler 재랭킹</span><strong>페르소나 · 선호 카테고리 · 노출 이력</strong></div>
            <i></i>
            <div><span>추천 결과</span><strong>순위 · 점수 · 재랭킹 근거</strong></div>
            <i></i>
            <div><span>보완재 추천</span><strong>함께 탐색할 상품</strong></div>
        </div>
        """,
        ).strip(),
        unsafe_allow_html=True,
    )

    # ── 데모에서 확인할 수 있는 것: 체험(이 섹션)과 실험(아래 "모델 실험" 섹션)을
    # 분리한다(요청 반영: "체험 영역과 실험 영역을 분리하는 편이 좋습니다") ──
    st.markdown(
        '<div class="intro-section-label">데모에서 확인할 수 있는 것</div>',
        unsafe_allow_html=True,
    )
    watch_cols = st.columns(4)
    for col, (title, desc) in zip(watch_cols, _WATCH_ITEMS):
        with col:
            st.markdown(
                dedent(
                    f"""
                <div class="intro-check-card">
                    <strong>{title}</strong>
                    <p>{desc}</p>
                </div>
                """
                ).strip(),
                unsafe_allow_html=True,
            )
