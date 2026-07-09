"""팀 소개 페이지.

기여자 카드(캐릭터 이미지 + GitHub 링크 + 소속/목표/역할) + 팀 소개 한 줄로 구성.
캐릭터 이미지는 app/static/images/character/의 기존 4장을 GitHub 계정명으로 매핑해 사용한다.
"""

import streamlit as st

TEAM_URL_DIR = "app/static/images/character"  # enableStaticServing 기준 상대 URL(다른 static 이미지와 동일 관례)

TEAM_MEMBERS: list[dict] = [
    {
        "name": "24기 박현서",
        "github": "gustj1819",
        "image": "gustj1819.png",
        "affiliations": ["이화여자대학교", "통계학과", "불어불문학과"],
        "goal": "데이터를 전략의 뼈대로 <br> 세우고 싶다",
        "goal_note": "이탈 문제의 궁극적 해결책",
        "roles": ["데이터 전처리, EDA", "보완재 파이프라인"],
    },
    {
        "name": "25기 김지은",
        "github": "j2nii",
        "image": "j2nii.png",
        "affiliations": ["덕성여자대학교", "디지털소프트웨어공학부"],
        "goal": "직접 추천시스템을 구현하고 <br> 경험하고 싶다",
        "goal_note": "이론을 실전으로",
        "roles": ["ALS 모델링", "UI 프로토타입 구현"],
    },
    {
        "name": "25기 이정연",
        "github": "JungYeoni",
        "image": "JungYeoni.png",
        "affiliations": ["세종대학교", "데이터사이언스학과", "경제학과"],
        "goal": "사용자의 디깅을 돕는 <br> 추천을 하고 싶다",
        "goal_note": "탐색 비용을 줄이는 디깅 조력자",
        "roles": ["LightGCN 모델링", "페르소나 생성"],
    },
    {
        "name": "25기 이용혁",
        "github": "eric010314-sys",
        "image": "eric010314-sys.png",
        "affiliations": ["인하대학교", "산업경영공학과"],
        "goal": "보이지 않는 취향까지 <br> 반영하고 싶다",
        "goal_note": "수치를 넘어선 감성공학적 해석",
        "roles": ["데이터 전처리, EDA", "페르소나 생성"],
    },
]


def _render_team_card(member: dict) -> None:
    """팀원 한 명의 카드. 원형 아바타는 product_card._circle()과 동일하게 단일 <img> 태그로
    렌더링해 마크다운 파싱 문제 없이 표시한다.
    """
    with st.container(border=True):
        st.markdown(
            f'<div class="team-avatar-wrap">'
            f'<img src="{TEAM_URL_DIR}/{member["image"]}" alt="{member["name"]}" '
            f'class="team-avatar" /></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="team-name">{member["name"]}</div>', unsafe_allow_html=True)
        university, *departments = member["affiliations"]
        st.markdown(
            f'<div class="team-affil">{university}<br>{" · ".join(departments)}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="team-goal">"{member["goal"]}"<br>'
            f'<span class="team-goal-note">({member["goal_note"]})</span></div>',
            unsafe_allow_html=True,
        )

        roles_html = "".join(f"<li>{role}</li>" for role in member["roles"])
        st.markdown(f'<ul class="team-role-list">{roles_html}</ul>', unsafe_allow_html=True)

        # st.link_button은 텍스트만 지원해 GitHub 로고를 못 넣는다 — 요청대로 실제 로고
        # 이미지(app/static/images/github_logo.png)를 쓰는 커스텀 <a> 앵커로 대체
        # (스타일은 style.css의 .team-github-link).
        st.markdown(
            f'<a href="https://github.com/{member["github"]}" target="_blank" '
            f'class="team-github-link">'
            f'<img src="app/static/images/github_logo.png" alt="GitHub" />'
            f'GitHub</a>',
            unsafe_allow_html=True,
        )


def render_team_page() -> None:
    st.title("👥 팀 소개")
    st.markdown(
        '<div class="team-intro">'
        '<span class="team-intro-title">추크크✨ Recommendation Creator Crew</span><br>'
        '<span class="team-intro-desc">탐색 비용 감소를 위한 페르소나 기반 개인화 및 연관 상품 재순위화 연구를 함께 진행한 팀입니다.</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    members = TEAM_MEMBERS
    cols = st.columns(len(members))
    for i, member in enumerate(members):
        with cols[i]:
            _render_team_card(member)
