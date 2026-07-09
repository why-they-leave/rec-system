"""팀 소개 페이지.

Q&A 배너(팀 소개 한 줄) + 원형 사진 그리드(이름만) + 팀원별 자기소개 문단 + GitHub 링크로
구성한다(요청 반영: 참고 이미지 스타일). 자기소개 문단은 기존에 갖고 있던 데이터(소속,
목표, 역할)를 1인칭 서술로 재구성한 것이라 새로운 사실을 지어내지 않는다.
"""

import streamlit as st

TEAM_URL_DIR = "app/static/images/character"  # enableStaticServing 기준 상대 URL(다른 static 이미지와 동일 관례)

TEAM_MEMBERS: list[dict] = [
    {
        "name": "24기 박현서",
        "github": "gustj1819",
        "image": "gustj1819.png",
        "speech": "hyunseo_speech.png",
        "affiliations": ["이화여자대학교", "통계학과", "불어불문학과"],
        "goal": "데이터를 전략의 뼈대로 세우고 싶다",
        "goal_note": "이탈 문제의 궁극적 해결책",
        "roles": ["데이터 전처리, EDA", "보완재 파이프라인"],
    },
    {
        "name": "25기 김지은",
        "github": "j2nii",
        "image": "j2nii.png",
        "speech": "jieun_speech.png",
        "affiliations": ["덕성여자대학교", "디지털소프트웨어공학부"],
        "goal": "직접 추천시스템을 구현하고 경험하고 싶다",
        "goal_note": "이론을 실전으로",
        "roles": ["ALS 모델링", "UI 프로토타입 구현"],
    },
    {
        "name": "25기 이정연",
        "github": "JungYeoni",
        "image": "JungYeoni.png",
        "speech": "jungyeoni_speech.png",
        "affiliations": ["세종대학교", "데이터사이언스학과", "경제학과"],
        "goal": "사용자의 디깅을 돕는 추천을 하고 싶다",
        "goal_note": "탐색 비용을 줄이는 디깅 조력자",
        "roles": ["LightGCN 모델링", "페르소나 생성"],
    },
    {
        "name": "25기 이용혁",
        "github": "eric010314-sys",
        "image": "eric010314-sys.png",
        "speech": "yonghyeok_speech.png",
        "affiliations": ["인하대학교", "산업경영공학과"],
        "goal": "보이지 않는 취향까지 반영하고 싶다",
        "goal_note": "수치를 넘어선 감성공학적 해석",
        "roles": ["데이터 전처리, EDA", "페르소나 생성"],
    },
]


def _member_bio_text(member: dict) -> str:
    """팀원 소개 문단. 소속/목표/역할 데이터를 1인칭 서술로 풀어 쓴다."""
    university, *departments = member["affiliations"]
    roles_text = ", ".join(member["roles"])
    return (
        f'{university} {" · ".join(departments)}에서 온 {member["name"]}입니다. '
        f'"{member["goal"]}"는 마음으로 참여했고({member["goal_note"]}), '
        f"이번 프로젝트에서는 {roles_text}를 맡았습니다."
    )


def _render_github_link(member: dict) -> None:
    # st.link_button은 텍스트만 지원해 GitHub 로고를 못 넣는다 — 실제 로고 이미지
    # (app/static/images/github_logo.png)를 쓰는 커스텀 <a> 앵커로 대체.
    st.markdown(
        f'<a href="https://github.com/{member["github"]}" target="_blank" '
        f'class="team-github-link">'
        f'<img src="app/static/images/github_logo.png" alt="GitHub" />'
        f"GitHub</a>",
        unsafe_allow_html=True,
    )


def render_team_page() -> None:
    st.title("팀 소개")

    st.markdown(
        '<div class="team-banner">'
        '<div class="team-banner-body">'
        "안녕하세요! 추크크 팀입니다. 저희는 빅데이터 연합동아리 투빅스 소속으로, "
        "탐색 비용 감소를 위한 페르소나 기반 개인화 및 연관 상품 재순위화 연구를 함께 "
        "진행했습니다.<br><br>"
        "학교도 전공도 서로 다르지만, "
        '"사용자가 덜 헤매고 더 잘 찾을 수 있게 만들자"는 마음 하나로 모였습니다.'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    members = TEAM_MEMBERS
    photo_cols = st.columns(len(members))
    for col, member in zip(photo_cols, members):
        with col:
            speech_html = ""
            if member.get("speech"):
                # 캐릭터 사진 위에 말풍선을 얹는다(요청 반영) — 사진을 감싼
                # .team-photo-circle-wrap을 기준(position:relative)으로 말풍선을
                # absolute 배치한다(style.css의 .team-photo-speech 참고).
                speech_html = (
                    f'<img src="{TEAM_URL_DIR}/{member["speech"]}" alt="" '
                    f'class="team-photo-speech" />'
                )
            st.markdown(
                f'<div class="team-photo-item">'
                f'<div class="team-photo-circle-wrap">'
                f"{speech_html}"
                f'<img src="{TEAM_URL_DIR}/{member["image"]}" alt="{member["name"]}" '
                f'class="team-photo-circle" />'
                f"</div>"
                f'<div class="team-photo-name">{member["name"]}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="team-divider">', unsafe_allow_html=True)

    for member in members:
        st.markdown(f'<div class="team-bio-name">{member["name"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="team-bio-text">{_member_bio_text(member)}</p>', unsafe_allow_html=True
        )
        _render_github_link(member)
        st.markdown('<div class="team-bio-spacer"></div>', unsafe_allow_html=True)
