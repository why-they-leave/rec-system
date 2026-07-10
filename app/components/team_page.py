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
        # 본인이 직접 준 문구로 소개 문단 첫 문장만 교체(요청 반영) — 나머지 팀원은
        # affiliations 기반 자동 생성 문장을 그대로 쓴다.
        "bio_intro": "이화여자대학교 통계학과 · 불어불문과를 다니고 있는 24기 박현서입니다!",
        "goal": "데이터를 전략의 뼈대로 세우고 싶다",
        "goal_note": "이탈 문제의 궁극적 해결책",
        "roles": ["데이터 전처리, EDA", "보완재 파이프라인"],
        # 팀 발표 자료("07. 결론 - 느낀 점") 내용을 본인 1인칭 존댓말 톤으로 옮긴다
        # (요청 반영: 팀 소개 페이지에 발표자료 결론 슬라이드 내용 포함).
        "reflection": (
            "정교한 유저 행동 로그와 품질 높은 상품 정보를 모두 갖춘 데이터셋을 찾기 "
            "어려워, 프로젝트의 한계도 비교적 명확했습니다. 추천 로직을 더 깊이 "
            "검증하고 풍부한 인사이트를 끌어내기에는 데이터 측면의 제약이 컸다는 점이 "
            "아쉬웠습니다."
        ),
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
        "reflection": (
            "퍼널 데이터가 잘 확보되면서도 프로젝트 진행에 결함이 없는 데이터셋을 찾기가 "
            "쉽지 않았습니다. 다양한 데이터셋을 고려했지만 끝내 해결하지 못해 한계점으로 "
            "남기게 된 점이 조금 아쉽습니다."
        ),
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
        "reflection": (
            "데이터 특성상 상품 이미지나 유저 정보가 누락된 경우가 많아, 깊이 있는 "
            "인사이트를 끌어내는 데에는 한계가 있었습니다. 그럼에도 LLM을 활용해 제한된 "
            "데이터에 설명력을 더하고, 추천 결과의 다양성을 보여줄 새로운 접근을 "
            "시도했다는 점에서 의미 있는 경험이었습니다."
        ),
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
        "reflection": (
            "페르소나를 추출하기에 적합한 데이터셋을 찾는 일이 가장 어려웠습니다. "
            "그럼에도 정량 위주 추천을 넘어 정성적 보완을 시도했다는 점에서 의미가 "
            "깊었던 프로젝트였습니다."
        ),
    },
]


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
        # 발표자료("07. 결론 - 느낀 점") 내용으로 소개 문단을 교체한다(요청 반영:
        # "지금 들어가있는 문장을 저걸로 교체해달라는 건" — 별도 카드로 추가하는 게
        # 아니라 기존 자기소개 문단 자리를 대체).
        st.markdown(f'<p class="team-bio-text">{member["reflection"]}</p>', unsafe_allow_html=True)
        _render_github_link(member)
        st.markdown('<div class="team-bio-spacer"></div>', unsafe_allow_html=True)
