"""프로젝트 소개 페이지 — 이 프로젝트를 왜, 어떻게 만들었는지 설명한다.

데모를 어떻게 쓰는지는 components.project_intro("데모 안내")로 분리했다(요청 반영:
"차라리 프로젝트에서 우리가 뭘 한건지 소개하고, 데모소개를 따로 빼줘").
"""

from textwrap import dedent

import streamlit as st

# 팀 발표 자료(Canva, "01. 문제의식")의 문제정의 슬라이드를 이 사이트 톤(1인칭, 평이한
# 말투)에 맞게 재구성한다(요청 반영: "피피티의 내용을 그대로 써달라는게 아니야. 적절히
# 재구성해줘" — 슬라이드 문구를 그대로 옮기지 않고 세 카드의 문장 구조도 서로 다르게 썼다).
_PROBLEM_POINTS = [
    (
        "데이터는 쌓이는데 맥락이 없다",
        "클릭·구매 로그는 숫자로만 남아서, 유저가 그 상품을 왜 원했는지는 알려주지 않습니다.",
    ),
    (
        "과거 이력에 갇힌 추천",
        "구매 확률 계산에만 의존하다 보니, 겉으로 드러나지 않는 취향과 소비 맥락은 "
        "반영하지 못합니다.",
    ),
    (
        "설명할 수 없는 임베딩",
        "유저 행동을 압축한 숫자 벡터만으로는, 왜 이 상품을 추천했는지 사람이 이해할 "
        "수 있는 말로 설명하기 어렵습니다.",
    ),
]

# 팀 발표 자료("02. 프로젝트 핵심 질문")의 질문·가설을 이 사이트 톤으로 재구성한다(요청
# 반영). "핵심 질문 N"/"가설 N" 같은 발표자료식 라벨 대신, 질문과 기대효과를 한 카드 안에
# 자연스러운 문장으로 묶었다.
_KEY_QUESTIONS = [
    (
        "추천 성능이 실제로 좋아질까",
        "생성한 페르소나를 기존 추천 알고리즘에 결합하면 추천이 더 정확해지는가?",
        "페르소나를 반영한 유저는 상품을 더 빨리 찾고, 클릭에서 구매로 이어지는 비율도 "
        "높아질 것이라 기대했습니다.",
    ),
    (
        "페르소나가 유저를 더 잘 표현할까",
        "LLM이 만든 페르소나가 기존 임베딩보다 유저의 소비 성향과 라이프스타일을 "
        "더 풍부하게 담아내는가?",
        "유저의 행동 흐름을 해석한 페르소나가 단순 임베딩보다 취향과 생활 방식을 더 "
        "정확히 반영할 것이라 기대했습니다.",
    ),
    (
        "효과가 오래 갈까",
        "페르소나 기반 추천은 시간이 지나도 품질과 만족도를 유지할 수 있는가?",
        "페르소나는 순간의 클릭이 아니라 장기적인 소비 성향을 반영하므로, 시간이 "
        "지나도 추천 일관성과 만족도가 기존보다 안정적으로 유지될 것이라 기대했습니다.",
    ),
]


def render_project_about() -> None:
    st.title("프로젝트 소개")

    # ── 문제정의 — 왜 이 프로젝트를 시작했는지(요청 반영: 팀 발표 자료 "01. 문제의식"
    # 슬라이드 내용 포함) ──
    st.markdown(
        '<div class="intro-section-label">왜 이 프로젝트를 시작했나요?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        dedent(
            """
            <p class="intro-about-text">
                상품 수가 급증할수록 유저는 원하는 상품을 찾기 더 어려워지고, 이는
                서비스 이탈로 이어집니다. 개인화 추천이 그 해답으로 떠올랐지만, 저희가
                살펴본 기존 방식에는 아직 풀리지 않은 한계가 있었습니다.
            </p>
            """
        ).strip(),
        unsafe_allow_html=True,
    )
    problem_cols = st.columns(3)
    for col, (title, desc) in zip(problem_cols, _PROBLEM_POINTS):
        with col:
            st.markdown(
                dedent(
                    f"""
                <div class="intro-problem-card">
                    <strong>{title}</strong>
                    <p>{desc}</p>
                </div>
                """,
                ).strip(),
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='intro-divider'>", unsafe_allow_html=True)

    # ── 프로젝트 동기 ──
    st.markdown(
        '<div class="intro-section-label">우리는 이런 걸 하고자 했어요</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        dedent(
            """
            <p class="intro-about-text">
                탐색 비용(원하는 상품을 찾기까지 드는 시간과 시행착오)을 줄이는 것이
                목표였습니다.
            </p>
            <p class="intro-about-text">
                ALS·LightGCN으로 만든 추천 후보를 페르소나 정보로 재정렬(Twiddler)했을
                때, 추천이 실제로 더 잘 맞고 다양해지는지를 수치와 화면으로 직접
                검증하고자 했습니다.
            </p>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='intro-divider'>", unsafe_allow_html=True)

    # ── 확인하고 싶었던 것 — 프로젝트를 이끈 질문(요청 반영: 팀 발표 자료 "02. 프로젝트
    # 핵심 질문" 슬라이드 내용을 이 사이트 톤으로 재구성). "핵심 질문 N"/"가설 N" 같은
    # 발표자료식 라벨은 빼고, 질문과 기대효과를 한 카드 안에 자연스러운 문장으로 묶었다. ──
    st.markdown('<div class="intro-section-label">확인하고 싶었던 것</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="intro-question-summary">실제 서비스 환경의 제약으로 두 번째·세 번째 '
        "질문은 정성적으로 확인하고, 첫 번째 질문(추천 성능)만 HR@K로 직접 "
        "측정했습니다.</p>",
        unsafe_allow_html=True,
    )
    question_cols = st.columns(3)
    for i, (col, (title, question, expectation)) in enumerate(
        zip(question_cols, _KEY_QUESTIONS), start=1
    ):
        with col:
            st.markdown(
                dedent(
                    f"""
                <div class="intro-question-card">
                    <span class="intro-question-num">{i}</span>
                    <strong>{title}</strong>
                    <p class="intro-question-text">{question}</p>
                    <p class="intro-question-expect">{expectation}</p>
                </div>
                """
                ).strip(),
                unsafe_allow_html=True,
            )

    # ── 모델 실험(검증) / 최종 설계(의사결정) — 바로 위 "확인하고 싶었던 것"(질문)과
    # 다른 파트인데도 구분 없이 이어져 있었다(요청 반영: "여기 파트 나눠야지. 모델
    # 실험이랑 최종 설계 부분은 확인하고 싶었던 것 아래 있을 항목들이 아니잖아") —
    # 별도 섹션 라벨로 명확히 나눈다. "성능이 항상 개선됐다"가 아니라 "불안정해서
    # 역할을 분리했다"는 결론으로 모델 실험과 최종 설계를 나눠 보여준다. ──
    st.markdown(
        '<div class="intro-section-label">검증 결과와 최종 설계</div>',
        unsafe_allow_html=True,
    )
    col_exp, col_final = st.columns(2, gap="large")
    with col_exp:
        st.markdown(
            dedent(
                """
            <div class="intro-final-note">
                <span>모델 실험</span>
                <strong>bi-graph vs. tri-graph 성능 비교</strong>
                <p>
                    페르소나 노드를 포함하지 않은 bi-graph와 페르소나 노드를 추가한
                    tri-graph의 HR@K와 NDCG@K를 비교했습니다.
                </p>
                <p>
                    일부 설정에서는 tri-graph의 성능이 개선됐지만, 하이퍼파라미터에
                    따른 변동성이 커 일관된 개선이라고 판단하기 어려웠습니다.
                </p>
                <p>
                    따라서 최종 데모에서는 페르소나를 후보 생성 모델에 직접
                    결합하기보다, 안정적으로 생성된 후보를 후처리 단계에서
                    재정렬하는 방식을 사용했습니다.
                </p>
            </div>
            """
            ).strip(),
            unsafe_allow_html=True,
        )
    with col_final:
        st.markdown(
            dedent(
                """
            <div class="intro-final-note">
                <span>최종 설계</span>
                <strong>안정적인 후보 생성 + 설명 가능한 후처리 재랭킹</strong>
                <ol class="intro-final-list">
                    <li>ALS와 LightGCN으로 추천 후보를 생성합니다.</li>
                    <li>Twiddler로 페르소나와 노출 이력을 반영합니다.</li>
                    <li>추천 순위가 변경된 이유를 화면에 함께 제공합니다.</li>
                </ol>
            </div>
            """
            ).strip(),
            unsafe_allow_html=True,
        )
