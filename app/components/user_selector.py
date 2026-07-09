import pandas as pd
import streamlit as st

# data/processed/segment_personas_train_only.json (KMeans 6-세그먼트, ALS와 동일
# train cutoff 2025-08-01 기준) 확정 결과 요약. 세그먼트명(persona_label)은 영문
# 원문을 그대로 쓰고, 설명만 한국어로 옮겼다. pct는 페르소나 카드의 "세그먼트 N%"
# 배지로 별도 표시하므로 desc 본문에서는 뺐다(요청 반영 — 배지+텍스트 중복 제거).
PERSONA_META: dict[str, dict] = {
    "Frequent Viewers with Consistent Purchases": {
        "pct": 19.5,
        "desc": "자주 조회·장바구니 담기를 하고 세션마다 꾸준히 구매하며, 조회 카테고리와 구매 카테고리가 거의 일치(매치율 100%)합니다.",
    },
    "Non-Purchasing Browsers": {
        "pct": 14.9,
        "desc": "조회·장바구니 활동은 활발하지만 구매로 이어진 적이 없는(구매율 0%) 유저군입니다.",
    },
    "Low-Engagement Non-Purchasers": {
        "pct": 5.3,
        "desc": "조회량이 적고 구매 전환도 거의 없으며(0.2%), 비활성 기간이 가장 긴 유저군입니다.",
    },
    "Frequent Browsers with Occasional Purchases": {
        "pct": 30.1,
        "desc": "여러 카테고리를 폭넓게 조회하지만 구매는 특정 카테고리에 편중되고, 조회-구매 카테고리 일치율은 0%에 가깝습니다.",
    },
    "High-Engagement Occasional Purchasers": {
        "pct": 9.9,
        "desc": "조회·장바구니 참여도는 높지만 구매는 간헐적이며, 조회-구매 카테고리 일치율은 52.5%로 중간 수준입니다.",
    },
    "High-Engagement Repeat Purchasers": {
        "pct": 20.4,
        "desc": "조회량·주문 횟수·구매 카테고리 다양성이 모두 높은 반복구매 유저군이지만, 최근 활동은 뜸한 편입니다.",
    },
}

# 드롭다운에서 영문 세그먼트명 옆에 짧게 붙일 한글 한 줄 해석 — PERSONA_META의 desc(상세
# 설명)와 별개로, 선택 전에 뜻을 훑어볼 수 있게 짧은 요약만 괄호로 덧붙인다(요청 반영).
PERSONA_KO: dict[str, str] = {
    "Frequent Viewers with Consistent Purchases": "꾸준한 구매형",
    "Non-Purchasing Browsers": "구경만 하는 유저",
    "Low-Engagement Non-Purchasers": "저활동 비구매",
    "Frequent Browsers with Occasional Purchases": "가끔 사는 탐색형",
    "High-Engagement Occasional Purchasers": "간헐적 구매형",
    "High-Engagement Repeat Purchasers": "반복구매 충성형",
}


def render_persona_and_user_selector(demo_users_df: pd.DataFrame) -> tuple[int, dict]:
    """메인 화면(유저 소개 영역)에 페르소나 선택과 유저 선택을 나란히(2열) 렌더링.

    (user_id, 유저 정보 dict) 반환.
    """
    personas = sorted(demo_users_df["persona_label"].unique().tolist())

    # 페르소나명이 영문 전체 문구(최대 44자)라 균등 2열이면 잘린다 — 유저 선택 쪽은
    # "User 00259 | HEAVY | 로그 43건"처럼 짧아서 폭이 덜 필요하다(요청 반영).
    col_persona, col_user = st.columns([2, 1])

    with col_persona:
        selected_persona = st.selectbox(
            "페르소나 선택",
            options=personas,
            format_func=lambda p: f"{p} ({PERSONA_KO.get(p, '')})" if PERSONA_KO.get(p) else p,
            key="selected_persona",
        )

    persona_users = demo_users_df[demo_users_df["persona_label"] == selected_persona]
    options = persona_users["user_id"].tolist()
    # "User "/"로그"/"건" 같은 수식어를 빼고 값만 나열 — 페르소나 쪽에 폭을 더 줘야 해서
    # 유저 선택 라벨은 짧을수록 좋다(요청 반영: 둘 다 안 잘리게).
    label_map = {
        int(
            row["user_id"]
        ): f"{int(row['user_id']):05d} · {row['user_type'].upper()} · {int(row['log_count']):,}건"
        for _, row in persona_users.iterrows()
    }

    with col_user:
        # 페르소나별로 key를 다르게 줘서(페르소나가 바뀌면 완전히 새 위젯으로 취급) 이전
        # 페르소나의 선택값이 옵션 목록이 바뀐 뒤에도 드롭다운 라벨에 남아있는 Streamlit
        # 프론트엔드 표시 지연 버그를 피한다. 같은 key를 재사용하면 실제 선택값(session_state)은
        # 바로 바뀌는데도 화면에 보이는 라벨 텍스트만 한 박자 늦게(다음 상호작용 때) 갱신됐다.
        selected_id = st.selectbox(
            "유저 선택",
            options=options,
            format_func=lambda uid: label_map[int(uid)],
            key=f"selected_user__{selected_persona}",
        )

    # 상세 페이지 등 다른 화면에서도 조회할 수 있도록 안정적인 키에 현재 선택값을 복사해 둔다.
    st.session_state["selected_user"] = int(selected_id)

    user_row = persona_users[persona_users["user_id"] == selected_id].iloc[0]
    user_type_label = "Heavy" if str(user_row["user_type"]).lower() == "heavy" else "Cold"
    meta = PERSONA_META.get(user_row["persona_label"], {"pct": 0.0, "desc": "설명 없음"})

    user_info = {
        "persona_label": user_row["persona_label"],
        "user_type_label": user_type_label,
        "log_count": int(user_row["log_count"]),
        "persona_pct": meta["pct"],
        "persona_desc": meta["desc"],
    }

    return int(selected_id), user_info


def get_current_user_selection(demo_users_df: pd.DataFrame) -> tuple[int, dict]:
    """현재 session_state에 저장된 페르소나/유저 선택값을 렌더링 없이 반환한다.

    추천 비교 화면에서 선택 UI를 별도 하위 페이지로 분리해도, 카드 비교 화면은 같은
    선택값을 계속 참조해야 하므로 렌더링 없는 조회 함수가 필요하다.
    """
    personas = sorted(demo_users_df["persona_label"].unique().tolist())
    selected_persona = st.session_state.get("selected_persona")
    if selected_persona not in personas:
        selected_persona = personas[0]
        st.session_state["selected_persona"] = selected_persona

    persona_users = demo_users_df[demo_users_df["persona_label"] == selected_persona]
    options = [int(uid) for uid in persona_users["user_id"].tolist()]
    user_key = f"selected_user__{selected_persona}"
    selected_id = int(st.session_state.get(user_key, options[0]))
    if selected_id not in options:
        selected_id = options[0]
        st.session_state[user_key] = selected_id
    st.session_state["selected_user"] = selected_id

    user_row = persona_users[persona_users["user_id"] == selected_id].iloc[0]
    user_type_label = "Heavy" if str(user_row["user_type"]).lower() == "heavy" else "Cold"
    meta = PERSONA_META.get(user_row["persona_label"], {"pct": 0.0, "desc": "설명 없음"})
    user_info = {
        "persona_label": user_row["persona_label"],
        "user_type_label": user_type_label,
        "log_count": int(user_row["log_count"]),
        "persona_pct": meta["pct"],
        "persona_desc": meta["desc"],
    }
    return selected_id, user_info


def render_user_summary_card(
    user_id: int,
    user_info: dict,
    twiddler_status: str | None = None,
) -> None:
    """개별 유저 카드 + 페르소나 특성 카드를 하나로 통합(요청 반영: 같은 유저에 대한
    정보인데 카드 2개로 나뉘어 있어 중복처럼 보인다는 UI 피드백).

    카드 내부 순서는 페르소나 특성(페르소나가 같으면 유저가 바뀌어도 그대로) → 개별
    유저(유저마다 바뀜)로 둔다(요청 반영: 이 카드 위의 페르소나 선택과, 아래의 Twiddler
    재랭킹 근거가 각각 "페르소나 단위"·"유저 단위" 정보라 그 사이에 낀 이 카드도 같은
    기준으로 위/아래를 나눠야 전체 화면이 페르소나 정보 묶음 → 유저별로 바뀌는 정보
    묶음 순서로 일관된다).
    twiddler_status가 없으면(모델/phase 미확정 시점) 그 부분만 생략한다.
    """
    persona_label = user_info["persona_label"]
    persona_ko = PERSONA_KO.get(persona_label, "")
    persona_name = f"{persona_label} ({persona_ko})" if persona_ko else persona_label
    status_part = f" · Twiddler: {twiddler_status}" if twiddler_status else ""
    st.markdown(
        f'<div class="user-summary-card">'
        f'<div class="persona-card-name">{persona_name}</div>'
        f'<div class="persona-card-desc">{user_info["persona_desc"]}</div>'
        f'<div class="user-summary-divider"></div>'
        f'<div class="user-summary-header">'
        f'<div class="user-summary-avatar">{user_id}</div>'
        f"<div>"
        f'<div class="user-summary-label">🧑 개별 유저</div>'
        f'<div class="user-summary-name">User {user_id:03d} · {persona_label}</div>'
        f'<div class="user-summary-meta">'
        f'{user_info["user_type_label"].upper()} · 로그 {user_info["log_count"]:,}건{status_part}'
        f"</div></div>"
        f'<span class="badge badge-segment user-summary-segment">'
        f'세그먼트 {user_info["persona_pct"]:.1f}%</span>'
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
