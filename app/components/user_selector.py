import pandas as pd
import streamlit as st

# data/processed/segment_personas_train_only.json (KMeans 6-세그먼트, ALS와 동일
# train cutoff 2025-08-01 기준) 확정 결과 요약. 세그먼트명(persona_label)은 영문
# 원문을 그대로 쓰고, 설명만 한국어로 옮겼다.
PERSONA_DESC: dict[str, str] = {
    "Frequent Viewers with Consistent Purchases": "전체의 19.5%. 자주 조회·장바구니 담기를 하고 세션마다 꾸준히 구매하며, 조회 카테고리와 구매 카테고리가 거의 일치(매치율 100%)합니다.",
    "Non-Purchasing Browsers": "전체의 14.9%. 조회·장바구니 활동은 활발하지만 구매로 이어진 적이 없는(구매율 0%) 유저군입니다.",
    "Low-Engagement Non-Purchasers": "전체의 5.3%. 조회량이 적고 구매 전환도 거의 없으며(0.2%), 비활성 기간이 가장 긴 유저군입니다.",
    "Frequent Browsers with Occasional Purchases": "전체의 30.1%(최대 세그먼트). 여러 카테고리를 폭넓게 조회하지만 구매는 특정 카테고리에 편중되고, 조회-구매 카테고리 일치율은 0%에 가깝습니다.",
    "High-Engagement Occasional Purchasers": "전체의 9.9%. 조회·장바구니 참여도는 높지만 구매는 간헐적이며, 조회-구매 카테고리 일치율은 52.5%로 중간 수준입니다.",
    "High-Engagement Repeat Purchasers": "전체의 20.4%. 조회량·주문 횟수·구매 카테고리 다양성이 모두 높은 반복구매 유저군이지만, 최근 활동은 뜸한 편입니다.",
}


def render_persona_and_user_selector(demo_users_df: pd.DataFrame) -> tuple[int, dict]:
    """메인 화면(유저 소개 영역)에 페르소나 선택과 유저 선택을 나란히(2열) 렌더링.

    (user_id, 유저 정보 dict) 반환.
    """
    personas = sorted(demo_users_df["persona_label"].unique().tolist())

    col_persona, col_user = st.columns(2)

    with col_persona:
        selected_persona = st.selectbox(
            "🧭 페르소나 선택",
            options=personas,
            key="selected_persona",
        )

    persona_users = demo_users_df[demo_users_df["persona_label"] == selected_persona]
    options = persona_users["user_id"].tolist()
    label_map = {
        int(row["user_id"]): f"User {int(row['user_id']):05d} | {row['user_type'].upper()} | 로그 {int(row['log_count']):,}건"
        for _, row in persona_users.iterrows()
    }

    with col_user:
        # 페르소나별로 key를 다르게 줘서(페르소나가 바뀌면 완전히 새 위젯으로 취급) 이전
        # 페르소나의 선택값이 옵션 목록이 바뀐 뒤에도 드롭다운 라벨에 남아있는 Streamlit
        # 프론트엔드 표시 지연 버그를 피한다. 같은 key를 재사용하면 실제 선택값(session_state)은
        # 바로 바뀌는데도 화면에 보이는 라벨 텍스트만 한 박자 늦게(다음 상호작용 때) 갱신됐다.
        selected_id = st.selectbox(
            "👤 유저 선택",
            options=options,
            format_func=lambda uid: label_map[int(uid)],
            key=f"selected_user__{selected_persona}",
        )

    # 상세 페이지 등 다른 화면에서도 조회할 수 있도록 안정적인 키에 현재 선택값을 복사해 둔다.
    st.session_state["selected_user"] = int(selected_id)

    st.info(PERSONA_DESC.get(selected_persona, "설명 없음"), icon="📖")

    user_row = persona_users[persona_users["user_id"] == selected_id].iloc[0]
    user_type_label = "Heavy" if str(user_row["user_type"]).lower() == "heavy" else "Cold"

    user_info = {
        "persona_label": user_row["persona_label"],
        "user_type_label": user_type_label,
        "log_count": int(user_row["log_count"]),
    }

    return int(selected_id), user_info
