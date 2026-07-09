"""유저 목록 페이지 — 데모 유저 28명을 한 번에 필터·검색하며 훑어보는 표 화면.

참고 이미지(어드민 청구 관리 테이블)의 "상태 배지/청구 정보" 같은 도메인 특화 컬럼은
우리 데이터에 없어 만들지 않고, 실제로 가진 값(페르소나, 유형, 로그건수)만으로 채운다.
행을 클릭하면 그 유저를 선택한 채 Twiddler 재랭킹 화면으로 바로 이동한다.
"""

import pandas as pd
import streamlit as st
from components.user_selector import PERSONA_KO


def _persona_display(label: str) -> str:
    ko = PERSONA_KO.get(label, "")
    return f"{label} ({ko})" if ko else label


def render_user_list(demo_users_df: pd.DataFrame) -> None:
    st.title("유저 목록")
    st.caption(f"데모 유저 전체 {len(demo_users_df)}명입니다. 행을 클릭하면 그 유저로 이동합니다.")

    # 카테고리 필터(_apply_filters)와 동일한 UX 규칙: 기본값은 "아무것도 선택 안 함 = 전체".
    # multiselect에 옵션 전체를 default로 미리 채우면 pill이 다 채워져 지저분해 보인다는
    # 피드백 반영 — 빈 선택으로 시작해 고른 것만 필터링한다.
    personas = sorted(demo_users_df["persona_label"].unique().tolist())
    col_persona, col_type, col_search = st.columns([2, 1, 1])
    with col_persona:
        selected_personas = st.multiselect(
            "페르소나",
            options=personas,
            format_func=_persona_display,
            key="user_list_persona_filter",
            placeholder="전체",
        )
    with col_type:
        selected_types = st.multiselect(
            "유형",
            options=["heavy", "cold"],
            format_func=lambda t: t.upper(),
            key="user_list_type_filter",
            placeholder="전체",
        )
    with col_search:
        search_id = st.text_input("유저 ID 검색", key="user_list_search", placeholder="예: 259")

    filtered = demo_users_df
    if selected_personas:
        filtered = filtered[filtered["persona_label"].isin(selected_personas)]
    if selected_types:
        filtered = filtered[filtered["user_type"].isin(selected_types)]
    if search_id.strip():
        filtered = filtered[filtered["user_id"].astype(str).str.contains(search_id.strip())]

    if filtered.empty:
        st.info("조건에 맞는 유저가 없습니다.")
        return

    display_df = pd.DataFrame(
        {
            "유저 ID": filtered["user_id"].apply(lambda x: f"{x:05d}"),
            "페르소나": filtered["persona_label"].apply(_persona_display),
            "유형": filtered["user_type"].str.upper(),
            "로그 건수": filtered["log_count"],
        }
    )

    event = st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="user_list_table",
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        picked_user_id = int(filtered.iloc[selected_rows[0]]["user_id"])
        st.success(f"User {picked_user_id:05d} 선택됨. 추천 비교 화면으로 이동합니다.")
        st.session_state["selected_persona"] = filtered.iloc[selected_rows[0]]["persona_label"]
        st.session_state[f"selected_user__{st.session_state['selected_persona']}"] = picked_user_id
        st.session_state["selected_user"] = picked_user_id
        st.session_state["main_tab"] = "rerank"
        st.session_state["view"] = "main"
        st.rerun()
