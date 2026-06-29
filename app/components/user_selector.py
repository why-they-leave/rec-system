import pandas as pd
import streamlit as st


def render_user_selector(demo_users_df: pd.DataFrame) -> int:
    """사이드바에 유저 드롭다운 + 유저 정보 렌더링 — HTML 없음."""
    options = demo_users_df["user_id"].tolist()
    label_map = {
        int(row["user_id"]): f"User {int(row['user_id']):03d} | {row['persona_label']}"
        for _, row in demo_users_df.iterrows()
    }

    selected_id = st.sidebar.selectbox(
        "👤 유저 선택",
        options=options,
        format_func=lambda uid: label_map[int(uid)],
        key="selected_user",
    )

    user_row = demo_users_df[demo_users_df["user_id"] == selected_id].iloc[0]
    user_type_label = "Heavy" if str(user_row["user_type"]).lower() == "heavy" else "Cold"

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 유저 정보**")
    st.sidebar.write(f"페르소나: **{user_row['persona_label']}**")
    st.sidebar.write(f"유저 유형: **{user_type_label}**")
    st.sidebar.write(f"행동 로그: **{int(user_row['log_count']):,}건**")

    return int(selected_id)
