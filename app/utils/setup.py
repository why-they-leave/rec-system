"""공통 사이드바 설정 — 모든 페이지에서 최상단에 호출."""
import sys
from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# 카테고리별 이모지 (표시 순서 고정)
CATEGORY_EMOJI: dict[str, str] = {
    "Electronics":    "💻",
    "Home & Kitchen": "🏠",
    "Beauty":         "💄",
    "Sports":         "⚽",
    "Fashion":        "👗",
    "Books":          "📚",
    "Toys":           "🎮",
}
ALL_CATEGORIES = list(CATEGORY_EMOJI.keys())


def common_setup() -> int | None:
    """CSS 주입 + 사이드바(유저 선택 + 카테고리 필터) 렌더링.
    선택된 user_id 반환 (데이터 없으면 None).
    선택된 카테고리 목록은 st.session_state['selected_categories'] 에 저장.
    """
    from utils.style_loader import load_css
    from utils.data_loader import load_demo_users
    from components.user_selector import render_user_selector

    load_css("app/static/style.css")

    st.sidebar.title("🛍️ 추천 시스템 데모")
    st.sidebar.markdown("---")

    # ── 유저 선택 ────────────────────────────────────────────────────────────
    try:
        demo_users = load_demo_users()
        user_id = render_user_selector(demo_users)
    except FileNotFoundError:
        st.sidebar.error("❌ `data/dashboard/demo_users.csv` 없음")
        user_id = None

    # ── 카테고리 필터 ────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🗂️ 카테고리**")

    selected_categories: list[str] = []
    for cat, emoji in CATEGORY_EMOJI.items():
        # session_state 키를 통해 상태 유지
        key = f"cat__{cat}"
        if key not in st.session_state:
            st.session_state[key] = True
        checked = st.sidebar.checkbox(f"{emoji}  {cat}", key=key)
        if checked:
            selected_categories.append(cat)

    # 전부 해제되면 전체 표시 (빈 결과 방지)
    if not selected_categories:
        selected_categories = ALL_CATEGORIES[:]

    st.session_state["selected_categories"] = selected_categories

    return user_id
