import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="추천 시스템 데모 v2", page_icon="🛍️", layout="wide")

_ROOT = Path(__file__).parent.parent  # rec-system/
_APP_DIR = _ROOT / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import pandas as pd
from utils.style_loader import load_css
from utils.data_loader import (
    load_demo_users,
    load_recommendations,
    load_detail_recommendations,
    load_products,
    get_user_recommendations,
)
from components.user_selector import render_user_selector
from components.product_card import render_product_card, render_current_product_card
from components.metric_chart import render_bump_chart, COLOR_UP, COLOR_DOWN, COLOR_SAME

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


# ── 사이드바 ───────────────────────────────────────────────────────────────────

def _setup_sidebar() -> int | None:
    """유저 선택 + 화면 이동 버튼 사이드바 렌더링."""
    load_css("app/static/style.css")
    st.sidebar.title("🛍️ 추천 시스템 데모")
    st.sidebar.markdown("---")

    try:
        demo_users = load_demo_users()
        user_id = render_user_selector(demo_users)
    except FileNotFoundError:
        st.sidebar.error("❌ `data/dashboard/demo_users.csv` 없음")
        user_id = None

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🗂️ 화면 이동**")

    current_view = st.session_state.get("view", "main")

    if st.sidebar.button(
        "📊 메인 추천",
        use_container_width=True,
        type="primary" if current_view == "main" else "secondary",
    ):
        st.session_state["view"] = "main"
        st.rerun()

    if st.sidebar.button(
        "🔄 Twiddler 비교",
        use_container_width=True,
        type="primary" if current_view == "twiddler" else "secondary",
    ):
        st.session_state["view"] = "twiddler"
        st.rerun()

    return user_id


# ── 카테고리 필터 ──────────────────────────────────────────────────────────────

def _render_category_filter() -> list[str]:
    """메인 화면 상단 가로 카테고리 필터. 단일 선택, 전체 포함."""
    options = ["전체"] + ALL_CATEGORIES
    labels = {cat: f"{CATEGORY_EMOJI[cat]} {cat}" for cat in ALL_CATEGORIES}
    labels["전체"] = "🏷️ 전체"

    selected = st.radio(
        "카테고리",
        options,
        format_func=lambda c: labels[c],
        horizontal=True,
        key="cat_filter_v2",
        label_visibility="collapsed",
    )

    return ALL_CATEGORIES[:] if selected == "전체" else [selected]


# ── 메인 추천 화면 ─────────────────────────────────────────────────────────────

def _render_main_grid(items_df: pd.DataFrame, common_ids: set, user_id: int, algo: str) -> None:
    if items_df.empty:
        st.info("선택한 카테고리에 해당하는 추천 상품이 없습니다.")
        return
    rows = [items_df.iloc[i : i + 3] for i in range(0, len(items_df), 3)]
    for row_chunk in rows:
        cols = st.columns(3)
        for col, (_, item) in zip(cols, row_chunk.iterrows()):
            with col:
                badge = "🔁 공통 추천" if item["item_id"] in common_ids else None
                render_product_card(item, int(item["rank"]), badge)
                if st.button(
                    "🔍 상세 추천",
                    key=f"detail_{user_id}_{int(item['item_id'])}_{algo}",
                    use_container_width=True,
                ):
                    st.session_state["view"] = "detail"
                    st.session_state["selected_item_id"] = int(item["item_id"])
                    st.rerun()


def _render_main_recommend(user_id: int) -> None:
    st.title("📊 메인 추천")

    try:
        rec_df      = load_recommendations()
        products_df = load_products()
    except FileNotFoundError as e:
        st.error(f"데이터 파일을 찾을 수 없습니다: `{e}`")
        return

    # 알고리즘 토글
    algo = st.radio(
        "알고리즘",
        ["ALS", "LightGCN"],
        horizontal=True,
        key="main_algo_v2",
        captions=["베이스라인", "GNN 기반"],
    )

    # 카테고리 필터
    st.markdown("---")
    selected_categories = _render_category_filter()
    st.markdown("---")

    st.subheader(f"User {user_id:03d}  ·  {algo}")

    recs = get_user_recommendations(rec_df, user_id, algo, "before")
    other_algo = "LightGCN" if algo == "ALS" else "ALS"
    other_recs = get_user_recommendations(rec_df, user_id, other_algo, "before")

    all_ids       = set(recs["item_id"].tolist())
    all_other_ids = set(other_recs["item_id"].tolist())
    common_ids    = all_ids & all_other_ids
    union_ids     = all_ids | all_other_ids
    jaccard       = len(common_ids) / len(union_ids) if union_ids else 0.0

    items = (
        recs.merge(products_df, on="item_id", how="left")
        .pipe(lambda df: df[df["category"].isin(selected_categories)])
    )

    _render_main_grid(items, common_ids, user_id, algo)

    # 하단 지표
    st.divider()
    user_type_val = "—"
    user_rows = rec_df[rec_df["user_id"] == user_id]
    if not user_rows.empty:
        user_type_val = user_rows["user_type"].iloc[0].upper()

    m1, m2, m3 = st.columns(3)
    m1.metric("공통 추천 상품 (vs 반대 알고리즘)", f"{len(common_ids)}개 / 10개")
    m2.metric("Jaccard 유사도", f"{jaccard:.3f}")
    m3.metric("유저 유형", user_type_val)


# ── 상세 추천 화면 ─────────────────────────────────────────────────────────────

def _render_detail_recommend() -> None:
    if st.button("← 돌아가기"):
        st.session_state["view"] = "main"
        st.rerun()

    st.title("🛒 연관 상품 추천")

    item_id = st.session_state.get("selected_item_id")
    if item_id is None:
        st.warning("메인 화면에서 상품 카드를 클릭해 주세요.")
        return

    try:
        detail_df   = load_detail_recommendations()
        products_df = load_products()
    except FileNotFoundError as e:
        st.error(f"데이터 파일을 찾을 수 없습니다: `{e}`")
        return

    current_item = products_df[products_df["item_id"] == item_id]
    if current_item.empty:
        st.warning("상품 정보를 찾을 수 없습니다.")
        return

    render_current_product_card(current_item.iloc[0])

    item_detail = detail_df[detail_df["item_id"] == item_id]

    content_recs = (
        item_detail[item_detail["rec_type"] == "content"]
        .sort_values("rank")
        .head(8)
        .merge(
            products_df,
            left_on="rec_item_id",
            right_on="item_id",
            how="left",
            suffixes=("_detail", ""),
        )
    )
    cf_recs = (
        item_detail[item_detail["rec_type"] == "cf"]
        .sort_values("rank")
        .head(8)
        .merge(
            products_df,
            left_on="rec_item_id",
            right_on="item_id",
            how="left",
            suffixes=("_detail", ""),
        )
    )

    def _render_rec_grid(recs_df: pd.DataFrame, cols_per_row: int = 4) -> None:
        if recs_df.empty:
            st.info("연관 상품 데이터가 없습니다.")
            return
        rows = [recs_df.iloc[i : i + cols_per_row] for i in range(0, len(recs_df), cols_per_row)]
        for row_chunk in rows:
            cols = st.columns(cols_per_row)
            for col, (_, item) in zip(cols, row_chunk.iterrows()):
                with col:
                    render_product_card(item, int(item["rank"]))

    col_content, col_cf = st.columns(2)
    with col_content:
        st.markdown("### 🔄 대체재 (콘텐츠 기반)")
        _render_rec_grid(content_recs)
    with col_cf:
        st.markdown("### 🛒 보완재 (Item-based CF)")
        _render_rec_grid(cf_recs)


# ── Twiddler 화면 ──────────────────────────────────────────────────────────────

def _render_twiddler(user_id: int) -> None:
    st.title("🔄 Twiddler 전/후 비교")

    try:
        rec_df      = load_recommendations()
        products_df = load_products()
    except FileNotFoundError as e:
        st.error(f"데이터 파일을 찾을 수 없습니다: `{e}`")
        return

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        model_type = st.selectbox("모델 선택", ["ALS", "LightGCN"], key="twiddler_model_v2")
    with ctrl2:
        phase = st.radio(
            "Twiddler",
            ["Before", "After"],
            horizontal=True,
            key="twiddler_phase_v2",
            captions=["적용 전", "적용 후"],
        )

    st.subheader(f"User {user_id:03d}  ·  {model_type}  ·  {phase}")

    before_recs = get_user_recommendations(rec_df, user_id, model_type, "before")
    after_recs  = get_user_recommendations(rec_df, user_id, model_type, "after")

    if before_recs.empty or after_recs.empty:
        st.warning("해당 유저의 추천 데이터가 없습니다.")
        return

    # Bump Chart — 항상 전체 순위 변화 표시
    st.markdown("#### 📈 순위 변화 Bump Chart")
    legend_cols = st.columns([1, 1, 1, 4])
    legend_cols[0].markdown(
        f'<span style="color:{COLOR_UP}; font-weight:700;">▲ 순위 상승</span>',
        unsafe_allow_html=True,
    )
    legend_cols[1].markdown(
        f'<span style="color:{COLOR_DOWN}; font-weight:700;">▼ 순위 하락</span>',
        unsafe_allow_html=True,
    )
    legend_cols[2].markdown(
        f'<span style="color:{COLOR_SAME}; font-weight:700;">➡ 유지</span>',
        unsafe_allow_html=True,
    )
    fig = render_bump_chart(before_recs, after_recs)
    st.plotly_chart(fig, width="stretch")

    # 토글에 따라 Before / After 카드 그리드 전환
    st.divider()
    current_recs = before_recs if phase == "Before" else after_recs
    current_items = current_recs.merge(products_df, on="item_id", how="left")

    # 순위 변화 배지 계산
    merged = (
        before_recs[["item_id", "rank"]]
        .rename(columns={"rank": "rank_before"})
        .merge(
            after_recs[["item_id", "rank"]].rename(columns={"rank": "rank_after"}),
            on="item_id",
            how="outer",
        )
    )
    rank_change = {
        int(row["item_id"]): int(row["rank_before"]) - int(row["rank_after"])
        for _, row in merged.dropna().iterrows()
    }

    if current_items.empty:
        st.info("추천 상품 데이터가 없습니다.")
        return

    rows = [current_items.iloc[i : i + 3] for i in range(0, len(current_items), 3)]
    for row_chunk in rows:
        cols = st.columns(3)
        for col, (_, item) in zip(cols, row_chunk.iterrows()):
            with col:
                delta = rank_change.get(int(item["item_id"]), 0)
                if delta > 0:
                    badge = f"▲ +{delta}"
                elif delta < 0:
                    badge = f"▼ {delta}"
                else:
                    badge = "➡ 유지"
                render_product_card(item, int(item["rank"]), badge)


# ── 라우터 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if "view" not in st.session_state:
        st.session_state["view"] = "main"

    user_id = _setup_sidebar()

    if user_id is None:
        st.warning("사이드바에서 유저를 선택해 주세요.")
        return

    view = st.session_state["view"]

    if view == "main":
        _render_main_recommend(user_id)
    elif view == "detail":
        _render_detail_recommend()
    elif view == "twiddler":
        _render_twiddler(user_id)


main()
