import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="추천 시스템 데모", page_icon="🛍️", layout="wide")

_APP_DIR = Path(__file__).parent
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

_STATIC_DIR = _APP_DIR / "static"

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

# st.pills 레이블 → 실제 카테고리명 역매핑
_PILL_TO_CAT: dict[str, str] = {
    f"{CATEGORY_EMOJI[c]} {c}": c for c in ALL_CATEGORIES
}
_PILL_OPTIONS = ["🏷️ 전체"] + list(_PILL_TO_CAT.keys())

# ALS 컬럼 마커 — CSS ::after 가상 요소가 gap 정중앙에 구분선을 그리도록 앵커 역할
_ALGO_LEFT_MARKER = '<div data-algo-left style="display:none"></div>'
_ALGO_RIGHT_MARKER = '<div style="display:none;"></div>'


# ── 사이드바 ───────────────────────────────────────────────────────────────────

def _setup_sidebar() -> int | None:
    """사이드바 초기화. 선택된 user_id 반환."""
    load_css(str(_STATIC_DIR / "style.css"))

    st.sidebar.title("🛍️ 추천 시스템 데모")
    st.sidebar.markdown("---")

    # ── 유저 선택 ────────────────────────────────────────────────────────────
    try:
        demo_users = load_demo_users()
        user_id = render_user_selector(demo_users)
    except FileNotFoundError:
        st.sidebar.error("❌ `data/dashboard/demo_users.csv` 없음")
        user_id = None

    # ── 페르소나 설명 플레이스홀더 ────────────────────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar.expander("📖 페르소나 설명", expanded=False):
        st.info("페르소나 확정 후 유형별 설명이 추가될 예정입니다.", icon="ℹ️")
        # TODO: 페르소나 확정 후 아래 딕셔너리를 채우고 st.info 교체
        # PERSONA_DESC: dict[str, str] = {
        #     "페르소나A": "설명...",
        #     "페르소나B": "설명...",
        # }
        # if user_id is not None:
        #     label = demo_users[demo_users["user_id"] == user_id]["persona_label"].values[0]
        #     st.write(PERSONA_DESC.get(label, "설명 없음"))

    # ── 상세 화면 전용: 메인 복귀 버튼 ─────────────────────────────────────
    if st.session_state.get("view") == "detail":
        st.sidebar.markdown("---")
        if st.sidebar.button("← 메인으로 돌아가기", use_container_width=True):
            st.session_state["view"] = "main"
            st.rerun()

    return user_id


# ── 메인 추천 화면 ─────────────────────────────────────────────────────────────

def _make_rank_note(item_id: int, before_map: dict, after_map: dict) -> str | None:
    """ALS 카드용 Twiddler 전후 순위 변화 텍스트."""
    b = before_map.get(item_id)
    a = after_map.get(item_id)
    if b is None or a is None:
        return None
    delta = b - a  # 양수 = 순위 상승 (숫자 낮을수록 좋음)
    if delta > 0:
        return f"{b}→{a}위  ▲ +{delta}"
    if delta < 0:
        return f"{b}→{a}위  ▼ {delta}"
    return f"rank {b}  →  유지"


def _render_grid(
    items_df: pd.DataFrame,
    common_ids: set,
    user_id: int,
    model_key: str,
    rank_notes: dict | None = None,
) -> None:
    """3열 상품 카드 그리드 + 상세 추천 버튼."""
    if items_df.empty:
        st.info("해당 카테고리 추천 결과 없음")
        return
    rows = [items_df.iloc[i : i + 3] for i in range(0, len(items_df), 3)]
    for row_chunk in rows:
        cols = st.columns(3)
        for col, (_, item) in zip(cols, row_chunk.iterrows()):
            with col:
                iid = int(item["item_id"])
                badge = "🔁 공통 추천" if item["item_id"] in common_ids else None
                rank_note = rank_notes.get(iid) if rank_notes else None
                render_product_card(item, int(item["rank"]), badge, rank_note)
                if st.button(
                    "🔍 상세 추천",
                    key=f"detail_{user_id}_{iid}_{model_key}",
                    use_container_width=True,
                ):
                    st.session_state["view"] = "detail"
                    st.session_state["selected_item_id"] = iid
                    st.rerun()


def _render_main_recommend(user_id: int) -> None:
    st.title("📊 메인 추천")

    try:
        rec_df      = load_recommendations()
        products_df = load_products()
    except FileNotFoundError as e:
        st.error(f"데이터 파일을 찾을 수 없습니다: `{e}`")
        return

    # ── 카테고리 필터 (pills) ────────────────────────────────────────────────
    selected_pill = st.pills(
        "카테고리",
        _PILL_OPTIONS,
        default="🏷️ 전체",
        key="cat_pill",
        label_visibility="collapsed",
    )
    if selected_pill is None or selected_pill == "🏷️ 전체":
        selected_categories = ALL_CATEGORIES[:]
    else:
        selected_categories = [_PILL_TO_CAT[selected_pill]]

    st.markdown("---")
    st.subheader(f"User {user_id:03d}")

    # ── 유저 유형 판별 (Twiddler 게이팅) ────────────────────────────────────
    user_rows = rec_df[rec_df["user_id"] == user_id]
    user_type_val = user_rows["user_type"].iloc[0].upper() if not user_rows.empty else "—"
    is_cold = (user_type_val == "COLD")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    # Cold 유저는 Twiddler 미적용 → "before"(인기도 기반)로 고정
    twiddler_phase = "Before" if is_cold else st.session_state.get("als_twiddler_phase", "Before")

    als_recs = get_user_recommendations(rec_df, user_id, "ALS",      twiddler_phase.lower())
    gcn_recs = get_user_recommendations(rec_df, user_id, "LightGCN", "before")

    # Twiddler 전후 순위 매핑 (rank_note 계산용)
    als_before = get_user_recommendations(rec_df, user_id, "ALS", "before")
    als_after  = get_user_recommendations(rec_df, user_id, "ALS", "after")
    rank_before_map = dict(zip(als_before["item_id"].astype(int), als_before["rank"]))
    rank_after_map  = dict(zip(als_after["item_id"].astype(int),  als_after["rank"]))
    rank_notes = {
        iid: _make_rank_note(iid, rank_before_map, rank_after_map)
        for iid in set(rank_before_map) | set(rank_after_map)
    }

    # Jaccard — 현재 선택된 ALS 결과 vs LightGCN
    common_ids = set(als_recs["item_id"]) & set(gcn_recs["item_id"])
    union_ids  = set(als_recs["item_id"]) | set(gcn_recs["item_id"])
    jaccard    = len(common_ids) / len(union_ids) if union_ids else 0.0

    # 카테고리 필터 적용
    als_items = (
        als_recs.merge(products_df, on="item_id", how="left")
        .pipe(lambda df: df[df["category"].isin(selected_categories)])
    )
    gcn_items = (
        gcn_recs.merge(products_df, on="item_id", how="left")
        .pipe(lambda df: df[df["category"].isin(selected_categories)])
    )

    # ── 2열 레이아웃: ALS | LightGCN (구분선은 CSS ::after 로 gap 정중앙에 배치) ──
    col_als, col_gcn = st.columns(2)

    with col_als:
        # 마커: CSS [data-testid="stColumn"]:has([data-algo-left])::after 로 구분선 생성
        st.markdown(_ALGO_LEFT_MARKER, unsafe_allow_html=True)
        st.markdown("### 🤖 ALS")

        if is_cold:
            st.caption("🧊 Cold 유저: Twiddler 미적용 — 인기도 기반 추천")
        else:
            st.caption("🔥 Heavy 유저: Twiddler 적용 효과 비교")

        st.radio(
            "Twiddler",
            ["Before", "After"],
            horizontal=True,
            key="als_twiddler_phase",
            format_func=lambda x: f"{x}  ({'적용 전' if x == 'Before' else '적용 후'})",
            label_visibility="collapsed",
            disabled=is_cold,
        )

        _render_grid(als_items, common_ids, user_id, f"als_{twiddler_phase.lower()}", rank_notes) 



    with col_gcn:
        st.markdown(_ALGO_RIGHT_MARKER, unsafe_allow_html=True)
        st.markdown("### 🚀 LightGCN")
        st.caption("페르소나 기반 삼분 그래프 (유저-아이템-페르소나)학습 결과")
        st.html("<div style='height: 0px;'></div>")
        st.write("GNN 기반 협업 필터링 · Twiddler 미적용")
        _render_grid(gcn_items, common_ids, user_id, "gcn")

    # ── 하단 지표 ────────────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric(f"공통 추천 상품 (ALS {twiddler_phase} vs LightGCN)", f"{len(common_ids)}개 / 10개")
    m2.metric("Jaccard 유사도", f"{jaccard:.3f}")
    m3.metric("유저 유형", user_type_val)


# ── 상세 추천 화면 ─────────────────────────────────────────────────────────────

def _render_detail_recommend() -> None:
    if st.button("← 메인으로 돌아가기"):
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
    st.markdown("### 🛒 함께 구매하면 좋은 상품 (보완재)")

    cf_recs = (
        detail_df[
            (detail_df["item_id"] == item_id)
            & (detail_df["rec_type"] == "cf")
        ]
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

    if cf_recs.empty:
        st.info("연관 상품 데이터가 없습니다.")
        return

    rows = [cf_recs.iloc[i : i + 4] for i in range(0, len(cf_recs), 4)]
    for row_chunk in rows:
        cols = st.columns(4)
        for col, (_, item) in zip(cols, row_chunk.iterrows()):
            with col:
                render_product_card(item, int(item["rank"]))


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


main()
