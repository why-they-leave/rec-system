import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="추천 시스템 데모", page_icon="🛍️", layout="wide")

_APP_DIR = Path(__file__).parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import pandas as pd
from utils.style_loader import load_css
from utils.api_client import BackendUnavailableError
from utils.data_loader import (
    load_demo_users,
    load_products,
    get_main_recommendations,
    get_detail_recommendations,
)
from utils.rank_delta import get_rank_delta
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

    # ── 상세 화면 전용: 메인 복귀 버튼 ─────────────────────────────────────
    if st.session_state.get("view") == "detail":
        st.sidebar.markdown("---")
        if st.sidebar.button("← 메인으로 돌아가기", use_container_width=True):
            st.session_state["view"] = "main"
            st.rerun()

    return user_id


# ── 추천 그리드 ─────────────────────────────────────────────────────────────────

def _render_recommend_grid(
    items_df: pd.DataFrame,
    *,
    id_key: str = "item_id",
    common_ids: set | None = None,
    user_id: int | None = None,
    model_key: str = "",
    rank_before_map: dict | None = None,
    plain_rank_mode: bool = False,
    ncols: int = 3,
    show_detail_button: bool = True,
) -> None:
    """상품 카드 그리드 렌더링. Twiddler 순위 변동 배지는 공유 유틸(get_rank_delta)로 계산.

    rank_before_map: {id: before_rank} — 있으면 카드마다 순위 변동 배지(▲/▼/–)를 계산해 표시.
    plain_rank_mode: True면 방향 계산 없이 회색 "N위" 배지만 표시("적용 전" 상태).
    """
    if items_df.empty:
        st.info("추천 결과가 없습니다.")
        return

    common_ids = common_ids or set()
    rows = [items_df.iloc[i : i + ncols] for i in range(0, len(items_df), ncols)]
    for row_chunk in rows:
        cols = st.columns(ncols)
        for col, (_, item) in zip(cols, row_chunk.iterrows()):
            with col:
                iid = int(item[id_key])
                badge = "🔁 공통 추천" if iid in common_ids else None
                score = float(item["score"])
                rank = int(item["rank"])

                if plain_rank_mode:
                    render_product_card(item, rank, badge, score=score, plain_rank_badge=rank)
                else:
                    rank_before = rank_before_map.get(iid) if rank_before_map else None
                    rank_delta = get_rank_delta(rank_before, rank) if rank_before is not None else None
                    render_product_card(
                        item, rank, badge, score=score, rank_delta=rank_delta, rank_before=rank_before,
                    )

                if show_detail_button:
                    if st.button(
                        "🔍 상세 추천",
                        key=f"detail_{user_id}_{iid}_{model_key}",
                        use_container_width=True,
                    ):
                        st.session_state["view"] = "detail"
                        st.session_state["selected_item_id"] = iid
                        st.rerun()


def _render_model_status_or_grid(
    status: str,
    message: str | None,
    items_df: pd.DataFrame,
    **grid_kwargs,
) -> None:
    """모델이 아직 준비되지 않은 경우(not_implemented) 안내 문구를, 아니면 카드 그리드를 렌더링."""
    if status == "not_implemented":
        st.info(f"🚧 {message}" if message else "🚧 준비 중입니다.")
        return
    _render_recommend_grid(items_df, **grid_kwargs)


# ── 메인 추천 화면 ─────────────────────────────────────────────────────────────

def _render_main_recommend(user_id: int) -> None:
    st.title("📊 메인 추천")

    try:
        products_df = load_products()
        als_before_df, before_status, before_message = get_main_recommendations(user_id, "ALS", "before")
    except (FileNotFoundError, BackendUnavailableError) as e:
        st.error(f"데이터를 불러올 수 없습니다: `{e}`")
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
    user_type_val = als_before_df["user_type"].iloc[0].upper() if not als_before_df.empty else "—"
    is_cold = (user_type_val == "COLD")

    # Cold 유저는 Twiddler 미적용 → "before"(인기도 기반)로 고정
    twiddler_phase = "Before" if is_cold else st.session_state.get("als_twiddler_phase", "After")

    try:
        als_after_df, after_status, after_message = get_main_recommendations(user_id, "ALS", "after")
        gcn_tri_recs, gcn_tri_status, gcn_tri_message = get_main_recommendations(
            user_id, "LightGCN", graph_type="tripartite"
        )
        gcn_bi_recs, gcn_bi_status, gcn_bi_message = get_main_recommendations(
            user_id, "LightGCN", graph_type="bipartite"
        )
    except BackendUnavailableError as e:
        st.error(f"데이터를 불러올 수 없습니다: `{e}`")
        return

    if twiddler_phase == "Before":
        als_recs, als_status, als_message = als_before_df, before_status, before_message
    else:
        als_recs, als_status, als_message = als_after_df, after_status, after_message

    rank_before_map = dict(zip(als_before_df["item_id"].astype(int), als_before_df["rank"]))

    # Jaccard — 현재 선택된 ALS 결과 vs LightGCN(삼분그래프, 페르소나 포함 — 주 비교 대상)
    common_ids = set(als_recs["item_id"]) & set(gcn_tri_recs["item_id"])
    union_ids  = set(als_recs["item_id"]) | set(gcn_tri_recs["item_id"])
    jaccard    = len(common_ids) / len(union_ids) if union_ids else 0.0

    # 카테고리 필터 적용
    als_items = (
        als_recs.merge(products_df, on="item_id", how="left")
        .pipe(lambda df: df[df["category"].isin(selected_categories)])
    )
    gcn_tri_items = (
        gcn_tri_recs.merge(products_df, on="item_id", how="left")
        .pipe(lambda df: df[df["category"].isin(selected_categories)])
    )
    gcn_bi_items = (
        gcn_bi_recs.merge(products_df, on="item_id", how="left")
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
            index=1,  # 기본값 "적용 후"
            horizontal=True,
            key="als_twiddler_phase",
            format_func=lambda x: f"{x}  ({'적용 전' if x == 'Before' else '적용 후'})",
            label_visibility="collapsed",
            disabled=is_cold,
        )

        _render_model_status_or_grid(
            als_status, als_message, als_items,
            id_key="item_id", common_ids=common_ids, user_id=user_id,
            model_key=f"als_{twiddler_phase.lower()}",
            rank_before_map=None if twiddler_phase == "Before" else rank_before_map,
            plain_rank_mode=(twiddler_phase == "Before"),
        )

    with col_gcn:
        st.markdown(_ALGO_RIGHT_MARKER, unsafe_allow_html=True)
        st.markdown("### 🚀 LightGCN")
        st.write("GNN 기반 협업 필터링 · Twiddler 미적용")

        st.radio(
            "그래프 종류",
            ["tripartite", "bipartite"],
            index=0,  # 기본값 "삼분그래프"
            horizontal=True,
            key="gcn_graph_type",
            format_func=lambda x: (
                "삼분그래프 (페르소나 포함)" if x == "tripartite" else "이분그래프 (페르소나 미포함)"
            ),
            label_visibility="collapsed",
        )
        graph_phase = st.session_state.get("gcn_graph_type", "tripartite")

        if graph_phase == "tripartite":
            gcn_status, gcn_message, gcn_items = gcn_tri_status, gcn_tri_message, gcn_tri_items
            gcn_common_ids = common_ids
        else:
            gcn_status, gcn_message, gcn_items = gcn_bi_status, gcn_bi_message, gcn_bi_items
            gcn_common_ids = set()  # 공통 추천 배지는 삼분그래프 기준(주 비교 대상)으로만 계산

        _render_model_status_or_grid(
            gcn_status, gcn_message, gcn_items,
            id_key="item_id", common_ids=gcn_common_ids, user_id=user_id,
            model_key=f"gcn_{graph_phase}",
        )

    # ── 하단 지표 ────────────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric(f"공통 추천 상품 (ALS {twiddler_phase} vs LightGCN 삼분그래프)", f"{len(common_ids)}개 / 10개")
    m2.metric("Jaccard 유사도", f"{jaccard:.3f}")
    m3.metric("유저 유형", user_type_val)


# ── 상세 추천 화면 ─────────────────────────────────────────────────────────────

def _render_detail_recommend(user_id: int) -> None:
    if st.button("← 메인으로 돌아가기"):
        st.session_state["view"] = "main"
        st.rerun()

    st.title("🛒 연관 상품 추천")

    item_id = st.session_state.get("selected_item_id")
    if item_id is None:
        st.warning("메인 화면에서 상품 카드를 클릭해 주세요.")
        return

    try:
        products_df = load_products()
        cf_before_df, before_status, before_message = get_detail_recommendations(
            item_id, top_n=8, user_id=user_id, twiddler="before"
        )
    except BackendUnavailableError as e:
        st.error(f"데이터를 불러올 수 없습니다: `{e}`")
        return

    current_item = products_df[products_df["item_id"] == item_id]
    if current_item.empty:
        st.warning("상품 정보를 찾을 수 없습니다.")
        return

    render_current_product_card(current_item.iloc[0])
    st.markdown("### 🛒 함께 구매하면 좋은 상품 (보완재)")

    if before_status == "not_implemented":
        st.info(f"🚧 {before_message}" if before_message else "🚧 준비 중입니다.")
        return

    try:
        cf_after_df, after_status, after_message = get_detail_recommendations(
            item_id, top_n=8, user_id=user_id, twiddler="after"
        )
    except BackendUnavailableError as e:
        st.error(f"데이터를 불러올 수 없습니다: `{e}`")
        return

    st.radio(
        "Twiddler",
        ["Before", "After"],
        index=1,  # 기본값 "적용 후"
        horizontal=True,
        key="cf_twiddler_phase",
        format_func=lambda x: f"{x}  ({'적용 전' if x == 'Before' else '적용 후'})",
        label_visibility="collapsed",
    )
    twiddler_phase = st.session_state.get("cf_twiddler_phase", "After")

    if twiddler_phase == "Before":
        cf_df, cf_status, cf_message = cf_before_df, before_status, before_message
    else:
        cf_df, cf_status, cf_message = cf_after_df, after_status, after_message

    rank_before_map = dict(zip(cf_before_df["rec_item_id"].astype(int), cf_before_df["rank"]))

    if cf_status == "not_implemented":
        st.info(f"🚧 {cf_message}" if cf_message else "🚧 준비 중입니다.")
        return

    cf_recs = cf_df.merge(
        products_df,
        left_on="rec_item_id",
        right_on="item_id",
        how="left",
        suffixes=("_detail", ""),
    )

    if cf_recs.empty:
        st.info("연관 상품 데이터가 없습니다.")
        return

    _render_recommend_grid(
        cf_recs, id_key="rec_item_id", user_id=user_id, model_key=f"cf_{twiddler_phase.lower()}",
        rank_before_map=None if twiddler_phase == "Before" else rank_before_map,
        plain_rank_mode=(twiddler_phase == "Before"),
        ncols=4, show_detail_button=False,
    )


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
        _render_detail_recommend(user_id)


main()
