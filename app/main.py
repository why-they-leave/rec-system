import os
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="추천 시스템 데모", page_icon="🛍️", layout="wide")

_APP_DIR = Path(__file__).parent
_ROOT_DIR = _APP_DIR.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_ROOT_DIR) not in sys.path:
    # data_loader.py가 backend.api.core를 직접 import하므로(HTTP 대신 in-process 호출),
    # 리포 루트도 sys.path에 있어야 backend/src 패키지를 찾을 수 있다.
    sys.path.insert(0, str(_ROOT_DIR))

from utils.data_bootstrap import GDRIVE_FILE_ID_ENV, ensure_data_downloaded

# Streamlit Community Cloud 콜드스타트 대응: 필요한 데이터 파일이 없으면 GDrive에서 자동으로 받는다.
# 파일 ID는 로컬 개발 시 환경변수로, Cloud 배포 시 st.secrets로 넘긴다(둘 다 없으면 조용히 건너뛰고
# 아래 각 로더의 FileNotFoundError 처리로 자연스럽게 안내한다).
_data_file_id = os.environ.get(GDRIVE_FILE_ID_ENV)
if not _data_file_id:
    try:
        _data_file_id = st.secrets.get(GDRIVE_FILE_ID_ENV)
    except Exception:
        _data_file_id = None
if _data_file_id:
    ensure_data_downloaded(_data_file_id)

import pandas as pd
from utils.style_loader import load_css
from utils.data_loader import (
    load_demo_users,
    load_products,
    get_main_recommendations,
    get_detail_recommendations,
)
from utils.rank_delta import get_rank_delta
from components.user_selector import render_persona_and_user_selector
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

# 카드 그리드 마커 — CSS :has() 가 이 마커를 감싼 컨테이너를 찾아 카드 높이를 통일하도록 앵커 역할
_CARD_GRID_MARKER = '<div data-card-grid style="display:none"></div>'


# ── 사이드바 ───────────────────────────────────────────────────────────────────

def _setup_sidebar() -> tuple[list[str], pd.DataFrame | None]:
    """사이드바 초기화. (선택된 카테고리 목록, demo_users_df) 반환.

    페르소나/유저 선택은 메인 화면(render_persona_and_user_selector)에서 렌더링하고,
    "메인으로 돌아가기" 버�른 상세 화면 본문에 이미 있으므로(_render_detail_recommend),
    사이드바는 카테고리 필터(메인 화면 전용)만 담당한다. 상세 화면에서는 사이드바에
    보여줄 내용이 없으므로 타이틀 아래 구분선 하나만 남기고 아무것도 추가하지 않는다
    (예전엔 divider + 빈 공간 + divider + 버튼 구조라 사이드바가 비어 보이면서도
    구분선이 두 번 나오는 게 어색했다).
    """
    load_css(str(_STATIC_DIR / "style.css"))

    st.sidebar.title("🛍️ 추천 시스템 데모")
    st.sidebar.markdown("---")

    try:
        demo_users = load_demo_users()
    except FileNotFoundError:
        st.sidebar.error("❌ `data/dashboard/demo_users.csv` 없음")
        demo_users = None

    # ── 카테고리 필터 (메인 화면 전용) ───────────────────────────────────────
    selected_categories = ALL_CATEGORIES[:]
    if st.session_state.get("view", "main") != "detail":
        st.sidebar.markdown("**🏷️ 카테고리 필터**")
        selected_pill = st.sidebar.pills(
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

    return selected_categories, demo_users


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
    with st.container():
        # 마커: CSS [data-testid="stVerticalBlock"]:has([data-card-grid]) 가 이 컨테이너를
        # 찾아 카드 행의 높이를 통일한다(상품명 길이에 따라 카드 높이가 달라지는 문제 방지).
        st.markdown(_CARD_GRID_MARKER, unsafe_allow_html=True)
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
                            "🔍 연관 상품 보기",
                            key=f"detail_{user_id}_{iid}_{model_key}",
                            use_container_width=True,
                        ):
                            st.session_state["view"] = "detail"
                            st.session_state["selected_item_id"] = iid
                            st.rerun()


def _render_twiddler_toggle(key: str, effective_phase: str, *, disabled: bool = False) -> None:
    """Twiddler 적용 전/후 토글 버튼. 라디오 대신 클릭 한 번으로 상태를 전환한다.

    effective_phase: 현재 유효한 phase("Before"/"After") — Cold 유저 게이팅 등으로 인해
    session_state 저장값과 다를 수 있어 호출부가 계산한 값을 그대로 받는다.
    """
    is_after = effective_phase == "After"
    label = "🔄 Twiddler 적용 중.. (적용 후)" if is_after else "⏸️ Twiddler 미적용 (적용 전)"
    if st.button(label, key=f"{key}_toggle_btn", use_container_width=True, disabled=disabled):
        st.session_state[key] = "Before" if is_after else "After"
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

def _render_main_recommend(selected_categories: list[str], demo_users_df: pd.DataFrame) -> None:
    st.title("📊 메인 추천")

    # ── 유저 소개 (페르소나 선택 + 유저 선택 — 둘 다 사이드바에서 이동, 나란히 배치) ──
    st.markdown("---")
    user_id, user_info = render_persona_and_user_selector(demo_users_df)

    try:
        products_df = load_products()
        als_before_df, before_status, before_message = get_main_recommendations(user_id, "ALS", "before")
    except FileNotFoundError as e:
        st.error(f"데이터를 불러올 수 없습니다: `{e}`")
        return

    st.subheader(f"User {user_id:03d}")
    st.markdown(
        f"**페르소나:** {user_info['persona_label']}  \n"
        f"**유저 유형:** {user_info['user_type_label']}  \n"
        f"**행동 로그:** {user_info['log_count']:,}건"
    )

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
    except FileNotFoundError as e:
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

    # ── 핵심 지표 (대시보드 성격에 맞게 화면 상단으로 배치) ──────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric(f"공통 추천 상품 (ALS {twiddler_phase} vs LightGCN 삼분그래프)", f"{len(common_ids)}개 / 10개")
    m2.metric("Jaccard 유사도", f"{jaccard:.3f}")
    m3.metric("유저 유형", user_type_val)
    st.divider()

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

        _render_twiddler_toggle("als_twiddler_phase", twiddler_phase, disabled=is_cold)

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


# ── 상세 추천 화면 ─────────────────────────────────────────────────────────────

def _render_detail_recommend(demo_users_df: pd.DataFrame) -> None:
    if st.button("← 메인으로 돌아가기"):
        st.session_state["view"] = "main"
        st.rerun()

    st.title("🛒 연관 상품 추천")

    item_id = st.session_state.get("selected_item_id")
    if item_id is None:
        st.warning("메인 화면에서 상품 카드를 클릭해 주세요.")
        return

    # ── 유저 소개 — 메인 화면과 동일한 선택 위젯을 그대로 재사용해, 같은 상품에 대해
    #    페르소나/유저를 바꿔가며 보완재 추천 결과를 비교할 수 있게 한다 ──────────
    user_id, user_info = render_persona_and_user_selector(demo_users_df)
    st.caption(
        f"페르소나: {user_info['persona_label']} · 유저 유형: {user_info['user_type_label']} "
        f"· 행동 로그: {user_info['log_count']:,}건"
    )

    try:
        products_df = load_products()
        cf_before_df, before_status, before_message = get_detail_recommendations(
            item_id, top_n=8, user_id=user_id, twiddler="before"
        )
    except FileNotFoundError as e:
        st.error(f"데이터를 불러올 수 없습니다: `{e}`")
        return

    current_item = products_df[products_df["item_id"] == item_id]
    if current_item.empty:
        st.warning("상품 정보를 찾을 수 없습니다.")
        return

    render_current_product_card(current_item.iloc[0])
    st.markdown("#### 함께 구매하면 좋은 상품 (보완재)")

    if before_status == "not_implemented":
        st.info(f"🚧 {before_message}" if before_message else "🚧 준비 중입니다.")
        return

    try:
        cf_after_df, after_status, after_message = get_detail_recommendations(
            item_id, top_n=8, user_id=user_id, twiddler="after"
        )
    except FileNotFoundError as e:
        st.error(f"데이터를 불러올 수 없습니다: `{e}`")
        return

    twiddler_phase = st.session_state.get("cf_twiddler_phase", "After")
    _render_twiddler_toggle("cf_twiddler_phase", twiddler_phase)

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

    selected_categories, demo_users_df = _setup_sidebar()

    if demo_users_df is None:
        st.warning("데이터를 불러올 수 없어 유저를 선택할 수 없습니다.")
        return

    view = st.session_state["view"]
    if view == "main":
        _render_main_recommend(selected_categories, demo_users_df)
    elif view == "detail":
        _render_detail_recommend(demo_users_df)


main()
