import os
import sys
from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).parent
_STATIC_DIR = _APP_DIR / "static"
_LOGO_PATH = _STATIC_DIR / "logo2.png"
# st.logo()의 image 인자는 사이드바가 펼쳐진 동안 항상 표시되는데(API 제약 — icon_image를
# 줘도 image 자체를 숨기는 옵션은 없음), 사이드바 안에 이미 큰 로고(st.sidebar.image)를
# 따로 두므로 여기서는 완전 투명 1x1 placeholder를 넣어 사실상 보이지 않게 하고,
# icon_image로만 실제 로고를 넘겨 "사이드바 접힘 시에만" 작은 아이콘이 뜨게 한다(요청 반영).
_LOGO_BLANK_PATH = _STATIC_DIR / "logo_blank.png"

st.set_page_config(
    page_title="추크크✨",
    page_icon=str(_LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

_ROOT_DIR = _APP_DIR.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_ROOT_DIR) not in sys.path:
    # data_loader.py가 backend.api.core를 직접 import하므로(HTTP 대신 in-process 호출),
    # 리포 루트도 sys.path에 있어야 backend/src 패키지를 찾을 수 있다.
    sys.path.insert(0, str(_ROOT_DIR))

from utils.data_bootstrap import GDRIVE_FILE_ID_ENV, ensure_data_downloaded  # noqa: E402

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

import pandas as pd  # noqa: E402
from components.eval_metrics_table import (  # noqa: E402
    render_eval_metrics,
    render_lightgcn_persona_accuracy,
    render_lightgcn_persona_rank_shift_and_category,
    render_user_twiddler_case,
)
from components.glossary import render_glossary  # noqa: E402
from components.product_card import (  # noqa: E402
    product_icon_html,
    render_current_product_card,
    render_product_card,
)
from components.project_about import render_project_about  # noqa: E402
from components.project_intro import render_project_intro  # noqa: E402
from components.team_page import render_team_page  # noqa: E402
from components.user_graph import render_user_graph  # noqa: E402
from components.user_list import render_user_list  # noqa: E402
from components.user_selector import (  # noqa: E402
    get_current_user_selection,
    render_persona_and_user_selector,
    render_user_summary_card,
)
from utils.category_emoji import (  # noqa: E402
    CATEGORY_EMOJI,
    CATEGORY_SUBTYPES,
    category_icon_url,
    product_type_from_name,
)
from utils.data_loader import (  # noqa: E402
    get_detail_recommendations,
    get_main_recommendations,
    load_demo_users,
    load_products,
    reset_user_exposure,
    simulate_next_session,
)
from utils.product_icons import icon_slug_for, icon_url  # noqa: E402
from utils.rank_delta import get_rank_delta  # noqa: E402
from utils.style_loader import load_css  # noqa: E402

from src.modeling.twiddler.rerank import POOL_MULTIPLIER  # noqa: E402

ALL_CATEGORIES = list(CATEGORY_EMOJI.keys())

# 카드 그리드 마커 — CSS :has() 가 이 마커를 감싼 컨테이너를 찾아 카드 높이를 통일하도록 앵커 역할
_CARD_GRID_MARKER = '<div data-card-grid style="display:none"></div>'

# 화면에 실제로 보여줄 카드 개수. Twiddler "after"는 backend에서 이 값의
# POOL_MULTIPLIER배(= backend/api/core.py 의 fetch_n)만큼 넓은 풀에서 재랭킹하므로,
# "before" 순위 추적(rank_before_map)도 같은 폭으로 가져와야 풀 밖에서 승격된 상품에
# 정확한 "▲" 배지를 매길 수 있다 — 좁게 가져오면 그 상품은 배지 없이 "새 항목"처럼
# 보이고, 반대로 밀려난 자리만 "▼"로 보여 마치 내려간 쪽만 있고 올라간 쪽은 없는
# 것처럼 보이는 착시가 생긴다.
_MAIN_TOP_N = 10
_DETAIL_TOP_N = 8

# 새로고침 시뮬레이션 라운드 수 — evaluate_twiddler.py의 T_ROUNDS(5)와 동일한 값으로 맞춘다.
_SIM_ROUNDS = 5


# ── 상단 네비게이션 바 ─────────────────────────────────────────────────────────


def _render_top_navbar() -> None:
    """로고+브랜드(맨 위 별도 줄) + 탭 버튼 필 카드(그 아래 줄)를 본문 최상단에 배치.

    원래는 로고+브랜드와 탭 버튼이 같은 필 카드 한 줄에 같이 있었는데, 로고가 nav 링크
    사이에 끼어 있는 것처럼 보여 어색하다는 피드백으로(요청 반영) 로고+브랜드를 카드
    밖 맨 위 줄로 분리했다. 탭 버튼 로직은 그대로 두고 위치만 옮긴 것이라 각 버튼은
    기존과 동일하게 session_state["main_tab"]만 바꾸고 st.rerun()한다.
    """
    # 새로고침(세션 초기화) 시 처음 보이는 화면은 "추천 비교"가 아니라 "프로젝트 소개"
    # 여야 한다(요청 반영) — 데모 체험 전에 프로젝트 맥락부터 보게 한다.
    current_tab = st.session_state.get("main_tab", "project")

    # ── 로고 + 브랜드명 — 맨 위, nav 카드 밖에 독립된 줄로 ────────────────────────
    col_logo, col_brand, _col_brand_spacer = st.columns([0.35, 1.2, 4], vertical_alignment="center")
    with col_logo:
        st.image(str(_LOGO_PATH), width=56)
    with col_brand:
        st.markdown(
            '<div class="topnav-brand">추크크'
            '<div class="topnav-brand-subtitle">Recommendation Creator Crew</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    # ── 탭 버튼 — 기능명보다 데모 체험 순서가 먼저 보이도록 단계형 라벨을 쓴다.
    st.markdown('<div class="topnav-marker"></div>', unsafe_allow_html=True)
    col_b0, col_b1, col_b2, col_b3, col_spacer = st.columns(
        [0.8, 0.85, 0.85, 0.7, 3], vertical_alignment="bottom"
    )
    with col_b0:
        if st.button(
            "프로젝트 소개",
            key="topnav_project",
            type="primary" if current_tab == "project" else "tertiary",
        ):
            st.session_state["main_tab"] = "project"
            st.session_state["project_page"] = "about"
            st.session_state["view"] = "main"
            st.rerun()
    with col_b1:
        if st.button(
            "추천 비교",
            key="topnav_rerank",
            type="primary" if current_tab == "rerank" else "tertiary",
        ):
            st.session_state["main_tab"] = "rerank"
            st.rerun()
    with col_b2:
        if st.button(
            "효과 해석",
            key="topnav_persona",
            type="primary" if current_tab == "persona" else "tertiary",
        ):
            st.session_state["main_tab"] = "persona"
            st.rerun()
    with col_b3:
        if st.button(
            "유저 목록",
            key="topnav_userlist",
            type="primary" if current_tab == "userlist" else "tertiary",
        ):
            st.session_state["main_tab"] = "userlist"
            st.rerun()


# ── 사이드바 ───────────────────────────────────────────────────────────────────


def _setup_sidebar() -> tuple[list[str], set[str] | None, pd.DataFrame | None]:
    """사이드바 초기화. (선택된 카테고리 목록, demo_users_df) 반환.

    탭 네비게이션은 상단바(_render_top_navbar)로 이동했고(요청 반영), 사이드바는 현재
    화면의 보조 메뉴와 카테고리 필터를 담당한다.
    """
    load_css(str(_STATIC_DIR / "style.css"))

    # 사이드바가 접혔을 때 본문 좌상단에 뜨는 작은 아이콘 — 로고/브랜드 본문은 상단바로
    # 옮겼으므로 image에는 투명 placeholder만 남긴다.
    st.logo(str(_LOGO_BLANK_PATH), icon_image=str(_LOGO_PATH), size="large")
    current_tab = st.session_state.get("main_tab", "project")

    try:
        demo_users = load_demo_users()
    except FileNotFoundError:
        st.sidebar.error("❌ `data/dashboard/demo_users.csv` 없음")
        demo_users = None

    # ── 카테고리 필터 — 카테고리별 아코디언 + 하위 타입 체크박스 ─────────────────
    # 카테고리를 펼치면 그 카테고리의 상품 타입(서브카테고리)이 체크박스로 나온다.
    # 아무것도 체크 안 하면 전체(카테고리 레벨 필터만 적용 안 함, selected_types=None).
    selected_categories = ALL_CATEGORIES[:]
    selected_types: set[str] = set()
    if current_tab == "project":
        project_pages = [
            ("프로젝트 소개", "about"),
            ("데모 안내", "intro"),
            ("용어 해석", "glossary"),
            ("팀 소개", "team"),
        ]
        current_project_page = st.session_state.get("project_page", "about")
        # 항상 펼쳐진(expanded=True) expander는 접었다 펼 일이 없는데도 체브론+박스
        # 테두리가 붙어 "카테고리 필터"(제목만 있는 plain markdown)와 톤이 안 맞고,
        # 버튼 사이 기본 세로 gap도 커서 부자연스럽다는 피드백(요청 반영) — 다른 사이드바
        # 섹션과 같은 plain 제목 + 촘촘한 리스트로 바꾼다(gap은 .st-key-project_menu
        # 스코프로 style.css에서 좁힌다).
        st.sidebar.markdown("**데모 안내**")
        with st.sidebar.container(key="project_menu"):
            for label, page_key in project_pages:
                if st.button(
                    label,
                    key=f"project_page_{page_key}",
                    type="primary" if current_project_page == page_key else "tertiary",
                    width="stretch",
                ):
                    st.session_state["project_page"] = page_key
                    st.rerun()
    if current_tab == "rerank" and st.session_state.get("view", "main") == "detail":
        detail_pages = [
            ("> 추천 결과 비교", "compare"),
            ("> 오프라인 성능 지표", "metrics"),
        ]
        current_detail_page = st.session_state.get("rerank_detail_page", "compare")
        with st.sidebar.expander("추천 비교 · 연관 상품", expanded=True):
            for label, page_key in detail_pages:
                if st.button(
                    label,
                    key=f"rerank_detail_page_{page_key}",
                    type="primary" if current_detail_page == page_key else "tertiary",
                    width="stretch",
                ):
                    st.session_state["rerank_detail_page"] = page_key
                    st.rerun()

    if current_tab == "rerank" and st.session_state.get("view", "main") != "detail":
        # 순서상 유저를 먼저 골라야 나머지(비교/지표)가 의미가 있어서 "페르소나 및 유저
        # 선택"을 맨 위로 올렸다(요청 반영). "데모 안내"와 같은 plain 리스트 톤으로
        # 맞추고(요청 반영: "디자인 통일") 라벨 앞 문자 ">"는 빼고 활성 항목 표시는
        # style.css의 [class*="st-key-rerank_page_"][kind="primary"] 왼쪽 인디고
        # 보더로만 준다. 섹션 제목도 하위 항목(옛 "추천 결과 비교")과 겹치지 않게 상단
        # nav 탭 이름과 맞춘다.
        rerank_pages = [
            ("페르소나 및 유저 선택", "user"),
            ("추천 결과 비교", "compare"),
            ("오프라인 성능 지표", "metrics"),
        ]
        current_rerank_page = st.session_state.get("rerank_page", "user")
        st.sidebar.markdown("**Twiddler 재랭킹**")
        with st.sidebar.container(key="rerank_menu"):
            for label, page_key in rerank_pages:
                if st.button(
                    label,
                    key=f"rerank_page_{page_key}",
                    type="primary" if current_rerank_page == page_key else "tertiary",
                    width="stretch",
                ):
                    st.session_state["rerank_page"] = page_key
                    st.rerun()

        if current_rerank_page == "compare":
            st.sidebar.markdown("**카테고리 필터**")
            for category in ALL_CATEGORIES:
                icon_url = category_icon_url(category)
                header = (
                    f"![]({icon_url}) {category}"
                    if icon_url
                    else f"{CATEGORY_EMOJI[category]} {category}"
                )
                with st.sidebar.expander(header):
                    subtypes = CATEGORY_SUBTYPES.get(category, [])
                    all_key = f"subcat_all_{category}"

                    def _toggle_all(category=category, subtypes=subtypes, all_key=all_key):
                        checked = st.session_state[all_key]
                        for subtype in subtypes:
                            st.session_state[f"subcat_{category}_{subtype}"] = checked

                    st.checkbox("전체", key=all_key, on_change=_toggle_all)
                    st.markdown("<hr style='margin:0.1rem 0 0.4rem;'>", unsafe_allow_html=True)
                    for subtype in subtypes:
                        if st.checkbox(subtype, key=f"subcat_{category}_{subtype}"):
                            selected_types.add(subtype)
        if selected_types:
            selected_categories = [
                c
                for c in ALL_CATEGORIES
                if any(t in selected_types for t in CATEGORY_SUBTYPES.get(c, []))
            ]

    if current_tab == "persona":
        # "효과 해석"은 유저 선택 → bi/tri 정량 비교 → 순위·카테고리 비교 → 추천 근거
        # 그래프까지 하나의 논증 흐름이라, "추천 비교" 탭처럼 별도 페이지로 쪼개면
        # 스토리가 끊긴다(요청 논의 반영) — 대신 페이지는 그대로 두고 사이드바에서
        # 클릭하면 같은 페이지 안에서 앵커(#section-*)로 스크롤만 이동하게 한다.
        # st.button 대신 plain <a href="#...">를 써서 rerun 없이 브라우저 네이티브
        # 스크롤만 일어나게 한다(사이드바 목록과 톤은 .sidebar-anchor-menu로 맞춤).
        st.sidebar.markdown("**효과 해석**")
        st.sidebar.markdown(
            '<div class="sidebar-anchor-menu">'
            '<a href="#section-persona-user" id="nav-persona-user">유저 선택</a>'
            '<a href="#section-persona-accuracy" id="nav-persona-accuracy">bi/tri 정량 비교</a>'
            '<a href="#section-persona-rankshift" id="nav-persona-rankshift">순위·카테고리 비교</a>'
            '<a href="#section-persona-graph" id="nav-persona-graph">추천 근거 그래프</a>'
            "</div>",
            unsafe_allow_html=True,
        )
        # 다른 사이드바 메뉴(추천 비교 등)는 항상 "현재 페이지"가 흰 박스로 켜져
        # 있는데, 앵커 링크는 그런 상태가 없어 톤이 안 맞아 보였다(요청 반영: "다른
        # 사이드바랑 똑같이 맞춰줘야지") — 스크롤 위치를 보고 어느 섹션이 보이는지
        # 계산해 해당 링크에 .active를 붙인다.
        # 처음엔 IntersectionObserver를 썼는데, 이 스크립트는 st.iframe의
        # 0px 높이 iframe 안에서 실행돼 new IntersectionObserver(...)의 암묵적
        # root(뷰포트)가 "부모 페이지"가 아니라 "이 0px iframe 자신"이 되어버려
        # 교차 판정이 전혀 안 됐다(요청으로 재현·확인 — 스크롤해도 활성 링크가 항상
        # 첫 항목에 고정됨). getBoundingClientRect() 좌표를 직접 비교하는 방식은
        # 어느 document에서 계산하든 뷰포트 기준 좌표라 이 문제가 없다 — window.parent
        # 기준으로 "화면 위쪽 25% 지점을 지나간 섹션 중 가장 가까운 것"을 활성으로
        # 고른다. 실제 스크롤은 window가 아니라 Streamlit 본문의 내부 컨테이너가
        # 담당해 window.scrollY는 안 바뀌므로(요청으로 확인), scroll 이벤트를
        # document에 capture:true로 걸어 내부 컨테이너의 스크롤도 잡아낸다.
        st.iframe(
            """
            <script>
            (function () {
                const doc = window.parent.document;
                const win = window.parent;
                const pairs = [
                    ["section-persona-user", "nav-persona-user"],
                    ["section-persona-accuracy", "nav-persona-accuracy"],
                    ["section-persona-rankshift", "nav-persona-rankshift"],
                    ["section-persona-graph", "nav-persona-graph"],
                ];
                function setActive(navId) {
                    pairs.forEach(([, nid]) => {
                        const a = doc.getElementById(nid);
                        if (a) a.classList.toggle("active", nid === navId);
                    });
                }
                function updateActive() {
                    const refY = win.innerHeight * 0.25;
                    let bestId = pairs[0][1];
                    let bestTop = -Infinity;
                    pairs.forEach(([sid, nid]) => {
                        const el = doc.getElementById(sid);
                        if (!el) return;
                        const top = el.getBoundingClientRect().top;
                        if (top <= refY && top > bestTop) {
                            bestTop = top;
                            bestId = nid;
                        }
                    });
                    setActive(bestId);
                }
                let scheduled = false;
                function onScroll() {
                    if (scheduled) return;
                    scheduled = true;
                    win.requestAnimationFrame(() => {
                        scheduled = false;
                        updateActive();
                    });
                }
                function init() {
                    const anyExists = pairs.some(([sid]) => doc.getElementById(sid));
                    if (!anyExists) {
                        setTimeout(init, 300);
                        return;
                    }
                    doc.addEventListener("scroll", onScroll, true);
                    win.addEventListener("scroll", onScroll, true);
                    updateActive();
                }
                init();
            })();
            </script>
            """,
            height=1,
        )

    return selected_categories, (selected_types or None), demo_users


def _apply_filters(
    df: pd.DataFrame,
    selected_categories: list[str],
    selected_types: set[str] | None,
) -> pd.DataFrame:
    """카테고리(필수) + 하위 타입(선택 시에만) 순으로 필터링. 4곳(ALS/LightGCN bipartite/
    시뮬레이션/persona 탭)에서 공유하는 필터 로직을 한 곳으로 모은다."""
    out = df[df["category"].isin(selected_categories)]
    if selected_types:
        out = out[out["name"].apply(product_type_from_name).isin(selected_types)]
    return out


# ── 추천 그리드 ─────────────────────────────────────────────────────────────────


def _related_preview_icon_urls(iid: int, products_df: pd.DataFrame | None) -> list[str]:
    """실제 보완재 top-3의 라인아트 아이콘 URL 목록. 없으면 빈 리스트."""
    urls = []
    if products_df is None:
        return urls
    try:
        preview_df, preview_status, _ = get_detail_recommendations(iid, top_n=3)
    except FileNotFoundError:
        return urls
    if preview_status == "not_implemented" or preview_df.empty:
        return urls
    preview_items = preview_df.merge(
        products_df,
        left_on="rec_item_id",
        right_on="item_id",
        how="left",
        suffixes=("", "_p"),
    )
    for _, row in preview_items.iterrows():
        if pd.isna(row.get("name")):
            continue
        slug = icon_slug_for(product_type_from_name(row["name"]))
        if slug:
            urls.append(icon_url(slug))
    return urls


def _render_related_preview_button(
    iid: int,
    products_df: pd.DataFrame | None,
    key: str,
) -> bool:
    """보완재 top-3 아이콘을 각각 별도 칸(테두리 있는 정사각형)에 나눠 보여준 뒤(요청
    반영 — 한 줄에 뭉쳐 보이지 않게), 그 아래 기존 "연관 상품 보기" 텍스트 버튼을 그대로 둔다.

    st.button 라벨은 unsafe_allow_html을 지원하지 않아 칸 구분(테두리)을 버튼 안에 못
    넣는다 — 그래서 미리보기 칸들과 버튼을 st.columns로 한 줄에 나란히 배치한다(요청 반영).
    """
    icon_urls = _related_preview_icon_urls(iid, products_df)
    if icon_urls:
        # 버튼 칸이 좁으면 "같이 구매한 상품 ›" 문구가 두 줄로 잘려 보인다는 피드백
        # 반영 — 버튼 쪽 비중을 넉넉히 키운다.
        preview_col, button_col = st.columns([len(icon_urls) * 0.9, 5], vertical_alignment="center")
        with preview_col:
            cells = "".join(
                f'<div class="related-preview-cell"><img src="{u}" /></div>' for u in icon_urls
            )
            st.markdown(f'<div class="related-preview-row">{cells}</div>', unsafe_allow_html=True)
        with button_col:
            return st.button("같이 구매한 상품 ›", key=key, width="stretch")
    return st.button("같이 구매한 상품 ›", key=key, width="stretch")


def _render_recommend_grid(
    items_df: pd.DataFrame,
    *,
    id_key: str = "item_id",
    common_ids: set | None = None,
    user_id: int | None = None,
    model_key: str = "",
    rank_before_map: dict | None = None,
    plain_rank_mode: bool = False,
    ncols: int = 5,
    show_detail_button: bool = True,
    products_df: pd.DataFrame | None = None,
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

                    footer = None
                    if show_detail_button:
                        # "연관 상품 보기" 버튼을 카드 테두리 안쪽에 넣어달라는 요청 반영 —
                        # render_product_card가 container(border=True) 블록을 닫기 전에
                        # 이 콜백을 호출하도록 넘긴다.
                        def footer(iid=iid):
                            clicked = _render_related_preview_button(
                                iid,
                                products_df,
                                key=f"detail_{user_id}_{iid}_{model_key}",
                            )
                            if clicked:
                                st.session_state["view"] = "detail"
                                st.session_state["selected_item_id"] = iid
                                st.rerun()

                    if plain_rank_mode:
                        render_product_card(
                            item,
                            rank,
                            badge,
                            score=score,
                            plain_rank_badge=rank,
                            footer=footer,
                        )
                    else:
                        # rank_before_map이 주어졌다는 건 "비교 대상이 있다"는 뜻이므로, 그
                        # 안에 이 상품이 없으면 "비교 안 함"이 아니라 "직전 대비 새로 진입"이다
                        # — 구분 없이 배지를 비워두면 유지/상승/하락 어디에도 안 걸리는 상품이
                        # 생긴다(요청으로 발견, 특히 새로고침 시뮬레이션에서 자주 발생 — 이전
                        # 라운드 top-10 밖에 있던 상품이 새로 진입하는 게 정상 시나리오).
                        if rank_before_map is not None:
                            rank_before = rank_before_map.get(iid)
                            if rank_before is not None:
                                rank_delta = get_rank_delta(rank_before, rank)
                            else:
                                rank_delta = {"direction": "new", "label": "신규"}
                        else:
                            rank_before, rank_delta = None, None
                        render_product_card(
                            item,
                            rank,
                            badge,
                            score=score,
                            rank_delta=rank_delta,
                            rank_before=rank_before,
                            footer=footer,
                        )


def _render_twiddler_toggle(key: str, effective_phase: str, *, disabled: bool = False) -> None:
    """Twiddler 적용 전/후를 고르는 압축형 Before/After 세그먼트 토글.

    기존에는 상태에 따라 라벨이 바뀌는 넓은 버튼 1개("🔄 Twiddler 적용 중.. (적용 후)")
    였는데, 프로젝트 소개 페이지 미리보기 목업의 필(pill) 토글이 더 직관적이라는 피드백으로
    실제 페이지에도 반영한다(요청 반영) — Before/After 버튼 2개를 나란히 두고 현재 phase만
    채워진 필로 강조한다. effective_phase: 현재 유효한 phase("Before"/"After") — Cold 유저
    게이팅 등으로 인해 session_state 저장값과 다를 수 있어 호출부가 계산한 값을 그대로 받는다.
    """
    is_after = effective_phase == "After"
    st.markdown('<div class="twiddler-toggle-marker"></div>', unsafe_allow_html=True)
    col_before, col_after = st.columns(2, gap="small")
    with col_before:
        if st.button(
            "Before",
            key=f"twidtoggle_{key}_before",
            width="stretch",
            type="secondary" if is_after else "primary",
            disabled=disabled,
            # Before/After가 뭘 뜻하는지 안 보인다는 피드백(요청 반영) — hover 툴팁으로
            # 짧게 설명한다(버튼 옆에 캡션을 추가하면 줄이 하나 더 늘어나 자리를 차지함).
            help="Twiddler 재랭킹을 적용하기 전 원래 모델 추천 순위입니다.",
        ):
            st.session_state[key] = "Before"
            st.rerun()
    with col_after:
        if st.button(
            "After",
            key=f"twidtoggle_{key}_after",
            width="stretch",
            type="primary" if is_after else "secondary",
            disabled=disabled,
            help="Twiddler 재랭킹을 적용해 페르소나 선호·노출 이력을 반영한 뒤의 순위입니다.",
        ):
            st.session_state[key] = "After"
            st.rerun()


def _render_refresh_simulation_button(
    user_id: int,
    disabled: bool,
    *,
    session_prefix: str = "als",
    model_type: str = "ALS",
    graph_type: str = "tripartite",
    exposure_context: str = "main",
) -> int:
    """반복 새로고침(다양성 효과)을 라이브로 보여주는 시뮬레이션 버튼.

    Tab1의 핵심 가치인 "반복 방문 다양성"이 지금까지는 사전계산된 population 평균 표로만
    보였다 — 이 버튼은 선택된 유저 1명에 대해 실제로 exposure_service 노출 이력을 누적시켜가며
    "새로고침을 반복하면 카드가 실제로 바뀐다"는 걸 라이브로 보여준다(요청 반영).
    클릭마다 라운드가 1씩 증가하고, _SIM_ROUNDS(5)를 넘으면 라운드 0(평소 before/after 토글
    화면)으로 돌아간다. 라운드 0→1로 새로 시작할 때만 노출 이력을 리셋해 매번 깨끗한
    상태에서 시작한다 — 그 외에는 세션 상태의 결과를 그대로 재사용해 매 rerun마다 시뮬레이션이
    다시 진행되지 않도록 한다(버튼을 실제로 눌렀을 때만 다음 라운드로 전진).

    각 라운드 결과를 sim_history_{user_id}_{session_prefix} 리스트에 누적해 두어, 호출부가
    "직전 라운드 대비 순위 변동(▲/▼)" 배지를 계산할 수 있게 한다(요청 반영 — 이전엔 라운드마다
    순위만 보여주고 변동 방향은 없었다).

    session_prefix/model_type/graph_type/exposure_context: ALS와 LightGCN bipartite처럼
    Twiddler 탭 안에 여러 모델 섹션이 공존할 때 라운드 상태·노출 이력이 서로 독립되도록
    분리하는 키(모델별로 다른 값을 넘긴다).

    반환: 현재 라운드(0이면 시뮬레이션 비활성 상태).
    """
    round_key = f"sim_round_{user_id}_{session_prefix}"
    history_key = f"sim_history_{user_id}_{session_prefix}"
    current_round = st.session_state.get(round_key, 0)
    label = (
        f"새로고침 시뮬레이션 ({current_round}/{_SIM_ROUNDS}회차)"
        if current_round
        else "새로고침 시뮬레이션 시작"
    )
    if st.button(
        label,
        key=f"sim_btn_{user_id}_{session_prefix}",
        width="stretch",
        disabled=disabled,
        # 버튼이 뭘 하는지 안 보인다는 피드백(요청 반영) — Before/After 툴팁과 같은 방식.
        help=(
            "클릭할 때마다 같은 유저가 다시 방문한 것처럼 노출 이력을 누적시켜, "
            "반복 노출된 상품의 추천 점수가 실제로 낮아지는지 라이브로 보여줍니다."
        ),
    ):
        next_round = current_round + 1 if current_round < _SIM_ROUNDS else 0
        st.session_state[round_key] = next_round
        if next_round == 1:
            reset_user_exposure(user_id, exposure_context)
            st.session_state[history_key] = []
        if next_round > 0:
            result = simulate_next_session(
                user_id, top_n=_MAIN_TOP_N, model_type=model_type, graph_type=graph_type
            )
            st.session_state.setdefault(history_key, []).append(result)
        st.rerun()
    return current_round


def _render_rerank_model_toggle() -> str:
    """추천 비교 탭 상단의 모델 좌우 토글 — ALS ↔ LightGCN bipartite.

    반환: 현재 선택된 모델("ALS" 또는 "LightGCN-bipartite").
    """
    current = st.session_state.get("rerank_model_type", "ALS")
    col_als, col_lgcn = st.columns(2)
    with col_als:
        if st.button(
            "ALS",
            key="rerank_model_als_btn",
            width="stretch",
            type="primary" if current == "ALS" else "secondary",
        ):
            st.session_state["rerank_model_type"] = "ALS"
            st.rerun()
    with col_lgcn:
        if st.button(
            # Before/After 토글과 한 줄에 들어가면서 폭이 좁아져 "bipartite"까지 쓰면
            # 줄바꿈/잘림이 생긴다(요청 반영: 한 줄로 합치기) — 짧게 "LightGCN"만 표시.
            "LightGCN",
            key="rerank_model_lgcn_btn",
            width="stretch",
            type="primary" if current == "LightGCN-bipartite" else "secondary",
        ):
            st.session_state["rerank_model_type"] = "LightGCN-bipartite"
            st.rerun()
    return st.session_state.get("rerank_model_type", "ALS")


def _render_rank_summary_list(
    items_df: pd.DataFrame,
    id_key: str = "item_id",
    rank_before_map: dict | None = None,
    plain_rank_mode: bool = False,
) -> None:
    """카드 그리드 위에 순위·순위변동만 압축해서 먼저 보여주는 요약 리스트.

    카드 안 순위 배지("🏆 N위 · 전 순위 N위" + 작은 "+2" 배지)가 작고 회색이라 순위가
    바뀐 걸 알아채기 어렵다는 피드백(요청 반영: "카드 그리드 위에 요약 리스트 추가로") —
    get_rank_delta와 같은 전/후 비교를 재사용하되, "+2" 대신 "2단계 상승"처럼 방향이
    말로 바로 드러나는 문구를 쓰고 색으로도 강조한다.
    """
    if items_df.empty:
        return
    rows_html = []
    for _, item in items_df.iterrows():
        iid = int(item[id_key])
        rank = int(item["rank"])
        score = item.get("score")
        match_html = f"<em>매칭 {float(score) * 100:.0f}%</em>" if pd.notna(score) else "<em></em>"

        delta_html = ""
        if not plain_rank_mode and rank_before_map is not None:
            rank_before = rank_before_map.get(iid)
            if rank_before is None:
                delta_html = '<span class="rank-summary-delta rank-summary-delta-new">신규</span>'
            else:
                delta = rank_before - rank
                if delta > 0:
                    delta_html = (
                        f'<span class="rank-summary-delta rank-summary-delta-up">'
                        f"▲ {delta}단계 상승</span>"
                    )
                elif delta < 0:
                    delta_html = (
                        f'<span class="rank-summary-delta rank-summary-delta-down">'
                        f"▼ {abs(delta)}단계 하락</span>"
                    )
                else:
                    delta_html = (
                        '<span class="rank-summary-delta rank-summary-delta-same">유지</span>'
                    )

        rows_html.append(
            f'<div class="rank-summary-row">'
            f'<div class="rank-summary-rank">{rank}</div>'
            f"{product_icon_html(item['name'], size=26)}"
            f'<div class="rank-summary-copy">'
            f"<strong>{item['name']}</strong><span>{item['category']}</span></div>"
            f"{match_html}"
            f"{delta_html}"
            f"</div>"
        )
    st.markdown(
        f'<div class="rank-summary-panel">{"".join(rows_html)}</div>',
        unsafe_allow_html=True,
    )


def _render_model_status_or_grid(
    status: str,
    message: str | None,
    items_df: pd.DataFrame,
    **grid_kwargs,
) -> None:
    """모델이 아직 준비되지 않은 경우(not_implemented) 안내 문구를, 아니면 요약
    리스트 + 카드 그리드를 렌더링."""
    if status == "not_implemented":
        st.info(f"🚧 {message}" if message else "🚧 준비 중입니다.")
        return
    _render_rank_summary_list(
        items_df,
        id_key=grid_kwargs.get("id_key", "item_id"),
        rank_before_map=grid_kwargs.get("rank_before_map"),
        plain_rank_mode=grid_kwargs.get("plain_rank_mode", False),
    )
    _render_recommend_grid(items_df, **grid_kwargs)


# ── Tab 1: Twiddler 재랭킹 — 메인 추천 화면 ────────────────────────────────────────


def _render_model_twiddler_block(
    *,
    model_type: str,
    graph_type: str,
    session_prefix: str,
    exposure_context: str,
    section_title: str,
    eval_label: str,
    eval_context: str,
    gate_cold_users: bool,
    user_id: int,
    user_info: dict,
    products_df: pd.DataFrame,
    selected_categories: list[str],
    selected_types: set[str] | None,
) -> None:
    """모델 1종(ALS 또는 LightGCN bipartite)의 Before/After Twiddler 비교 섹션 전체를 렌더링.

    ALS와 LightGCN bipartite 두 섹션이 완전히 동일한 로직(풀 조회 → 유저 유형/Twiddler
    phase 판정 → Before/After 토글 → 새로고침 시뮬레이션 → 카드 그리드 → 오프라인 지표)을
    공유하므로 model_type/graph_type/session_prefix로 파라미터화해 중복 없이 재사용한다
    (reports/LIGHTGCN_BIPARTITE_TWIDDLER_PLAN.md).

    gate_cold_users: ALS는 cold 유저에게 인기도 폴백만 주고 Twiddler를 건너뛰는 게이팅이
    있지만, LightGCN bipartite는 precomputed 추천이 전체 유저를 커버해 이런 구분이 없다
    (False로 호출 — 토글이 항상 활성화된다).
    session_prefix/exposure_context: 두 모델 섹션의 세션 상태(Before/After, 새로고침 라운드)와
    노출 이력이 서로 독립적으로 유지되도록 분리하는 키.
    """
    try:
        # 순위 추적용으로 실제 표시 개수보다 넓은 풀을 가져온다(POOL_MULTIPLIER 설명은
        # 상단 _MAIN_TOP_N 주석 참고). 화면에는 head(_MAIN_TOP_N)만 보여준다.
        before_pool_df, before_status, before_message = get_main_recommendations(
            user_id,
            model_type,
            "before",
            top_n=_MAIN_TOP_N * POOL_MULTIPLIER,
            graph_type=graph_type,
        )
        before_df = before_pool_df.head(_MAIN_TOP_N)
    except FileNotFoundError as e:
        st.error(f"데이터를 불러올 수 없습니다: `{e}`")
        return

    # ── 유저 유형 판별 (Twiddler 게이팅) ────────────────────────────────────
    if gate_cold_users:
        user_type_val = before_df["user_type"].iloc[0].upper() if not before_df.empty else "—"
        is_cold = user_type_val == "COLD"
    else:
        user_type_val, is_cold = "—", False

    # Cold 유저는 Twiddler 미적용 → "before"(인기도 기반)로 고정
    phase_key = f"{session_prefix}_twiddler_phase"
    twiddler_phase = "Before" if is_cold else st.session_state.get(phase_key, "After")

    # 개별 유저 카드는 _render_rerank_main 상단(페르소나 특성 카드 위)으로 옮겨서
    # 여기서는 중복 렌더링하지 않는다(요청 반영).

    try:
        after_df, after_status, after_message = get_main_recommendations(
            user_id,
            model_type,
            "after",
            top_n=_MAIN_TOP_N,
            graph_type=graph_type,
        )
    except FileNotFoundError as e:
        st.error(f"데이터를 불러올 수 없습니다: `{e}`")
        return

    if twiddler_phase == "Before":
        recs, status, message = before_df, before_status, before_message
    else:
        recs, status, message = after_df, after_status, after_message

    # 풀 전체(_MAIN_TOP_N * POOL_MULTIPLIER) 기준 순위로 추적 — Twiddler가 상위 노출권
    # 밖에서 끌어올린 상품도 정확한 "▲" 배지를 받도록 한다.
    rank_before_map = dict(zip(before_pool_df["item_id"].astype(int), before_pool_df["rank"]))

    # 카테고리 필터 적용
    items = recs.merge(products_df, on="item_id", how="left").pipe(
        _apply_filters, selected_categories, selected_types
    )

    # 모델 토글 바로 아래라 같은 모델명을 헤딩으로 또 띄우면 "ALS"가 중복으로 보인다.
    # Heavy 유저 캡션("🔥 Heavy 유저: Twiddler 적용 효과 비교")은 페이지 상단 캡션
    # ("User 00259 · HEAVY · ... 기준으로 Before/After 추천 순위를 비교합니다")과
    # 내용이 겹쳐서(요청 반영: "차라리 여기에 붙었으면") 제거 — Cold 유저는 상단
    # 캡션에 없는 새 정보(Twiddler 미적용·인기도 폴백)라 그대로 남긴다. LightGCN
    # bipartite 캡션("GNN 기반 협업 필터링(페르소나 미결합) · Twiddler 적용 효과 비교")도
    # 위 "모델 선택" 캡션과 페이지 상단 캡션 둘 다와 내용이 겹쳐서 같은 이유로 제거한다
    # (요청 반영: "빼던지 적절한위치에 넣던지").
    if is_cold:
        st.caption("🧊 Cold 유저: Twiddler 미적용, 인기도 기반 추천")

    # 모델 좌우 토글을 예전엔 이 위에 독립된 한 줄(+구분선)로 뒀는데, 세로 공간을
    # 줄이려는 목적으로(요청 반영: "als lightgcn 선택부를 줄일 목적으로 Twiddler
    # 비포애프터랑 한줄로") Before/After·새로고침 버튼과 같은 줄 3열로 합쳤다. 버튼
    # 클릭은 session_state를 바꾸고 즉시 st.rerun()하므로, 이 함수가 지금 어떤
    # model_type로 호출됐는지와 무관하게 다음 rerun에서 _render_rerank_main이 새로
    # 선택된 모델의 블록을 그린다 — 렌더링 위치를 옮겨도 로직에는 영향이 없다.
    # "Twiddler 재랭킹 근거"와 같은 크기의 소제목을 붙여 이 줄이 뭘 고르는 곳인지
    # 표시한다(요청 반영: "모델 선택 이런 거 위에 대제목이랑 크기 비슷하게").
    st.markdown('<div class="section-label">추천 결과 비교</div>', unsafe_allow_html=True)
    # st.caption + markdown **bold**는 실제로 font-weight:600까지만 올라가고 캡션
    # 기본 크기(0.875rem)에 묻혀 잘 안 보였다(요청으로 재확인: "볼드처리", "글씨체
    # 크기도 다른곳이랑 통일") — 페이지 상단 캡션과 같은 톤(.section-caption, 1rem
    # /700)으로 맞춘다.
    st.markdown(
        '<div class="section-caption">ALS와 LightGCN bipartite 중 추천에 사용할 모델을 선택합니다.</div>',
        unsafe_allow_html=True,
    )
    col_model, col_toggle, col_sim = st.columns([1.3, 1, 1.1])
    with col_model:
        _render_rerank_model_toggle()
    with col_toggle:
        _render_twiddler_toggle(phase_key, twiddler_phase, disabled=is_cold)
    with col_sim:
        sim_round = _render_refresh_simulation_button(
            user_id,
            is_cold,
            session_prefix=session_prefix,
            model_type=model_type,
            graph_type=graph_type,
            exposure_context=exposure_context,
        )

    # 새로고침 시뮬레이션은 Twiddler exposure decay(Rule 2)를 라운드마다 누적시키는
    # 기능이라 "Twiddler 미적용(Before)" 상태에서는 보여줄 게 없다 — 토글로 Before를
    # 선택하면 시뮬레이션 진행 중이어도 평소의 정적 "적용 전" 결과를 보여준다(요청 반영:
    # 이전엔 sim_round>0이면 토글 상태를 무시하고 항상 시뮬레이션 결과만 보여줬음).
    if sim_round > 0 and twiddler_phase != "Before":
        sim_history = st.session_state[f"sim_history_{user_id}_{session_prefix}"]
        sim_df, sim_status, sim_message = sim_history[sim_round - 1]
        sim_items = sim_df.merge(products_df, on="item_id", how="left").pipe(
            _apply_filters, selected_categories, selected_types
        )
        # 1회차는 "적용 전(before)" 전체 풀 순위 대비, 2회차부터는 직전 회차 대비 변동을
        # 보여준다 — 매 라운드 실제로 무엇이 오르고 내렸는지 명시적으로 드러난다(요청 반영).
        if sim_round == 1:
            sim_rank_before_map = rank_before_map
        else:
            prev_df = sim_history[sim_round - 2][0]
            sim_rank_before_map = dict(zip(prev_df["item_id"].astype(int), prev_df["rank"]))
        st.caption(
            f"새로고침 시뮬레이션 {sim_round}/{_SIM_ROUNDS}회차 (노출 이력이 실제로 누적됩니다)"
        )
        _render_model_status_or_grid(
            sim_status,
            sim_message,
            sim_items,
            id_key="item_id",
            user_id=user_id,
            model_key=f"{session_prefix}_sim_round{sim_round}",
            rank_before_map=sim_rank_before_map,
            products_df=products_df,
        )
    else:
        if sim_round > 0:
            st.caption(
                "⏸️ Twiddler 미적용 상태입니다. 새로고침 시뮬레이션은 Twiddler 적용(After) 상태에서만 순위가 변합니다."
            )
        _render_model_status_or_grid(
            status,
            message,
            items,
            id_key="item_id",
            user_id=user_id,
            model_key=f"{session_prefix}_{twiddler_phase.lower()}",
            rank_before_map=None if twiddler_phase == "Before" else rank_before_map,
            plain_rank_mode=(twiddler_phase == "Before"),
            products_df=products_df,
        )


def _render_rerank_main(
    selected_categories: list[str],
    selected_types: set[str] | None,
    demo_users_df: pd.DataFrame,
) -> None:
    """Tab 1(Twiddler 재랭킹) 메인페이지 — ALS ↔ LightGCN bipartite 좌우 토글로 전환.

    reports/UI_TAB_RESTRUCTURE_PLAN.md 기준 재구성: 기존 LightGCN 비교 블록과
    "ALS vs LightGCN 공통 추천 상품 수 / Jaccard 유사도" 지표는 서로 다른 독립변수를
    섞는 지표라 제거했다(사용자 확인 완료) — LightGCN tripartite 비교는 _render_persona_tab으로
    이관돼 있다. 이 화면에는 LightGCN bipartite(페르소나 미결합)만 ALS와 나란히 비교
    노출한다(reports/LIGHTGCN_BIPARTITE_TWIDDLER_PLAN.md).
    """
    rerank_page = st.session_state.get("rerank_page", "user")
    user_id, user_info = get_current_user_selection(demo_users_df)

    if rerank_page == "user":
        st.title("페르소나 및 유저 선택")
        st.caption("추천 비교에 사용할 페르소나와 데모 유저를 선택합니다.")
        user_id, user_info = render_persona_and_user_selector(demo_users_df)
        render_user_summary_card(user_id, user_info)

        # 선택 화면이 텍스트만 있어 밋밋하다는 피드백으로, 이 유저에게 Twiddler 적용
        # 전(before) 원래 보여지는 추천 상품을 미리보기로 깔아준다(요청 반영: "무작위로
        # 원래 보여지는 상품들"). 클릭 상호작용은 "추천 비교" 화면 몫이라 상세보기
        # 버튼 없이 카드만 조용히 보여준다(show_detail_button=False).
        st.divider()
        # 이 미리보기는 항상 ALS 기준으로 고정돼 있다(사이드바 모델 선택과 무관) —
        # 오해 없도록 라벨에 명시한다(요청 반영: "ALS 모델이라고 쓰라").
        # 글자 크기를 한 단계 줄여달라는 요청 반영 — 기본 굵은 텍스트(1rem)보다 한
        # 단계 작은 0.9rem으로.
        st.markdown(
            '<span style="font-size:0.9rem; font-weight:700;">'
            "Twiddler 적용 전 추천 (ALS 기준)</span>",
            unsafe_allow_html=True,
        )
        try:
            products_df = load_products()
            preview_df, _, _ = get_main_recommendations(
                user_id, "ALS", "before", top_n=4, graph_type="tripartite"
            )
            preview_items = preview_df.sample(frac=1, random_state=None).merge(
                products_df, on="item_id", how="left"
            )
            _render_recommend_grid(
                preview_items,
                user_id=user_id,
                plain_rank_mode=True,
                ncols=4,
                show_detail_button=False,
                products_df=products_df,
            )
        except FileNotFoundError as e:
            st.error(f"데이터를 불러올 수 없습니다: `{e}`")
        return

    if rerank_page == "metrics":
        st.title("오프라인 성능 지표")
        st.caption("현재 선택된 모델과 페르소나 기준으로 Before/After 성능 지표를 확인합니다.")
        model_choice = _render_rerank_model_toggle()
        if model_choice == "ALS":
            eval_context, eval_label = "main", "ALS"
        else:
            eval_context, eval_label = "main_lightgcn_bipartite", "LightGCN bipartite"
        st.markdown(f"### {eval_label} only vs {eval_label}+Twiddler")
        render_eval_metrics(context=eval_context, persona_label=user_info["persona_label"])
        return

    st.title("추천 비교")
    # 유저 식별 정보(User/HEAVY/페르소나명)만 검정으로 강조하고 나머지 설명은 기존
    # 캡션 색(진한 슬레이트)을 유지한다(요청 반영: "이 부분은 검정색으로").
    st.caption(
        f'<span style="color:#000000;">User {user_id:05d} · '
        f"{user_info['user_type_label'].upper()} · {user_info['persona_label']}</span> "
        # 그냥 "Before/After"만 쓰면 뭐가 바뀌기 전/후인지 안 드러난다는 피드백
        # (요청 반영: "어떤 비포애프터를 말하는건지 안와닿음") — Twiddler 재랭킹을
        # 명시한다.
        "기준으로 Twiddler 재랭킹 적용 전(Before)/후(After) 추천 순위를 비교합니다.",
        unsafe_allow_html=True,
    )
    render_user_summary_card(user_id, user_info, show_persona=False)
    # "왜 이렇게 재정렬됐는지"(alpha/decay/선호 카테고리)는 실제 Before/After 비교
    # 결과 바로 위에 있어야 근거-결과가 붙어 보인다(요청 반영: "재랭킹근거를 추천 결과
    # 비교 페이지로 빼는게 더 맞지 않나?" — 기존엔 "유저 선택" 페이지에 따로 있어
    # 실제 비교 결과와 떨어져 보였음).
    render_user_twiddler_case(user_id)
    st.divider()

    try:
        products_df = load_products()
    except FileNotFoundError as e:
        st.error(f"데이터를 불러올 수 없습니다: `{e}`")
        return

    # 모델 토글 버튼 자체는 _render_model_twiddler_block 안에서 그린다(내부 호출) —
    # 여기서는 session_state에 저장된 현재 선택값만 읽어 어느 블록을 렌더링할지 정한다.
    model_choice = st.session_state.get("rerank_model_type", "ALS")

    if model_choice == "ALS":
        _render_model_twiddler_block(
            model_type="ALS",
            graph_type="tripartite",
            session_prefix="als",
            exposure_context="main",
            section_title="ALS",
            eval_label="ALS",
            eval_context="main",
            gate_cold_users=True,
            user_id=user_id,
            user_info=user_info,
            products_df=products_df,
            selected_categories=selected_categories,
            selected_types=selected_types,
        )
    else:
        _render_model_twiddler_block(
            model_type="LightGCN",
            graph_type="bipartite",
            session_prefix="lgcn_bipartite",
            exposure_context="main_lightgcn_bipartite",
            section_title="LightGCN bipartite",
            eval_label="LightGCN bipartite",
            eval_context="main_lightgcn_bipartite",
            gate_cold_users=False,
            user_id=user_id,
            user_info=user_info,
            products_df=products_df,
            selected_categories=selected_categories,
            selected_types=selected_types,
        )


# ── Tab 1: Twiddler 재랭킹 — 상세(연관 상품) 화면 ──────────────────────────────────


def _render_rerank_detail(demo_users_df: pd.DataFrame) -> None:
    """Tab 1(Twiddler 재랭킹) 상세페이지 서브탭 — 보완재 vs 보완재+Twiddler."""
    if st.button("← 메인으로 돌아가기"):
        st.session_state["view"] = "main"
        st.rerun()

    st.title("추천 비교 · 연관 상품")

    item_id = st.session_state.get("selected_item_id")
    if item_id is None:
        st.warning("메인 화면에서 상품 카드를 클릭해 주세요.")
        return

    # ── 유저 소개 — 메인 화면과 동일한 선택 위젯을 그대로 재사용해, 같은 상품에 대해
    #    페르소나/유저를 바꿔가며 보완재 추천 결과를 비교할 수 있게 한다. 좌측 사이드바의
    #    "추천 결과 비교"/"오프라인 성능 지표" 서브페이지와 무관하게 항상 공통으로 보여준다
    #    (요청 반영: 유저 선택만은 별도 탭으로 안 빼고 지금처럼 유지).
    user_id, user_info = render_persona_and_user_selector(demo_users_df)
    render_user_summary_card(user_id, user_info)
    render_user_twiddler_case(user_id)
    st.divider()

    detail_page = st.session_state.get("rerank_detail_page", "compare")
    if detail_page == "metrics":
        st.markdown("### 오프라인 성능 지표 (보완재 only vs 보완재+Twiddler)")
        render_eval_metrics(context="detail", persona_label=user_info["persona_label"])
        return

    try:
        products_df = load_products()
        # 순위 추적용으로 실제 표시 개수보다 넓은 풀을 가져온다(상단 _MAIN_TOP_N 주석의
        # POOL_MULTIPLIER 설명 참고). 화면에는 head(_DETAIL_TOP_N)만 보여준다.
        cf_before_pool_df, before_status, before_message = get_detail_recommendations(
            item_id, top_n=_DETAIL_TOP_N * POOL_MULTIPLIER, user_id=user_id, twiddler="before"
        )
        cf_before_df = cf_before_pool_df.head(_DETAIL_TOP_N)
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
            item_id, top_n=_DETAIL_TOP_N, user_id=user_id, twiddler="after"
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

    # 풀 전체(_DETAIL_TOP_N * POOL_MULTIPLIER) 기준 순위로 추적 — Twiddler가 상위 노출권
    # 밖에서 끌어올린 상품도 정확한 "▲" 배지를 받도록 한다.
    rank_before_map = dict(
        zip(cf_before_pool_df["rec_item_id"].astype(int), cf_before_pool_df["rank"])
    )

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
    # 카탈로그 스냅샷에 없는 rec_item_id(배치 산출물과 버전이 어긋난 경우)는 name이
    # NaN이 되어 product_card의 extract_color(name.split())에서 크래시하므로 제거한다
    # (요청으로 발견).
    cf_recs = cf_recs[cf_recs["name"].notna()]

    if cf_recs.empty:
        st.info("연관 상품 데이터가 없습니다.")
        return

    _render_recommend_grid(
        cf_recs,
        id_key="rec_item_id",
        user_id=user_id,
        model_key=f"cf_{twiddler_phase.lower()}",
        rank_before_map=None if twiddler_phase == "Before" else rank_before_map,
        plain_rank_mode=(twiddler_phase == "Before"),
        ncols=4,
        show_detail_button=False,
    )


# ── Tab 2: 페르소나 기여도 ──────────────────────────────────────────────────────


def _render_persona_tab(
    selected_categories: list[str],
    selected_types: set[str] | None,
    demo_users_df: pd.DataFrame,
) -> None:
    """Tab 2(효과 해석) — LightGCN bi-graph vs tri-graph 페르소나 기여도.

    추천 결과 카드 자체는 "추천 비교" 탭에서 이미 보여주므로 여기서 중복으로 그리지
    않는다(요청 반영: 팀원 피드백 — "추천 결과는 트위들러탭에서 보여주니까 효과해석은
    지금 비워진 두 부분만 들어가도 될 것 같다"). 두 정량 비교(bi vs tri HR@K/NDCG@K,
    페르소나 유무 순위·카테고리 분포)는 retail-clickstream-analysis #41(tri/bipartite
    CSV 파일명 분리) 이후 tri 학습 산출물이 생겨 계산 가능해졌다 —
    src/evaluation/evaluate_lightgcn_persona_effect.py 배치 스크립트 실행 결과를 읽어온다.
    """
    st.title("효과 해석")
    st.caption(
        "bi-graph는 유저-상품 관계만, tri-graph는 여기에 페르소나 노드를 추가로 연결해 "
        "학습합니다.<br>"
        "두 그래프의 추천 결과를 비교하면 페르소나 정보가 추천 정확도에 실제로 기여하는지 "
        "확인할 수 있습니다.",
        unsafe_allow_html=True,
    )

    # 사이드바 앵커 링크(#section-persona-*)가 스크롤해 갈 빈 표식 — 실제 텍스트 없이
    # id만 심어둔다(요청 반영: "효과 해석도 사이드바 목차 필요할 것 같은데" 논의 결과,
    # 페이지 분리 대신 같은 페이지 안 스크롤 이동으로 결정).
    st.markdown('<div id="section-persona-user"></div>', unsafe_allow_html=True)
    user_id, user_info = render_persona_and_user_selector(demo_users_df)
    render_user_summary_card(user_id, user_info)

    st.divider()
    st.markdown('<div id="section-persona-accuracy"></div>', unsafe_allow_html=True)
    st.markdown("### bi-graph vs tri-graph 정량 비교 (HR@K / NDCG@K)")
    render_lightgcn_persona_accuracy()

    st.divider()
    st.markdown('<div id="section-persona-rankshift"></div>', unsafe_allow_html=True)
    st.markdown("### 페르소나 유무에 따른 순위 · 카테고리 분포 비교")
    render_lightgcn_persona_rank_shift_and_category()

    st.divider()
    st.markdown('<div id="section-persona-graph"></div>', unsafe_allow_html=True)
    render_user_graph(user_id)


# ── 라우터 ─────────────────────────────────────────────────────────────────────


def main() -> None:
    if "view" not in st.session_state:
        st.session_state["view"] = "main"
    if "main_tab" not in st.session_state:
        # 접속 시 기본 진입 탭 — 프로젝트 소개(요청 반영: "새로고침 시 기본 페이지
        # 프로젝트 소개로 오게 수정" — 데모 체험 전에 프로젝트 맥락부터 보게 한다).
        st.session_state["main_tab"] = "project"
    if st.session_state.get("main_tab") in ("glossary", "team"):
        st.session_state["project_page"] = st.session_state["main_tab"]
        st.session_state["main_tab"] = "project"

    _render_top_navbar()
    selected_categories, selected_types, demo_users_df = _setup_sidebar()

    if demo_users_df is None:
        st.warning("데이터를 불러올 수 없어 유저를 선택할 수 없습니다.")
        return

    main_tab = st.session_state.get("main_tab", "project")
    view = st.session_state["view"]
    if main_tab == "rerank":
        if view == "main":
            _render_rerank_main(selected_categories, selected_types, demo_users_df)
        elif view == "detail":
            _render_rerank_detail(demo_users_df)
    elif main_tab == "persona":
        _render_persona_tab(selected_categories, selected_types, demo_users_df)
    elif main_tab == "userlist":
        render_user_list(demo_users_df)
    elif main_tab == "project":
        project_page = st.session_state.get("project_page", "about")
        if project_page == "glossary":
            render_glossary()
        elif project_page == "team":
            render_team_page()
        elif project_page == "intro":
            render_project_intro()
        else:
            render_project_about()


main()
