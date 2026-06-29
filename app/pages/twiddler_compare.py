import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Twiddler 비교", page_icon="🔄", layout="wide")

_APP_DIR = Path(__file__).parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils.setup import common_setup, ALL_CATEGORIES
from utils.data_loader import load_recommendations, load_products, get_user_recommendations
from components.product_card import render_product_card
from components.metric_chart import render_bump_chart, COLOR_UP, COLOR_DOWN, COLOR_SAME

user_id = common_setup()

# ── 본문 ───────────────────────────────────────────────────────────────────────
st.title("🔄 Twiddler 전/후 비교")

if user_id is None:
    st.warning("사이드바에서 유저를 선택해 주세요.")
    st.stop()

try:
    rec_df      = load_recommendations()
    products_df = load_products()
except FileNotFoundError as e:
    st.error(f"데이터 파일을 찾을 수 없습니다: `{e}`")
    st.stop()

selected_categories = st.session_state.get("selected_categories", ALL_CATEGORIES)
model_type = st.selectbox("모델 선택", ["ALS", "LightGCN"], key="twiddler_model")

st.subheader(f"User {user_id:03d} — {model_type}")

before_recs = get_user_recommendations(rec_df, user_id, model_type, "before")
after_recs  = get_user_recommendations(rec_df, user_id, model_type, "after")

if before_recs.empty or after_recs.empty:
    st.warning("해당 유저의 추천 데이터가 없습니다.")
    st.stop()

# ── Bump Chart ─────────────────────────────────────────────────────────────────
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

# ── 순위 변화별 카드 (카테고리 필터 적용) ─────────────────────────────────────
import pandas as pd  # noqa: E402

merged = (
    before_recs[["item_id", "rank"]]
    .rename(columns={"rank": "rank_before"})
    .merge(
        after_recs[["item_id", "rank"]].rename(columns={"rank": "rank_after"}),
        on="item_id",
        how="outer",
    )
    .merge(products_df, on="item_id", how="left")
    .pipe(lambda df: df[df["category"].isin(selected_categories)])
)

if merged.empty:
    st.info("선택한 카테고리에 해당하는 추천 상품이 없습니다.")
    st.stop()

up_items   = merged[merged["rank_after"] < merged["rank_before"]].sort_values("rank_after")
down_items = merged[merged["rank_after"] > merged["rank_before"]].sort_values("rank_after")
same_items = merged[merged["rank_after"] == merged["rank_before"]].sort_values("rank_after")

st.divider()
col_up, col_down, col_same = st.columns(3)

with col_up:
    st.markdown(
        f'<p style="color:{COLOR_UP}; font-weight:700; font-size:1rem;">🔼 순위 상승 ({len(up_items)})</p>',
        unsafe_allow_html=True,
    )
    for _, item in up_items.iterrows():
        delta = int(item["rank_before"]) - int(item["rank_after"])
        render_product_card(item, int(item["rank_after"]), f"▲ +{delta}")

with col_down:
    st.markdown(
        f'<p style="color:{COLOR_DOWN}; font-weight:700; font-size:1rem;">🔽 순위 하락 ({len(down_items)})</p>',
        unsafe_allow_html=True,
    )
    for _, item in down_items.iterrows():
        delta = int(item["rank_after"]) - int(item["rank_before"])
        render_product_card(item, int(item["rank_after"]), f"▼ -{delta}")

with col_same:
    st.markdown(
        f'<p style="color:{COLOR_SAME}; font-weight:700; font-size:1rem;">➡ 유지 ({len(same_items)})</p>',
        unsafe_allow_html=True,
    )
    for _, item in same_items.iterrows():
        render_product_card(item, int(item["rank_after"]), "➡ 유지")
