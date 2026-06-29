import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="연관 상품 추천", page_icon="🛒", layout="wide")

_APP_DIR = Path(__file__).parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils.setup import common_setup, ALL_CATEGORIES
from utils.data_loader import load_detail_recommendations, load_products
from components.product_card import render_product_card, render_current_product_card

common_setup()

# ── 본문 ───────────────────────────────────────────────────────────────────────
st.title("🛒 상세 페이지 연관 상품 추천")

try:
    detail_df   = load_detail_recommendations()
    products_df = load_products()
except FileNotFoundError as e:
    st.error(f"데이터 파일을 찾을 수 없습니다: `{e}`")
    st.stop()

selected_categories = st.session_state.get("selected_categories", ALL_CATEGORIES)

# 카테고리 필터 적용 후 상품 드롭다운
products_filtered = (
    products_df[products_df["category"].isin(selected_categories)]
    .sort_values(["category", "name"])
)

if products_filtered.empty:
    st.info("선택한 카테고리에 해당하는 상품이 없습니다. 사이드바에서 카테고리를 확인해 주세요.")
    st.stop()

item_options   = products_filtered["item_id"].tolist()
item_label_map = {
    int(row["item_id"]): f"{row['category']} > {row['name']}"
    for _, row in products_filtered.iterrows()
}

selected_item_id = st.selectbox(
    "상품 선택",
    options=item_options,
    format_func=lambda iid: item_label_map.get(int(iid), str(iid)),
    key="detail_item",
)

# 현재 상품 가로형 강조 카드
current_item = products_df[products_df["item_id"] == selected_item_id]
if current_item.empty:
    st.warning("상품 정보를 찾을 수 없습니다.")
    st.stop()

render_current_product_card(current_item.iloc[0])

# 연관 상품 필터 + 카테고리 적용
item_detail = detail_df[detail_df["item_id"] == selected_item_id]

content_recs = (
    item_detail[item_detail["rec_type"] == "content"]
    .sort_values("rank")
    .head(8)
    .merge(products_df, left_on="rec_item_id", right_on="item_id",
           how="left", suffixes=("_detail", ""))
    .pipe(lambda df: df[df["category"].isin(selected_categories)])
)
cf_recs = (
    item_detail[item_detail["rec_type"] == "cf"]
    .sort_values("rank")
    .head(8)
    .merge(products_df, left_on="rec_item_id", right_on="item_id",
           how="left", suffixes=("_detail", ""))
    .pipe(lambda df: df[df["category"].isin(selected_categories)])
)

col_content, col_cf = st.columns(2)


def _render_rec_grid(recs_df, cols_per_row: int = 4):
    if recs_df.empty:
        st.info("연관 상품 데이터가 없습니다.")
        return
    rows = [recs_df.iloc[i : i + cols_per_row] for i in range(0, len(recs_df), cols_per_row)]
    for row_chunk in rows:
        cols = st.columns(cols_per_row)
        for col, (_, item) in zip(cols, row_chunk.iterrows()):
            with col:
                render_product_card(item, int(item["rank"]))


with col_content:
    st.markdown("### 🔄 대체재 (콘텐츠 기반)")
    _render_rec_grid(content_recs)

with col_cf:
    st.markdown("### 🛒 보완재 (Item-based CF)")
    _render_rec_grid(cf_recs)
