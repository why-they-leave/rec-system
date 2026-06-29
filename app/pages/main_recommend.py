import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="메인 추천 비교", page_icon="📊", layout="wide")

_APP_DIR = Path(__file__).parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils.setup import common_setup, ALL_CATEGORIES
from utils.data_loader import load_recommendations, load_products, get_user_recommendations
from components.product_card import render_product_card

user_id = common_setup()

# ── 본문 ───────────────────────────────────────────────────────────────────────
st.title("📊 메인 화면 추천 비교")

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

st.subheader(f"User {user_id:03d}")

# twiddler='before' 기준으로 두 모델 비교
als_recs  = get_user_recommendations(rec_df, user_id, "ALS",      "before")
gcn_recs  = get_user_recommendations(rec_df, user_id, "LightGCN", "before")

# 상품 정보 병합 후 선택 카테고리로 필터
als_items = (
    als_recs.merge(products_df, on="item_id", how="left")
    .pipe(lambda df: df[df["category"].isin(selected_categories)])
)
gcn_items = (
    gcn_recs.merge(products_df, on="item_id", how="left")
    .pipe(lambda df: df[df["category"].isin(selected_categories)])
)

# 공통 추천 상품 및 Jaccard 유사도 계산 (카테고리 필터 적용 전 기준)
all_als_ids = set(als_recs["item_id"].tolist())
all_gcn_ids = set(gcn_recs["item_id"].tolist())
common_ids  = all_als_ids & all_gcn_ids
union_ids   = all_als_ids | all_gcn_ids
jaccard     = len(common_ids) / len(union_ids) if union_ids else 0.0

if als_items.empty and gcn_items.empty:
    st.info("선택한 카테고리에 해당하는 추천 상품이 없습니다. 사이드바에서 카테고리를 확인해 주세요.")
    st.stop()


def _render_grid(items_df, common):
    if items_df.empty:
        st.info("해당 카테고리 추천 결과 없음")
        return
    rows = [items_df.iloc[i : i + 3] for i in range(0, len(items_df), 3)]
    for row_chunk in rows:
        cols = st.columns(3)
        for col, (_, item) in zip(cols, row_chunk.iterrows()):
            with col:
                badge = "🔁 공통 추천" if item["item_id"] in common else None
                render_product_card(item, int(item["rank"]), badge)


col_als, col_gcn = st.columns(2)

with col_als:
    st.markdown("### 🤖 ALS (베이스라인)")
    _render_grid(als_items, common_ids)

with col_gcn:
    st.markdown("### 🚀 LightGCN")
    _render_grid(gcn_items, common_ids)

# ── 하단 지표 ──────────────────────────────────────────────────────────────────
st.divider()

user_type_val = "—"
user_rows = rec_df[rec_df["user_id"] == user_id]
if not user_rows.empty:
    user_type_val = user_rows["user_type"].iloc[0].upper()

m1, m2, m3 = st.columns(3)
m1.metric("공통 추천 상품", f"{len(common_ids)}개 / 10개")
m2.metric("Jaccard 유사도", f"{jaccard:.3f}")
m3.metric("유저 유형", user_type_val)
