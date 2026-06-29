import sys
from pathlib import Path

import streamlit as st

# 페이지 config는 가장 먼저 호출해야 함
st.set_page_config(
    page_title="추천 시스템 데모",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# app/ 디렉토리를 sys.path에 추가
_APP_DIR = Path(__file__).parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils.setup import common_setup

common_setup()

# ── 홈 페이지 본문 ─────────────────────────────────────────────────────────────
st.title("🛍️ 추천 시스템 데모")
st.markdown(
    """
이 데모는 이커머스 클릭스트림 데이터 기반의 **ALS(베이스라인)** 와 **LightGCN** 추천 시스템의
결과를 시각적으로 비교합니다. 왼쪽 사이드바에서 유저를 선택한 뒤 아래 페이지로 이동하세요.
"""
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 메인 추천 비교")
    st.markdown(
        "동일 유저에 대해 **ALS** 와 **LightGCN** 의 추천 상위 10개를 상품 카드로 나란히 비교합니다."
        " 공통 추천 상품 배지와 Jaccard 유사도를 함께 표시합니다."
    )

with col2:
    st.markdown("### 🔄 Twiddler 비교")
    st.markdown(
        "페르소나 후처리(**Twiddler**) 적용 전/후 추천 순위 변화를 **Bump Chart** 로 시각화합니다."
        " 순위 상승·하락·유지 항목을 색상으로 구분합니다."
    )

with col3:
    st.markdown("### 🛒 연관 상품 추천")
    st.markdown(
        "특정 상품을 선택하면 **대체재(콘텐츠 기반)** 와 **보완재(Item-based CF)** 연관 상품을"
        " 함께 표시합니다."
    )

st.markdown("---")
st.caption("데이터: 합성 이커머스 클릭스트림 | 모델: ALS · LightGCN | 후처리: Twiddler")
