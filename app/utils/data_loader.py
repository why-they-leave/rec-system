import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from . import api_client

DATA_SOURCE = "sqlite"  # "csv" 또는 "sqlite" — 카탈로그성 데이터(상품/유저)에만 적용.
                        # 추천 데이터는 더 이상 여기서 읽지 않고 backend API를 호출한다(api_client 참고).
                        # recommend.db는 scripts/generate_demo_data.py가 생성한다.
DATA_DIR = Path("data/dashboard")
SQLITE_PATH = DATA_DIR / "recommend.db"

# products.csv 원본 컬럼: product_id, category, name, price_usd, cost_usd, margin_usd
# UI에서는 item_id 로 통일하여 사용
_PRODUCTS_RENAME = {"product_id": "item_id"}
_PRODUCTS_COLS   = ["item_id", "name", "category", "price_usd"]

_MAIN_REC_COLUMNS   = ["user_id", "item_id", "score", "rank", "model_type", "twiddler", "user_type"]
_DETAIL_REC_COLUMNS = ["item_id", "rec_item_id", "score", "rank", "twiddler"]


@st.cache_data
def load_products() -> pd.DataFrame:
    """products 로드. product_id → item_id 변환, UI 필요 컬럼만 반환."""
    if DATA_SOURCE == "sqlite":
        with sqlite3.connect(SQLITE_PATH) as conn:
            df = pd.read_sql("SELECT * FROM products", conn)
    else:
        df = pd.read_csv(DATA_DIR / "products.csv")
    if "product_id" in df.columns:
        df = df.rename(columns=_PRODUCTS_RENAME)
    return df[_PRODUCTS_COLS]


@st.cache_data
def load_demo_users() -> pd.DataFrame:
    """demo_users 로드"""
    if DATA_SOURCE == "sqlite":
        with sqlite3.connect(SQLITE_PATH) as conn:
            return pd.read_sql("SELECT * FROM demo_users", conn)
    return pd.read_csv(DATA_DIR / "demo_users.csv")


@st.cache_data
def load_categories() -> list[str]:
    """products에서 유니크 카테고리 목록 반환 (정렬)"""
    return sorted(load_products()["category"].unique().tolist())


@st.cache_data(ttl=30)
def get_main_recommendations(
    user_id: int,
    model_type: str,
    twiddler: str = "before",
    top_n: int = 10,
    graph_type: str = "tripartite",
) -> tuple[pd.DataFrame, str, str | None]:
    """
    특정 유저/모델/twiddler 조합의 추천을 backend API에서 조회한다.
    graph_type은 model_type="LightGCN"일 때만 의미를 가진다("bipartite" | "tripartite").
    반환: (df, status, message) — status가 "not_implemented"면 df는 빈 DataFrame(컬럼은 유지).
    연결 실패 시 api_client.BackendUnavailableError가 그대로 전파된다.
    """
    data = api_client.get_main_recommendations(user_id, model_type, twiddler, top_n, graph_type)
    df = pd.DataFrame(data["items"], columns=_MAIN_REC_COLUMNS)
    return df, data["status"], data.get("message")


@st.cache_data(ttl=30)
def get_detail_recommendations(
    item_id: int,
    top_n: int = 8,
    user_id: int | None = None,
    twiddler: str = "before",
) -> tuple[pd.DataFrame, str, str | None]:
    """
    특정 상품의 보완재(함께 구매하면 좋은 상품) 추천을 backend API에서 조회한다.
    user_id가 주어지면 Twiddler(페르소나 재랭킹)를 적용할 수 있다(twiddler="before"|"after").
    반환: (df, status, message) — status가 "not_implemented"면 df는 빈 DataFrame(컬럼은 유지).
    """
    data = api_client.get_detail_recommendations(item_id, top_n, user_id, twiddler)
    df = pd.DataFrame(data["items"], columns=_DETAIL_REC_COLUMNS)
    return df, data["status"], data.get("message")
