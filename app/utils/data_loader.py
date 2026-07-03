import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_SOURCE = "csv"  # "csv" 또는 "sqlite"
DATA_DIR = Path("data/dashboard")
SQLITE_PATH = DATA_DIR / "recommend.db"

# products.csv 원본 컬럼: product_id, category, name, price_usd, cost_usd, margin_usd
# UI에서는 item_id 로 통일하여 사용
_PRODUCTS_RENAME = {"product_id": "item_id"}
_PRODUCTS_COLS   = ["item_id", "name", "category", "price_usd"]


@st.cache_data
def load_recommendations() -> pd.DataFrame:
    """PRED_MAIN_RECOMMEND 로드"""
    if DATA_SOURCE == "sqlite":
        with sqlite3.connect(SQLITE_PATH) as conn:
            return pd.read_sql("SELECT * FROM pred_main_recommend", conn)
    return pd.read_csv(DATA_DIR / "PRED_MAIN_RECOMMEND.csv")


@st.cache_data
def load_detail_recommendations() -> pd.DataFrame:
    """PRED_DETAIL_RECOMMEND 로드"""
    if DATA_SOURCE == "sqlite":
        with sqlite3.connect(SQLITE_PATH) as conn:
            return pd.read_sql("SELECT * FROM pred_detail_recommend", conn)
    return pd.read_csv(DATA_DIR / "PRED_DETAIL_RECOMMEND.csv")


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
def load_persona_labels() -> pd.DataFrame:
    """persona_labels 로드"""
    if DATA_SOURCE == "sqlite":
        with sqlite3.connect(SQLITE_PATH) as conn:
            return pd.read_sql("SELECT * FROM persona_labels", conn)
    return pd.read_csv(DATA_DIR / "persona_labels.csv")


@st.cache_data
def load_demo_users() -> pd.DataFrame:
    """demo_users 로드"""
    return pd.read_csv(DATA_DIR / "demo_users.csv")


@st.cache_data
def load_categories() -> list[str]:
    """products에서 유니크 카테고리 목록 반환 (정렬)"""
    return sorted(load_products()["category"].unique().tolist())


def get_user_recommendations(
    rec_df: pd.DataFrame,
    user_id: int,
    model_type: str,
    twiddler: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """특정 유저의 모델/트위들러 조합 추천 결과 반환"""
    return (
        rec_df[
            (rec_df["user_id"] == user_id)
            & (rec_df["model_type"] == model_type)
            & (rec_df["twiddler"] == twiddler)
        ]
        .sort_values("rank")
        .head(top_n)
        .reset_index(drop=True)
    )
