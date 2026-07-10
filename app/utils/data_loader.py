import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.api.core import (
    get_detail_recommendation_items,
    get_main_recommendation_items,
    get_user_subgraph_items,
)
from backend.api.core import get_user_twiddler_case as _get_user_twiddler_case_core
from backend.api.core import reset_user_exposure as _reset_user_exposure_core

DATA_SOURCE = "sqlite"  # "csv" 또는 "sqlite" — 카탈로그성 데이터(상품/유저)에만 적용.
# recommend.db는 scripts/generate_demo_data.py가 생성한다.
# 추천 데이터는 backend.api.core를 Streamlit 프로세스 안에서 직접 호출한다
# (HTTP 왕복 없음 — Streamlit Community Cloud처럼 프로세스를 하나만
# 띄울 수 있는 환경에서도 동작하게 하기 위함).
DATA_DIR = Path("data/dashboard")
SQLITE_PATH = DATA_DIR / "recommend.db"
EVAL_DIR = Path("data/outputs/eval")

# products.csv 원본 컬럼: product_id, category, name, price_usd, cost_usd, margin_usd
# UI에서는 item_id 로 통일하여 사용
_PRODUCTS_RENAME = {"product_id": "item_id"}
_PRODUCTS_COLS = ["item_id", "name", "category", "price_usd"]

_MAIN_REC_COLUMNS = ["user_id", "item_id", "score", "rank", "model_type", "twiddler", "user_type"]
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


@st.cache_data
def load_twiddler_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    """src/evaluation/evaluate_twiddler.py가 생성한 사전계산 정확도/다양성 지표 CSV를 읽는다.

    유저별 실시간 조회가 아니라 population 전체 평균(aggregate) 값이라 backend/api/core를
    거치지 않고 이 파일에서 직접 CSV를 읽는다(load_products()와 동일한 카탈로그성 데이터 취급).
    """
    accuracy_df = pd.read_csv(EVAL_DIR / "twiddler_accuracy.csv")
    diversity_df = pd.read_csv(EVAL_DIR / "twiddler_diversity.csv")
    return accuracy_df, diversity_df


@st.cache_data
def load_lightgcn_persona_accuracy() -> pd.DataFrame:
    """src/evaluation/evaluate_lightgcn_persona_effect.py가 생성한 bi/tri HR@K 비교 CSV."""
    return pd.read_csv(EVAL_DIR / "lightgcn_persona_accuracy.csv")


@st.cache_data
def load_lightgcn_persona_rank_shift() -> pd.DataFrame:
    """evaluate_lightgcn_persona_effect.py가 생성한 bi/tri 순위 이동 비교 CSV(단일 행)."""
    return pd.read_csv(EVAL_DIR / "lightgcn_persona_rank_shift.csv")


@st.cache_data
def load_lightgcn_persona_category_share() -> pd.DataFrame:
    """evaluate_lightgcn_persona_effect.py가 생성한 bi/tri 카테고리 구성비 비교 CSV."""
    return pd.read_csv(EVAL_DIR / "lightgcn_persona_category_share.csv")


def _main_rec_df(user_id: int, items: list[dict], model_type: str, twiddler: str) -> pd.DataFrame:
    rows = [
        {
            "user_id": user_id,
            "item_id": item["item_id"],
            "score": item["score"],
            "rank": item["rank"],
            "model_type": model_type,
            "twiddler": twiddler,
            "user_type": item["user_type"],
        }
        for item in items
    ]
    return pd.DataFrame(rows, columns=_MAIN_REC_COLUMNS)


@st.cache_data(ttl=30)
def get_main_recommendations(
    user_id: int,
    model_type: str,
    twiddler: str = "before",
    top_n: int = 10,
    graph_type: str = "tripartite",
) -> tuple[pd.DataFrame, str, str | None]:
    """
    특정 유저/모델/twiddler 조합의 추천을 backend.api.core에서 직접 조회한다(HTTP 없음).
    graph_type은 model_type="LightGCN"일 때만 의미를 가진다("bipartite" | "tripartite").
    반환: (df, status, message) — status가 "not_implemented"면 df는 빈 DataFrame(컬럼은 유지).
    """
    items, status, message, response_model_type, response_twiddler = get_main_recommendation_items(
        user_id, model_type, graph_type, twiddler, top_n
    )
    df = _main_rec_df(user_id, items, response_model_type, response_twiddler)
    return df, status, message


def simulate_next_session(
    user_id: int,
    top_n: int = 10,
    model_type: str = "ALS",
    graph_type: str = "tripartite",
) -> tuple[pd.DataFrame, str, str | None]:
    """새로고침 시뮬레이션 전용 — 캐시를 붙이지 않는다.

    get_main_recommendations()는 @st.cache_data(ttl=30)라 30초 내 같은 인자로 다시 부르면
    캐시된 결과만 돌려주고 실제로는 재계산하지 않는다 — twiddler_service.apply_twiddler가
    호출되지 않으니 exposure_service.record_exposure도 호출 안 돼 "버튼을 누를 때마다 다음
    라운드로 진행"하는 시뮬레이션 의도가 깨진다. 이 함수는 캐시 없이 매 호출마다 실제로
    backend.api.core를 다시 타서 노출 이력이 실제로 누적되게 한다.

    model_type/graph_type: ALS(기본값) 또는 LightGCN-bipartite(model_type="LightGCN",
    graph_type="bipartite") 새로고침 시뮬레이션에서 재사용한다.
    """
    items, status, message, response_model_type, response_twiddler = get_main_recommendation_items(
        user_id, model_type, graph_type, twiddler="after", top_n=top_n
    )
    df = _main_rec_df(user_id, items, response_model_type, response_twiddler)
    return df, status, message


def reset_user_exposure(user_id: int, context: str = "main") -> None:
    """새로고침 시뮬레이션을 매번 라운드 0(깨끗한 노출 이력)에서 시작하기 위한 리셋."""
    _reset_user_exposure_core(user_id, context)


@st.cache_data(ttl=30)
def get_detail_recommendations(
    item_id: int,
    top_n: int = 8,
    user_id: int | None = None,
    twiddler: str = "before",
) -> tuple[pd.DataFrame, str, str | None]:
    """
    특정 상품의 보완재(함께 구매하면 좋은 상품) 추천을 backend.api.core에서 직접 조회한다(HTTP 없음).
    user_id가 주어지면 Twiddler(페르소나 재랭킹)를 적용할 수 있다(twiddler="before"|"after").
    반환: (df, status, message) — status가 "not_implemented"면 df는 빈 DataFrame(컬럼은 유지).
    """
    items, status, message, response_twiddler = get_detail_recommendation_items(
        item_id, top_n, user_id, twiddler
    )
    rows = [
        {
            "item_id": item_id,
            "rec_item_id": item["rec_item_id"],
            "score": item["score"],
            "rank": item["rank"],
            "twiddler": response_twiddler,
        }
        for item in items
    ]
    df = pd.DataFrame(rows, columns=_DETAIL_REC_COLUMNS)
    return df, status, message


@st.cache_data(ttl=30)
def get_user_subgraph(user_id: int, hops: int = 1) -> tuple[dict, str, str | None]:
    """유저 중심 추천 근거 서브그래프(유저-상품-세그먼트)를 backend.api.core에서 직접 조회한다(HTTP 없음)."""
    return get_user_subgraph_items(user_id, hops)


@st.cache_data(ttl=30)
def get_user_twiddler_case(user_id: int) -> dict | None:
    """선택 유저 1명의 Twiddler 재랭킹 근거(alpha/decay/선호 카테고리)를 backend.api.core에서 조회한다."""
    return _get_user_twiddler_case_core(user_id)
