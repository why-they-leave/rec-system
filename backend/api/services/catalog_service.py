"""
상품 카탈로그(카테고리) 조회 서비스.

data/dashboard/products.csv(product_id, category, ...)를 기동 시 1회 로드해
item_id -> category 매핑을 메모리에 캐시한다. Twiddler Rule 1(페르소나 카테고리
가중치)이 아이템 카테고리를 알아야 하므로 필요하다 — 지금까지는 카테고리 조인이
app/ 쪽에서만 표시용으로 일어났고 백엔드에는 없었다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_TABLE_PATH = Path(__file__).resolve().parents[3] / "data" / "dashboard" / "products.csv"

_category_map: dict[int, str] | None = None


def get_category_map() -> dict[int, str]:
    global _category_map
    if _category_map is None:
        df = pd.read_csv(_TABLE_PATH)
        _category_map = dict(zip(df["product_id"], df["category"]))
    return _category_map
