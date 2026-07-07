"""
보완재(complementary item) 추천 서비스.

data/outputs/complementary/detail_cf.csv(item_id, rec_item_id, score, rank)를 기동 시
로드해 item_id 기준으로 조회한다. scripts/run_bowanjae_pipeline.py를 배치 실행해
이 파일을 만들기 전까지는 파일이 없으므로 not_implemented로 응답한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.modeling.twiddler.rerank import LOW_EXPOSURE_PERCENTILE

_TABLE_PATH = Path(__file__).resolve().parents[3] / "data" / "outputs" / "complementary" / "detail_cf.csv"

_table: pd.DataFrame | None = None
_loaded = False


def _load_table() -> pd.DataFrame | None:
    global _table, _loaded
    if not _loaded:
        _table = pd.read_csv(_TABLE_PATH) if _TABLE_PATH.exists() else None
        _loaded = True
    return _table


def get_low_exposure_items(threshold_percentile: float = LOW_EXPOSURE_PERCENTILE) -> set[int]:
    """rec_item_id 등장 빈도 하위 분위수를 저노출(신상품 근사) 상품으로 반환한다."""
    table = _load_table()
    if table is None:
        return set()

    counts = table["rec_item_id"].value_counts()
    threshold = counts.quantile(threshold_percentile)
    return set(counts[counts <= threshold].index)


def get_recommendations(item_id: int, top_n: int = 8) -> tuple[list[dict], str, str | None]:
    """반환: (items, status, message). status는 "ok" 또는 "not_implemented"."""
    table = _load_table()
    if table is None:
        return [], "not_implemented", "보완재 추천 테이블이 아직 생성되지 않았습니다."

    # detail_cf.csv의 rank는 배치 파이프라인이 rank(method="min")으로 계산해 동점 시
    # 같은 값을 공유한다(예: 1,2,3,4,4 — 5위가 아예 존재하지 않음). Twiddler "after"
    # 결과는 항상 1..N 밀집 순위로 재부여되므로(als_service와 동일 관례), 원본 rank를
    # 그대로 돌려주면 before/after 비교 시 gap 있는 순위와 밀집 순위를 비교하게 돼
    # 순위 배지가 어긋난다. rec_item_id를 동점 타이브레이커로 써서 재현 가능하게
    # 정렬한 뒤 rank를 밀집 값으로 재부여한다.
    rows = table[table["item_id"] == item_id].sort_values(["rank", "rec_item_id"]).head(top_n)
    items = [
        {"rec_item_id": int(row.rec_item_id), "score": float(row.score), "rank": rank}
        for rank, row in enumerate(rows.itertuples(index=False), start=1)
    ]
    return items, "ok", None
