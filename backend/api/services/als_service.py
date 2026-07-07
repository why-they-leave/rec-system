"""
ALS 추천 서비스 — 앱 기동 시 models/ALS/als_model.pkl을 메모리에 로드해 두고,
요청이 오면 그 자리에서 model.recommend()를 호출해 추천을 반환한다(재학습 없음).

현재 로드되는 아티팩트에 matrix/popular_items/user_type_map이 없을 수 있다
(과거에 저장된 pickle과 src/modeling/als/model.py가 새로 저장하는 pickle의 스키마가 다름).
그 경우 already-liked 필터링과 cold 유저 인기도 폴백은 건너뛰고 known user에 한해서만
개인화 추천을 제공한다 — reports/BACKEND_INTEGRATION_PLAN.md Phase 1 참고.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import scipy.sparse as sparse

_ARTIFACT_PATH = Path(__file__).resolve().parents[3] / "models" / "ALS" / "als_model.pkl"

_artifact: dict | None = None


def _load_artifact() -> dict:
    global _artifact
    if _artifact is None:
        with open(_ARTIFACT_PATH, "rb") as f:
            _artifact = pickle.load(f)
    return _artifact


def get_recommendations(user_id: int, top_n: int = 10) -> tuple[list[dict], str, str | None]:
    """반환: (items, status, message). status는 "ok" 또는 "not_implemented"."""
    artifact = _load_artifact()
    user_enc = artifact["user_enc"]
    item_dec = artifact["item_dec"]
    model = artifact["model"]
    matrix = artifact.get("matrix")
    popular_items = artifact.get("popular_items")
    user_type_map = artifact.get("user_type_map") or {}

    # cold_threshold 미만 이벤트를 가진 cold 유저도 train에 이벤트가 1건 이상 있으면
    # user_enc에는 포함된다(build_sparse_matrix가 heavy/cold 구분 없이 행렬을 만들기 때문).
    # "학습 데이터에 아예 없는 유저"(user_id not in user_enc)와 "cold 유저"(user_type_map
    # 기준)를 구분해야 model.py::generate_cold_recommendations()와 동일하게 인기도 기반
    # 추천이 나간다 — 아니면 cold 유저도 본인의 희소한 상호작용 행으로 개인화 추천을 받아
    # 유저마다 결과가 달라진다(원래 설계와 불일치).
    is_cold = user_id not in user_enc or user_type_map.get(user_id) == "cold"

    if is_cold:
        if popular_items is None:
            return [], "not_implemented", "이 유저는 학습 데이터에 없고, 인기도 기반 폴백 데이터도 아직 없습니다."
        seen_items = set()
        if user_id in user_enc and matrix is not None:
            row = matrix[user_enc[user_id]]
            seen_items = {item_dec[idx] for idx in row.indices}
        recs = popular_items[~popular_items["item_id"].isin(seen_items)].head(top_n)
        items = [
            {
                "item_id": int(row.item_id),
                "score": float(row.total_score),
                "rank": rank,
                "user_type": "cold",
            }
            for rank, row in enumerate(recs.itertuples(index=False), start=1)
        ]
        return items, "ok", None

    user_idx = user_enc[user_id]
    if matrix is not None:
        user_items = matrix[user_idx]
        filter_liked = True
    else:
        user_items = sparse.csr_matrix((1, len(item_dec)))
        filter_liked = False

    item_indices, scores = model.recommend(
        user_idx, user_items, N=top_n, filter_already_liked_items=filter_liked
    )
    items = [
        {
            "item_id": int(item_dec[idx]),
            "score": float(score),
            "rank": rank,
            "user_type": "heavy",
        }
        for rank, (idx, score) in enumerate(zip(item_indices, scores), start=1)
    ]
    return items, "ok", None
