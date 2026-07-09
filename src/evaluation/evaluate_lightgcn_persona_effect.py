"""
LightGCN bi-graph vs tri-graph 페르소나 효과 정량 비교 — 배치 사전계산.

Usage:
    python -m src.evaluation.evaluate_lightgcn_persona_effect

reports/UI_TAB_RESTRUCTURE_PLAN.md에서 Tab2(효과 해석) 상단 두 섹션으로 미뤄뒀던 항목이다
("LightGCN 모델 학습 완료 후"로 보류). retail-clickstream-analysis #41에서 tri/bipartite
추천 CSV 파일명이 분리되고 실제 tri 학습 산출물이 생기면서 계산 가능해졌다.
evaluate_twiddler.py와 목적이 다르다(twiddler 재랭킹 효과가 아니라 그래프에 페르소나
노드를 추가했을 때의 효과) — 별도 스크립트로 분리한다.

계산 두 가지:
    1. HR@K/Recall@K/NDCG@K — bipartite vs tripartite, 공유 정답셋(lightgcn_test.csv) 기준.
       src.modeling.als.evaluate의 hit_rate_at_k/recall_at_k/ndcg_at_k 재사용(재구현 없음).
    2. 순위 이동 + 카테고리 구성비 — 같은 유저에게 두 그래프가 겹쳐 추천한 아이템의 순위가
       얼마나 다른지(rank shift), top-K 추천의 카테고리 구성비가 얼마나 다른지.

출력 (data/outputs/eval/):
    lightgcn_persona_accuracy.csv       — condition(bipartite/tripartite), k, HR, Recall,
                                           NDCG, eval_users
    lightgcn_persona_rank_shift.csv     — k, mean_abs_rank_shift, mean_overlap_ratio,
                                           n_users, n_overlapping_pairs
    lightgcn_persona_category_share.csv — condition, category, count, share
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.api.services import catalog_service
from backend.api.services.lightgcn_service import resolve_artifact_path
from src.modeling.als.evaluate import hit_rate_at_k, ndcg_at_k, recall_at_k

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
LIGHTGCN_OUTPUT_DIR = ROOT_DIR / "data" / "outputs" / "LightGCN"
EVAL_OUTPUT_DIR = ROOT_DIR / "data" / "outputs" / "eval"

K_LIST = [5, 10, 20]
CATEGORY_K = 20  # 순위 이동/카테고리 구성비 비교에 쓸 top-K (HR@20과 동일 기준)

TEST_FILENAME = "lightgcn_test.csv"


def _load_recs_by_user(condition: str) -> dict[int, list[dict]]:
    """graph_type별 추천 CSV를 유저별 rank순 [{"item_id", "rank"}, ...]로 로드한다.

    lightgcn_service.resolve_artifact_path()를 재사용해 파일명 후보 목록(신규/레거시
    이름)을 서비스 코드와 동일하게 유지한다 — 여기서 따로 하드코딩하면 나중에 파일명
    관례가 바뀔 때 두 곳을 따로 고쳐야 하는 드리프트가 생긴다.
    """
    artifact_path = resolve_artifact_path(condition)
    if artifact_path is None:
        raise FileNotFoundError(f"LightGCN({condition}) 추천 결과 CSV를 찾을 수 없습니다.")
    df = pd.read_csv(artifact_path)
    return {
        int(uid): g.sort_values("rank")[["item_id", "rank"]].to_dict("records")
        for uid, g in df.groupby("user_id")
    }


def _load_ground_truth() -> dict[int, set]:
    test_df = pd.read_csv(LIGHTGCN_OUTPUT_DIR / TEST_FILENAME)
    return test_df.groupby("user_id")["item_id"].apply(set).to_dict()


def compute_accuracy_rows(
    recs_by_user: dict[int, list[dict]],
    ground_truth: dict[int, set],
    condition: str,
    k_list: list[int],
) -> list[dict]:
    """HR@K/Recall@K/NDCG@K 계산(als/evaluate.py의 지표 함수 재사용)."""
    rows = []
    for k in k_list:
        hr, recall, ndcg = [], [], []
        for uid, true_items in ground_truth.items():
            user_recs = recs_by_user.get(uid)
            if user_recs is None:
                continue
            recommended = [r["item_id"] for r in user_recs if r["rank"] <= k]
            hr.append(hit_rate_at_k(recommended, true_items))
            recall.append(recall_at_k(recommended, true_items))
            ndcg.append(ndcg_at_k(recommended, true_items))
        rows.append(
            {
                "condition": condition,
                "k": k,
                "HR": round(float(np.mean(hr)), 4) if hr else 0.0,
                "Recall": round(float(np.mean(recall)), 4) if recall else 0.0,
                "NDCG": round(float(np.mean(ndcg)), 4) if ndcg else 0.0,
                "eval_users": len(hr),
            }
        )
    return rows


def compute_rank_shift(
    bi_recs: dict[int, list[dict]], tri_recs: dict[int, list[dict]], k: int
) -> dict:
    """두 그래프의 top-k 추천에서 같은 유저·같은 아이템의 순위가 얼마나 다른지 계산.

    겹치는 아이템이 없는 유저-아이템 쌍은 순위 이동 평균에서 제외되지만(정의상 계산
    불가), overlap_ratio에는 0으로 반영된다 — bi/tri 양쪽에 다 등장하는 유저만 집계 대상.
    """
    rank_shifts = []
    overlap_ratios = []
    for uid, bi_items in bi_recs.items():
        tri_items = tri_recs.get(uid)
        if tri_items is None:
            continue
        bi_top = {r["item_id"]: r["rank"] for r in bi_items if r["rank"] <= k}
        tri_top = {r["item_id"]: r["rank"] for r in tri_items if r["rank"] <= k}
        overlap = set(bi_top) & set(tri_top)
        overlap_ratios.append(len(overlap) / k)
        rank_shifts.extend(abs(bi_top[iid] - tri_top[iid]) for iid in overlap)

    return {
        "k": k,
        "mean_abs_rank_shift": round(float(np.mean(rank_shifts)), 4) if rank_shifts else 0.0,
        "mean_overlap_ratio": round(float(np.mean(overlap_ratios)), 4) if overlap_ratios else 0.0,
        "n_users": len(overlap_ratios),
        "n_overlapping_pairs": len(rank_shifts),
    }


def compute_category_share(
    recs_by_user: dict[int, list[dict]], k: int, category_map: dict, condition: str
) -> pd.DataFrame:
    """top-k 추천에 등장한 아이템의 카테고리 구성비(population 전체 집계)."""
    counts: dict[str, int] = {}
    total = 0
    for items in recs_by_user.values():
        for r in items:
            if r["rank"] > k:
                continue
            category = category_map.get(r["item_id"])
            if category is None:
                continue
            counts[category] = counts.get(category, 0) + 1
            total += 1

    rows = [
        {
            "condition": condition,
            "category": category,
            "count": count,
            "share": round(count / total, 4) if total else 0.0,
        }
        for category, count in counts.items()
    ]
    return pd.DataFrame(rows, columns=["condition", "category", "count", "share"])


def main() -> None:
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ground_truth = _load_ground_truth()
    logger.info("[정답셋] 평가 대상 유저 수: %s명", len(ground_truth))

    category_map = catalog_service.get_category_map()

    accuracy_rows = []
    recs_by_condition = {}
    for condition in ("bipartite", "tripartite"):
        recs_by_user = _load_recs_by_user(condition)
        recs_by_condition[condition] = recs_by_user
        accuracy_rows += compute_accuracy_rows(recs_by_user, ground_truth, condition, K_LIST)
        logger.info("[%s] 정확도 계산 완료", condition)

    accuracy_path = EVAL_OUTPUT_DIR / "lightgcn_persona_accuracy.csv"
    pd.DataFrame(accuracy_rows).to_csv(accuracy_path, index=False)
    logger.info("[저장] %s", accuracy_path)

    rank_shift_row = compute_rank_shift(
        recs_by_condition["bipartite"], recs_by_condition["tripartite"], CATEGORY_K
    )
    rank_shift_path = EVAL_OUTPUT_DIR / "lightgcn_persona_rank_shift.csv"
    pd.DataFrame([rank_shift_row]).to_csv(rank_shift_path, index=False)
    logger.info("[저장] %s", rank_shift_path)

    category_dfs = [
        compute_category_share(recs_by_condition[c], CATEGORY_K, category_map, c)
        for c in ("bipartite", "tripartite")
    ]
    category_path = EVAL_OUTPUT_DIR / "lightgcn_persona_category_share.csv"
    pd.concat(category_dfs, ignore_index=True).to_csv(category_path, index=False)
    logger.info("[저장] %s", category_path)

    logger.info("===== LightGCN 페르소나 효과 비교 완료 =====")


if __name__ == "__main__":
    main()
