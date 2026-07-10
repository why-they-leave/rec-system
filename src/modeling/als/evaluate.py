"""
evaluate.py
ALS 추천 결과 평가 코드 — 사전 계산된 추천 결과(CSV) 기반
모델 재추론 없이 rank 컬럼으로 K를 조정하며 평가 가능

Usage:
    python -m src.modeling.als.evaluate --dataset full
    python -m src.modeling.als.evaluate --dataset us --k 5 10 20
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ============================================================
# 경로 설정 (--dataset 인자에 따라 자동 결정)
# ============================================================
# src/modeling/als/evaluate.py 기준 parents[3] == 리포 루트.
PARAMS_PATH = Path(__file__).resolve().parents[3] / "configs" / "als" / "params.yaml"

PATHS = {
    "full": {
        "rec_file": "PRED_MAIN_RECOMMEND.csv",
        "test_file": "als_test.csv",
        "eval_file": "eval_results.csv",
    },
    "us": {
        "rec_file": "PRED_MAIN_RECOMMEND_us.csv",
        "test_file": "als_test_us.csv",
        "eval_file": "eval_results_us.csv",
    },
}

OUTPUT_DIR = "data/outputs/ALS"
LOG_DIR = "logs/ALS"


def setup_logging(log_dir: str, dataset: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"als_evaluate_{dataset}_{timestamp}.log")

    logger = logging.getLogger(f"als_evaluate.{dataset}")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# 1. 추천 결과 및 테스트 데이터 로드
def load_artifacts(rec_path: str, test_path: str, logger: logging.Logger):
    recs_df = pd.read_csv(rec_path)
    test_df = pd.read_csv(test_path)
    logger.info(
        f"[로드] 추천 결과: {len(recs_df):,}개 레코드 (유저 수: {recs_df['user_id'].nunique():,}명)"
    )
    logger.info(
        f"[로드] 테스트 데이터: {len(test_df):,}개 user-item 쌍 (유저 수: {test_df['user_id'].nunique():,}명)"
    )
    return recs_df, test_df


# 2. 유저별 정답 아이템 집합 생성
def build_ground_truth(test_df: pd.DataFrame, logger: logging.Logger) -> dict:
    ground_truth = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    logger.info(f"[정답셋] 평가 대상 유저 수: {len(ground_truth):,}명")
    return ground_truth


# 3. 지표 계산 함수
def hit_rate_at_k(recommended: list, ground_truth: set) -> float:
    """추천 상위 K개 중 정답이 1개라도 있으면 1, 없으면 0"""
    return 1.0 if len(set(recommended) & ground_truth) > 0 else 0.0


def recall_at_k(recommended: list, ground_truth: set) -> float:
    """실제 소비 아이템 중 추천 K개 안에 포함된 비율"""
    if len(ground_truth) == 0:
        return 0.0
    return len(set(recommended) & ground_truth) / len(ground_truth)


def ndcg_at_k(recommended: list, ground_truth: set) -> float:
    """추천 순위를 고려한 품질 지표 — 정답이 상위에 있을수록 높은 점수"""
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, item in enumerate(recommended, start=1)
        if item in ground_truth
    )
    ideal_hits = min(len(ground_truth), len(recommended))
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# 4. K별 평가 실행
def evaluate_at_k(
    recs_df: pd.DataFrame,
    ground_truth: dict,
    k: int,
    logger: logging.Logger,
) -> dict:
    """저장된 추천 결과에서 rank <= k 필터링 후 지표 계산"""
    max_rank = recs_df["rank"].max()
    if k > max_rank:
        logger.warning(
            f"K={k}가 저장된 최대 추천 수({max_rank})를 초과합니다. 결과가 부정확할 수 있습니다."
        )

    user_recs = (
        recs_df[recs_df["rank"] <= k]
        .sort_values("rank")
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

    hr_list, recall_list, ndcg_list = [], [], []
    skipped = 0

    for user_id, true_items in ground_truth.items():
        if user_id not in user_recs:
            skipped += 1
            continue
        recommended = user_recs[user_id]
        hr_list.append(hit_rate_at_k(recommended, true_items))
        recall_list.append(recall_at_k(recommended, true_items))
        ndcg_list.append(ndcg_at_k(recommended, true_items))

    logger.info(
        f"[K={k:2d}] 평가 유저: {len(hr_list):,}명 / 스킵(추천 없음): {skipped:,}명 | "
        f"HR@{k}={np.mean(hr_list):.4f}  Recall@{k}={np.mean(recall_list):.4f}  NDCG@{k}={np.mean(ndcg_list):.4f}"
    )

    return {
        "k": k,
        "HR": round(np.mean(hr_list), 4),
        "Recall": round(np.mean(recall_list), 4),
        "NDCG": round(np.mean(ndcg_list), 4),
        "eval_users": len(hr_list),
        "skipped": skipped,
    }


# 5. 평가 결과 저장
def save_eval_results(
    results: list,
    eval_path: str,
    dataset: str,
    logger: logging.Logger,
):
    df_results = pd.DataFrame(results)
    df_results.insert(0, "dataset", dataset)
    df_results.to_csv(eval_path, index=False)
    logger.info(f"[저장] 평가 결과 → {eval_path}")


# 메인 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALS 추천 결과 평가")
    parser.add_argument(
        "--dataset", choices=["full", "us"], default="full", help="평가 데이터셋 선택 (full / us)"
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=None,
        help="평가할 K 값 목록 (예: --k 5 10 20). 미입력 시 params.yaml의 eval.k_list 사용",
    )
    args = parser.parse_args()

    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    paths = PATHS[args.dataset]
    logger = setup_logging(LOG_DIR, args.dataset)

    k_list = args.k if args.k is not None else params["eval"]["k_list"]
    logger.info(f"===== ALS 평가 시작 | dataset={args.dataset}, K={k_list} =====")

    # 1. 로드
    rec_path = os.path.join(OUTPUT_DIR, paths["rec_file"])
    test_path = os.path.join(OUTPUT_DIR, paths["test_file"])
    recs_df, test_df = load_artifacts(rec_path, test_path, logger)

    # 2. 정답셋 생성
    ground_truth = build_ground_truth(test_df, logger)

    # 3. K별 평가
    results = [evaluate_at_k(recs_df, ground_truth, k, logger) for k in k_list]

    # 4. 결과 저장
    eval_path = os.path.join(OUTPUT_DIR, paths["eval_file"])
    save_eval_results(results, eval_path, args.dataset, logger)

    logger.info("===== ALS 평가 완료 =====")
