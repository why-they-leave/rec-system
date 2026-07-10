"""
als_model.py
ALS 기반 추천 모델 학습 코드

Usage:
    python -m src.modeling.als.model --dataset full
    python -m src.modeling.als.model --dataset us
"""

import argparse
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import implicit
import pandas as pd
import scipy.sparse as sparse
import yaml
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

# ============================================================
# 경로 설정 (--dataset 인자에 따라 자동 결정)
# ============================================================
# src/modeling/als/model.py 기준 parents[3] == 리포 루트.
# (구 backend/ALS/als_model.py는 디렉토리 depth가 달라 parents[3]이 루트 밖을 가리키는 버그가 있었음)
PARAMS_PATH = Path(__file__).resolve().parents[3] / "configs" / "als" / "params.yaml"

PATHS = {
    "full": {
        "mart": "data/interim/als_events.csv",  # 원본 데이터마트 생성 코드로 만든 실제 마트(사용자 제공)
        "rec_file": "PRED_MAIN_RECOMMEND.csv",
        "test_file": "als_test.csv",
        "model_file": "als_model.pkl",
    },
    "us": {
        "mart": "data/interim/als_events_us.csv",
        "rec_file": "PRED_MAIN_RECOMMEND_us.csv",
        "test_file": "als_test_us.csv",
        "model_file": "als_model_us.pkl",
    },
}

OUTPUT_DIR = "data/outputs/ALS"
MODEL_DIR = "models/ALS"
LOG_DIR = "logs/ALS"


def setup_logging(log_dir: str, dataset: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"als_model_{dataset}_{timestamp}.log")

    logger = logging.getLogger(f"als_model.{dataset}")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_params(params_path: Path) -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 1. 이벤트 데이터 로드
def load_events(path: str, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info(f"[로드] 총 {len(df):,}개 이벤트 레코드")
    return df


# 2. 이벤트 레벨 Train / Test Split
def split_events(df: pd.DataFrame, split_date: str, logger: logging.Logger):
    """
    원본 이벤트 레벨에서 split — 집계 전에 먼저 수행
    이후 train/test 각각에서 user-item 점수를 집계
    """
    train = df[df["timestamp"] < split_date].copy()
    test = df[df["timestamp"] >= split_date].copy()
    logger.info(f"[Split] Train: {len(train):,}개 / Test: {len(test):,}개 (기준: {split_date})")
    return train, test


# 3. user-item 단위 점수 집계
def aggregate_scores(events_df: pd.DataFrame) -> pd.DataFrame:
    """이벤트 레벨 → user-item 단위 점수 합산"""
    return events_df.groupby(["user_id", "item_id"]).agg(total_score=("score", "sum")).reset_index()


# 4. 유저 세그먼트 분리
def segment_users(train_events: pd.DataFrame, threshold: int, logger: logging.Logger):
    """
    train 이벤트 수(행 수) 기준으로 Heavy / Cold 유저 분리
    집계 전 이벤트 레벨 데이터를 사용해야 정확한 로그 수 집계 가능
    """
    user_event_counts = train_events.groupby("user_id")["item_id"].count()
    heavy_users = user_event_counts[user_event_counts >= threshold].index.tolist()
    cold_users = user_event_counts[user_event_counts < threshold].index.tolist()
    logger.info(f"[세그먼트] Heavy 유저: {len(heavy_users):,}명 / Cold 유저: {len(cold_users):,}명")
    return heavy_users, cold_users


# 5. 희소행렬 변환
def build_sparse_matrix(train_agg: pd.DataFrame, logger: logging.Logger):
    """[user_id, item_id, total_score] → CSR 희소행렬"""
    user_ids = train_agg["user_id"].unique()
    item_ids = train_agg["item_id"].unique()

    user_enc = {u: i for i, u in enumerate(user_ids)}
    item_enc = {it: i for i, it in enumerate(item_ids)}
    user_dec = {i: u for u, i in user_enc.items()}
    item_dec = {i: it for it, i in item_enc.items()}

    rows = train_agg["user_id"].map(user_enc)
    cols = train_agg["item_id"].map(item_enc)

    matrix = sparse.csr_matrix(
        (train_agg["total_score"].astype(float), (rows, cols)), shape=(len(user_ids), len(item_ids))
    )
    logger.info(f"[희소행렬] shape: {matrix.shape} / nonzero: {matrix.nnz:,}")
    return matrix, user_enc, item_enc, user_dec, item_dec


# 5-1. (선택) 신뢰도 행렬 재가중치 — train_agg 기반 matrix에만 적용, test 미접촉
def apply_weighting(
    matrix: sparse.csr_matrix,
    method: str,
    logger: logging.Logger,
    K1: float = 100,
    B: float = 0.8,
) -> sparse.csr_matrix:
    """
    build_sparse_matrix()가 만든 train-only 행렬에 재가중치를 적용한다.
    row=user(document), col=item(term) 방향을 그대로 사용한다 — implicit의
    BM25Recommender.fit()처럼 .T로 전치할 필요가 없다. 전치는 그쪽이 아이템-아이템
    유사도 계산을 위해 반대 방향(item-user)을 요구하기 때문이며, 우리 matrix는
    이미 AlternatingLeastSquares.fit()이 기대하는 user_items 방향과 일치해
    bm25_weight()의 row(=문서=유저) 길이정규화 / col(=단어=아이템) IDF 가정이
    바로 들어맞는다.
    """
    if method == "none":
        return matrix
    if method == "tfidf":
        weighted = tfidf_weight(matrix).tocsr()
        logger.info("[가중치] TF-IDF 적용")
        return weighted
    if method == "bm25":
        weighted = bm25_weight(matrix, K1=K1, B=B).tocsr()
        logger.info(f"[가중치] BM25 적용 (K1={K1}, B={B})")
        return weighted
    raise ValueError(f"알 수 없는 weighting.method: {method}")


# 6. ALS 학습
def train_als(matrix: sparse.csr_matrix, params: dict, logger: logging.Logger):
    als_params = params["als"]
    model = implicit.als.AlternatingLeastSquares(
        factors=als_params["factors"],
        iterations=als_params["iterations"],
        alpha=als_params["alpha"],
        regularization=als_params["regularization"],
        random_state=als_params["random_state"],
    )
    model.fit(matrix)
    logger.info(
        f"[ALS] 학습 완료 | factors={als_params['factors']}, "
        f"iterations={als_params['iterations']}, alpha={als_params['alpha']}, "
        f"regularization={als_params['regularization']}"
    )
    return model


# 7. Heavy 유저 추천 생성
def generate_heavy_recommendations(
    model,
    matrix: sparse.csr_matrix,
    heavy_users: list,
    user_enc: dict,
    item_dec: dict,
    top_n: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Heavy 유저 ALS 추천 — 배치 처리(recommend)로 일괄 추천"""
    valid_users = [u for u in heavy_users if u in user_enc]
    user_indices = [user_enc[u] for u in valid_users]

    # model.recommend_all()은 implicit>=0.6부터 deprecated이며 scores 없이 ids만 반환해
    # (ids, scores) 언패킹이 실패한다. model.recommend(userid, user_items, ...)를 사용해야 한다.
    # userid에는 sub_matrix의 로컬 행 순서(0..k-1)가 아니라 학습 시점의 인코딩 인덱스를
    # 그대로 넘겨야 한다 — model.user_factors는 그 인덱스로 색인되기 때문이다.
    sub_matrix = matrix[user_indices]
    all_item_indices, all_scores = model.recommend(
        user_indices, sub_matrix, N=top_n, filter_already_liked_items=True
    )

    records = []
    for i, user_id in enumerate(valid_users):
        for rank, (item_idx, score) in enumerate(zip(all_item_indices[i], all_scores[i]), start=1):
            records.append(
                {
                    "user_id": user_id,
                    "item_id": item_dec[item_idx],
                    "score": round(float(score), 6),
                    "rank": rank,
                    "user_type": "heavy",
                }
            )

    df_rec = pd.DataFrame(records)
    logger.info(f"[Heavy 추천] {len(valid_users):,}명 → {len(df_rec):,}개 레코드")
    return df_rec


# 8. Cold 유저 인기도 기반 Fallback 추천
def generate_cold_recommendations(
    train_agg: pd.DataFrame,
    cold_users: list,
    top_n: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Cold 유저에게 전체 train 기준 인기 상품 추천
    유저별로 이미 소비한 아이템은 제외
    """
    popular_items = (
        train_agg.groupby("item_id")["total_score"].sum().sort_values(ascending=False).reset_index()
    )
    user_seen = train_agg.groupby("user_id")["item_id"].apply(set).to_dict()

    records = []
    for user_id in cold_users:
        seen = user_seen.get(user_id, set())
        user_recs = popular_items[~popular_items["item_id"].isin(seen)].head(top_n)
        for rank, row in enumerate(user_recs.itertuples(index=False), start=1):
            records.append(
                {
                    "user_id": user_id,
                    "item_id": row.item_id,
                    "score": round(float(row.total_score), 6),
                    "rank": rank,
                    "user_type": "cold",
                }
            )

    df_cold = pd.DataFrame(records)
    logger.info(f"[Cold 추천] {len(cold_users):,}명 → {len(df_cold):,}개 레코드")
    return df_cold


# 9. 결과 저장
def save_outputs(
    df_rec: pd.DataFrame,
    model,
    user_enc: dict,
    item_enc: dict,
    user_dec: dict,
    item_dec: dict,
    test_pairs: pd.DataFrame,
    paths: dict,
    output_dir: str,
    model_dir: str,
    logger: logging.Logger,
    matrix: sparse.csr_matrix | None = None,
    popular_items: pd.DataFrame | None = None,
    user_type_map: dict | None = None,
):
    """
    모델/인코더에 더해 실시간 서빙에 필요한 아티팩트(matrix, popular_items, user_type_map)를
    함께 pickle에 담는다 — backend의 als_service.py가 already-liked 필터링과 cold 폴백을
    하려면 학습 스크립트 밖에서는 재계산할 수 없는 정보이기 때문.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    rec_path = os.path.join(output_dir, paths["rec_file"])
    df_rec.to_csv(rec_path, index=False)
    logger.info(f"[저장] 추천 결과 → {rec_path} ({len(df_rec):,}개 레코드)")

    test_path = os.path.join(output_dir, paths["test_file"])
    test_pairs.to_csv(test_path, index=False)
    logger.info(f"[저장] 테스트 데이터 → {test_path} ({len(test_pairs):,}개 user-item 쌍)")

    model_path = os.path.join(model_dir, paths["model_file"])
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "user_enc": user_enc,
                "item_enc": item_enc,
                "user_dec": user_dec,
                "item_dec": item_dec,
                "matrix": matrix,
                "popular_items": popular_items,
                "user_type_map": user_type_map,
            },
            f,
        )
    logger.info(f"[저장] 모델 → {model_path}")


# 메인 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALS 추천 모델 학습")
    parser.add_argument(
        "--dataset", choices=["full", "us"], default="full", help="학습 데이터셋 선택 (full / us)"
    )
    args = parser.parse_args()

    params = load_params(PARAMS_PATH)
    paths = PATHS[args.dataset]
    logger = setup_logging(LOG_DIR, args.dataset)

    logger.info(f"===== ALS 학습 시작 | dataset={args.dataset} =====")
    logger.info(
        f"[파라미터] split_date={params['split_date']}, cold_threshold={params['cold_threshold']}, top_n={params['top_n']}"
    )

    # 1. 이벤트 로드
    events_df = load_events(paths["mart"], logger)

    # 2. 이벤트 레벨 split
    train_events, test_events = split_events(events_df, params["split_date"], logger)

    # 3. train 집계 (ALS 학습용)
    train_agg = aggregate_scores(train_events)
    logger.info(f"[Train 집계] {len(train_agg):,}개 user-item 쌍")

    # 4. 유저 세그먼트 (이벤트 수 기준)
    heavy_users, cold_users = segment_users(train_events, params["cold_threshold"], logger)
    user_type_map = {u: "heavy" for u in heavy_users}
    user_type_map.update({u: "cold" for u in cold_users})

    # 5. 희소행렬
    matrix, user_enc, item_enc, user_dec, item_dec = build_sparse_matrix(train_agg, logger)

    # 5-1. (선택) 신뢰도 행렬 재가중치 — params.yaml에 weighting 키가 없으면 no-op(method="none")
    weighting_cfg = params.get("weighting", {"method": "none"})
    matrix = apply_weighting(
        matrix,
        weighting_cfg.get("method", "none"),
        logger,
        K1=weighting_cfg.get("K1", 100),
        B=weighting_cfg.get("B", 0.8),
    )

    # 6. ALS 학습
    model = train_als(matrix, params, logger)

    # 7. Heavy 유저 추천
    df_heavy = generate_heavy_recommendations(
        model, matrix, heavy_users, user_enc, item_dec, params["top_n"], logger
    )

    # 8. Cold 유저 추천 (+ 서빙용 인기도 테이블)
    df_cold = generate_cold_recommendations(train_agg, cold_users, params["top_n"], logger)
    popular_items = (
        train_agg.groupby("item_id")["total_score"].sum().sort_values(ascending=False).reset_index()
    )

    # 9. 합치기
    df_all = pd.concat([df_heavy, df_cold], ignore_index=True)
    logger.info(f"[최종] 전체 추천 레코드: {len(df_all):,}개")

    # 10. 테스트 ground truth 준비 (purchase 이벤트만, train에 등장한 user/item만)
    # checkout은 order_items로 검증 불가한 high-intent signal이므로 학습에만 사용.
    # 평가 정답은 실제 구매(purchase)만 사용해 지표 부풀림을 방지한다.
    test_pairs = (
        test_events[test_events["event_type"] == "purchase"][["user_id", "item_id"]]
        .drop_duplicates()
        .loc[lambda x: x["user_id"].isin(user_enc) & x["item_id"].isin(item_enc)]
    )
    new_user_cnt = test_events[~test_events["user_id"].isin(user_enc)]["user_id"].nunique()
    new_item_cnt = test_events[~test_events["item_id"].isin(item_enc)]["item_id"].nunique()
    logger.info(f"[신규 유저] train 미등장: {new_user_cnt:,}명 (평가에서 제외)")
    logger.info(f"[신규 아이템] train 미등장: {new_item_cnt:,}개 (평가에서 제외)")
    logger.info(f"[테스트 쌍] purchase 기준, 필터링 후: {len(test_pairs):,}개")

    # 11. 저장
    save_outputs(
        df_all,
        model,
        user_enc,
        item_enc,
        user_dec,
        item_dec,
        test_pairs,
        paths,
        OUTPUT_DIR,
        MODEL_DIR,
        logger,
        matrix=matrix,
        popular_items=popular_items,
        user_type_map=user_type_map,
    )

    logger.info("===== ALS 학습 완료 =====")
