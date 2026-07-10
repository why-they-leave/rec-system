import logging
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def run_modeling(df_integrated_logs, products):
    """행동 가중치 기반 보완재 추천 모델 생성 함수"""
    logger.info("Step 3: Modeling and calculating conditional probabilities.")

    # 최소 2개 이상의 상품을 가진 세션 추출 및 Train/Test 분리
    session_product_counts = df_integrated_logs.groupby("session_id")["product_id"].nunique()
    valid_session_ids = session_product_counts[session_product_counts >= 2].index.tolist()

    train_session_ids, test_session_ids = train_test_split(
        valid_session_ids, test_size=0.2, random_state=42
    )

    train_df = df_integrated_logs[df_integrated_logs["session_id"].isin(train_session_ids)].copy()
    test_df = df_integrated_logs[df_integrated_logs["session_id"].isin(test_session_ids)].copy()

    # 가중치 정의
    action_weights = {"purchase": 0.6, "add_to_cart": 0.3, "page_view": 0.1}
    product_scores = Counter()
    pair_scores = Counter()

    # 세션별 누적 가중치 집계
    for session_id, g_df in train_df.groupby("session_id"):
        sess_prod_weights = (
            g_df.groupby("product_id")["event_type"]
            .apply(lambda x: sum(action_weights.get(et, 0.1) for et in x))
            .to_dict()
        )
        unique_products = list(sess_prod_weights.keys())

        if len(unique_products) < 2:
            continue

        for prod, w in sess_prod_weights.items():
            product_scores[prod] += w

        for i in range(len(unique_products)):
            for j in range(len(unique_products)):
                if i == j:
                    continue
                p1, p2 = unique_products[i], unique_products[j]
                pair_weight = min(sess_prod_weights[p1], sess_prod_weights[p2])
                pair_scores[(p1, p2)] += pair_weight

    # 조건부 확률 P(B|A) 계산
    product_category_map = products.set_index("product_id")["category"].to_dict()

    score_list = []
    for (prod_A, prod_B), count_AB in pair_scores.items():
        count_A = product_scores[prod_A]
        p_B_given_A = count_AB / count_A if count_A > 0 else 0
        score_list.append({"prod_A": prod_A, "prod_B": prod_B, "score": p_B_given_A})

    df_weighted_scores = pd.DataFrame(score_list)
    df_weighted_scores["cat_A"] = df_weighted_scores["prod_A"].map(product_category_map)
    df_weighted_scores["cat_B"] = df_weighted_scores["prod_B"].map(product_category_map)

    # 비즈니스 룰: 동일 카테고리 제외 (보완재 목적)
    df_recs = df_weighted_scores[df_weighted_scores["cat_A"] != df_weighted_scores["cat_B"]].copy()

    # 타겟 상품별 순위 매기기 및 Top 5 추출
    df_recs["rank"] = (
        df_recs.groupby("prod_A")["score"].rank(ascending=False, method="min").astype(int)
    )
    df_recs = df_recs.sort_values(by=["prod_A", "rank"], ascending=[True, True])
    top_n_recs = df_recs.groupby("prod_A").head(5)

    return top_n_recs, df_recs, train_df, test_df
