import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compute_hit_rate(recs_dict: dict, test_df: pd.DataFrame, top_n: int = 5) -> dict:
    """recs_dict({target_prod: [rec_prod, ...]})와 test_df를 받아 Hit Rate@top_n을 계산한다.

    baseline(df_recs 기반 recs_dict)과 Twiddler 재랭킹 결과(별도 조립한 recs_dict) 양쪽에서
    재사용해 동일한 세션/타겟 상품 기준으로 공정하게 비교할 수 있도록 분리했다.
    """
    hit_count = 0
    total_cases = 0
    test_sessions = test_df.groupby('session_id')['product_id'].apply(list).values

    for products_list in test_sessions:
        unique_products = list(set(products_list))
        if len(unique_products) < 2:
            continue

        for i, target_prod in enumerate(unique_products):
            if target_prod not in recs_dict:
                continue
            actual_complements = unique_products[:i] + unique_products[i+1:]
            predicted_recs = recs_dict[target_prod][:top_n]

            has_hit = any(prod in actual_complements for prod in predicted_recs)
            if has_hit:
                hit_count += 1
            total_cases += 1

    hit_rate = hit_count / total_cases if total_cases > 0 else 0
    return {"hit_rate": hit_rate, "hit_count": hit_count, "total_cases": total_cases}


def evaluate_model(train_df, test_df, df_recs, top_n: int = 5) -> dict:
    """가설 검증 데이터 불일치 리포트 및 Hit Rate를 검증하는 함수. 지표를 dict로 반환한다."""
    logger.info("Step 4: Validating model hypotheses and calculating metrics.")
    
    train_customers = set(train_df['customer_id'].unique())
    test_customers = set(test_df['customer_id'].unique())
    train_products = set(train_df['product_id'].unique())
    test_products = set(test_df['product_id'].unique())

    overlap_customers = train_customers.intersection(test_customers)
    only_test_customers = test_customers - train_customers
    customer_overlap_ratio = len(overlap_customers) / len(test_customers) * 100 if test_customers else 0
    customer_cold_start_ratio = len(only_test_customers) / len(test_customers) * 100 if test_customers else 0

    overlap_products = train_products.intersection(test_products)
    only_test_products = test_products - train_products
    product_overlap_ratio = len(overlap_products) / len(test_products) * 100 if test_products else 0
    product_cold_start_ratio = len(only_test_products) / len(test_products) * 100 if test_products else 0

    logger.info("=== Hypothesis Verification: Train vs Test Mismatch Report ===")
    logger.info("1. Customer Side:")
    logger.info(f"   - Total test customers: {len(test_customers)}")
    logger.info(f"   - Overlapping customer ratio: {customer_overlap_ratio:.2f}%")
    logger.info(f"   - New customer (Cold Start) ratio: {customer_cold_start_ratio:.2f}% (Data Mismatch Detected)")
    logger.info("2. Product Side:")
    logger.info(f"   - Total test products: {len(test_products)}")
    logger.info(f"   - Overlapping product ratio: {product_overlap_ratio:.2f}%")
    logger.info(f"   - New product ratio: {product_cold_start_ratio:.2f}% (Non-recommendable)")

    # HIT_RATE 계산
    recs_dict = df_recs.groupby('prod_A')['prod_B'].apply(list).to_dict()
    metrics = compute_hit_rate(recs_dict, test_df, top_n=top_n)
    logger.info(f"Base Algorithm HIT_RATE: {metrics['hit_rate']:.4f}")

    return {
        "customer_overlap_ratio": customer_overlap_ratio,
        "customer_cold_start_ratio": customer_cold_start_ratio,
        "product_overlap_ratio": product_overlap_ratio,
        "product_cold_start_ratio": product_cold_start_ratio,
        **metrics,
    }