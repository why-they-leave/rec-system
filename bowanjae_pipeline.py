import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# =========================================================================
# [설정] 데이터 파일들이 위치한 폴더 경로 (상대 경로 권장)
# =========================================================================
# 스크립트와 같은 폴더에 데이터가 있다면 ""로 두고, 특정 폴더 안에 있다면 이름을 적으세요.
DATA_DIR = r"C:\Users\USER\OneDrive\Desktop\dataset"

def load_data():
    """데이터셋을 로드하는 함수"""
    print("▶ 1. 데이터 로딩 중...")
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))
    sessions = pd.read_csv(os.path.join(DATA_DIR, "sessions.csv"))
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    order_items = pd.read_csv(os.path.join(DATA_DIR, "order_items.csv"))
    return products, events, sessions, orders, order_items

def preprocess_data(events, sessions, orders, order_items):
    """행동 로그 데이터를 수직 결합 및 정제하는 함수"""
    print("▶ 2. 데이터 전처리 및 유저 여정 통합 중...")
    
    # 1) 시간 변수 datetime 변경
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    sessions['start_time'] = pd.to_datetime(sessions['start_time'])
    orders['order_time'] = pd.to_datetime(orders['order_time'])

    # 2) 기본 베이스: 세션과 유저 ID 결합
    df_base_events = pd.merge(
        events[['session_id', 'event_id', 'timestamp', 'event_type', 'product_id']], 
        sessions[['session_id', 'customer_id']], 
        on='session_id', 
        how='left'
    )

    # 3) 비-구매 이벤트 분리
    df_non_purchase = df_base_events[df_base_events['event_type'].isin(['add_to_cart', 'page_view'])].copy()

    # 4) 구매(purchase) 이벤트 전처리 및 결측치 채우기
    df_purchase_logs = df_base_events[df_base_events['event_type'] == 'purchase'].copy()
    
    df_purchase_with_order_id = pd.merge(
        df_purchase_logs,
        orders[['order_id', 'customer_id', 'order_time']],
        left_on=['customer_id', 'timestamp'],
        right_on=['customer_id', 'order_time'],
        how='inner'
    )

    df_purchase_with_items = pd.merge(
        df_purchase_with_order_id,
        order_items[['order_id', 'product_id']],
        on='order_id',
        how='left'
    )

    df_purchase_final = df_purchase_with_items.drop(columns=['product_id_x', 'order_id', 'order_time'])
    df_purchase_final = df_purchase_final.rename(columns={'product_id_y': 'product_id'})

    # 5) 수직 결합 및 시간순 정렬
    common_columns = ['customer_id', 'session_id', 'event_id', 'timestamp', 'event_type', 'product_id']
    df_integrated_logs = pd.concat([
        df_non_purchase[common_columns],
        df_purchase_final[common_columns]
    ], axis=0, ignore_index=True)

    df_integrated_logs['product_id'] = df_integrated_logs['product_id'].astype(int)
    df_integrated_logs = df_integrated_logs.sort_values(by=['customer_id', 'timestamp']).reset_index(drop=True)
    
    return df_integrated_logs

def run_modeling(df_integrated_logs, products):
    """행동 가중치 기반 보완재 추천 모델 생성 함수"""
    print("▶ 3. 모델링 및 조건부 확률 계산 중...")
    
    # 최소 2개 이상의 상품을 가진 세션 추출 및 Train/Test 분리
    session_product_counts = df_integrated_logs.groupby('session_id')['product_id'].nunique()
    valid_session_ids = session_product_counts[session_product_counts >= 2].index.tolist()

    train_session_ids, test_session_ids = train_test_split(
        valid_session_ids, test_size=0.2, random_state=42
    )

    train_df = df_integrated_logs[df_integrated_logs['session_id'].isin(train_session_ids)].copy()
    test_df = df_integrated_logs[df_integrated_logs['session_id'].isin(test_session_ids)].copy()

    # 가중치 정의
    action_weights = {'purchase': 0.6, 'add_to_cart': 0.3, 'page_view': 0.1}
    product_scores = Counter()
    pair_scores = Counter()

    # 세션별 누적 가중치 집계
    for session_id, g_df in train_df.groupby('session_id'):
        sess_prod_weights = g_df.groupby('product_id')['event_type'].apply(
            lambda x: sum(action_weights.get(et, 0.1) for et in x)
        ).to_dict()
        unique_products = list(sess_prod_weights.keys())
        
        if len(unique_products) < 2:
            continue
            
        for prod, w in sess_prod_weights.items():
            product_scores[prod] += w
            
        for i in range(len(unique_products)):
            for j in range(len(unique_products)):
                if i == j: continue
                p1, p2 = unique_products[i], unique_products[j]
                pair_weight = min(sess_prod_weights[p1], sess_prod_weights[p2])
                pair_scores[(p1, p2)] += pair_weight

    # 조건부 확률 P(B|A) 계산
    product_category_map = products.set_index('product_id')['category'].to_dict()
    product_name_map = products.set_index('product_id')['name'].to_dict()

    score_list = []
    for (prod_A, prod_B), count_AB in pair_scores.items():
        count_A = product_scores[prod_A]
        p_B_given_A = count_AB / count_A if count_A > 0 else 0
        score_list.append({
            'prod_A': prod_A, 'prod_B': prod_B, 'co_occurrence_score': count_AB, 'score': p_B_given_A
        })

    df_weighted_scores = pd.DataFrame(score_list)
    df_weighted_scores['cat_A'] = df_weighted_scores['prod_A'].map(product_category_map)
    df_weighted_scores['cat_B'] = df_weighted_scores['prod_B'].map(product_category_map)
    df_weighted_scores['prod_A_name'] = df_weighted_scores['prod_A'].map(product_name_map)
    df_weighted_scores['prod_B_name'] = df_weighted_scores['prod_B'].map(product_name_map)

    # 비즈니스 룰: 동일 카테고리 제외 (보완재 목적)
    df_recs = df_weighted_scores[df_weighted_scores['cat_A'] != df_weighted_scores['cat_B']].copy()

    # 타겟 상품별 순위 매기기 및 Top 5 추출
    df_recs['rank'] = df_recs.groupby('prod_A')['score'].rank(ascending=False, method='min').astype(int)
    df_recs = df_recs.sort_values(by=['prod_A', 'rank'], ascending=[True, True])
    top_n_recs = df_recs.groupby('prod_A').head(5)
    
    return top_n_recs, df_recs, train_df, test_df

def evaluate_model(train_df, test_df, df_recs):
    """가설 검증 데이터 불일치 리포트 및 Hit Rate를 검증하는 함수"""
    print("▶ 4. 모델 가설 검증 및 지표 산출 중...")
    
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

    print("\n====== 📊 [가설 검증] Train vs Test 데이터 불일치 리포트 ======")
    print(f"1. 유저(Customer) 측면:")
    print(f"   - Test셋 전체 유저 수: {len(test_customers)}명")
    print(f"   - Train셋과 '겹치는' 유저 비율: {customer_overlap_ratio:.2f}% (이 유저들은 취향 학습됨)")
    print(f"   - Train셋에 '없는' 신규 유저 비율: {customer_cold_start_ratio:.2f}% ⚠️ (유저 불일치 발생)")
    print(f"2. 상품(Product) 측면:")
    print(f"   - Test셋 전체 상품 수: {len(test_products)}개")
    print(f"   - Train셋과 '겹치는' 상품 비율: {product_overlap_ratio:.2f}%")
    print(f"   - Train셋에 '없는' 신규 상품 비율: {product_cold_start_ratio:.2f}% ⚠️ (추천 불가 상품)\n")

    # HIT_RATE 계산
    recs_dict = df_recs.groupby('prod_A')['prod_B'].apply(list).to_dict()
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
            predicted_recs = recs_dict[target_prod][:5]
            
            has_hit = any(prod in actual_complements for prod in predicted_recs)
            if has_hit:
                hit_count += 1
            total_cases += 1

    hit_rate = hit_count / total_cases if total_cases > 0 else 0
    print(f"🏆 기본 알고리즘 HIT_RATE: {hit_rate:.4f}\n")

# =========================================================================
# 메인 실행 흐름 제어
# =========================================================================
if __name__ == "__main__":
    # 1. 데이터 로드
    products, events, sessions, orders, order_items = load_data()
    
    # 2. 전처리
    df_integrated_logs = preprocess_data(events, sessions, orders, order_items)
    
    # 3. 모델링
    top_n_recs, df_recs, train_df, test_df = run_modeling(df_integrated_logs, products)
    
    # 4. 검증 리포트 및 Hit Rate 출력
    evaluate_model(train_df, test_df, df_recs)
    
    # 5. 최종 결과 CSV 파일 저장
    final_columns = ['prod_A_name', 'prod_B_name', 'score', 'rank']
    output_df = top_n_recs[final_columns]
    
    output_path = os.path.join(DATA_DIR, "bowanjae.csv")
    output_df.to_csv(output_path, index=False)
    print(f"✅ 추출 완료! 최종 추천 파일이 저장되었습니다 ➡️ {output_path}")