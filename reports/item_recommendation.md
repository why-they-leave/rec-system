# [ML] 가중치 기반 보완재 추천 시스템 성능 평가 — 2026-07-01

## 1. 프로젝트 개요 및 실험 설계
본 프로젝트는 이커머스 세션 로그 데이터를 활용하여 유저가 탐색 중인 상품의 **보완재(Cross-selling)**를 정교하게 제안하는 추천 엔진 개발을 목표로 합니다. 

* **분석 단위:** `session_id` 기준 유저 행동 그룹화 (단기 맥락 파악)
* **비즈니스 제약 룰:** 대체재 추천 방지를 위한 이종 카테고리 매칭 필터링 (cat_A != cat_B)
* **검증 평가 설계:** 데이터 Leakage나 인위적 왜곡을 방지하기 위해, 최소 2개 이상의 상품과 상호작용이 발생한 유효 세션을 기준으로 **80%의 학습 데이터(Train)와 20%의 평가 데이터(Test)로 무작위 분할(Random Split)**하여 정석적인 오프라인 평가를 진행함.

---

## 2. 핵심 지표 및 실험 결과 (KPI)

| 지표명 | 베이스라인 모델 (v1) | 가중치 합산 고도화 모델 (v2) | 성능 변화율 |
| :--- | :--- | :--- | :--- |
| **최종 HIT_RATE** | **0.0083 (0.83%)** | **0.0175 (1.75%)** | **+ 110.8% 상승 (2.1배)** |
| **정답 매칭 시도 횟수** | - | 104,567 건 | - |
| **정확한 매칭(Hit) 횟수** | - | 1,833 건 | - |

> **💡 실무적 평가:**
> 단순 통계 기반의 초기 모델(0.83%) 대비, 유저의 행동 깊이를 반영한 가중치 모델(1.75%)이 **2.1배의 폭발적인 성능 향상**을 기록했습니다. 10만 건이 넘는 방대한 테스트 매칭 시도 속에서 유저의 다음 보완 행동을 정밀 타격하여 맞춘 고무적인 성과입니다.

---

## 3. 알고리즘 방법론 및 핵심 로직

### 3.1 행동별 가중치(Weight) 스코어링
단순 구매(`purchase`) 데이터만 활용할 경우 발생하는 심각한 데이터 희소성(Sparsity) 문제를 해결하기 위해, 장바구니 담기(`add_to_cart`) 및 상세페이지 조회(`page_view`) 로그를 통합하고 유저의 의지 강도에 따라 차등적인 가중치를 부여했습니다.
* `purchase` (최종 결제 완료) : **0.6 점**
* `add_to_cart` (구매 의도 장착) : **0.3 점**
* `page_view` (단순 탐색 및 비교) : **0.1 점**

### 3.2 세션 내 가중치 합산 방식 (Sum)
한 세션 내에서 특정 상품에 발생한 모든 유저 행동의 흔적을 소수점 점수로 누적 합산하여 **유저의 몰입도(Engagement)**를 반영했습니다.

$$
\text{SessionProductWeight}(A) = \sum_{et \in \text{events}(A)} \text{actionWeight}(et)
$$

* **효과:** 단순히 상품을 1번 슬쩍 보고 지나간 경우보다, 여러 번 조회하고 장바구니에 담은 뒤 구매한 상품 조합에 훨씬 묵직한 가중치를 부여해 롱테일 노이즈(co_occurrence = 1)를 통계적으로 억제합니다.

### 3.3 상품 쌍 결합 점수 산출
세션 내 상품 순열을 생성할 때, 두 상품 간의 결속력은 **둘 중 더 낮은 가중치(min)**를 기준으로 엄격하게 산출하여 점수 뻥튀기를 방지합니다.

$$
\text{PairWeight}(A, B) = \min(\text{SessionProductWeight}(A), \text{SessionProductWeight}(B))
$$

---

## 4. [심층 분석] 성능 평가 스코어에 대한 과학적 해석과 반전

### ❓ 의문 제기
> "Train과 Test 데이터를 완전히 무작위로 분할했기 때문에, Train 셋에 없는 신규 유저나 신상품의 교차 공백(Cold Start) 현상 때문에 Hit Rate가 낮게 잡힌 것이 아닐까?"

### 🧪 가설 검증 결과 (Train vs Test 데이터 교집합 분석)
가설 검증을 위해 두 데이터셋 간의 고유 ID Overlap 비율을 정량적으로 측정한 결과는 다음과 같습니다.
* **유저(Customer) 측면:** Test셋 전체 유저 중 Train셋과 **겹치는 유저 비율은 98.70%**에 달함 (신규 유저 1.30% 불과)
* **상품(Product) 측면:** Test셋 전체 상품 중 Train셋과 **겹치는 상품 비율은 100.00%**로 완벽히 일치함

### 🎯 진짜 원인 도출 (Real Insight)
교집합 검증 결과, 점수가 1.75%로 기록된 것은 데이터 분할의 오류나 신규 유저/상품의 공백(Cold Start) 때문이 아님을 증명했습니다. 진짜 원인은 **"세션(Session)의 단발적 독립성"**과 **"추천 후보 공간의 희소성"**에 있습니다.

1. **세션의 독립성:** 유저 ID는 98.7% 일치하지만, 유저가 접속할 때마다 새로 생성되는 `session_id`는 완벽히 독립적입니다. 과거 세션에서 유저의 취향을 학습했더라도, 새로운 세션(Test)에서 유저가 완전히 다른 목적의 카테고리를 탐색하면 고정된 규칙으로는 예측이 빗나갑니다.
2. **추천 후보 공간의 희소성:** 1,197개의 광범위한 상품 풀 중에서 단 5개를 추천하여 유저가 세션 내에서 소비할 1~2개의 정답을 맞춰야 하는 극도로 희소한 확률 게임입니다. 
3. **페르소나 분할의 부재:** 전체 유저를 하나의 풀로 묶다 보니 추천 규칙이 대중적인 상품 위주로 뭉툭해져(Diluted) 정밀 타격 능력이 떨어집니다.

---

## 5. 결론 및 차세대 고도화 방향
본 실험은 가중치 모델을 도입하여 베이스라인(purchase 만 포함했을때) 대비 **2.1배의 성능 향상**을 이루어내는 데 성공했습니다. 데이터 불일치 이슈가 없음을 정량적으로 확인한 만큼, 향후 스코어를 10~20%대 이상으로 혁신적으로 끌어올리기 위해 **"유저별 페르소나(refined_segment) 기반 세그먼트 분할 모델"**로 고도화를 진행할 것을 제안합니다.

---

## 6. rec-system 데모 연결 지점

본 모델의 추론 결과물은 데모 UI **Page 3 — 연관 상품 추천**의 보완재(Item-based CF) 섹션에 직접 연결됩니다.

| 항목 | 내용 |
| :--- | :--- |
| **대상 파일** | `data/dashboard/PRED_DETAIL_RECOMMEND.csv` |
| **연결 컬럼** | `rec_type = "cf"` |
| **입력 키** | `item_id` (조회 중인 상품) |
| **출력** | `rec_item_id`, `score`, `rank` (상위 5개 추천 보완재) |

현재 데모에는 더미 데이터가 채워져 있으며, 본 모델의 실제 추론 결과(`top_n_recs`)를 아래 포맷으로 변환하여 교체하면 됩니다.

```python
# top_n_recs 컬럼: prod_A, prod_B, score, rank
result = (
    top_n_recs
    .rename(columns={"prod_A": "item_id", "prod_B": "rec_item_id"})
    .assign(rec_type="cf")
    [["item_id", "rec_item_id", "score", "rank", "rec_type"]]
)
result.to_csv("data/dashboard/PRED_DETAIL_RECOMMEND.csv", index=False)
```

---

## 7. 부록 — 코드 전체 흐름 (`bowanjae_pipeline.py`)

현재 `bowanjae_pipeline.py`(레포 루트)는 아래 4개 함수 + `main` 실행부로 구성된 단일 스크립트다. 유지보수성 개선(모듈 분리)은 [이슈 #7](https://github.com/why-they-leave/rec-system/issues/7)에서 별도로 추적 중이며, 여기서는 현재 코드가 실제로 어떻게 동작하는지 흐름을 정리한다.

### 7.1 파이프라인 다이어그램

```
[1] load_data()
    products.csv, events.csv, sessions.csv, orders.csv, order_items.csv
        │
        ▼
[2] preprocess_data(events, sessions, orders, order_items)
    ├─ timestamp/start_time/order_time → datetime 변환
    ├─ events + sessions  → customer_id 결합 (df_base_events)
    ├─ event_type 분기
    │   ├─ add_to_cart / page_view → df_non_purchase
    │   └─ purchase → orders/order_items와 조인해 실제 product_id 복원 → df_purchase_final
    └─ 두 그룹을 세로 결합 후 (customer_id, timestamp) 정렬
        → df_integrated_logs  (공통 컬럼: customer_id, session_id, event_id, timestamp, event_type, product_id)
        │
        ▼
[3] run_modeling(df_integrated_logs, products)
    ├─ session_id 기준 상품 2개 이상인 유효 세션만 추출
    ├─ train_test_split(test_size=0.2, random_state=42) → train_df / test_df
    ├─ action_weights = {purchase: 0.6, add_to_cart: 0.3, page_view: 0.1}
    ├─ train_df를 세션 단위로 순회하며
    │   ├─ product_scores[product] 누적  (세션 내 가중치 합)
    │   └─ pair_scores[(A, B)] 누적       (min(weight_A, weight_B))
    ├─ P(B|A) = pair_scores[(A,B)] / product_scores[A]
    ├─ cat_A != cat_B 필터링 (동일 카테고리 제외 → 대체재 아닌 보완재만)
    └─ prod_A 기준 score 내림차순 rank → Top 5 추출
        → top_n_recs, df_recs, train_df, test_df
        │
        ▼
[4] evaluate_model(train_df, test_df, df_recs)
    ├─ train/test 간 customer_id, product_id overlap 비율 출력 (Cold Start 점검)
    └─ test_df 세션별로 실제 동시 등장 상품(actual_complements) vs
       추천 top5(predicted_recs) 비교 → HIT_RATE 산출 및 출력
        │
        ▼
[main]
    output_df = top_n_recs[["prod_A_name", "prod_B_name", "score", "rank"]]
    → {DATA_DIR}/bowanjae.csv 로 저장
```

### 7.2 함수별 입출력 요약

| 함수 | 입력 | 출력 | 비고 |
| :--- | :--- | :--- | :--- |
| `load_data` | 없음 (`DATA_DIR` 하드코딩 경로에서 CSV 5종 로드) | `products, events, sessions, orders, order_items` | 경로가 로컬 절대경로로 고정돼 있어 실행 환경에 따라 수정 필요 |
| `preprocess_data` | `events, sessions, orders, order_items` | `df_integrated_logs` | purchase 이벤트는 `(customer_id, timestamp)` 기준으로 orders와 매칭 후 실제 구매 상품으로 복원 |
| `run_modeling` | `df_integrated_logs, products` | `top_n_recs, df_recs, train_df, test_df` | 학습은 `train_df`에서만 수행 (leakage 방지), `cat_A != cat_B` 룰로 보완재만 남김 |
| `evaluate_model` | `train_df, test_df, df_recs` | 없음 (콘솔 출력만) | Hit Rate·Cold Start 비율을 반환하지 않고 print만 하므로, 리포트 자동화 시 반환값 형태로 변경 필요 |

### 7.3 현재 구조에서 눈에 띄는 유지보수 포인트
- `evaluate_model`이 지표를 반환하지 않고 `print`만 하므로, 이 결과(Hit Rate 1.75% 등 위 2장 수치)는 매 실행마다 콘솔 로그를 수동으로 옮겨 적어야 리포트에 반영 가능
- `DATA_DIR`이 Windows 로컬 절대경로로 고정 — 다른 팀원 환경이나 CI에서 그대로 실행 불가
- 로딩·전처리·모델링·평가가 한 파일에 있어 ALS 등 다른 추천 모델과 구조가 다름 → 모듈 분리 및 `data/dashboard/` 연동 작업은 [이슈 #7](https://github.com/why-they-leave/rec-system/issues/7)에서 진행 예정

---

