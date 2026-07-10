# Twiddler 탭에 LightGCN bipartite vs bipartite+Twiddler 비교 추가

## Context

retail-clickstream-analysis #34에서 LightGCN bipartite(페르소나 미결합) vs tri(페르소나 결합) 그래프 임베딩을 비교한 결과, 페르소나를 그래프에 결합하는 효과가 하이퍼파라미터에 따라 뒤집힐 만큼 약하고 불안정하다는 결론이 나왔다. rec-system에는 이미 "ALS vs ALS+Twiddler"(rule-based 재랭킹) 오프라인 비교가 있고, 여기서는 정확도는 K마다 방향이 엇갈리지만 다양성(반복 노출 중복률, 누적 카테고리 수)은 뚜렷이 개선된다는 결론이었다. 그래프 임베딩 방식(bipartite)에도 Twiddler를 적용했을 때 같은 패턴(다양성 개선, 정확도 불명확)이 재현되는지 확인하려는 것이 이번 작업의 목적이다.

`backend/api/services/lightgcn_service.py`는 현재 `not_implemented`만 반환하는 스텁이고, `backend/api/core.py`는 `model_type="LightGCN"`일 때 Twiddler 재랭킹 자체를 강제로 건너뛴다(`response_twiddler = "before"` 고정). 이번 작업은 `graph_type="bipartite"`에 한해 이 두 가지를 모두 채우고, Streamlit의 **Twiddler 재랭킹 탭**(`_render_rerank_main`)에 ALS ↔ LightGCN bipartite를 좌우 버튼으로 전환하는 UI를 추가한다. `_render_persona_tab`(페르소나 기여도 탭, tripartite 관련 로직)은 이번 범위에서 건드리지 않는다.

## 데이터 (이미 확보됨)

아래 두 파일이 `data/outputs/LightGCN/`에 이미 준비되어 있다(검증 완료, ALS/complementary와 동일한 `data/outputs/<MODEL>/` 컨벤션이라 별도 재배치 불필요):

- `data/outputs/LightGCN/PRED_MAIN_RECOMMEND.csv` — 컬럼 `user_id, item_id, score, rank, model_type`. 20,000명 × 100개(rank 1~100), `model_type`은 전부 `"LightGCN"` 고정값(ALS의 `user_type` 컬럼과 달리 cold/heavy 구분 없음).
- `data/outputs/LightGCN/lightgcn_test.csv` — 컬럼 `user_id, item_id`(정답 구매 쌍), 1,465명 평가 대상. 정답셋 소스는 rec-system 자체 ALS 테스트셋과 동일한 방법론(동일 split_date 2025-08-01)으로 이미 생성돼 도착했으므로 그대로 사용한다.
- 유저 ID 공간 검증: `data/dashboard/demo_users.csv`의 전체 유저, `lightgcn_test.csv`의 전체 유저가 `PRED_MAIN_RECOMMEND.csv`에 100% 포함됨을 확인했다 — retail-clickstream-analysis와 동일 소스(세그먼트/split_date)를 공유하므로 ID 불일치 문제 없음.

## 백엔드 변경

### 1. `backend/api/services/lightgcn_service.py`
`graph_type="bipartite"`만 구현하고 `"tripartite"`는 기존 stub 그대로 유지한다.

- 모듈 전역 캐시(`als_service.py`의 `_load_artifact()` 패턴과 동일)로 `data/outputs/LightGCN/PRED_MAIN_RECOMMEND.csv`를 한 번만 읽어 `user_id -> [{item_id, score, rank}, ...]` (rank 순 정렬) 딕셔너리로 보관.
- `get_recommendations(user_id, top_n=10, graph_type="tripartite")`:
  - `graph_type != "bipartite"` → 기존과 동일하게 `[], "not_implemented", _MESSAGES[graph_type]` 반환.
  - `graph_type == "bipartite"`: 캐시에서 `user_id` 조회 → 없으면 `not_implemented`(친절한 메시지), 있으면 상위 `top_n`개를 `{"item_id": int, "score": float, "rank": int, "user_type": "all"}` 리스트로 반환(status `"ok"`). `user_type`은 `schemas.MainRecommendItem`이 요구하는 필드를 채우기 위한 고정값이며, cold/heavy 개념이 없으므로 이후 로직에서 읽지 않는다.

### 2. `backend/api/core.py::get_main_recommendation_items`
분기를 3단으로 정리:
- `model_type == "ALS"` → 기존 그대로.
- `model_type == "LightGCN" and graph_type == "bipartite"` → ALS 분기와 동일한 패턴으로 새로 추가: `twiddler == "after"`면 `top_n * POOL_MULTIPLIER`만큼 pool을 가져와 `lightgcn_service.get_recommendations`로 조회 → 성공 시 `twiddler_service.apply_twiddler(items, twiddler, user_id, id_key="item_id", context="main_lightgcn_bipartite", top_k=top_n)` 호출. `response_twiddler = twiddler`(더 이상 강제 "before" 아님).
- 그 외(`LightGCN` + `tripartite`) → 기존 그대로 유지(Twiddler 미적용, `not_implemented`).

`context="main_lightgcn_bipartite"`는 `exposure_service`가 `(user_id, context)` 튜플로 자유롭게 키를 잡으므로 ALS의 `"main"`과 완전히 독립적인 노출 이력을 갖는다 — 두 모델 섹션의 새로고침 시뮬레이션이 서로 간섭하지 않는다.

## 오프라인 평가 파이프라인

### 3. `src/evaluation/evaluate_twiddler.py`
`_load_main_context`, `_main_accuracy_rows`, `_main_diversity_rows`를 파라미터화해 ALS/LightGCN 양쪽에 재사용:
- `_load_main_context(rec_filename, test_filename, output_dir)`로 시그니처 확장(기본값은 기존 ALS 경로 그대로 유지 — 하위 호환).
- `_main_accuracy_rows(ctx, context_label="main")`, `_main_diversity_rows(ctx, context_label="main")`로 `context_label`을 행에 반영.
- `main()`에서 LightGCN bipartite용 컨텍스트를 `data/outputs/LightGCN/{PRED_MAIN_RECOMMEND.csv,lightgcn_test.csv}`로 추가 로드해 `context_label="main_lightgcn_bipartite"`로 계산.
- **주의**: rec-system 로컬에는 ALS 자체의 `data/outputs/ALS/{PRED_MAIN_RECOMMEND.csv,als_test.csv}`가 현재 없다(모델 아티팩트 `models/ALS/als_model.pkl`만 존재, CSV는 gitignore 대상이라 로컬에 없을 수 있음 — 검증 완료). 따라서 `main()`은 ALS/보완재 컨텍스트 로드가 `FileNotFoundError`면 기존에 커밋된 `twiddler_accuracy.csv`/`twiddler_diversity.csv`에서 해당 `context`(`"main"`, `"detail"`) 행을 그대로 유지하고, 새로 계산 가능한 LightGCN bipartite 행만 교체/추가하는 머지 방식으로 저장한다(기존 행을 갈아엎지 않음).

### 4. `app/components/eval_metrics_table.py`
`_CONDITION_LABELS`에 항목 추가:
```python
"main_lightgcn_bipartite": {"baseline": "LightGCN bipartite only", "twiddler": "LightGCN bipartite+Twiddler"},
```
`render_eval_metrics(context="main_lightgcn_bipartite", persona_label=...)` 호출만으로 population/세그먼트 breakdown 표·그래프가 기존과 동일하게 렌더링된다(코드 변경 불필요, 데이터만 있으면 동작).

## 프론트엔드 변경 (`app/main.py`, Twiddler 탭만)

### 5. 모델 좌우 토글 추가
사이드바의 `type="primary"/"secondary"` 버튼 페어 패턴(117-131행)을 재사용하되 `st.columns(2)`로 좌우 배치하는 `_render_rerank_model_toggle()`을 추가. 세션 키 `"rerank_model_type"`, 값 `"ALS"` | `"LightGCN-bipartite"`. 위치는 스크린샷과 동일하게 `render_user_twiddler_case(user_id)` + `st.divider()` 바로 다음, 기존 `st.markdown("### 🤖 ALS")` 자리.

### 6. 모델별 렌더링 블록 공통화
현재 `_render_rerank_main`의 359~409행(모델 heading/caption, Before/After 토글, 새로고침 시뮬레이션, 카드 그리드, 오프라인 지표)을 파라미터화된 헬퍼로 추출해 ALS/LightGCN 양쪽에서 재사용(중복 제거):

```python
def _render_model_twiddler_block(
    *, model_type, graph_type, session_prefix, exposure_context,
    section_title, caption_text, eval_caption, eval_context,
    user_id, products_df, selected_categories, is_cold,
) -> None: ...
```
- `session_prefix`로 `{prefix}_twiddler_phase`, `sim_round_{user_id}_{prefix}`, `sim_history_{user_id}_{prefix}` 키를 분리(ALS: `"als"`, LightGCN: `"lgcn_bipartite"`).
- ALS 호출: `is_cold`는 기존처럼 `als_before_df["user_type"]` 기반 계산. LightGCN 호출: `is_cold=False` 고정(precomputed CSV가 20,000명 전원을 커버하므로 cold 폴백 개념 없음 — cold 캡션/게이팅 없이 항상 토글 활성화).
- `eval_context`: ALS는 `"main"`, LightGCN은 `"main_lightgcn_bipartite"` → `render_eval_metrics(context=eval_context, ...)` 호출.

`_render_rerank_main`은 이제 유저 소개/카드/Twiddler 근거까지는 공통으로 렌더링한 뒤, `_render_rerank_model_toggle()`로 선택된 쪽에 따라 위 헬퍼를 ALS 또는 LightGCN 파라미터로 1회 호출.

### 7. `app/utils/data_loader.py::simulate_next_session`
`model_type`/`graph_type` 파라미터 추가(기본값 `"ALS"`/`"tripartite"`로 하위 호환 유지), `get_main_recommendation_items`에 그대로 전달. LightGCN 새로고침 시뮬레이션 버튼이 이 함수를 `model_type="LightGCN", graph_type="bipartite"`로 호출.

### 8. `_render_refresh_simulation_button`
`session_prefix`, `model_type`, `graph_type`, `exposure_context` 파라미터를 받도록 확장 — `round_key`/`history_key`에 `session_prefix`를 포함시키고, `reset_user_exposure(user_id, exposure_context)` 및 `simulate_next_session(..., model_type=..., graph_type=...)` 호출에 반영.

## 노트북

### 9. `notebooks/20260709_ML_lightgcn_bipartite_vs_twiddler_offline.ipynb`
`notebooks/20260708_ML_als_vs_als_twiddler_offline.ipynb`와 동일한 구조·방법론으로 신규 작성(같은 seed=42, 같은 import: `persona_service`, `catalog_service`, `rerank_mod`, `src.modeling.als.evaluate`의 `hit_rate_at_k`/`recall_at_k`/`ndcg_at_k` — 이름은 als지만 범용 함수라 그대로 재사용):
1. 데이터 = `data/outputs/LightGCN/PRED_MAIN_RECOMMEND.csv` + `lightgcn_test.csv`. 로드/검증 섹션에서 ALS 스키마와의 차이(`user_type` 컬럼 없음, `model_type` 컬럼 있음) 명시.
2. 단일 세션 정확도 HR/Recall/NDCG @ K=5,10,20: `bipartite_only` vs `bipartite_twiddler`.
3. 반복 새로고침(5회) 다양성: 노출 중복률(`repetition_rate`), 누적 카테고리 수 — 기존 노트북의 `diversity_metrics()`/`decay_and_record()` 로직 그대로 재사용.
4. 유저 3명 가중치 의미 예시(`persona_service.get_user_alpha`/`get_user_decay`/`get_user_affinity`) — 모델 무관 로직이라 기존 노트북과 동일하게 구성.
5. 결과 요약: ALS 노트북 결론(정확도 불명확·다양성 개선) 및 retail-clickstream-analysis #34의 "페르소나의 그래프 임베딩 효과는 약하고 불안정" 결론과 연결지어 해석.

## 검증 방법

- `lightgcn_service.get_recommendations(user_id, 10, "bipartite")`를 데모 유저 ID 몇 개로 직접 호출해 `status="ok"`와 정상 스키마 확인, 없는 유저 ID로 `not_implemented` 확인.
- `backend/api/core.get_main_recommendation_items(user_id, "LightGCN", "bipartite", "after", 10)` 호출 → Twiddler 적용 전/후 순위가 실제로 바뀌는지 확인.
- Streamlit 앱 실행(`streamlit run app/main.py`) 후 Twiddler 탭에서: ALS ↔ LightGCN bipartite 좌우 버튼 전환, Before/After 토글, 새로고침 시뮬레이션(5회, 카드가 실제로 바뀌는지), 오프라인 성능 지표 그래프(population + 페르소나 breakdown)가 각 모델 선택 상태에서 독립적으로 정상 렌더링되는지 브라우저로 직접 확인.
- `python -m src.evaluation.evaluate_twiddler` 실행 후 `data/outputs/eval/twiddler_accuracy.csv`/`twiddler_diversity.csv`에 `context="main_lightgcn_bipartite"` 행이 추가되고 기존 `"main"`/`"detail"` 행이 보존되는지 확인.
- 새 노트북을 처음부터 끝까지 실행해 에러 없이 완료되는지 확인.
