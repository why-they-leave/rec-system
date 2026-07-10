# Twiddler 자연어 설명(XAI) 기능 구현 계획

## Context

`configs/twiddler/params.yaml` / `src/modeling/twiddler/rerank.py`에 정의된 Twiddler 재랭킹(Rule 1: 페르소나
카테고리 가중치, Rule 2: 노출 이력 패널티)은 완전히 투명한 수식이지만, "왜 이 순위로 추천됐는지"가 UI 어디에도
노출되지 않는다. 이를 해결하기 위해 GitHub 이슈로 규칙 기반(rule-based, LLM 미사용) 자연어 설명 기능을 제안했고,
협업자 리뷰에서 두 가지를 지적받았다:

1. **유저 레벨 문장 반복**: 프로토타입(`notebooks/20260708_ML_twiddler_final_design.ipynb`)의
   `explain_recommendation()`은 유저의 탐색 성향 문장을 아이템마다 반복 생성해 UI에서 지루해 보인다.
2. **Faithfulness 테스트 공백**: "동일 입력→동일 출력"(결정론성) 테스트는 설명에 적힌 수치가 실제 랭킹에
   쓰인 계산값과 일치함을 증명하지 못한다. 프로토타입은 `multiplier`를 `rerank.py`와 별도로 재계산해
   설명 문장에 넣고 있어, 두 코드가 갈라질(drift) 위험이 실제로 존재한다.

이 계획은 두 지적을 구조적으로 해결하면서, 이미 존재하는 "유저 1명당 1회" 근거 블록
(`backend/api/core.py::get_user_twiddler_case` + `app/components/eval_metrics_table.py::render_user_twiddler_case`)을
재사용하고, 아이템별 설명은 순위가 크게 바뀐 카드에만 배지 형태로 붙이기로 사용자와 합의했다(아래 결정 사항 참고).

## 사용자와 합의된 설계 결정

- **아이템별 설명 대상**: `|rank_before - rank_after| >= 임계값(기본 3)`인 카드만 선정, 최대 5개까지만 설명 부여.
  전체 후보가 아니라 "순위가 크게 바뀐" 카드에 한정 — 배지 존재 자체가 "이 카드는 재랭킹 효과가 컸다"는
  신호가 되도록 한다.
- **UI 표현**: 카드에 작은 "💡 추천 이유" 배지만 보이다가, 클릭하면 문장이 펼쳐진다 → `st.popover`로 구현
  (Streamlit 1.56 설치 확인, `st.expander`는 그리드 정렬을 깨므로 배제).
- **HTTP 스키마(`schemas.py`) 변경은 생략**: `app/utils/data_loader.py`가 이미 "Streamlit Community
  Cloud처럼 프로세스 하나만 띄우는 환경에서도 동작하도록" HTTP를 거치지 않고 `backend.api.core`를
  in-process로 직접 호출하는 구조로 설계돼 있다(파일 상단 주석 참고) — 향후 Streamlit Cloud 배포도 이
  경로를 그대로 쓰므로 스키마 변경이 필요 없다. 별도의 실제 HTTP API 소비자(예: 모바일 앱)가 생기면 그때
  `MainRecommendItem`/`DetailRecommendItem`에 `explanation` 필드와 라우터 매핑을 추가한다.
- **유저 레벨 캡션**: 기존 `render_user_twiddler_case()`의 사실 기반 캡션(페르소나/편차 수치)은 그대로 두고,
  새 자연어 문장(고정 템플릿, 탐색성향 3구간 분기)을 그 앞에 추가한다.

## 파일별 변경 사항

### 1. `src/modeling/twiddler/rerank.py`
`apply_persona_weight`/`apply_exposure_penalty`가 계산 후 버리던 중간값을 item dict에 그대로 보존한다
(score 변경 로직 자체는 무변경):

```python
RULE1_DEVIATION_KEY = "rule1_deviation"
RULE1_MULTIPLIER_KEY = "rule1_multiplier"
RULE2_COUNT_KEY = "rule2_count"
RULE2_DECAY_FACTOR_KEY = "rule2_decay_factor"
```

- `apply_persona_weight`: `item[RULE1_DEVIATION_KEY] = deviation`, `item[RULE1_MULTIPLIER_KEY] = multiplier` 추가.
- `apply_exposure_penalty`: `item[RULE2_COUNT_KEY] = count`, `item[RULE2_DECAY_FACTOR_KEY] = decay ** count` 추가.
- `exposure_counts`가 falsy면 `apply_exposure_penalty` 자체가 호출 안 되므로 `RULE2_*` 키가 아예 없을 수
  있음 — 이는 "아직 노출 이력 없음(첫 조회)"을 뜻하는 정상 상태로 문서화(읽는 쪽은 `.get(key, 기본값)`으로
  처리).
- 기존 컨슈머(`data_loader.py`, `product_card.py`, `schemas.py`)는 item dict를 필드별로 명시적으로
  구성하므로 새 키 추가로 인한 충돌 없음(직접 확인 완료).

새 상수 2개도 이 파일이 로드하는 `params.yaml`에 추가(§5 참고), `EXPLAIN_RANK_DELTA_THRESHOLD`/
`EXPLAIN_MAX_ITEMS`로 모듈에 노출 — `twiddler_service.py`가 `rerank_mod.EXPLAIN_MAX_ITEMS` 형태로 그대로 참조.

### 2. `src/modeling/twiddler/explain.py` (신규)
I/O 없는 순수 함수 모듈(`rerank.py`와 동일한 패턴). **재계산 금지 — 오직 rerank.py가 저장한 값만 읽는다**:

```python
def explain_item(item: dict, category: str | None) -> str:
    """카테고리 편차/배율 + 노출 감쇠 사실만 담은 짧은 문장(유저 스타일 문장 없음).
    item에 rerank.py가 실제로 저장한 rule1_*/rule2_* 값만 읽는다 — alpha/affinity로
    재계산하지 않는다(재계산 시 rerank.py 공식이 바뀌면 설명이 따라가지 못하는 drift 위험)."""

def explain_user_style(exploration_tendency: float) -> str:
    """유저 1명당 1회만 노출할 탐색 성향 문장 — 고정 템플릿 3분기(<0.3/0.3~0.7/>0.7)."""
```
- 임계값(`_DEVIATION_NOTABLE=0.05`, `_EXPLORATION_LOW=0.3`, `_EXPLORATION_HIGH=0.7`)은 표현(copy) 분기점이지
  스코어링 파라미터가 아니므로 모듈 상수로 유지(params.yaml에 안 넣음 — rerank.py 스코프와 분리).
- 문구는 노트북 프로토타입 파트 2/3을 기반으로 하되 카드에 맞게 축약.

### 3. `backend/api/services/twiddler_service.py`
`apply_twiddler()`에 순위-변동 기반 설명 선정 로직 추가. 핵심: **rerank() 호출 전** 원본 `items` 순서가
곧 "Twiddler 적용 전 순위"이므로 별도 재조회 없이 그 자리에서 계산한다:

```python
rank_before_by_id = {item[id_key]: i + 1 for i, item in enumerate(items)}
reranked = rerank_mod.rerank(...)  # 기존 호출 그대로

deltas = [
    (item, abs(rank_before_by_id.get(item[id_key], item["rank"]) - item["rank"]))
    for item in reranked
]
candidates = [(item, d) for item, d in deltas if d >= rerank_mod.EXPLAIN_RANK_DELTA_THRESHOLD]
candidates.sort(key=lambda pair: pair[1], reverse=True)
for item, delta in candidates[: rerank_mod.EXPLAIN_MAX_ITEMS]:
    category = category_map.get(item[id_key])
    rank_before = rank_before_by_id[item[id_key]]
    item["explanation"] = (
        f"{rank_before}위 → {item['rank']}위로 이동. " + explain_mod.explain_item(item, category)
    )
```
- `context="main"`(`id_key="item_id"`)과 `context="detail"`(`id_key="rec_item_id"`) 양쪽에 동일하게 동작
  (category_map이 이미 두 컨텍스트 모두 product id로 키잉되어 있어 분기 불필요 — 기존 `apply_persona_weight`
  호출과 동일한 패턴).
- `phase=="before"` 또는 persona 없음으로 조기 반환하는 경로는 rerank() 자체가 호출 안 되므로 이 로직도
  실행 안 됨 → 해당 아이템들은 `"explanation"` 키가 아예 없음(정상, "설명 없음"으로 취급).

### 4. `backend/api/services/persona_service.py` + `backend/api/core.py`
`persona_service.py`에 기존 `get_user_alpha`/`get_user_decay`와 동일 패턴으로 getter 추가:
```python
def get_user_exploration_tendency(user_id: int) -> float:
    """존재하지 않으면 _build_features()의 fillna(0.5) 중립값과 동일하게 0.5 반환."""
```
`core.py::get_user_twiddler_case()`에 `exploration_tendency`, `user_sentence`(=
`explain_mod.explain_user_style(exploration_tendency)`) 키 추가. 이 함수는 FastAPI 라우터를 거치지 않고
`app/utils/data_loader.py`가 직접 호출하는 순수 dict 반환 함수라 스키마 영향 없음(확인 완료).

### 5. `configs/twiddler/params.yaml`
파일 끝에 표시(display) 전용 섹션을 스코어링 섹션과 명확히 분리해 추가:
```yaml
# 설명(XAI) 배지 표시 기준 — 재랭킹 스코어링에는 전혀 영향 없음, UI에 몇 개/어떤 카드에
# "추천 이유" 배지를 붙일지만 결정한다.
explain_rank_delta_threshold: 3   # |Twiddler 적용 전후 순위 변동| 이 이 값 이상인 카드만 설명 대상
explain_max_items: 5              # 한 응답당 설명을 붙이는 최대 카드 수
```
`rerank.py`가 이미 이 파일을 로드하므로 `EXPLAIN_RANK_DELTA_THRESHOLD`/`EXPLAIN_MAX_ITEMS` 모듈 상수로
같이 노출(기존 `MULTIPLIER_FLOOR` 등과 동일한 방식).

### 6. `app/utils/data_loader.py`
`_MAIN_REC_COLUMNS`/`_DETAIL_REC_COLUMNS`에 `"explanation"` 추가하고, `_main_rec_df()`와
`get_detail_recommendations()`의 row dict comprehension에 `"explanation": item.get("explanation")` 추가
(둘 다 필드별 명시적 구성이라 한 줄씩만 추가하면 됨, 확인 완료).

### 7. `app/main.py::_render_recommend_grid`
`explanation = item.get("explanation")` 추출, `None if pd.isna(explanation) else explanation`로 가드 후
`render_product_card(...)` 호출에 전달(두 호출 분기 — `plain_rank_mode`/일반 모두).

### 8. `app/components/product_card.py::render_product_card`
`explanation: str | None = None` 파라미터 추가. 기존 `_badge_widget`/`_corner_badge`의 "None이면 동일 높이
플레이스홀더" 관례를 그대로 따르는 `_explanation_widget()` 추가:
```python
def _explanation_widget(explanation: str | None) -> None:
    if explanation is None:
        st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)  # 그리드 정렬용 플레이스홀더
        return
    with st.popover("💡 추천 이유", use_container_width=False):
        st.write(explanation)
```
`_badge_widget(badge)` 호출 다음 줄에 `_explanation_widget(explanation)` 추가.

### 9. `app/components/eval_metrics_table.py::render_user_twiddler_case`
기존 3개 `st.metric` 타일은 그대로 유지. 마지막 `st.caption(...)` 앞에 `st.write(case["user_sentence"])`
(또는 `st.info`) 한 줄 추가 — 기존 사실 기반 캡션은 삭제하지 않고 그 위에 자연어 문장을 덧붙인다(합의된
결정: 대체가 아니라 추가).

## 테스트

### `tests/test_rerank.py` (신규)
`tests/test_features.py`의 pytest-fixture 스타일을 따르되, 이 파일은 닫힌 형태(closed-form) 수식 검증이라
랜덤 데이터 대신 손으로 만든 소규모 fixture 사용:
- `apply_persona_weight`/`apply_exposure_penalty`가 `RULE1_*`/`RULE2_*` 키를 저장하는지 기본 확인.
- **Faithfulness 테스트(리뷰 지적 2 직접 대응)**: 공개 공식(`max(floor, min(ceiling, 1+alpha*deviation))`,
  `decay**count`)으로 기대값을 독립적으로 계산해 `item[RULE1_MULTIPLIER_KEY]`/`RULE2_DECAY_FACTOR_KEY`와
  일치하는지 assert — "동일 입력→동일 출력"이 아니라 "저장된 값=실제 공식값"을 검증.
- **구조적 결속 테스트**: `item["score"] == pytest.approx(원본_score * rule1_multiplier * rule2_decay_factor)`
  — 저장된 값이 실제로 score를 만든 인자임을 직접 증명(요청보다 한 단계 더 강한 보장).
- `exposure_counts`가 없을 때 `RULE2_*` 키가 아예 없는지 확인(엣지 케이스 문서화).
- 결정론성 테스트도 유지하되, 주석으로 "이것만으로는 faithfulness를 증명 못 함" 명시.

### `tests/test_explain.py` (신규)
- `explain_item`이 일부러 비현실적인 `rule1_multiplier`(예: 2.5, 정상 범위 밖)를 넣은 가짜 item dict를
  받았을 때 그 값을 그대로 문장에 반영하는지 확인 — 재계산하지 않음을 증명.
- `RULE2_*` 키가 없는 item(노출 이력 없음 시뮬레이션)에서 KeyError 없이 노출 감쇠 문장이 생략되는지 확인.
- `explain_user_style`의 3구간 분기(< 0.3 / 0.3~0.7 / > 0.7) 각각 올바른 문구 선택 확인.

### `tests/test_twiddler_service.py` (신규 또는 확장)
- `persona_service`/`exposure_service`/`catalog_service`를 monkeypatch해, 순위 변동이 임계값(3) 이상인
  아이템에만, 최대 5개까지만 `"explanation"` 키가 붙는지 확인 — 새로 추가되는 선정 로직 자체가 이 계획에서
  가장 복잡한 신규 로직이므로 별도 검증 필요.

## 검증 방법
1. `pytest tests/` 전체 통과 확인(신규 3개 파일 포함).
2. `ruff check src/modeling/twiddler/explain.py backend/api/services/twiddler_service.py backend/api/services/persona_service.py backend/api/core.py app/components/product_card.py app/components/eval_metrics_table.py app/utils/data_loader.py app/main.py` — 프로젝트 코드 스타일(라인 100자, import 순서) 준수 확인.
3. Streamlit 앱을 로컬 실행(`streamlit run app/main.py`)해 실제로 확인:
   - 순위가 크게 바뀐 유저를 선택해 "재랭킹 효과" 탭에서 (a) 유저 레벨 자연어 문장이 근거 블록에 1회만
     보이는지, (b) 순위 변동이 큰 카드에만 "💡 추천 이유" 배지가 보이고 클릭 시 문장이 펼쳐지는지,
     (c) 배지 없는 카드도 그리드 높이가 정렬되는지(플레이스홀더 확인).
   - main(ALS)과 detail(보완재) 컨텍스트 양쪽에서 동일하게 동작하는지 확인.
4. 기존 회귀 확인: `apply_twiddler`의 반환 스코어/순위 자체(`score`, `rank`)는 이번 변경으로 전혀 바뀌지
   않으므로, 기존 오프라인 평가 노트북들(`notebooks/20260708_ML_als_vs_als_twiddler_offline.ipynb` 등)의
   수치가 그대로 재현되는지 스팟 체크.
