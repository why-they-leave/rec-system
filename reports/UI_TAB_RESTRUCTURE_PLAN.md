# 추천 시스템 데모 UI 재구성 — 검증 질문별 탭 분리 — 2026-07-08

> 이 문서는 `app/main.py`의 메인 화면(현재 ALS vs LightGCN을 한 화면에서 나란히 비교하는 구조)을 "검증하려는 질문별로 탭을 나누는" 구조로 재구성하는 계획이다. 어느 세션에서든 이 문서를 읽고 이어서 작업할 수 있도록 `reports/`에 저장한다. **코드 구현은 아직 착수 전이며, 이 문서 작성까지가 이번 요청 범위다.**

## Context

현재 `_render_main_recommend()`(`app/main.py:199-343`)는 ALS(+Twiddler before/after) 카드와 LightGCN(tri/bi) 카드를 2열로 나란히 보여주고, 상단에 "ALS vs LightGCN 공통 추천 상품 수 / Jaccard 유사도"까지 계산한다. 문제는 **하나의 화면에 독립변수 세 개(알고리즘 종류 / Twiddler 유무 / 그래프 구조 유무)가 섞여 있어, 결과가 어느 변수 때문에 그런지 해석하기 어렵다**는 점이다.

재구성 원칙(요청 그대로): **검증하려는 질문별로 탭을 나눈다.**
- **Tab 1 — Twiddler 재랭킹**: ALS vs ALS+Twiddler, 보완재 vs 보완재+Twiddler (같은 알고리즘 내에서 Twiddler 유무만 비교)
- **Tab 2 — 페르소나 기여도**: LightGCN bi-graph vs tri-graph (같은 LightGCN 내에서 페르소나 결합 유무만 비교) — 상단 정량비교 + 하단 서브그래프 시각화([reports/USER_GRAPH_VIZ_PLAN.md](USER_GRAPH_VIZ_PLAN.md)가 여기 들어감)

**[확정] 기존 "ALS vs LightGCN 공통 추천 상품 수 / Jaccard 유사도" 지표(`app/main.py:265-268, 286-287`)는 제거한다** — ALS(Tab1의 변수)와 LightGCN(Tab2의 변수)을 직접 비교하는 지표라 "질문별로 섞지 않는다"는 원칙과 정면으로 배치되어 제거를 제안했고 사용자 확인 완료.

### 오늘 이미 수행된 관련 분석 (재사용 대상)

Tab1에 필요한 지표(HR@K/Recall@K/NDCG@K 정확도, 반복방문 중복률/누적 카테고리 수 다양성)는 오늘 작성된 노트북 3개에서 **이미 실제 서빙 코드를 그대로 import해 계산 완료**했다 — 새로 만들 필요 없이 로직만 스크립트로 이관하면 된다:

| 노트북 | 담당 범위 | 재사용하는 실서빙 코드 |
|---|---|---|
| `notebooks/20260708_ML_twiddler_final_design.ipynb` | 현재 프로덕션 Twiddler 로직(Rule1 v3 + Rule2 v3, Rule3 제거) 확정 근거 | `src/modeling/twiddler/rerank.py` |
| `notebooks/20260708_ML_als_vs_als_twiddler_offline.ipynb` | **Tab1 메인페이지** 지표 원본 (ALS only vs ALS+Twiddler) | `persona_service.py`, `rerank.py`, `exposure_service.py`, `src/modeling/als/evaluate.py`의 `hit_rate_at_k`/`recall_at_k`/`ndcg_at_k` |
| `notebooks/20260708_ML_complementary_vs_twiddler_offline.ipynb` | **Tab1 상세페이지** 지표 원본 (보완재 only vs 보완재+Twiddler) | 위와 동일 + `src/modeling/complementary/model.py` |

두 "offline" 노트북 모두 "로직을 새로 구현하지 않고 실서빙 코드를 그대로 import한다"는 원칙을 이미 지키고 있어, 배치 스크립트로의 이관이 곧 그대로 옮기는 작업에 가깝다(로직 변경 없음).

### 설계 결정 — 지표 계산 방식: 배치 사전계산

Tab1의 HR@K/NDCG@K/다양성 지표는 **population 전체 평균(aggregate)** 값이라 사이드바에서 어떤 유저를 고르든 항상 동일한 표를 보여주면 된다 — 즉 "선택된 유저"에 의존하지 않는다. 따라서 배치로 한 번 계산해 CSV로 저장해두는 방식이 "어떤 유저를 고르든 항상 같은 결과를 보여줄 수 있는가"라는 조건을 자명하게 만족한다(애초에 유저별로 달라지는 값이 아님). → **배치 사전계산 + Streamlit은 읽기만** 채택. 반대로 오늘 노트북의 "가중치 의미 예시 — 유저 3명" 같은 유저별 케이스 스터디는 이번 Tab1 요구사항에 명시적으로 포함되지 않아 구현 범위에서 제외했다.

### 설계 결정 — Tab2 상단(bi vs tri 정량비교)은 이번 범위 제외

`src/modeling/lightgcn/model.py`가 여전히 `NotImplementedError` 스텁이라 학습된 bi/tri 추천 결과 자체가 없다. "페르소나 유무가 순위·카테고리 분포에 미치는 영향을 보여주는 케이스"도 마찬가지로 bi/tri 양쪽 모델 출력이 있어야 계산 가능하다. 둘 다 **모델 학습 완료 후로 미루고**, 지금 데이터가 준비된 하단(tri-graph 서브그래프 시각화, [USER_GRAPH_VIZ_PLAN.md](USER_GRAPH_VIZ_PLAN.md))만 우선 구현한다.

## 전체 구조

사이드바 최상단(카테고리 필터보다 위)에 탭 전환 버튼 추가 — Streamlit 기본 `st.tabs`는 상단 가로형이라 "좌측 탭"을 네이티브로 지원하지 않으므로, 기존 `_render_twiddler_toggle`과 동일한 "버튼 클릭 → `session_state` 갱신 → `st.rerun()`" 패턴을 재사용한다(라디오 대신 버튼으로 사용자 확정, `type="primary"`로 현재 탭만 강조).

```python
# app/main.py::_setup_sidebar() 상단에 추가
st.sidebar.markdown("**검증 화면**")
current_tab = st.session_state.get("main_tab", "rerank")
if st.sidebar.button("Twiddler 재랭킹", use_container_width=True,
                      type="primary" if current_tab == "rerank" else "secondary"):
    st.session_state["main_tab"] = "rerank"
    st.rerun()
if st.sidebar.button("페르소나 기여도", use_container_width=True,
                      type="primary" if current_tab == "persona" else "secondary"):
    st.session_state["main_tab"] = "persona"
    st.rerun()
st.sidebar.markdown("---")
```
`main()`은 `view`와 동일하게 `main_tab`도 `if "main_tab" not in st.session_state:` 로 기본값(`"rerank"`)을 미리 초기화한다(버튼은 라디오처럼 `key=`로 상태를 자동 관리하지 않으므로).

`main_tab`(어떤 검증 질문을 볼지)과 `view`(main/detail 중 뭘 볼지)는 서로 다른 축이라 독립적으로 유지한다. `main()`의 라우팅:

```python
def main() -> None:
    ...
    main_tab = st.session_state.get("main_tab", "rerank")
    view = st.session_state["view"]
    if main_tab == "rerank":
        if view == "main":
            _render_rerank_main(selected_categories, demo_users_df)   # 기존 _render_main_recommend에서 LightGCN 블록 제거
        elif view == "detail":
            _render_rerank_detail(demo_users_df)                      # 기존 _render_detail_recommend, 이름만 변경
    elif main_tab == "persona":
        _render_persona_tab(demo_users_df)                            # 신규 — LightGCN 카드 이관 + 서브그래프
```

## Tab 1 — Twiddler 재랭킹 (`_render_rerank_main`, `_render_rerank_detail`)

기존 함수명을 유지하되(`_render_main_recommend`→`_render_rerank_main`, `_render_detail_recommend`→`_render_rerank_detail`) 내부 로직은 거의 그대로 둔다. 변경점은:

1. **LightGCN 블록 제거**: `app/main.py:246-251`(gcn_tri/gcn_bi 조회), `265-268`(Jaccard), `279-282`(gcn 카테고리 필터), `286-287`(공통상품/Jaccard 메트릭), `314-343`(`col_gcn` 전체) 삭제. `col_als, col_gcn = st.columns(2)`(`292`)는 단일 컬럼 또는 폭 조정.
2. **[신규] 정량 지표 표 추가** — 카드 그리드 아래에 `render_eval_metrics(context="main")`(메인) / `render_eval_metrics(context="detail")`(상세) 호출 한 줄 추가.
3. 상세페이지(`_render_rerank_detail`)는 이미 보완재 before/after 토글이 있으므로(§Context 확인됨) 그 아래에 동일하게 지표 표만 추가.

### 데이터 파이프라인 — `src/evaluation/evaluate_twiddler.py` (신규 배치 스크립트)

CLAUDE.md 역할 분리 원칙("평가 지표 계산: `src/evaluation/`")에 따른 위치. `src/modeling/als/evaluate.py`와 동일한 CLI 스크립트 패턴(`python -m src.evaluation.evaluate_twiddler`)으로, 오늘 노트북 두 개의 셀 코드를 그대로 옮긴다(신규 로직 없음):

- **메인(ALS)**: `20260708_ML_als_vs_als_twiddler_offline.ipynb`의 §3(단일 세션 정확도) + §4(5회 새로고침 다양성) 셀.
- **상세(보완재)**: `20260708_ML_complementary_vs_twiddler_offline.ipynb`의 동일 구조 셀.
- 출력:
  - `data/outputs/eval/twiddler_accuracy.csv` — 컬럼: `context`(main/detail), `condition`(baseline/twiddler), `k`, `segment`, `HR`, `Recall`, `NDCG`, `eval_users`
  - `data/outputs/eval/twiddler_diversity.csv` — 컬럼: `context`, `condition`, `k`, `segment`, `repetition_rate`, `unique_item_ratio`, `categories_first`, `categories_cumulative`, `n_users`
  - `context` 컬럼으로 메인/상세를 한 파일에 통합(`als/evaluate.py`가 `dataset` 컬럼으로 full/us를 구분하는 것과 동일한 패턴).
- 재현성: `CUTOFF_DATE=2025-08-01`, `random_state=42` — 두 노트북과 동일 값을 상수로 고정.
- **[갱신] 3단 구성으로 확장**(population 지표 + 세그먼트 breakdown + 선택 유저 케이스, 사용자 요청 반영):
  - `segment` 컬럼 추가 — 유저별 지표를 `"ALL"`(population 전체)과 그 유저의 세그먼트(persona_service.get_persona 반환값) 양쪽에 동시 적립해 재계산 없이 한 번의 순회로 두 레벨을 모두 만든다(`ALL_SEGMENTS_LABEL` 상수).
  - 선택 유저 1명분 케이스(alpha/decay/선호 카테고리)는 population 시뮬레이션과 무관하게 가벼워 배치가 아니라 `backend/api/core.py::get_user_twiddler_case`가 라이브로 계산 — HR/NDCG는 유저 1명 기준으로는 통계적으로 의미가 없어(0 또는 1) "정확도"로 보여주지 않고, 실제 재랭킹 파라미터 숫자만 보여준다.
  - UI: `render_eval_metrics(context, persona_label)`가 ①population 정확도 ②population 다양성 ③선택 페르소나 breakdown을, `render_user_twiddler_case(user_id)`가 선택 유저 케이스를 렌더링(`app/components/eval_metrics_table.py`).
  - **[갱신] 표 → 그래프**: ①②③ 전부 `st.dataframe` 대신 Plotly 그룹 막대그래프(`_grouped_bar_figure`, condition=baseline/twiddler 고정 2색 `#0173B2`/`#DE8F05`, 지표별 서브플롯, dual-axis 없음)로 렌더링, 원본 표는 `st.expander("표로 보기")`에 접어서 접근성용으로 유지(dataviz 스킬의 "표 뷰는 항상 존재" 원칙).
  - **[갱신] 배치 위치**: `render_user_twiddler_case`를 화면 하단(①②③ 이후)에서 "유저 유형/Twiddler 상태" 지표 바로 아래(ALS 카드 그리드 시작 전)로 이동 — ①②③보다 먼저 보이므로 헤더에서 번호(④)를 떼고 "🔍 선택 유저 Twiddler 재랭킹 근거"로 표기.

### `app/utils/data_loader.py`에 로더 추가

```python
EVAL_DIR = Path("data/outputs/eval")

@st.cache_data
def load_twiddler_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    """evaluate_twiddler.py가 생성한 사전계산 정확도/다양성 지표 CSV를 읽는다."""
    accuracy_df = pd.read_csv(EVAL_DIR / "twiddler_accuracy.csv")
    diversity_df = pd.read_csv(EVAL_DIR / "twiddler_diversity.csv")
    return accuracy_df, diversity_df
```
CSV가 없으면(배치 스크립트 미실행) `FileNotFoundError` — 기존 `_render_rerank_main`의 `try/except FileNotFoundError` 블록에 자연스럽게 흡수되므로 신규 예외 처리 불필요.

### `app/components/eval_metrics_table.py` (신규) — `render_eval_metrics(context: str) -> None`

`load_twiddler_eval()`에서 `context`로 필터링한 두 표(정확도/다양성)를 `st.dataframe`으로 렌더링. 순수 렌더링만 담당 — 데이터는 backend/api/services를 거치지 않고 `data_loader.py`가 CSV를 직접 읽는다(카탈로그성 정적 데이터와 동일한 대우, `load_products()`와 같은 패턴 — 유저별 실시간 조회가 아니라 population aggregate이므로 backend 서비스 레이어가 필요 없음).

## Tab 2 — 페르소나 기여도 (`_render_persona_tab`, 신규)

### 상단 — 이번 범위 제외
```python
st.info("🚧 LightGCN 모델 학습 완료 후 bi-graph vs tri-graph 정량 비교(HR@K/NDCG@K)와 "
        "페르소나 유무에 따른 순위·카테고리 분포 비교 케이스가 추가될 예정입니다.")
```

### 중단 — LightGCN 카드 그리드 (기존 로직 이관, 변경 없음)
`app/main.py:246-251`(gcn_tri/gcn_bi 조회), `279-282`(카테고리 필터), `314-343`(`col_gcn` 블록 전체, 단 `col_als, col_gcn = st.columns(2)` 대신 단일 컬럼)을 그대로 옮긴다 — 그래프 종류 라디오(`tripartite`/`bipartite`), `_render_model_status_or_grid` 호출 등 로직/카드 컴포넌트는 손대지 않고 위치만 이동.

### 하단 — 서브그래프 시각화
[reports/USER_GRAPH_VIZ_PLAN.md](USER_GRAPH_VIZ_PLAN.md)의 구현 방식(1~6)을 그대로 따른다. `render_persona_and_user_selector()` 호출 직후, LightGCN 카드 그리드 앞 또는 뒤(레이아웃은 구현 시 결정)에 `render_user_graph(user_id)` 호출을 추가한다. "유저 선택 시 자동 연동"은 Streamlit이 selectbox 변경마다 자동 rerun하는 기본 동작을 그대로 활용하므로 별도 연동 로직이 필요 없다.

> `USER_GRAPH_VIZ_PLAN.md` §5("app/main.py 연결")는 이 문서 기준으로 갱신 필요 — 연결 대상이 `_render_main_recommend()`가 아니라 신규 `_render_persona_tab()`이 된다.

## Critical Files

- `app/main.py` — 사이드바 라디오 추가, `main()` 라우팅 분기, `_render_main_recommend`→`_render_rerank_main`(LightGCN 블록 제거) 리네임, `_render_detail_recommend`→`_render_rerank_detail` 리네임, `_render_persona_tab`(신규, LightGCN 블록 이관 + 서브그래프)
- `src/evaluation/evaluate_twiddler.py` (신규 배치 스크립트)
- `app/utils/data_loader.py` — `load_twiddler_eval()` 추가
- `app/components/eval_metrics_table.py` (신규)
- `data/outputs/eval/twiddler_accuracy.csv`, `twiddler_diversity.csv` (신규, 배치 스크립트 산출물)
- [reports/USER_GRAPH_VIZ_PLAN.md](USER_GRAPH_VIZ_PLAN.md) — Tab2 하단 그대로 편입, §5 연결 지점만 갱신 필요

## 진행 순서 제안

1. 사이드바 탭 라디오 + `main()` 라우팅 골격 — 기존 화면을 그대로 두 탭으로 나눠 배치만 바꿈(지표/그래프 없이 우선 골격만, 회귀 확인 쉬움)
2. `_render_main_recommend`를 `_render_rerank_main`/`_render_persona_tab`으로 분리(LightGCN 블록 이관, Jaccard 지표 제거)
3. `src/evaluation/evaluate_twiddler.py` 작성 + 1회 실행 → `data/outputs/eval/*.csv` 생성
4. `eval_metrics_table.py` 컴포넌트 → Tab1 메인/상세 양쪽에 연결
5. Tab2 하단 서브그래프 — `USER_GRAPH_VIZ_PLAN.md` 그대로 구현(`graph_service.py`~`user_graph.py`)
6. 회귀 확인 — 기존 before/after 카드, Twiddler 토글, LightGCN tri/bi 라디오가 탭 이동 후에도 동일하게 동작하는지, 사이드바 카테고리 필터가 두 탭 모두에서 정상 작동하는지

## 진행 상태

- [x] 이 문서 작성
- [x] 사이드바 탭 골격 — 라디오 대신 좌측 버튼 2개(`type="primary"`로 현재 탭 강조)로 사용자 확인 후 변경
- [x] `_render_rerank_main`/`_render_rerank_detail`/`_render_persona_tab` 분리, Jaccard 지표 제거, 미사용 CSS 마커(`_ALGO_LEFT/RIGHT_MARKER`) 정리, `AppTest`로 3개 화면(재랭킹 메인/상세, 페르소나) 무예외 확인
- [x] `src/evaluation/evaluate_twiddler.py` 작성 + 실행 → `data/outputs/eval/twiddler_{accuracy,diversity}.csv` 생성, 노트북 수치와 정확히 일치 확인(예: 메인 HR@10 0.0369→0.0335, 상세 K=1 반복률 100%→34.2%)
- [x] `app/components/eval_metrics_table.py` 구현 + `_render_rerank_main`/`_render_rerank_detail`에 연결, `AppTest`로 두 화면 모두 표 렌더링 무예외 확인
- [x] Tab2 서브그래프 구현(`USER_GRAPH_VIZ_PLAN.md` 그대로 — `graph_service.py`/`user_graph.py` 완료, `pyvis` 의존성 추가 후 `uv sync`, 실제 유저로 709KB HTML 생성 확인)
- [x] 회귀 확인(`AppTest`로 재랭킹 메인/상세/페르소나 3개 화면 + 탭 버튼 클릭 플로우 + hop2 토글까지 무예외 확인, 브라우저 시각적 검토는 미실시)

**남은 것**: 브라우저로 직접 열어 pyvis 그래프의 실제 시각적 레이아웃(물리 시뮬레이션 안정성, 텍스트 겹침 등)과 사이드바 탭 버튼 스타일(강조색 대비 등)을 눈으로 확인하는 작업만 남았다 — `AppTest`는 예외 발생 여부만 확인하고 렌더링 품질은 검증하지 못한다.

### [추가] UX 피드백 반영 (2026-07-08 후속)

- [x] **새로고침 시뮬레이션 버튼** (Tab1 메인, 토글 버튼 우측) — 클릭마다 `exposure_service` 노출 이력을 실제로 누적시켜(캐시 없는 `data_loader.simulate_next_session`) 5라운드까지 카드가 실제로 바뀌는 걸 라이브로 보여준다. 라운드 0→1 전환 시 `exposure_service.reset()`으로 매번 깨끗하게 시작. `AppTest`로 3회 연속 클릭 시 라운드 0→1→2→3 정상 전진 확인.
- [x] **페르소나 설명 위치 이동** — 두 드롭다운 아래 전체 폭 → 페르소나 selectbox 바로 아래(`col_persona` 안)로 이동. UX 근거: 설명이 "페르소나 선택" 시점에 바로 보여야 그다음 "유저 선택"에 도움이 된다 — 두 선택을 다 마친 뒤 보여주는 건 사후 확인용일 뿐 의사결정에 도움 안 됨.
- [x] **Plotly 툴바 숨김** — `st.plotly_chart(..., config={"displayModeBar": False})`.
- [x] **그래프 라벨 겹침 개선** — 상품 라벨(상품명)을 상시 표시에서 hover 전용으로 전환(요청: "hover 시에만 보이게" 채택 — 노드 수가 많아 상시 라벨이 가장 큰 혼잡 원인이었음), "소속 세그먼트" 엣지 라벨도 hover 전용으로 전환(세그먼트 노드 라벨과 겹침), `physics.barnesHut` 반발력(`gravitationalConstant` -8000→-15000, `springLength` 160→220)을 더 키움.
- [x] **범례 우측 고정 배치** — `st.columns([5, 1.3])`로 그래프 옆에 세로형 박스(색상 스와치 + 라벨, 테두리/배경 있는 카드) 배치. 기존 하단 한 줄 텍스트보다 가독성 개선.
- [x] `AppTest`로 재랭킹 메인(시뮬레이션 클릭 포함)/상세/페르소나 3개 화면 재확인, 브라우저 시각 검토는 미실시.

### [추가] 스크린샷 피드백 반영 (2026-07-08 재후속)

- [x] **시뮬레이션 라운드별 순위 배지** — `sim_history_{user_id}` 리스트로 라운드별 결과를 누적 보관, 1회차는 "적용 전(before)" 전체 풀 순위 대비, 2회차부터는 직전 회차 대비 ▲/▼ 배지를 보여준다(기존 `rank_before_map`/`get_rank_delta` 재사용, 신규 로직 없음). round1→round2 실측으로 방향(down/same) 정상 계산 확인.
- [x] **페르소나 설명 위치 재조정** — 스크린샷의 두 후보(① 페르소나 드롭다운 바로 아래=현재, ② "User XXX" 요약 블록의 "페르소나: X" 줄 옆=제안) 중 ②를 채택. 근거: ①은 컬럼이 좌우 비대칭 레이아웃이 되고, 추상적 카테고리 설명을 유저 선택 전에 보여줘 즉각적 의사결정 도움이 적음. ②는 "이 구체적 유저가 왜 이렇게 행동하는지"의 근거로 읽혀 더 유용하고 레이아웃도 안 깨짐. `render_persona_and_user_selector`는 이제 `st.info`를 직접 렌더링하지 않고 `user_info["persona_desc"]`로 값만 반환 — 호출부 3곳(재랭킹 메인/상세, 페르소나 탭)이 각자 "페르소나: X" 줄 옆에 `st.caption`으로 붙인다.
- [x] **유저 노드 아이콘** — 상품 노드와 동일한 `_emoji_circle_image` 재사용, `shape="dot"` → `"circularImage"`로 바꿔 사람 이모지(👤)를 색상 원 위에 표시.
- [x] `AppTest`로 4개 시나리오(재랭킹 메인/상세/페르소나 화면, 시뮬레이션 6연속 클릭으로 5라운드→시작 복귀까지) 전부 무예외 확인.

### [추가] 글씨 크기·유저 노드 대비 조정 (2026-07-08 3차 후속)

- [x] **`st.metric` 값 글씨 축소** — 앱 전체에 metric 호출이 5곳(유저 유형/Twiddler 상태, Rule 1/2 강도·선호 카테고리)뿐이라 `app/static/style.css`에 전역 `[data-testid="stMetricValue"]` 규칙(1.3rem, subheader보다 살짝 작게)을 추가해도 다른 곳과 안 섞임.
- [x] **유저 노드 대비 개선** — `_COLOR_USER`를 `#6366f1`(config.toml primaryColor)→`#818cf8`(indigo-400, 한 단계 밝게)로, `_USER_EMOJI`를 `👤`(bust in silhouette, 대부분 폰트에서 검정 실루엣)→`🧑`(사람, 기본 노란빛 피부톤)로 교체 — 배경과 아이콘 둘 다 어두워 안 보이던 문제(요청으로 발견) 해결. 범례 스와치도 같은 상수를 쓰므로 함께 밝아짐.

### [추가] 글씨 크기/페르소나 박스 재조정 (2026-07-08 4차 후속)

- [x] **`st.metric` 값 1.3rem → 1.7rem + bold** — "너무 작아졌다"는 피드백으로 다시 키우고 `font-weight: 700` 추가.
- [x] **페르소나 설명 박스 원복** — 위치(User 요약 블록의 "페르소나: X" 줄 옆)는 유지하되, 스타일을 `st.caption`(회색 텍스트)에서 `st.info(icon="📖")`(파란 배경 박스)로 되돌림 — "위치는 좋은데 배경색/글씨 크기는 이전처럼" 피드백 반영. 3개 화면(재랭킹 메인/상세, 페르소나 탭) 전부 동일하게 적용.
- [x] **[갱신] 페르소나 박스 재배치 + 내용 포맷** — 유저 프로필 요약 카드(subheader/caption 블록) 위로 다시 이동, 내용은 `**{persona_label}**  \n{persona_desc}`(페르소나명 볼드 + 줄바꿈 + 설명)로 한 박스에 같이 표시. 3개 화면 모두 `AppTest`로 정확한 문자열 렌더링 확인.

### [추가] 새로고침 시뮬레이션 "빈 배지" 버그 수정 (2026-07-08 5차 후속)

- [x] **원인**: `_render_recommend_grid`가 `rank_before_map`에 없는 상품을 "비교 안 함"과 "비교했는데 못 찾음"을 구분하지 않고 똑같이 `rank_delta=None`으로 처리 — 카드가 배지를 아예 숨겨버려 유지/상승/하락 어디에도 안 걸리는 상품이 나왔다. 시뮬레이션은 라운드마다 top-10만 비교해서(직전 라운드 top-10 밖에 있던 상품이 새로 진입하는 게 정상 시나리오) 특히 자주 발생.
- [x] **수정**: `rank_before_map`이 주어졌는데 해당 상품이 없으면 `{"direction": "new", "label": "신규"}`로 명시적으로 표시(`app/main.py::_render_recommend_grid`). `product_card.py`에 `"new": "🆕"` 아이콘, `style.css`에 `.badge-new`(연파랑) 추가.
- [x] 실측 확인 — round1→round2 비교에서 이전엔 8/10개가 빈 배지였는데 수정 후 전부 `same`/`down`/`new` 중 하나로 표시(`None` 없음). `AppTest`로 3화면 + 시뮬레이션 3연속 클릭 무예외, 렌더링 텍스트에 "신규" 포함 확인.
