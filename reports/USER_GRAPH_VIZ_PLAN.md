# 유저 중심 추천 근거 서브그래프 시각화(pyvis) 구현 계획 — 2026-07-08

> 이 문서는 유저를 선택했을 때 "왜 이 상품이 추천됐는지"를 유저→상품→세그먼트 1~2홉 서브그래프(lift 가중치 포함)로 보여주는 신규 Streamlit 컴포넌트의 구현 계획이다. 어느 세션에서든 이 문서를 읽고 이어서 작업할 수 있도록 `reports/`에 저장한다. 데이터 준비(§Context 2)는 이미 완료된 상태이며, 코드 구현(§구현 방식 1~6)은 아직 착수 전이다.

## Context

`app/components/user_selector.py::render_persona_and_user_selector()`로 유저를 고르면 메인 화면에 페르소나 라벨과 추천 상품 카드는 보이지만, **"왜 이 유저에게 이 상품이 추천됐는지"에 대한 근거가 UI에 없다.** `retail-clickstream-analysis` 레포(#29)에서 만든 LightGCN용 tri-graph(유저-상품-세그먼트, lift 가중치 포함) 데이터가 이미 있는데도 지금은 어디에도 노출되지 않는다.

`backend/api/services/lightgcn_service.py`와 `src/modeling/lightgcn/model.py`는 아직 스텁이지만, tri-graph는 **학습된 모델 출력이 아니라 사전 계산된 그래프 구조**(co-occurrence + lift)이므로 모델 연결과 무관하게 지금 바로 시각화 프로토타입을 만들 수 있다. 이 판단은 조사로 확인됐다.

### 조사로 확인한 사실 (구현 전 반드시 알아야 할 것들)

1. **이슈 텍스트의 함수명이 실제와 다르다** — `render_user_selector()`가 아니라 `render_persona_and_user_selector(demo_users_df) -> (user_id: int, user_info: dict)`다. 그대로 재사용하면 된다.
2. **[완료] tri-graph 데이터를 생성해서 rec-system으로 복사 완료.** retail-clickstream-analysis에서 `uv run python -m src.datasets.make_lgcn_graph --mode tri` 실행(입력 파일 전부 이미 로컬에 있었음, user_num=20000/item_num=1197 확인) → 아래 5개 파일을 `rec-system/data/processed/`로 복사 완료:
   ```
   tri_graph_uidx2tidx_train.json   {uidx: [tidx, ...]}              # cutoff(2025-08-01) 이전 전체 상호작용(조회+장바구니+구매)
   tri_graph_tidx2pidx.json         {tidx: [[segment_id, lift], ...]}
   tri_graph_uidx2pidx.json         {uidx: [segment_id]}             # 유저 본인 세그먼트(단일 라벨)
   uidx_user_id_mapping.csv         컬럼: uidx, user_id
   tidx_product_id_mapping.csv      컬럼: tidx, product_id
   ```
   `segment_id`는 rec-system에 이미 있는 `data/processed/segment_personas_train_only.json` / `customer_segments_labeled_train_only.csv`와 동일 ID 체계라 별도 매핑이 필요 없다.

   **⚠️ `tri_graph_uidx2tidx_valid.json`은 의도적으로 제외했다.** 검증 결과 이 파일은 train의 부분집합이 아니라 **cutoff(2025-08-01) 이후(미래) 구매만 담은 모델 평가용 정답 라벨**이었다(1,465명 중 1,448명이 subset 관계 위반 — `order_time >= CUTOFF_DATE`로 별도 계산됨, `make_lgcn_graph.py` 확인). 이걸 "구매 여부" 판정에 썼다면 아직 일어나지 않은 미래 구매를 과거 행동처럼 보여주는 데이터 누수가 됐을 것이다. → **구매/조회 판정은 §구현 방식 1에서 별도 로직으로 대체.**
3. **시각화 라이브러리는 pyvis로 결정.** 현재 `pyproject.toml`/`uv.lock`/`.venv`에 `pyvis`, `networkx` 둘 다 없음 — 신규 의존성 추가 필요(`pyvis`가 `networkx`를 자체 의존성으로 끌고 옴).
4. 규모(유저 ~20,000 x 상품 1,197)에서 tri-graph JSON은 이미 adjacency-list(딕셔너리) 형태라 `uidx`로 O(1) 조회 가능 — networkx로 전체 그래프를 메모리에 올릴 필요 없음.

## 구현 방식

기존 3단 계층(`backend/api/services/*` → `backend/api/core.py` → `app/utils/data_loader.py` → `app/components/*`)을 그대로 따른다. `als_service.py`/`complementary_service.py`와 동일하게 **모듈 전역 lazy load + 캐시** 패턴을 쓰고, 파일이 없으면 예외 대신 `status="not_implemented"`를 반환한다(기존 관례).

### 1. `backend/api/services/graph_service.py` (신규)

`get_user_subgraph(user_id: int, hops: int = 1) -> tuple[dict, str, str | None]`

- 5개 tri-graph/mapping 파일 + `data/processed/df_integrated_logs.csv`(customer_id, event_type, product_id, timestamp — 이미 rec-system에 있음, cutoff 이전/이후 전체 커버 확인됨)를 첫 호출 시 1회 로드(JSON key는 문자열이므로 `int()` 캐스팅 필수).
- `user_id → uidx → tidx 목록(train) → product_id`, `tidx → [(segment_id, lift), ...]`, `uidx → 본인 segment_id` 를 조합해 노드/엣지 dict 리스트로 반환.
  - **구매 여부 판정**: `tri_graph_uidx2tidx_valid.json`은 쓰지 않는다(§Context 참고 — 미래 데이터라 부적합). 대신 `df_integrated_logs.csv`에서 `customer_id == user_id`(uidx 매핑 불필요, user_id 그대로 사용) & `timestamp < CUTOFF_DATE(2025-08-01)` & `event_type == "purchase"`로 필터링한 `product_id` 집합을 구해, hop1 상품이 이 집합에 속하면 "구매", 아니면 "조회 등"으로 라벨링한다. `CUTOFF_DATE`는 ALS(`configs/ALS/params.yaml`)·tri-graph 파이프라인과 동일한 값이므로 상수로 고정.
  - 노드 수 폭발 방지 상수: `MAX_PRODUCTS_HOP1=12`(구매 우선 정렬), `MAX_SEGMENTS_PER_PRODUCT=3`(세그먼트 총량이 6개뿐이라 실질 위험 낮음), `MAX_TOTAL_NODES=60`(안전판).
  - hop=2: 유저 본인 세그먼트 + lift 최고 세그먼트(`HOP2_MAX_EXPANDED_SEGMENTS=2`)에서 세그먼트당 lift 상위 인기 상품(`HOP2_MAX_PRODUCTS_PER_SEGMENT=5`)을 추가로 끌어옴 — hop1에 이미 나온 상품은 제외.
  - 콜드 유저(상호작용 0건) / uidx 매핑 없는 유저(cutoff 이후 신규 유저) → 빈 그래프 + 안내 메시지로 graceful 처리.
  - 파일 자체가 없으면 `status="not_implemented"`.

### 2. `backend/api/core.py`에 오케스트레이션 함수 추가

```python
from backend.api.services import als_service, complementary_service, graph_service, lightgcn_service, twiddler_service

def get_user_subgraph_items(user_id: int, hops: int = 1) -> tuple[dict, str, str | None]:
    return graph_service.get_user_subgraph(user_id, max(1, min(hops, 2)))
```

### 3. `app/utils/data_loader.py`에 얇은 래퍼 추가

기존 `get_main_recommendations`/`get_detail_recommendations`와 동일하게 `@st.cache_data(ttl=30)`로 `backend.api.core.get_user_subgraph_items`를 감싼다(HTTP 없이 in-process 호출, 기존 ttl 값과 통일).

### 4. `app/components/user_graph.py` (신규) — `render_user_graph(user_id: int) -> None`

- **pyvis-Streamlit 통합**: `Network(cdn_resources="in_line")`로 vis-network 리소스를 HTML에 인라인 임베드(디스크에 `lib/` 폴더나 임시파일을 쓰지 않음 — 제한된 파일시스템 환경에서도 안전). `net.generate_html(notebook=False)`로 HTML 문자열을 바로 받아 `st.components.v1.html(html, height=660, scrolling=True)`로 렌더링.
- **색상/스타일은 `.streamlit/config.toml`의 라이트 테마 고정값**(`primaryColor #6366f1` 등, 다크모드 토글 없음 확인됨)과 맞춰 상수로 하드코딩. 유저(중심, 강조색, `physics=False`로 위치 고정) → 상품(구매=굵은 실선/초록, 조회=점선/회색, hop2 확장 상품=옅은 회색) → 세그먼트(다이아몬드 모양, 본인 세그먼트는 빨강 강조, lift 값이 엣지 라벨+두께로 표시).
- 상품 라벨/카테고리 표시는 `load_products()`(이미 캐시됨, `item_id` 컬럼 확인됨)를 재사용 — 그래프 서비스가 카탈로그를 중복으로 알 필요 없음.
- hop=1/2 전환은 컴포넌트 내부 `st.toggle`로 처리(시그니처는 `user_id`만 받도록 유지).

### 5. `app/main.py` 연결

`_render_main_recommend()`의 유저 정보 표시 블록(`app/main.py:213-218`, persona/user_type/log_count 마크다운) 바로 다음, Twiddler 게이팅(`app/main.py:220`) 이전에 `render_user_graph(user_id)` 한 줄 추가. `_render_detail_recommend()`는 상품 중심 화면이라 그래프 맥락이 흐려지므로 기본적으로 넣지 않는다(이슈 체크리스트도 main만 언급).

### 6. 의존성

`pyproject.toml`의 `dependencies`에 `"pyvis>=0.3.2",` 한 줄 추가(별도 섹션 주석과 함께), `requirements.txt`도 동일하게 미러링, 이후 `uv lock`/`uv sync`로 잠금 갱신.

## 검증 방법

1. [완료] tri-graph 5개 파일이 `data/processed/`에 이미 있음 — `uv sync`로 pyvis만 설치하면 됨.
2. `/run` 스킬 또는 `streamlit run app/main.py`로 앱 기동.
3. Heavy 유저 1명, Cold 유저 1명을 각각 선택해:
   - 그래프가 렌더링되는지(유저→상품→세그먼트 노드/엣지, lift 라벨 표시).
   - hop=2 토글 시 세그먼트 연관 인기 상품이 추가로 나타나는지.
   - Cold 유저(상호작용 없음)에서 "상호작용 데이터가 없습니다" 안내가 뜨는지(에러 아님).
4. 파일이 아직 없는 상태(현재 상태)에서 앱을 켜면 "🚧 그래프 데이터가 아직 준비되지 않았습니다" 안내만 뜨고 나머지 화면(ALS/LightGCN 카드 등)은 정상 동작하는지 회귀 확인.
5. `tests/test_graph_service.py` 신규 작성 — 소규모 fixture로 정상 케이스/uidx 없는 유저/콜드 유저/파일 없음/cap 동작을 검증(`tests/` 기존 테스트 패턴 참고).

## Critical Files

- `backend/api/services/graph_service.py` (신규)
- `backend/api/core.py`
- `app/utils/data_loader.py`
- `app/components/user_graph.py` (신규)
- `app/main.py` (`app/main.py:218` 부근에 연결)
- `pyproject.toml`, `requirements.txt`
- `data/processed/` — tri-graph 5개 파일 [복사 완료], `df_integrated_logs.csv`(기존, 구매/조회 판정용으로 신규 사용)

## 진행 상태

- [x] 데이터 준비(생성 + 복사 + valid.json 누수 이슈 발견 및 회피)
- [ ] `graph_service.py` 구현
- [ ] `core.py` / `data_loader.py` 연결
- [ ] `user_graph.py` 컴포넌트 구현
- [ ] `main.py` 연결
- [ ] `pyproject.toml` 의존성 추가 + `uv sync`
- [ ] 앱 기동 후 Heavy/Cold 유저 시각 확인
- [ ] `tests/test_graph_service.py` 작성
