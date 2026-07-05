# app ↔ backend(ALS·Twiddler·LightGCN·보완재) 연결 계획 — 2026-07-05

> 이 문서는 `app/`(Streamlit UI)을 `backend/`의 4가지 추천 방식(ALS, ALS+Twiddler, LightGCN, 보완재)에 실제로 연결하기 위한 구현 계획이다. 어느 세션에서든 이 문서를 읽고 이어서 작업할 수 있도록 리포 안(`reports/`)에 저장한다. 원본 계획 승인 시점 기준이며, 진행 상황에 따라 이 문서를 계속 갱신한다.

## Context

`app/main.py`는 현재 `data/dashboard/*.csv`(사전 계산된 더미 데이터, `scripts/generate_demo_data.py`가 `np.random.seed(42)`로 생성)만 읽는 구조다. `backend/`에는 4가지 추천 방식이 있어야 하는데 실제로는:

- **ALS**(`backend/ALS/als_model.py`, `als_evaluate.py`) — 유일하게 실제 코드가 있는 배치 스크립트. `PARAMS_PATH = Path(__file__).parents[3]/...`가 리포 루트 밖을 가리키는 버그가 있고, `configs/ALS/params.yaml`과 학습 입력 `data/processed/als_events.csv`가 리포에 존재하지 않으며, `implicit` 라이브러리도 의존성에 선언돼 있지 않다.
  - **다만 `backend/ALS/als_model.pkl`에 이미 학습된 모델 아티팩트가 있음을 pickle opcode 검사로 확인** — `implicit`이 설치돼 있지 않아 직접 언피클은 못 했지만, 바이트 구조상 `als_model.py::save_outputs()`가 만드는 것과 동일한 `{model, user_enc, item_enc, user_dec, item_dec}` 딕셔너리이고, `model`은 `implicit.cpu.als.AlternatingLeastSquares`이며 `item_factors` shape `(1197, 16)`, `user_factors` shape `(19930, 16)`이다. 1197은 `products.csv`의 실제 상품 수와 일치하고, 19,930명은 데모용 10명(`demo_users.csv`)보다 훨씬 큰 실제 규모의 유저 베이스다 — **더미가 아닌 실제 학습 결과물**로 판단됨.
  - 단, 이 pickle에는 `model/user_enc/item_enc/user_dec/item_dec`만 있고 **학습 시점 상호작용 행렬(matrix)·인기도(popular_items) 테이블·유저 heavy/cold 구분 정보는 포함돼 있지 않다** — 이 셋은 `save_outputs()`가 현재 저장하지 않기 때문. Phase 1에서 이 한계를 어떻게 다룰지 명시한다.
- **Twiddler**(`backend/Twiddler/`) — 빈 폴더. `app/main.py`의 Before/After 라디오는 CSV의 `twiddler` 컬럼 값을 읽을 뿐, 실제 페르소나 가중치 재랭킹 로직은 없다.
- **LightGCN**(`backend/LightGCN/`) — 빈 폴더. 코드/의존성(torch 등) 전무.
- **TF-IDF**(`backend/TF-IDF/bowanjae.csv`) — 실제로는 TF-IDF가 아니라 리포 루트의 `bowanjae_pipeline.py`가 만든 **상품명 기반 co-occurrence 보완재(complementary item) 추천 결과**다. 진짜 콘텐츠 기반(대체재) 모델은 노트북에 죽은 import만 남아있고 구현된 적이 없다.
  - 다만 **`src/modeling/complementary_recommender.py`(`run_modeling`)와 `src/evaluation/evaluate_complementary.py`(`evaluate_model`)에 이미 `bowanjae_pipeline.py`의 모델링/평가 로직이 역할 분리되어 리팩터링돼 있음을 확인** — `reports/item_recommendation.md`가 언급한 "모듈 분리 작업(이슈 #7)"이 이미 절반은 진행된 상태다. `prod_A_name`/`prod_B_name` 대신 `prod_A`/`prod_B`(원본 id)를 그대로 유지하고, `logging` 기반으로 정리돼 있어 재사용에 적합하다. **이번 계획은 이 두 파일을 새로 만들지 않고 그대로 재사용한다.**

사용자 결정사항(질의응답 결과):
1. **서빙 방식**: Streamlit이 CSV를 직접 읽는 대신, **실시간 추론 서비스(API)**를 신설해 앱이 요청 시점에 백엔드에 질의한다.
2. **이번 계획의 범위**: LightGCN·진짜 콘텐츠 기반 모델·Twiddler 재랭킹 알고리즘을 전부 새로 구현하는 것은 범위 밖. **레포/파이프라인 구조 정리 + ALS 실연결을 우선**하고, 나머지 3개는 동일 인터페이스를 갖춘 스텁으로 남겨 추후 채워 넣을 수 있게 한다.
3. **폴더 정리**: 보완재 로직은 이미 존재하는 `src/modeling/complementary_recommender.py` / `src/evaluation/evaluate_complementary.py`를 그대로 활용하고, `backend/TF-IDF/`라는 이름과 위치는 폐기한다(모델 코드는 `src/`, API 코드만 `backend/`에 남긴다는 원칙과 배치되므로).
4. **저장소 전환 대비**: 지금은 모든 산출물이 CSV지만 **추후 SQLite(`recommend.db`)로 전환 예정** — `app/utils/data_loader.py`에 이미 있는 `DATA_SOURCE = "csv"/"sqlite"` 토글과 동일한 원칙을, 새로 만드는 backend 서비스의 아티팩트/사전계산 테이블 로딩부에도 적용해 전환 지점을 한 곳으로 모아둔다.
5. **ALS 가중치**: `backend/ALS/als_model.pkl`에 이미 학습된 가중치가 있으므로 이를 그대로 서빙에 활용한다(재학습 아님).

> 용어 정리: "실시간 추론 서비스"는 매 요청마다 모델을 재학습한다는 뜻이 아니라, **오프라인으로 학습된 모델/아티팩트를 FastAPI 프로세스가 메모리에 상주시켜 두고, 요청이 오면 그 자리에서 추천을 계산/조회해 반환**한다는 뜻이다. 학습(배치)과 서빙(실시간 API)을 분리하는 것이 핵심 변경점이며, 지금처럼 Streamlit이 사전 산출 CSV를 직접 읽는 방식과 구분된다.

## 목표 디렉토리 구조

```
src/
  features/
    build_complementary_features.py   # 신규 — bowanjae_pipeline.py의 load_data/preprocess_data를
                                        # 하드코딩 경로 없이 이관 (역할 분리 원칙: 로딩/전처리는 features/)
  modeling/
    als/
      model.py            # backend/ALS/als_model.py 이관 + 버그 수정 + 서빙용 아티팩트 확장
      evaluate.py          # backend/ALS/als_evaluate.py 이관
    complementary_recommender.py   # 이미 존재 — 그대로 재사용 (run_modeling)
    twiddler/
      rerank.py             # 페르소나 가중치 재랭킹 — 인터페이스만 정의, 본 구현은 TODO 스텁
    lightgcn/
      model.py              # 인터페이스만 정의, 본 구현은 TODO 스텁
  evaluation/
    evaluate_complementary.py   # 이미 존재 — 그대로 재사용 (evaluate_model)

configs/
  als/params.yaml          # 신규 — factors/iterations/alpha/regularization/random_state/split_date/cold_threshold/top_n/weighting
  complementary/params.yaml   # 신규 — action_weights, top_n (기존 bowanjae_pipeline.py 상수를 설정으로 분리)
  (lightgcn, twiddler는 스텁 단계이므로 파라미터 파일 보류)

backend/                   # FastAPI 서빙 레이어 (Streamlit이 호출하는 API만, 모델/학습 코드는 두지 않음)
  main.py                   # FastAPI 앱, 시작 시 아티팩트 로딩(lifespan)
  api/
    schemas.py               # PRED_MAIN/PRED_DETAIL과 동일한 응답 스키마(Pydantic)
    routers/
      recommend_main.py       # GET /recommend/main
      recommend_detail.py     # GET /recommend/detail
    services/
      als_service.py          # src.modeling.als 아티팩트 로드 + 추천 조회
      complementary_service.py  # src.modeling.complementary_recommender 산출 테이블 로드 + item_id 조회
      twiddler_service.py      # 스텁: "after" 요청 시 재랭킹 미구현 → before로 폴백 + 플래그 반환
      lightgcn_service.py      # 스텁: 미구현 상태를 명시적으로 응답(빈 리스트 + status="not_implemented")
    # 각 서비스는 "아티팩트/테이블을 어디서 읽어오는가"를 함수 하나로 캡슐화한다.
    # 지금은 그 함수가 CSV/pickle을 읽지만, 추후 recommend.db(SQLite)로 옮길 때
    # 이 함수 내부만 바꾸면 되도록 호출부(router)와 분리해 둔다 — app/utils/data_loader.py의
    # 기존 DATA_SOURCE="csv"/"sqlite" 토글과 동일한 사고방식.

models/ALS/als_model.pkl     # (gitignore) 학습된 pickle 아티팩트 — backend/ALS/als_model.pkl 이관
                              # (model, user_enc, item_enc, user_dec, item_dec; 19,930명×1,197개 상품 실 데이터)

data/
  outputs/
    complementary/detail_cf.csv   # complementary_recommender 산출물 (item_id, rec_item_id, score, rank)
                                    # → 추후 recommend.db 테이블로 전환 예정

app/
  utils/
    api_client.py            # 신규 — BACKEND_API_URL 설정 + requests 래퍼, 타임아웃/에러 처리
    data_loader.py            # 카탈로그성 데이터(products/demo_users/persona_labels)는 CSV/SQLite 유지,
                                # 추천 데이터(load_recommendations/load_detail_recommendations/
                                # get_user_recommendations)는 api_client 호출로 대체
  main.py                    # 호출 시그니처 최대한 유지, 백엔드 다운 시 에러 배너 처리 추가
```

`backend/TF-IDF/`(및 `bowanjae.csv`)와 리포 루트의 `bowanjae_pipeline.py`는 위 구조로 로직이 완전히 이관된 뒤 삭제한다 — 모델 코드가 `src/`에, 산출물이 `data/outputs/`에 자리 잡으므로 더 이상 필요 없다.

## Phase 1 — ALS 실연결 (우선순위 최상)

**이미 학습된 `backend/ALS/als_model.pkl`을 그대로 서빙에 사용한다 — 지금 당장 재학습은 하지 않는다.** 원본 이벤트 데이터(`data/processed/als_events.csv`)가 리포에 없어도 이 아티팩트만으로 Phase 1을 진행할 수 있다.

1. **아티팩트 배치**: `backend/ALS/als_model.pkl`을 `models/ALS/als_model.pkl`로 이동(다른 산출물과 위치 일관성 유지 — `als_model.py`의 `MODEL_DIR = "models/ALS"` 규칙과 일치). `backend/`에는 모델 코드/아티팩트를 두지 않는다는 원칙 유지.
2. **`als_model.py` 버그 수정 및 이관**: `PARAMS_PATH`를 `configs/als/params.yaml`(리포 루트 기준)로 고정, `src/modeling/als/model.py`로 이동. `configs/als/params.yaml` 신규 작성(factors/iterations/alpha/regularization/random_state/split_date/cold_threshold/top_n/weighting) — 이후 **재학습이 필요해질 때**를 대비한 정리이며, 지금 당장 재실행 대상은 아니다.
3. **의존성 추가**: 이 pickle을 로드/추론하려면 `implicit`이 필요하다(현재 환경에 미설치 — `ModuleNotFoundError`로 확인함). `pyproject.toml`/`requirements.txt`에 `implicit`, `fastapi`, `uvicorn` 추가.
4. **현재 아티팩트의 한계와 서빙 시 처리 방식**: pickle에는 `model/user_enc/item_enc/user_dec/item_dec`만 있고 학습 시점 상호작용 행렬·인기도 테이블·heavy/cold 구분이 없다. 초기 버전의 `als_service.py`는:
   - `model.recommend(userid, user_items, N, filter_already_liked_items=False)`처럼 **already-liked 필터링은 비활성화**(빈/0 sparse row를 넘겨 shape만 맞춤)하고,
   - `user_id`가 `user_enc`에 있으면 정상적으로 개인화 추천을 반환하고, **없으면(cold) `status="not_implemented"` + 안내 메시지**로 응답한다(인기도 테이블이 아직 없으므로).
   - 추후 원본 이벤트 데이터를 확보해 `als_model.py`를 재실행하게 되면, `save_outputs()`에 sparse matrix·`popular_items`·`user_type` 맵을 추가로 pickle에 포함하도록 확장해 already-liked 필터링과 cold 폴백을 완성한다(후속 개선 항목, 이번 1차 연결의 필수 조건 아님).
5. **`backend/api/services/als_service.py`**: 앱 기동 시 위 아티팩트를 로드해 메모리에 보관. `get_recommendations(user_id, twiddler_phase) -> list[{item_id, score, rank, user_type}]` 형태의 함수 제공.
6. **`backend/api/routers/recommend_main.py`**: `GET /recommend/main?user_id=&model_type=ALS&twiddler=before|after` 엔드포인트로 위 서비스를 호출해 `PRED_MAIN_RECOMMEND.csv`와 동일한 필드(`user_id, item_id, score, rank, model_type, twiddler, user_type`)를 JSON으로 반환.
7. **앱 연동**: `app/utils/api_client.py`에서 위 엔드포인트를 호출하고, `data_loader.py`의 `get_user_recommendations()`를 API 응답 기반으로 재구현(반환 스키마는 기존과 동일하게 유지해 `main.py` 변경 최소화).

## Phase 2 — 보완재(complementary item, 구 backend/TF-IDF) 연결

이미 존재하는 `src/modeling/complementary_recommender.py::run_modeling`과 `src/evaluation/evaluate_complementary.py::evaluate_model`을 **그대로 재사용**한다(새로 작성하지 않음). 부족한 것은 이 둘을 잇는 데이터 로딩/실행 스크립트와 배치→API 연결 부분뿐이다.

1. `bowanjae_pipeline.py`의 `load_data`/`preprocess_data`(원본 이벤트·세션·주문 로그 → `df_integrated_logs` 생성)를 `src/features/build_complementary_features.py`로 이관하면서 하드코딩된 `DATA_DIR`(`C:\Users\USER\OneDrive\Desktop\dataset`)을 인자/설정으로 분리한다.
2. 이 셋(`build_complementary_features` → `complementary_recommender.run_modeling` → `evaluate_complementary.evaluate_model`)을 순서대로 호출하는 배치 실행 스크립트(예: `src/modeling/run_complementary_pipeline.py` 또는 CLI 진입점)를 추가한다.
3. `run_modeling`이 반환하는 `top_n_recs`(`prod_A, prod_B, score, rank`, id 그대로 보존됨)를 `item_id, rec_item_id, score, rank` 컬럼으로 rename해 `data/outputs/complementary/detail_cf.csv`에 저장(`reports/item_recommendation.md`가 이미 문서화한 변환과 동일).
4. `backend/api/services/complementary_service.py`가 기동 시 이 테이블을 로드하고, `GET /recommend/detail?item_id=&rec_type=cf` 요청에 대해 `item_id` 기준 top-N을 조회해 반환. 로딩 함수를 한 곳에 모아 두어 추후 `recommend.db` SQLite 테이블로 바꿀 때 이 함수만 교체하면 되게 한다.
5. `rec_type="content"`(대체재/진짜 콘텐츠 기반)는 알고리즘 자체가 없으므로 이번 범위에서는 **빈 결과 + `status="not_implemented"`** 로 응답하는 스텁만 만들고, 화면에서는 "준비 중" 안내로 처리.
6. 이관 완료 후 `backend/TF-IDF/`(`bowanjae.csv` 포함)와 리포 루트 `bowanjae_pipeline.py`는 삭제한다.

## Phase 3 — Twiddler (스텁)

- `src/modeling/twiddler/rerank.py`에 `rerank(als_recs: list, persona_label: str) -> list` 시그니처만 정의하고 본문은 `raise NotImplementedError`/pass-through(입력을 그대로 반환).
- `backend/api/services/twiddler_service.py`: `twiddler=after` 요청이 오면 실제 재랭킹 대신 `before`와 동일한 결과에 `"twiddler_status": "not_implemented"` 플래그를 얹어 반환 — `app/main.py`의 Before/After 라디오가 깨지지 않고, UI에서 "재랭킹 준비 중" 캡션을 보여줄 수 있게 함.

## Phase 4 — LightGCN (스텁)

- `src/modeling/lightgcn/model.py`에 `train(...)`/`recommend(user_id, top_n)` 시그니처만 정의, 본문은 TODO.
- `backend/api/services/lightgcn_service.py`가 `GET /recommend/main?model_type=LightGCN`에 대해 빈 리스트 + `status="not_implemented"`를 반환 → `app/main.py`의 LightGCN 컬럼에 "모델 준비 중" 안내를 표시(현재처럼 카드가 비어 에러 나는 상황 방지).
- PyTorch/torch-geometric 등 의존성 추가는 실제 구현 착수 시점으로 보류.

## app 측 변경 요약

- `app/utils/api_client.py`(신규): `BACKEND_API_URL`(환경변수 또는 `.streamlit/secrets.toml`) 기반 `requests` 래퍼. 타임아웃/커넥션 에러 시 `main.py`가 처리할 수 있는 예외로 통일.
- `app/utils/data_loader.py`: `load_products/load_demo_users/load_persona_labels`는 카탈로그·데모 메타데이터이므로 **그대로 CSV 유지**(모델 산출물이 아님). `load_recommendations/load_detail_recommendations/get_user_recommendations`만 API 호출로 교체하되 **반환 DataFrame 스키마는 기존과 동일하게 유지**해 `app/main.py`/`product_card.py` 등 하위 코드 변경을 최소화.
- `app/main.py`: 백엔드 응답에 `status="not_implemented"`가 포함된 경우(LightGCN, Twiddler-after, content 상세) UI에 "준비 중" 안내를 추가하는 부분만 손댐. 나머지 라우팅/레이아웃 로직은 유지.
- 백엔드 연결 실패(서버 미기동 등) 시 현재의 `FileNotFoundError` 처리 대신 API 타임아웃/연결 에러를 잡아 동일한 형태의 에러 메시지를 보여주도록 예외 처리 교체.

## 검증 방법

1. `uvicorn backend.main:app --reload`로 API 기동 후 `curl`/httpie로 `/recommend/main`, `/recommend/detail` 각각 정상 유저·아이템/존재하지 않는 유저·아이템 케이스 확인.
2. `streamlit run app/main.py` 실행 후 브라우저에서: (a) ALS 컬럼이 실제 API 응답으로 카드가 채워지는지, (b) LightGCN/Twiddler-after가 "준비 중" 안내로 정상적으로 표시되는지(에러로 죽지 않는지), (c) 상세 추천 화면에서 cf(보완재) 카드가 뜨는지 확인.
3. `backend/ALS/als_evaluate.py`(이관 후 `src/modeling/als/evaluate.py`)로 HR/Recall/NDCG가 산출되는지 확인해 실제 학습된 모델의 품질을 검증.
4. 백엔드 프로세스를 끈 상태에서 Streamlit을 열어 에러 배너가 사용자 친화적으로 표시되는지 확인(백엔드 다운 시나리오).

## 미해결 전제조건 (진행 전 확인 필요)

- ALS는 `backend/ALS/als_model.pkl`(학습 완료된 아티팩트)을 그대로 활용하므로 더 이상 원본 이벤트 데이터가 없어도 Phase 1 진행에 지장 없음. 다만 already-liked 필터링·cold 유저 폴백까지 완전히 재현하려면 추후 원본 이벤트 데이터(`events.csv, sessions.csv, orders.csv, order_items.csv`류)로 재학습이 필요 — 이건 후속 개선 항목으로 남겨둠.
- 보완재(complementary) 로직에 필요한 원본 로그(`bowanjae_pipeline.py`가 참조하던 `events.csv, sessions.csv, orders.csv, order_items.csv`, 현재 로컬 경로 `C:\Users\USER\OneDrive\Desktop\dataset`)는 여전히 리포 안에 없음 — `data/raw/`로 옮겨야 Phase 2의 배치 재실행(`build_complementary_features` → `run_modeling`)이 가능함. 이미 산출된 `backend/TF-IDF/bowanjae.csv`(상품명 기준)가 있으므로, 원본 로그 확보 전까지는 이 파일을 임시로 item_id 매핑해 사용하는 것도 가능(단, `products.csv`의 상품명이 유일함을 전제로 함 — 중복 상품명이 있으면 매핑이 부정확해질 수 있어 확인 필요).

## 진행 상황

- [x] 계획 수립 및 승인
- [ ] Phase 1 — ALS 실연결
- [ ] Phase 2 — 보완재 연결
- [ ] Phase 3 — Twiddler 스텁
- [ ] Phase 4 — LightGCN 스텁
- [ ] app 측 연동(api_client.py, data_loader.py, main.py)
- [ ] 검증
