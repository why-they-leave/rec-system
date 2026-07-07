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
  modeling/
    als/
      model.py            # backend/ALS/als_model.py 이관 + 버그 수정 + 서빙용 아티팩트 확장
      evaluate.py          # backend/ALS/als_evaluate.py 이관
    complementary/
      model.py              # 기존 src/modeling/complementary_recommender.py 이관 (run_modeling, 재사용/무수정)
    twiddler/
      rerank.py             # 페르소나 가중치 재랭킹 — 인터페이스만 정의, 본 구현은 TODO 스텁
    lightgcn/
      model.py              # 인터페이스만 정의, 본 구현은 TODO 스텁
  evaluation/
    evaluate_complementary.py   # 이미 존재 — 그대로 재사용 (evaluate_model)

configs/
  als/params.yaml          # 신규 — factors/iterations/alpha/regularization/random_state/split_date/cold_threshold/top_n/weighting
  (complementary/lightgcn/twiddler는 하드코딩 재사용/스텁 단계이므로 파라미터 파일 없음)

scripts/
  run_bowanjae_pipeline.py   # 사용자가 직접 작성한 배치 실행 스크립트(기존 파일) — data/processed/df_integrated_logs.csv
                              # + data/raw/products.csv를 읽어 complementary.model.run_modeling →
                              # evaluate_complementary.evaluate_model을 실행하고
                              # data/outputs/complementary/detail_cf.csv에 저장하도록 출력 경로만 수정.
                              # 보완재 배치 파이프라인의 유일한 진입점 — src/ 쪽에 별도 오케스트레이션
                              # 스크립트를 중복으로 두지 않는다.
  convert_bowanjae_to_detail_cf.py   # 일회성 변환 스크립트 — 아래 참고

backend/                   # FastAPI 서빙 레이어 (Streamlit이 호출하는 API만, 모델/학습 코드는 두지 않음)
  main.py                   # FastAPI 앱, 시작 시 아티팩트 로딩(lifespan)
  api/
    schemas.py               # PRED_MAIN/PRED_DETAIL과 동일한 응답 스키마(Pydantic)
    routers/
      recommend_main.py       # GET /recommend/main
      recommend_detail.py     # GET /recommend/detail
    services/
      als_service.py          # src.modeling.als 아티팩트 로드 + 추천 조회
      complementary_service.py  # data/outputs/complementary/detail_cf.csv 로드 + item_id 조회
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
    complementary/detail_cf.csv   # scripts/run_bowanjae_pipeline.py 산출물 (item_id, rec_item_id, score, rank)
                                    # → 추후 recommend.db 테이블로 전환 예정

app/
  utils/
    api_client.py            # 신규 — BACKEND_API_URL 설정 + requests 래퍼, 타임아웃/에러 처리
    data_loader.py            # 카탈로그성 데이터(products/demo_users/persona_labels)는 CSV/SQLite 유지,
                                # 추천 데이터(load_recommendations/load_detail_recommendations/
                                # get_user_recommendations)는 api_client 호출로 대체
  main.py                    # 호출 시그니처 최대한 유지, 백엔드 다운 시 에러 배너 처리 추가
```

`backend/TF-IDF/`(및 `bowanjae.csv`)와 리포 루트의 `bowanjae_pipeline.py`는 위 구조로 로직이 완전히 이관된 뒤 삭제한다 — 모델 코드가 `src/`에, 산출물이 `data/outputs/`에 자리 잡으므로 더 이상 필요 없다. `data/dashboard/PRED_MAIN_RECOMMEND.csv`/`PRED_DETAIL_RECOMMEND.csv`(구 더미 데이터, `scripts/generate_demo_data.py` 산출물)도 앱이 더 이상 읽지 않으므로 삭제했다.

> **주의**: `src/modeling/complementary/`에는 `model.py`(run_modeling)만 두고, 원본 로그 로딩·배치 오케스트레이션은 새로 만들지 않는다 — 그 역할은 사용자가 이미 갖고 있는 `scripts/run_bowanjae_pipeline.py`가 전담한다. 처음에는 `src/features/build_complementary_features.py` + `src/modeling/complementary/run_pipeline.py`를 별도로 만들었으나, `scripts/run_bowanjae_pipeline.py`와 입력 전제(원본 로그 vs 이미 전처리된 `df_integrated_logs.csv`)가 서로 달라 파이프라인이 두 갈래로 나뉘는 중복이 발생해 **삭제하고 사용자의 스크립트 하나로 일원화**했다.

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

이미 존재하는 `src/modeling/complementary/model.py::run_modeling`(구 `complementary_recommender.py`)과 `src/evaluation/evaluate_complementary.py::evaluate_model`을 **그대로 재사용**한다(새로 작성하지 않음). 배치 오케스트레이션은 사용자가 이미 갖고 있던 `scripts/run_bowanjae_pipeline.py`가 담당하므로, 이 파이프라인과 백엔드 API 사이의 **출력 경로만** 맞추면 된다.

1. `scripts/run_bowanjae_pipeline.py`(`data/processed/df_integrated_logs.csv` + `data/raw/products.csv` → `run_modeling` → `evaluate_model` → 저장)의 출력 경로를 `data/dashboard/PRED_DETAIL_RECOMMEND.csv`(옛 설계)에서 `data/outputs/complementary/detail_cf.csv`(백엔드가 읽는 경로)로 수정.
2. `backend/api/services/complementary_service.py`가 기동 시 이 테이블을 로드하고, `GET /recommend/detail?item_id=&rec_type=cf` 요청에 대해 `item_id` 기준 top-N을 조회해 반환. 로딩 함수를 한 곳에 모아 두어 추후 `recommend.db` SQLite 테이블로 바꿀 때 이 함수만 교체하면 되게 한다.
3. `rec_type="content"`(대체재/진짜 콘텐츠 기반)는 알고리즘 자체가 없으므로 이번 범위에서는 **빈 결과 + `status="not_implemented"`** 로 응답하는 스텁만 만들고, 화면에서는 "준비 중" 안내로 처리.
4. 이관 완료 후 `backend/TF-IDF/`(`bowanjae.csv` 포함)와 리포 루트 `bowanjae_pipeline.py`는 삭제한다.
5. `scripts/run_bowanjae_pipeline.py`가 필요로 하는 `data/processed/df_integrated_logs.csv`/`data/raw/products.csv`가 아직 없어 지금 당장 재실행할 수는 없다 — 확보되면 이 스크립트만 실행하면 된다(별도 오케스트레이션 스크립트를 만들지 않음).

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

- ~~ALS 재학습에 필요한 원본 이벤트 데이터가 없음~~ → **해결(2026-07-06)**: `data/raw/`에 원본 로그(`events.csv, sessions.csv, orders.csv, order_items.csv, products.csv, customers.csv, reviews.csv`)가 채워짐. `scripts/build_processed_data.py`로 `data/processed/als_events.csv` 생성 완료.
- ~~보완재 배치 재실행에 필요한 `df_integrated_logs.csv`/`products.csv`가 없음~~ → **해결(2026-07-06)**: 같은 스크립트로 `data/processed/df_integrated_logs.csv` 생성 완료, `data/raw/products.csv`도 이미 존재(내용은 `data/dashboard/products.csv`와 동일 확인).
- 다만 아직 **재학습/재생성 실행 자체는 보류 상태**(사용자 요청으로 데이터 생성까지만 진행) — `models/ALS/als_model.pkl`과 `data/outputs/complementary/detail_cf.csv`는 여전히 이전 산출물을 그대로 서빙 중. 아래 "생성된 학습 데이터" 절 참고.

## 진행 상황 (2026-07-06 기준)

- [x] 계획 수립 및 승인
- [x] Phase 1 — ALS 실연결: `models/ALS/als_model.pkl`로 아티팩트 이관, `src/modeling/als/{model,evaluate}.py`로 코드 이관(경로 버그 수정 + matrix/popular_items/user_type_map 저장 확장), `backend/api/services/als_service.py` 작성. 실제 pickle 로드해 유저 1번 추천 확인 완료(19,930명 × 1,197개 상품).
- [x] Phase 2 — 보완재 연결: `scripts/convert_bowanjae_to_detail_cf.py`로 기존 `backend/TF-IDF/bowanjae.csv`(상품명 기준)를 `data/outputs/complementary/detail_cf.csv`(item_id 기준)로 변환해 즉시 서빙 가능하게 함. `backend/TF-IDF/`, 루트 `bowanjae_pipeline.py` 삭제 완료.
- [x] 폴더 정리(1차): 기존 top-level `src/modeling/complementary_recommender.py`를 다른 알고리즘(als/twiddler/lightgcn)과 동일하게 `src/modeling/complementary/model.py`로 옮김.
- [x] 폴더 정리(2차, 2026-07-06): 처음엔 원본 로그 전처리용 `src/features/build_complementary_features.py`와 배치 오케스트레이션용 `src/modeling/complementary/run_pipeline.py`를 새로 만들었으나, 사용자가 이미 `scripts/run_bowanjae_pipeline.py`(전처리 완료 로그 `df_integrated_logs.csv` 기준)를 별도로 갖고 있어 파이프라인이 두 갈래로 나뉘는 중복이 발생 — **둘 다 삭제**하고, `scripts/run_bowanjae_pipeline.py`의 출력 경로만 `data/dashboard/PRED_DETAIL_RECOMMEND.csv` → `data/outputs/complementary/detail_cf.csv`로 수정해 백엔드와 일원화함.
- [x] Phase 3 — Twiddler 스텁: `src/modeling/twiddler/rerank.py`(NotImplementedError) + `backend/api/services/twiddler_service.py`(after 요청 시 before로 폴백 + not_implemented 플래그).
- [x] Phase 4 — LightGCN 스텁: `src/modeling/lightgcn/model.py`(NotImplementedError) + `backend/api/services/lightgcn_service.py`(not_implemented 고정 응답).
- [x] app 측 연동: `app/utils/api_client.py` 신규 작성, `app/utils/data_loader.py`/`app/main.py`를 API 호출 기반으로 재작성(카탈로그 데이터는 CSV 유지).
- [x] 검증: `uv`로 `.venv` 생성 후 `uvicorn backend.main:app`/`streamlit run app/main.py` 실제 기동 확인. `streamlit.testing.v1.AppTest`로 메인 추천(ALS 실데이터 렌더링, LightGCN/Twiddler-after not_implemented 안내 정상 표시)과 상세 추천(보완재 카드 렌더링) 페이지 모두 예외 없이 동작 확인.
- [x] 더 이상 앱이 읽지 않는 `data/dashboard/PRED_MAIN_RECOMMEND.csv`/`PRED_DETAIL_RECOMMEND.csv`(구 더미 데이터) 삭제.
- [x] 원본 로그 확보(2026-07-06): `data/raw/`에 실제 이벤트/세션/주문 로그(2020-01-01~2025-11-01)가 채워짐.
  - 처음엔 `scripts/build_processed_data.py`로 ALS 마트(`als_events.csv`)까지 직접 만들었으나(2025-08-01 이전만 필터링, event_type별 가중치도 임의 추정), **원래 별도의 데이터마트 생성 코드가 있었고 그 결과물을 사용자가 `data/interim/als_events.csv`로 직접 제공** — 820,336행, 유저 19,945명 × 상품 1,197개, `score`는 `page_view=1, add_to_cart=3, checkout=4, purchase=5`(checkout도 포함되어 있었음, 제 추측과 달랐음), 기간은 전체(~2025-11-01, `split_date=2025-08-01` 기준으로 train/test를 나누는 방식이었음 — 마트 자체를 자르는 게 아니었음). **제가 임의로 만든 버전은 삭제**하고 `src/modeling/als/model.py`의 `PATHS["full"]["mart"]`를 `data/interim/als_events.csv`로 수정해 실제 마트를 가리키도록 함.
  - `configs/als/params.yaml`은 사용자가 원본 설정값(`split_date=2025-08-01, cold_threshold=10, top_n=100`, als 하이퍼파라미터 `factors=16, iterations=20, alpha=0.5, regularization=10.0` — `notebooks/20260703_ML_als_hyperparameter_tuning.ipynb` 그리드서치 결과)으로 직접 교체 — 경로 주석만 이 리포 구조에 맞게 수정.
  - `scripts/build_processed_data.py`는 보완재용 `data/processed/df_integrated_logs.csv`만 생성하도록 범위를 좁힘(ALS 마트 생성 로직 제거). 날짜 필터도 제거하고 전체 기간(741,632행)으로 재생성 — ALS 마트가 전체 기간을 쓰는 것과 일관되게 맞춤.
  - 전처리 로직(특히 `purchase` 이벤트의 product_id 복원)은 기존 `bowanjae_pipeline.py`의 방식을 그대로 재현. `checkout`은 주문과 timestamp가 정확히 일치하지 않아(장바구니 이탈 등) 보완재 파이프라인에서는 이번에도 제외 — ALS 마트는 checkout을 포함하고 있어 두 파이프라인의 처리 방식이 다르다(각각의 원 설계를 따른 것).
- [x] **ALS 재학습 실행(2026-07-06)**: `python -m src.modeling.als.model --dataset full` 실행 완료.
  - Train 783,477개 / Test 36,859개(`split_date=2025-08-01` 기준 정상 분할). Heavy 19,292명 / Cold 638명(`cold_threshold=10`).
  - `models/ALS/als_model.pkl` 갱신(재학습 전 버전은 `models/ALS/als_model.pkl.bak`로 백업). 이제 `matrix`/`popular_items`/`user_type_map`이 모두 포함되어 already-liked 필터링과 cold 폴백이 정상 동작 확인(예: 실제 cold 유저 13/88/89번은 `status=ok, user_type=cold`로 인기도 기반 추천 반환).
  - **주의**: `data/dashboard/demo_users.csv`의 10명은 전부 실제로는 heavy임(실측 train 이벤트 수 18~62개, `cold_threshold=10` 훨씬 위) — 이 CSV의 `user_type=cold` 라벨(8,9,10번)은 `scripts/generate_demo_data.py`가 임의로 붙인 것이라 실제 데이터와 무관했음. 지금 데모 UI에서 고를 수 있는 10명 중에는 진짜 cold 유저가 없음 — cold 화면을 보여주려면 `demo_users.csv`에 실제 cold 유저 ID(예: 13, 88, 89 등, 총 638명 중 일부)를 추가해야 함.
  - `python -m src.modeling.als.evaluate --dataset full` 결과: HR@5=1.71%, HR@10=3.69%, HR@20=6.08% (Recall/NDCG는 `data/outputs/ALS/eval_results.csv` 참고) — 보완재 모델의 Hit Rate(1.75%)와 비슷한 자릿수, 두 모델 다 이 데이터셋에서는 예측이 어려운 편.
- [x] **보완재 배치 재실행(2026-07-06)**: `python scripts/run_bowanjae_pipeline.py` 실행 완료. `data/outputs/complementary/detail_cf.csv` 갱신(5,985행) — 전체 1,197개 상품 **100% 커버**(이전 임시본은 1,026개/171개 누락). Hit Rate 1.75%로 기존 `reports/item_recommendation.md` 문서화 수치와 일치.
- [x] 두 재학습 산출물로 `streamlit.testing.v1.AppTest` 재검증 — 예외 없음.
- [x] **데모 유저에 실제 cold 유저 3명 추가(2026-07-07)**: `demo_users.csv`에 실제 train 이벤트 수가 `cold_threshold=10` 미만인 유저 13/88/89(각 7/6/8건)를 추가하고 `als_service`가 정상적으로 `cold`로 분류하는지 curl/AppTest로 검증. 검증 중 오래전 세션에서 띄워둔 채 남아있던(재학습 전 구모델을 메모리에 물고 있던) 좀비 uvicorn 프로세스를 발견해 종료 후 재기동 — 이후 정상 확인.

## 페르소나 확정 반영 (2026-07-07)

`data/processed/segment_personas_train_only.json`(세그먼트 정의)과 `data/processed/customer_segments_labeled_train_only.csv`(customer_id→segment_id 매핑, 사용자 업로드)로 실제 KMeans 6-세그먼트 페르소나가 확정됨. 둘 다 형제 리포 `retail-clickstream-analysis`(`src/features/build_train_only_segments.py`, ALS와 동일한 train cutoff `2025-08-01` 기준, `random_state=42`)의 산출물이며, 원본 고객 데이터(`customers.csv`)가 두 리포에서 동일(md5 일치)해 `customer_id`가 그대로 호환된다.

기존 `demo_users.csv`/`persona_labels.csv`는 `scripts/generate_demo_data.py`가 `np.random.choice`로 지어낸 가상의 5종 한국어 라벨(가성비추구형/트렌드세터/브랜드충성형/충동구매형/비교분석형)이었음 — 이번에 완전히 교체:

- [x] `scripts/generate_demo_data.py`: 가짜 페르소나 생성 로직 제거. `customer_segments_labeled_train_only.csv` + 재학습된 `models/ALS/als_model.pkl`의 `user_type_map`(heavy/cold) + `data/interim/als_events.csv`(실제 train 로그 수 계산)을 조합해, 세그먼트당 heavy 3명·cold 최대 2명(있는 만큼)을 `random_state=42`로 샘플링 → `demo_users.csv`(28행, 실제 6개 세그먼트 전부 포함). `persona_labels.csv`도 동일 소스에서 100명 샘플로 재생성. 더 이상 쓰이지 않는 `PRED_MAIN_RECOMMEND.csv`/`PRED_DETAIL_RECOMMEND.csv` 생성 로직(섹션 3·4)도 함께 제거.
- [x] `app/components/user_selector.py`: 유저 단일 드롭다운 → **페르소나 선택 → (필터링된) 유저 선택** 2단계 드롭다운으로 변경. 페르소나를 바꾸면 `st.session_state["selected_user"]`를 명시적으로 초기화해, 이전 페르소나의 유저 id가 새 옵션 목록에 없어 Streamlit이 에러를 내는 것을 방지.
- [x] `app/main.py`: 사이드바의 "페르소나 확정 후 추가 예정" TODO 플레이스홀더를 실제 `PERSONA_DESC` 딕셔너리(6개 세그먼트 설명, 세그먼트명은 영문 원문 유지·설명만 한국어로 번역)로 교체.
- [x] `streamlit.testing.v1.AppTest`로 검증: 페르소나 전환 시 유저 목록 필터링·세션 상태 리셋·설명 패널 갱신이 예외 없이 동작, 실제 백엔드 API(HEAVY/COLD 분류) 응답까지 정상 확인.

## 버그 수정 — cold 유저가 인기도 기반이 아니라 유저마다 다른 개인화 추천을 받던 문제 (2026-07-07)

사용자가 "cold 유저는 인기도 기반이면 누굴 골라도 결과가 같아야 하는 거 아니냐"고 지적 — 확인해보니 실제로 유저마다 다른 결과가 나오는 버그였음.

**원인**: `backend/api/services/als_service.py`의 분기 조건이 `if user_id not in user_enc`였음 — "학습 데이터에 아예 없는 유저"만 인기도(`popular_items`) 폴백을 타도록 되어 있었다. 그런데 `cold_threshold=10` 기준 cold 유저(train 이벤트 1~9건)도 이벤트가 1건 이상이면 `user_enc`에는 포함된다(`model.py`의 `build_sparse_matrix`가 heavy/cold 구분 없이 train에 등장한 모든 유저로 행렬을 만들기 때문). 그래서 cold 유저도 `else` 분기로 빠져 본인의 희소한 상호작용 행(row)으로 `model.recommend()`를 호출해 **개인화 추천**을 받고 있었다 — UI 캡션("Cold 유저: 인기도 기반 추천")·배치 파이프라인(`model.py::generate_cold_recommendations`, 원래 인기도 순 동일 리스트) 설계와 불일치.

**수정**: `user_id not in user_enc`(진짜 미지의 유저) **또는** `user_type_map.get(user_id) == "cold"`(이벤트 수 기준 cold 유저) 둘 중 하나면 인기도 폴백을 타도록 조건 수정. 폴백 시에도 해당 유저가 이미 본 아이템은 제외(matrix row 기준, `model.py`의 `user_seen` 필터링과 동일 원칙). curl/직접 호출로 cold 유저 6명(12605/12951/13742/16200/16683/18472) 모두 동일한 인기도 리스트를 반환하는지, heavy 유저는 여전히 서로 다른 개인화 리스트를 받는지 검증 완료.

### 남은 후속 작업 (2026-07-07 시점, Twiddler 구현 이후 갱신)
- LightGCN 실제 학습/추론 구현 — 여전히 미착수(스텁 유지).
- `detail_cf.csv` 후보 풀 확대(`src/modeling/complementary/model.py:69`의 `head(5)` → 예: `head(20)`) + `scripts/run_bowanjae_pipeline.py` 재실행 — 아래 Twiddler Rule 2/3이 상세탭에서 재배치할 여지를 넓히기 위함. 재학습성 배치 작업이라 사용자 승인 후 진행.
- `configs/complementary/params.yaml`은 만들지 않음 — `src/modeling/complementary/model.py`(구 `complementary_recommender.py`)를 원본 그대로 재사용하기로 해 하드코딩된 action_weights/top_n을 그대로 두었고(사용자 지시: 파일을 그대로 활용), 실제로 쓰이지 않는 설정 파일을 미리 만들지 않았음.
- CSV → SQLite(`recommend.db`) 전환 시 `app/utils/data_loader.py`의 카탈로그 로더와 `backend/api/services/*`의 테이블 로딩 함수만 교체하면 되도록 설계해 둠. `backend/api/services/exposure_service.py`(Twiddler Rule 2 노출 이력)도 프로세스 메모리 상태라 이 전환 시 함께 옮길 대상.

## Twiddler 실제 구현 — 페르소나 3-rule 재랭킹, ALS·보완재 상세탭 공용화 (2026-07-07)

사용자가 "현재 ALS에만 고려 중인 Twiddler를 상세탭 보완재 추천에도 같은 로직으로 적용할 수 있는가"를 질문 — 답은 "가능"이었고, 이를 실제로 구현했다. 재랭킹 파이프라인은 사용자가 제시한 구조(원본 점수 → Rule 1 페르소나 가중치 곱 → Rule 2 노출 이력 패널티 → Rule 3 저노출 상품 최소 노출 보너스 → 재정렬 → top-K)를 그대로 채택.

**핵심 설계**:
- `src/modeling/twiddler/rerank.py`: I/O 없는 순수 함수 파이프라인(`apply_persona_weight`/`apply_exposure_penalty`/`apply_new_item_bonus` + `rerank()`). `id_key` 파라미터로 ALS(`item_id`)와 보완재(`rec_item_id`) 양쪽에 동일 함수를 재사용.
- Rule 1(페르소나 카테고리 가중치)은 자유 텍스트 evidence를 파싱하지 않고 `data/processed/customer_segments_labeled_train_only.csv`의 구조화 컬럼(`top_view_category`/`top_purchase_category`/`view_purchase_category_match`/`dominant_purchase_category_ratio`)에서 직접 재계산 — `backend/api/services/persona_service.py`(신규)가 담당.
- 신규 서비스 3개: `persona_service.py`(user_id→persona_label + 세그먼트 affinity/alpha), `catalog_service.py`(item_id→category, 기존엔 백엔드 어디에도 카테고리 조인이 없었음), `exposure_service.py`(Rule 2용 프로세스 메모리 노출 이력, DB 없음이 알려진 한계).
- `complementary_service.py`에 `get_low_exposure_items()` 추가(Rule 3용, `detail_cf.csv`의 `rec_item_id` 등장 빈도 하위 10%).
- `twiddler_service.apply_twiddler(items, phase, user_id, id_key, context, top_k)`를 `recommend_main.py`(`context="main"`)와 `recommend_detail.py`(`context="detail"`, `user_id`/`twiddler` 쿼리 파라미터 신규 추가) 양쪽이 동일하게 호출.
- `twiddler=after`일 때만 라우터가 `POOL_MULTIPLIER=3`배 넓은 후보 풀을 서비스에서 가져와 `rerank()`가 최종 top_k로 절단 — Rule 2/3이 화면 밖 후보를 끌어올릴 여지를 줌.
- 앱 쪽: `app/main.py` 상세탭에 ALS 컬럼과 동일한 Before/After 라디오 + `_make_rank_note` 재사용(순위 변화 배지)을 추가, `api_client.py`/`data_loader.py`에 `user_id`/`twiddler` 파라미터 threading.

**검증**: curl로 `/recommend/main`·`/recommend/detail` 양쪽 before/after 재정렬 확인(페르소나 top 카테고리 상품이 실제로 순위 상승하는 것을 원본 점수/순위와 대조해 확인), 동일 요청 반복 시 Rule 2로 순위가 점진적으로 바뀌는 것 확인, 미지의 user_id는 `not_implemented`+before 폴백으로 에러 없이 처리됨을 확인. `streamlit.testing.v1.AppTest`로 heavy/cold 유저 각각 메인·상세 탭 전환과 Before/After 토글까지 예외 없이 동작 확인.

**알려진 한계**: `detail_cf.csv`가 상품당 후보 5개뿐이라 상세탭 Rule 2/3의 체감 효과가 제한적(위 후속 작업 참고). "신상품"은 실제 출시일이 없어 저노출 proxy로 근사.

### UI 순위 배지 정합성 재검증 (2026-07-07)

`_make_rank_note`(순위 변화 배지, 예: "6→4위 ▲ +2")가 화면에 표시하는 값이 실제 backend 응답과 어긋나지 않는지 별도로 재검증했다. `requests.get`을 몽키패치해 `streamlit.testing.v1.AppTest` 실행 중 앱이 실제로 호출한 `/recommend/main`·`/recommend/detail` 요청·응답 JSON을 원본 그대로 가로채 기록한 뒤, 그 JSON에서 직접 계산한 before/after 순위 차이와 화면에 렌더링된 caption 텍스트를 1:1로 대조하는 방식(별도 curl 호출과 비교하는 방식은 Rule 2 노출 이력 때문에 호출 시점이 다르면 상태가 달라져 부정확할 수 있어 배제).

- ALS 메인 탭(user_id=259, "Frequent Browsers with Occasional Purchases"): before 10개 아이템 전부(`item_id` 215/958/154/772/891/937/1101/1012/922/1142)에 대해 계산한 기대 배지("1→2위 ▼ -1" 등)가 Before/After 화면 caption과 카드 순서까지 정확히 일치.
- 상세탭 보완재(item_id=958): before/after 응답 5개 아이템(`rec_item_id` 1083/420/612/1132/1160) 전부 배지·순서·표시 점수(반올림 포함)가 일치, tie-break로 밀린 1160("4→5위 ▼ -1")까지 정확히 반영됨.

두 탭 모두 10/10, 5/5 항목이 실제 데이터와 어긋남 없이 일치 — UI 순위 배지가 실제 재랭킹 결과를 그대로 반영하고 있음을 확인.

### 버그 수정 — 보완재 상세탭 순위 배지가 동점(tie) 상품에서 허위로 어긋나던 문제 (2026-07-07)

사용자가 item_id 958("Hardcover LightGreen 573") 상세탭에서 rec_item_id 1160("Board Game FireBrick 845")이 "★ 4→5위 ▼ -1"로 표시되는데 1~4위는 전부 "변화없음"이라 앞뒤가 안 맞는다고 지적.

**원인**: `data/outputs/complementary/detail_cf.csv`의 `rank`는 배치 파이프라인(`model.py`)이 `rank(method="min")`으로 계산해 동점 상품이 같은 값을 공유한다(예: 1,2,3,4,4 — "5위"가 아예 존재하지 않음). `complementary_service.get_recommendations()`는 이 원본 rank를 그대로 반환했는데, Twiddler "after" 결과는 `rerank()`가 항상 1..N 밀집 순위로 재부여한다(`als_service`가 이미 이렇게 동작하는 것과 동일 관례). 그 결과 "before"는 gap 있는 순위(1,2,3,4,4), "after"는 밀집 순위(1,2,3,4,5)를 비교하게 돼, 동점으로 밀린 쪽이 실제로는 "5위 항목이 원래 없었는데 생겨난 것처럼" 보이는 허위 변화 배지가 나왔다. 실제로 전체 1,197개 상품 중 **941개(79%)**가 동점 순위를 갖고 있어 드문 예외가 아니라 상세탭 대부분에서 나타날 수 있는 문제였다.

**수정**: `complementary_service.py::get_recommendations()`에서 `rec_item_id`를 타이브레이커로 재현 가능하게 정렬한 뒤, 원본 `rank` 컬럼 대신 `enumerate()`로 밀집 순위를 재부여하도록 변경. item 958로 재검증(before/after 모두 rec_item_id 1132/1160이 정확히 rank 4/5로 분리돼 배지가 "rank 5 → 유지"로 정상 표시됨) 및 `streamlit.testing.v1.AppTest`로 최종 확인.

### 파일 정리 — 안 쓰는 persona_labels.csv, 템플릿 잔재 configs (2026-07-07)

`app/main.py`/`backend/`(`persona_service.py`가 세그먼트 정보를 `customer_segments_labeled_train_only.csv`에서 직접 읽음) 어디에서도 `load_persona_labels()`/`persona_labels.csv`를 호출하지 않는 죽은 코드였음을 확인 후 삭제:
- `data/dashboard/persona_labels.csv` 삭제, `app/utils/data_loader.py::load_persona_labels()` 제거.
- `scripts/generate_demo_data.py`에서 `persona_labels.csv` 생성 로직/상수(`N_PERSONA_LABELS_SAMPLE`) 제거 — 이제 `demo_users.csv`만 생성.
- `configs/base.yaml`/`dev.yaml`/`prod.yaml` 삭제 — `da-template`(범용 데이터분석 템플릿) 잔재로, 이 추천 파이프라인 코드(`src/modeling/als`, `src/modeling/twiddler`, `backend/*`) 어디서도 참조하지 않음. 실사용 설정은 `configs/als/params.yaml`이 유일.
- 전체 `*.py` grep으로 참조 없음 확인, `python scripts/generate_demo_data.py` 재실행 + `streamlit.testing.v1.AppTest`로 앱이 예외 없이 동작하는지 재검증 완료.
- **범위 밖으로 남겨둔 것**: `README.md`가 `configs/base.yaml` 등 이번에 지운 파일뿐 아니라 `src/recommender/`, `download_data.py` 등 현재 코드베이스와 무관한 훨씬 오래된 구조를 통째로 설명하고 있어(전체가 이미 stale) 이번 정리 범위에서 제외 — 문서 전체를 다시 쓰는 별도 작업이 필요함.

## 결정 변경 — 콘텐츠 기반(대체재) 모델 폐기, 보완재만 제공 (2026-07-07)

Phase 2에서 `rec_type="content"`(대체재)를 "알고리즘 없음 → not_implemented 스텁 + 화면에 준비중 안내"로 남겨뒀는데(위 Phase 2 항목 3 참고), 실제로는 화면 쪽 "준비중" 안내조차 연결된 적이 없어(그때 `app/main.py`가 애초에 `rec_type="content"`를 요청한 적이 없음) 사용자 입장에서는 있는지도 알 수 없는 죽은 계약이었다. 대체재 모델(상품 속성/텍스트 유사도 기반)은 애초에 설계된 적 자체가 없고 노트북의 죽은 `TfidfVectorizer`/`cosine_similarity` import만 남아있던 상태 — 사용자가 이 기능 없이 보완재만 제공하기로 결정, 관련 잔재를 전부 제거:

- `backend/api/routers/recommend_detail.py`: `rec_type` 쿼리 파라미터·`content` 분기 제거, 무조건 `complementary_service` 호출.
- `backend/api/schemas.py::DetailRecommendItem`: `rec_type` 필드 제거.
- `app/utils/api_client.py`·`app/utils/data_loader.py::get_detail_recommendations()`: `rec_type` 파라미터 제거, `_DETAIL_REC_COLUMNS`에서도 제거.
- `app/main.py`: 호출부에서 `"cf"` 인자 제거.
- `scripts/run_bowanjae_pipeline.py`·`scripts/convert_bowanjae_to_detail_cf.py`: `detail_cf.csv`에 항상 `"cf"` 고정값으로 쓰던 `rec_type` 컬럼 조립 제거(어차피 `complementary_service.py`는 이 컬럼을 읽은 적이 없었음 — 응답의 `rec_type` 라벨은 라우터가 쿼리 파라미터 값을 그대로 되돌려주던 것뿐이었다). 기존에 이미 생성된 `data/outputs/complementary/detail_cf.csv` 파일 자체는 재생성하지 않음(남아있는 컬럼은 무해하게 무시됨).

### Twiddler 하이퍼파라미터 YAML 설정화 (2026-07-07)

`src/modeling/twiddler/rerank.py`의 `BASE_ALPHA`/`MULTIPLIER_FLOOR`/`EXPOSURE_DECAY`/`POOL_MULTIPLIER`/`LOW_EXPOSURE_PERCENTILE`/`RESERVED_SLOTS`/`NUM_CATEGORIES`가 코드에 하드코딩돼 있어 값 변경 시 코드 수정이 필요했던 것을, `configs/als/params.yaml`과 동일한 패턴으로 분리:

- `configs/twiddler/params.yaml` 신규 작성 — 위 7개 값을 그대로 옮김(값 자체는 변경 없음, 소스만 이동).
- `src/modeling/twiddler/rerank.py`: 모듈 상단에서 `_load_params()`로 YAML을 1회 읽어 동일한 이름의 모듈 상수로 재노출 — `persona_service.py`/`exposure_service.py`/`recommend_main.py`/`recommend_detail.py`의 `from src.modeling.twiddler.rerank import ...` 구문은 전혀 변경할 필요 없었음(같은 이름의 모듈 속성이라 import 인터페이스가 그대로 유지됨).
- 겸사겸사 `backend/api/services/complementary_service.py::get_low_exposure_items()`의 기본값 `threshold_percentile: float = 0.10`이 `LOW_EXPOSURE_PERCENTILE`과 같은 값을 별도로 하드코딩하고 있던 중복을 발견해 `LOW_EXPOSURE_PERCENTILE`을 import해서 참조하도록 정리(두 값이 따로 바뀌면 어긋날 수 있는 잠재 버그 제거).
- `app/`(Streamlit) 쪽은 이 상수들을 전혀 참조하지 않아(grep으로 확인) 변경 없음.
- 검증: 모든 상수가 YAML에서 로드한 값과 원래 하드코딩 값이 동일함을 직접 확인, curl로 `twiddler=after` 재랭킹이 기존과 동일하게 동작하는지, `streamlit.testing.v1.AppTest`로 앱이 예외 없이 도는지 확인.

### CSV → SQLite 마이그레이션(카탈로그 데이터) + GDrive 데이터 배포 (2026-07-07)

사용자가 "SQLite로 바꾸면 이 repo를 pull 받는 누구나 데이터를 바로 쓸 수 있냐"고 질문 — 답은 "아니다"였다. `.gitignore`가 `data/raw/`, `data/interim/`, `data/processed/`, `data/dashboard/*`, `models/`, `*.pkl`을 포맷과 무관하게 전부 제외하므로, SQLite로 바꿔도 그 파일이 이 경로 안에 있으면 여전히 git에 안 올라간다. 이후 두 가지를 분리해서 진행:

**1) SQLite 마이그레이션 — 범위는 앱 카탈로그 데이터(products/demo_users)로 한정**
- `als_model.pkl`(학습된 모델 객체+희소행렬)은 관계형 테이블로 옮기는 게 무의미해 제외 — pickle 유지가 맞다는 데 사용자도 동의.
- `als_events.csv`/`customer_segments_labeled_train_only.csv`/`detail_cf.csv`(대용량 파이프라인 산출물, `backend/api/services/*`가 직접 읽음)도 이번 마이그레이션 범위 밖 — `app/utils/data_loader.py`의 `DATA_SOURCE` 토글은 애초에 이 파이프라인 산출물이 아니라 소규모 카탈로그 데이터용으로 설계된 것이었음.
- `scripts/generate_demo_data.py`에 3단계 추가: `products`/`demo_users` 두 테이블을 `data/dashboard/recommend.db`에 `to_sql(if_exists="replace")`로 저장(CSV도 그대로 함께 생성 — 사람이 직접 열어보는 용도).
- `app/utils/data_loader.py`: `DATA_SOURCE` 기본값을 `"csv"→"sqlite"`로 변경, `load_demo_users()`에 없던 sqlite 분기 추가(`load_products()`엔 이미 있었음). `streamlit.testing.v1.AppTest`로 sqlite 경로에서 페르소나/유저 목록·상품 카드 렌더링까지 정상 확인.

**2) GDrive 데이터 배포 — 앱을 바로 구동하는 데 필요한 최소 파일만 번들링**
사용자가 GDrive에 직접 업로드하기로 함(Claude는 외부 클라우드에 업로드 권한이 없음). 필요한 파일 6개(`data/dashboard/{recommend.db,products.csv,demo_users.csv}`, `data/processed/customer_segments_labeled_train_only.csv`, `data/outputs/complementary/detail_cf.csv`, `models/ALS/als_model.pkl`)를 상대경로 그대로 압축해 `rec-system-data-required.zip`(리포 바로 바깥, ~3.9MB — 원본 총합 ~35MB에서 CSV/pickle 압축률 덕분에 크게 줄어듦)으로 만들어 `C:\Users\kmj\Desktop\why-they-leave\`에 저장해 둠. `scripts/download_data.py`(신규, `gdown` 사용 — `pyproject.toml`엔 이미 있던 의존성, `requirements.txt`에도 추가)가 GDrive 파일 ID로 이 zip을 받아 리포 루트에 압축 해제하도록 작성, 로컬 zip으로 압축 해제 경로가 정확히 맞는지 검증 완료.
- **범위 밖(선택)**: 재학습/재현까지 필요하면 `data/raw/`(52M) + `data/interim/als_events.csv`(34M) + `data/processed/df_integrated_logs.csv`(39M)도 별도 배포가 필요 — 이번엔 "바로 실행"만 목표라 포함하지 않음.

**완료(2026-07-07)**: 사용자가 GDrive에 업로드 후 파일 ID를 전달 — `scripts/download_data.py`를 코드에 ID를 하드코딩하지 않고 `REC_SYSTEM_DATA_FILE_ID` 환경변수로만 받도록 정리(모듈 상수 `GDRIVE_FILE_ID` 제거, `os.environ.get(GDRIVE_FILE_ID_ENV)`로 필수 체크). 격리된 임시 디렉터리에 먼저 내려받아 zip 내용(6개 파일, 크기까지) 원본과 1:1 일치 확인 후, 실제로 리포 루트에 대고 `REC_SYSTEM_DATA_FILE_ID=<FILE_ID> python scripts/download_data.py` 실행 → 6개 파일 전부 정상 압축 해제, 임시 zip 자동 삭제 확인. `streamlit.testing.v1.AppTest`로 재다운로드된 파일 기준 앱이 예외 없이 동작하는 것까지 재검증 완료 — GDrive 배포 파이프라인이 실제로 동작함을 end-to-end로 확인. 실제 파일 ID는 비공개 채널(팀 노션/채널)로 공유.

**부가 검증(2026-07-07)**: `data/`, `models/` 폴더 전체를 임시로 치워 완전히 빈 상태를 만든 뒤 `download_data.py`만으로 사이드바(페르소나/유저 선택)·메인 추천·상세(보완재) 화면까지 전부 정상 동작하는지 확인(원본 복원 완료, git 상태 변화 없음) — "이 6개 파일만 있으면 재현되는가"에 대한 직접 검증.

### 문서 정리 — Bump Chart·콘텐츠 기반(대체재) 제거 결정 반영 (2026-07-08)

`reports/STREAMLIT_UI_DESIGN.md`(원래 설계 명세서)가 실제로 만들지 않기로 결정한 두 기능을 여전히 "예정"인 것처럼 서술하고 있어 업데이트:
- **Bump Chart(`components/metric_chart.py`, `pages/2_twiddler_compare.py`)**: 제거 결정. 실제로는 카드 우상단 ▲/▼ 배지(`app/utils/rank_delta.py::get_rank_delta`)로만 순위 변화 표시.
- **콘텐츠 기반(대체재) 추천(`rec_type='content'`)**: 제거 결정. 상세 화면엔 보완재만 남음(이미 `backend/`에서 `rec_type` 삭제 완료 — 위 항목들 참고).

문서 상단에 결정 사항 요약을 추가하고, 관련 서술 10곳에 `[제거 결정]` 인라인 표시를 달아 원래 설계 의도는 기록으로 남기되 실제 구현과 다르다는 점이 명확히 보이도록 함(문서 전체를 새로 쓰지는 않음 — `pages/` 멀티페이지 구조 등 이 두 결정과 무관한 다른 stale한 부분은 이번 범위 밖).

### LightGCN 이분/삼분 그래프 비교 — UI·백엔드 배관만 선반영 (2026-07-08)

사용자가 "LightGCN도 페르소나를 넣은 삼분그래프와 안 넣은 이분그래프 학습 결과를 각각 카드로 보여주려 한다"고 요청 — 실제 학습/추론 로직은 채우지 않고(als_service처럼 완성하지 않고), **지금의 LightGCN 스텁과 동일한 수준**으로 두 변형을 구분할 수 있는 배관만 미리 준비함.

- `src/modeling/lightgcn/model.py`: `train()`/`recommend()`에 `graph_type: "bipartite" | "tripartite"` 파라미터 추가(둘 다 여전히 `NotImplementedError`). `GRAPH_TYPES` 상수로 유효값 노출.
- `backend/api/services/lightgcn_service.py`: `get_recommendations(user_id, top_n, graph_type)`로 확장, `graph_type`별로 다른 `not_implemented` 메시지 반환("이분그래프 · 페르소나 미포함" / "삼분그래프 · 페르소나 포함").
- `backend/api/routers/recommend_main.py`: `/recommend/main`에 `graph_type` 쿼리 파라미터 추가(`model_type="LightGCN"`일 때만 의미 있음, 기본값 `"tripartite"`). 응답의 `model_type` 필드를 `"LightGCN-bipartite"`/`"LightGCN-tripartite"`로 반환해 어떤 변형인지 구분(ALS 쪽은 기존과 동일하게 `model_type` 그대로).
- `app/utils/api_client.py`/`app/utils/data_loader.py::get_main_recommendations()`: `graph_type` 파라미터 threading(ALS 호출 경로는 영향 없음).
- `app/main.py`: 처음엔 LightGCN 컬럼 안에 "🕸️ 삼분그래프" / "🔗 이분그래프" 두 섹션을 위아래로 쌓았으나, 사용자 요청으로 ALS의 Twiddler Before/After와 동일한 `st.radio` 토글 패턴으로 변경(`gcn_graph_type` 세션 상태, 기본값 `tripartite`). ALS 컬럼·Before/After 토글은 손대지 않음(요청 범위 그대로 유지). Jaccard/공통 추천 지표는 토글 상태와 무관하게 삼분그래프(페르소나 포함, 주 비교 대상) 기준으로 고정 계산.
- 검증: 두 `graph_type` 쿼리 각각 다른 안내 메시지로 응답하는지 curl로 확인, ALS 엔드포인트·Before/After 토글이 그대로 동작하는지 확인, `streamlit.testing.v1.AppTest`로 두 섹션 헤딩("삼분그래프"/"이분그래프")과 각각의 안내 문구가 정상 렌더링되는지 확인.
- **후속 작업(실제 구현 시)**: `src/modeling/lightgcn/model.py`의 `train()`/`recommend()`를 `graph_type`별로 채우고, `lightgcn_service.py`가 `als_service.py`와 동일한 패턴(아티팩트 로드 + 조회)으로 두 아티팩트 경로를 나눠 로드하도록 확장.
