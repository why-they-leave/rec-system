# rec-system

ecommerce-funnel-analysis의 행동 프로필 피처를 입력으로 받아, 유저에게 카테고리·상품·재참여 콘텐츠를 추천하는 시스템.
분석가용 실험 탭과 이해관계자 데모 탭을 분리한 Streamlit 앱으로 제공한다.

---

## 시스템 개요

| 항목 | 내용 |
|------|------|
| 입력 | `overall_user_behavior_profile.csv`, `customer_category_behavior_features_v2.csv` |
| 출처 | [ecommerce-funnel-analysis](../ecommerce-funnel-analysis) → Google Drive |
| 추천 방식 | 코사인 유사도 기반 카테고리 추천 + 이탈 위험 룰 기반 재참여 추천 |
| 배포 | Streamlit Community Cloud |

---

## 빠른 시작

### 1. 환경 준비

Python 3.11 이상과 `uv`가 필요하다.

```bash
uv sync --extra dev
```

`pip` 환경:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. 데이터 다운로드

`.env.example`을 복사해 `.env`로 만들고, 팀에서 공유한 Google Drive 파일 ID를 입력한다.

```bash
cp .env.example .env
# .env 파일에 GDRIVE_PROFILE_ID, GDRIVE_CATEGORY_ID 입력
python scripts/download_data.py
```

### 3. 앱 실행

```bash
uv run streamlit run app.py
```

---

## 디렉터리 구조

```
rec-system/
├── app.py                          # Streamlit 홈 화면
├── pages/
│   ├── 01_experiment.py            # 분석가용: 파라미터 조작, 수치 확인
│   └── 02_demo.py                  # 이해관계자 데모용
│
├── src/
│   ├── data/
│   │   └── loader.py               # CSV 로드 + 기본 검증
│   ├── features/
│   │   └── preprocessor.py         # 피처 정규화·스케일링
│   ├── recommender/
│   │   ├── base.py                 # BaseRecommender 추상 클래스
│   │   ├── similarity.py           # 코사인 유사도 / KNN 추천
│   │   └── rule_based.py           # 이탈 위험 룰 기반 추천
│   └── evaluation/
│       └── metrics.py              # precision@k, coverage
│
├── data/
│   ├── raw/                        # GDrive에서 받은 원본 (gitignore)
│   └── processed/                  # 전처리 후 캐시 (gitignore)
│
├── configs/
│   ├── base.yaml                   # 공통 설정 (top_k, alpha, 피처 목록)
│   ├── dev.yaml
│   ├── prod.yaml
│   └── local.yaml                  # 로컬 경로 설정 (gitignore)
│
├── notebooks/                      # 알고리즘 실험용
├── scripts/
│   └── download_data.py            # GDrive → data/raw/ 다운로드
└── tests/
```

---

## 설정

`configs/base.yaml`에서 추천 파라미터를 조정한다.

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `recommender.top_k` | 10 | 추천 결과 수 |
| `recommender.alpha` | 0.5 | 프로필 유사도 가중치 (0=카테고리, 1=프로필) |
| `recommender.churn_days_threshold` | 90 | 이탈 위험 판단 기준 (일) |

---

## 테스트

```bash
uv run pytest tests/ -v --ignore=tests/test_features.py
```

---

## 품질 확인

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

---

## 배포

Streamlit Community Cloud에 GitHub 레포를 연결하고 `app.py`를 메인 파일로 지정한다.
의존성은 `requirements.txt`를 사용한다.

```bash
# requirements.txt 재생성
uv pip compile pyproject.toml -o requirements.txt
```

---

## 변경 이력

[CHANGELOG.md](CHANGELOG.md)
