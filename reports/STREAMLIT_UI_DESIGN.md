# 🛍️ 추천 시스템 데모 UI 설계 문서

> **목적:** 이 문서는 Streamlit 기반의 추천 시스템 데모 UI를 처음 구현하는 개발자(또는 LLM)가
> 전체 구조와 설계 의도를 파악하고 바로 구현에 착수할 수 있도록 작성된 설계 명세서입니다.

> **⚠️ 결정 사항 업데이트(2026-07-08)** — 아래 두 항목은 이 문서가 원래 계획했던 것과 달리 **의도적으로 제외**하기로 결정됨. 문서 곳곳에 남아있는 관련 서술은 원래 설계 의도를 보여주는 기록으로 남겨두고, 각 위치에 별도 표시를 추가함:
> - **Bump Chart(`components/metric_chart.py`, `pages/2_twiddler_compare.py`)**: 만들지 않기로 결정. Twiddler 전/후 순위 변화는 실제 구현에서 각 상품 카드 우상단의 ▲/▼ 배지(`app/utils/rank_delta.py::get_rank_delta`)로만 표시.
> - **콘텐츠 기반(대체재) 추천(`rec_type='content'`)**: 완전히 제거하기로 결정. 상세 화면엔 보완재(Item-based CF)만 남음 — `backend/api/schemas.py`/`recommend_detail.py`에서 `rec_type` 파라미터 자체를 삭제함(`reports/BACKEND_INTEGRATION_PLAN.md` 참고).
>
> 이 외에도 실제 구현은 `pages/` 멀티페이지 대신 `app/main.py` 단일 파일 + `st.session_state` 뷰 라우팅을 쓰는 등 이 문서 작성 이후 구조가 여러 곳에서 달라졌으나, 이번 업데이트는 위 두 결정 사항에 한정함.

---

## 1. 프로젝트 개요

### 1.1 목적
- 이커머스 클릭스트림 데이터를 기반으로 구현한 **ALS(베이스라인)** 와 **LightGCN(비교 모델)** 추천 시스템의 결과를 시각적으로 데모
- 페르소나가 다른 **5~20명의 대표 유저**를 선정하여, 유저별 추천 결과 차이를 직관적으로 비교
- **Twiddler(후처리) 적용 전/후** 추천 결과 변화를 시각화

### 1.2 핵심 비교 축
| 비교 축 | 내용 |
|---|---|
| **ALS vs LightGCN** | 동일 유저에 대해 두 모델의 추천 결과가 어떻게 다른지 |
| **Twiddler 전/후** | 후처리(페르소나 가중치 재정렬) 적용 전/후 추천 순위 변화 |

### 1.3 기술 스택
- **UI 프레임워크:** Streamlit
- **데이터베이스:** SQLite (선택적 연동)
- **시각화:** Plotly, Streamlit 네이티브 컴포넌트
- **데이터 소스:** 사전 연산된 CSV 결과물 (또는 SQLite DB)

---

## 2. 프로젝트 주요 구조(UI 위주)

```
rec-system/
│
├── app/                              # UI 관련 코드 전체
│   ├── main.py                       # 진입점 · 사이드바 · session_state 설정
│   │
│   ├── pages/                        # (미사용 — 실제로는 app/main.py 단일 파일 + session_state 라우팅)
│   │   ├── 1_main_recommend.py       # ALS vs LightGCN 상품 카드 비교
│   │   ├── 2_twiddler_compare.py     # Twiddler 전/후 Bump Chart (제거 결정 — 아래 §5.3 참고)
│   │   └── 3_detail_recommend.py     # 보완재 연관 상품 추천 (대체재는 제거 결정 — 아래 §5.4 참고)
│   │
│   ├── components/
│   │   ├── user_selector.py          # 유저 드롭다운 · 페르소나 정보 카드
│   │   ├── product_card.py           # 상품 카드 렌더링 · 색상 원형 아이콘 · 배지
│   │   └── metric_chart.py           # (제거 결정 — 아래 §6.3 참고) Plotly Bump Chart · 순위 변화 시각화
│   │
│   ├── static/
│   │   ├── style.css                 # 공통 스타일 · 카드 레이아웃 · CSS 변수
│   │   └── images/                   # 상품 종류별 대표 아이콘 이미지 (41종)
│   │
│   └── utils/
│       ├── data_loader.py            # CSV / SQLite 로드 · @cache_data 캐싱
│       └── style_loader.py           # CSS 파일 로드 · st.markdown 주입
│
├── data/                             # 사전 연산된 결과물 (모델 코드와 분리)
│   ├── PRED_MAIN_RECOMMEND.csv
│   ├── PRED_DETAIL_RECOMMEND.csv
│   ├── products.csv
│   ├── persona_labels.csv
│   ├── demo_users.csv
│   └── recommend.db                  # SQLite 연동 시 사용 (선택)
│
├── .streamlit/
│   └── config.toml                   # 테마 · 포트 등 Streamlit 설정
├── requirements.txt
└── README.md
```

---

## 3. 데이터 구조

### 3.1 주요 컬럼 정의

**`PRED_MAIN_RECOMMEND.csv`**
```
user_id        | 유저 식별자
item_id        | 추천 상품 식별자
score          | 모델 예측 점수
rank           | 추천 순위
model_type     | 'ALS' 또는 'LightGCN'
twiddler       | 'before' 또는 'after' (Twiddler 적용 여부)
user_type      | 'heavy' 또는 'cold'
```

**`PRED_DETAIL_RECOMMEND.csv`**
```
item_id        | 기준 상품 식별자
rec_item_id    | 추천 상품 식별자
score          | 유사도 점수
rank           | 추천 순위
rec_type       | (제거 결정) 'content' (대체재) 또는 'cf' (보완재) — 실제로는 rec_type 컬럼/파라미터 자체를 삭제하고 보완재(cf)만 제공
```

**`persona_labels.csv`**
```
user_id        | 유저 식별자
persona_label  | 페르소나 레이블 (예: '가성비추구형', '트렌드세터', '브랜드충성형' 등)
```

**`products.csv`**
```
item_id        | 상품 식별자
name           | 상품명 (예: SSD MediumBlue 149)
category       | 카테고리 (Electronics, Books 등 7종)
price_usd      | 가격
```

**`demo_users.csv`**
```
user_id        | 유저 식별자
persona_label  | 페르소나 레이블
user_type      | 'heavy' 또는 'cold'
log_count      | 총 행동 로그 수
```

### 3.2 SQLite 연동 (선택)
CSV 대신 SQLite를 사용할 경우 아래 테이블 구조를 사용:
```sql
CREATE TABLE pred_main_recommend (
    user_id      INTEGER,
    item_id      INTEGER,
    score        REAL,
    rank         INTEGER,
    model_type   TEXT,   -- 'ALS' | 'LightGCN'
    twiddler     TEXT,   -- 'before' | 'after'
    user_type    TEXT    -- 'heavy' | 'cold'
);

CREATE TABLE pred_detail_recommend (
    item_id      INTEGER,
    rec_item_id  INTEGER,
    score        REAL,
    rank         INTEGER,
    rec_type     TEXT    -- (제거 결정) 'content' | 'cf' — 실제로는 컬럼 자체 없이 보완재만 제공
);

CREATE TABLE persona_labels (
    user_id       INTEGER PRIMARY KEY,
    persona_label TEXT
);

CREATE TABLE products (
    item_id      INTEGER PRIMARY KEY,
    name         TEXT,
    category     TEXT,
    price_usd    REAL
);
```

---

## 4. 상품 카드 이미지 설계

### 4.1 배경 및 제약 조건
- 데이터는 **합성 데이터**로, 상품명이 `"SSD MediumBlue 149"` 형식 (종류 + CSS 색상명 + 번호)
- 상품별 실제 이미지 없음 → 종류별 대표 아이콘 이미지 41장 + 색상 정보 활용
- 같은 종류 + 같은 색상이지만 번호만 다른 상품이 약 117쌍 존재 → 이미지로는 구분 불가, **가격과 상품 ID 텍스트로 구분**

### 4.2 카드 디자인 방식: 원형 아이콘 + 색상 배경

**선택 근거 (UX):**
카드 배경 전체나 상단 1/3을 상품 색상으로 채우는 방식 대신, **아이콘을 원으로 감싸고 해당 원의 배경만 상품 색상으로 적용**하는 방식을 채택한다.

- 배경 전체 색상 → 텍스트 가독성 저하, 색상이 너무 강해 카드 구분 어려움
- 상단 1/3 컬러 헤더 → 아이콘과 조합 시 레이아웃이 어색함
- **원형 아이콘 방식** → 색상을 작은 영역에 집중시켜 카드 전체의 통일감 유지, Google Material / Apple HIG 표준 방식

**카드 구조:**
```
┌───────────────────────┐
│                       │
│      ╭───────╮        │
│      │  🖥️  │        │  ← 원형 배경색 = 상품명의 CSS 색상명
│      ╰───────╯        │    (예: MediumBlue → background: MediumBlue)
│                       │
│  SSD MediumBlue 149   │  ← 상품명 전체 표시
│  Electronics          │  ← 카테고리
│  $ 570.00             │  ← 가격 (같은 종류+색상 구분 핵심)
│  ★ rank: 1            │  ← 추천 순위
│  [🔁 공통 추천]        │  ← 선택적 배지
└───────────────────────┘
```

**구현 방법 (`product_card.py`):**
```python
def extract_color(name: str) -> str:
    """상품명에서 CSS 색상명 추출 (뒤에서 두 번째 단어)"""
    return name.split()[-2]  # "SSD MediumBlue 149" → "MediumBlue"

def extract_product_type(name: str) -> str:
    """상품명에서 종류 추출 (색상·번호 제외)"""
    return ' '.join(name.split()[:-2])  # "SSD MediumBlue 149" → "SSD"

def render_product_card(item: pd.Series, rank: int, badge: str = None):
    color = extract_color(item['name'])
    product_type = extract_product_type(item['name'])

    # 원형 아이콘 HTML: 아이콘 이미지 + 색상 원형 배경
    icon_html = f"""
    <div style="
        width: 64px; height: 64px; border-radius: 50%;
        background-color: {color};
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 8px auto;
    ">
        <img src="app/static/images/{product_type.lower().replace(' ', '_')}.png"
             width="36" height="36"
             onerror="this.style.display='none'"  />
    </div>
    """
    badge_html = f'<span class="badge">{badge}</span>' if badge else ''

    card_html = f"""
    <div class="product-card">
        {icon_html}
        <p class="product-name">{item['name']}</p>
        <p class="product-meta">{item['category']}</p>
        <p class="product-price">$ {item['price_usd']:.2f}</p>
        <p class="product-rank">★ rank: {rank}</p>
        {badge_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
```

**CSS (`static/style.css`):**
```css
.product-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
    background: #ffffff;
    transition: box-shadow 0.2s;
}
.product-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.product-name  { font-size: 0.8rem; font-weight: 600; margin: 4px 0; word-break: break-word; }
.product-meta  { font-size: 0.72rem; color: #888; margin: 2px 0; }
.product-price { font-size: 0.85rem; font-weight: 700; color: #1a1a1a; margin: 4px 0; }
.product-rank  { font-size: 0.72rem; color: #aaa; margin: 2px 0; }
.badge {
    display: inline-block;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 99px;
    background: #f0f4ff;
    color: #3355cc;
    margin-top: 6px;
}
```

---

## 5. 페이지별 상세 설계

### 5.1 `app/main.py` — 진입점 & 공통 사이드바

**역할:** 전체 앱의 레이아웃 기준점. 사이드바에서 유저를 선택하면 모든 페이지에 전역으로 반영됨.

**사이드바 구성:**
```
┌─────────────────────────────┐
│  🛍️ 추천 시스템 데모         │
│                             │
│  👤 유저 선택               │
│  [드롭다운: User 001 ▼]     │
│   "User 001 | 가성비추구형" │
│                             │
│  📋 유저 정보               │
│  - 페르소나: 가성비추구형    │
│  - 유저 유형: Heavy         │
│  - 총 행동 로그: 142건      │
│                             │
│  ─────────────────────────  │
│  📄 페이지 이동             │
│  > 메인 추천 비교           │
│  > Twiddler 비교            │
│  > 연관 상품 추천           │
└─────────────────────────────┘
```

**구현 포인트:**
- 유저 목록은 `demo_users.csv`에서 로드 (전체 유저가 아닌 대표 유저 5~20명만)
- 드롭다운 레이블: `"User 001 | 가성비추구형"` 형식으로 페르소나를 함께 표시
- 선택된 `user_id`를 `st.session_state.selected_user`에 저장하여 전 페이지에서 공유
- CSS 로드: `style_loader.py`의 함수를 호출하여 `static/style.css`를 `st.markdown`으로 주입

---

### 5.2 `pages/1_main_recommend.py` — ALS vs LightGCN 비교

**목적:** 동일 유저에 대해 두 모델의 추천 상위 10개를 나란히 비교

**레이아웃:**
```
┌──────────────────────────────────────────────────────────┐
│  메인 화면 추천 비교 — User 001 | 가성비추구형            │
├────────────────────────┬─────────────────────────────────┤
│   🤖 ALS (베이스라인)  │   🚀 LightGCN                   │
│                        │                                 │
│  [카드][카드][카드]    │  [카드][카드][카드]             │
│  [카드][카드][카드]    │  [카드][카드][카드]             │
│  [카드][카드][카드]    │  [카드][카드][카드]             │
│  [카드]                │  [카드]                         │
├────────────────────────┴─────────────────────────────────┤
│  📊 공통 추천 상품: 3개 / 10개 │ Jaccard 유사도: 0.23   │
└──────────────────────────────────────────────────────────┘
```

**구현 포인트:**
- `st.columns(2)`로 좌(ALS) / 우(LightGCN) 분리
- 각 컬럼 안에서 `st.columns(3)`으로 상품 카드 그리드 배치
- 두 모델에서 공통으로 추천된 상품은 카드에 `🔁 공통 추천` 배지 표시
- 하단에 Jaccard 유사도 지표 표시
- `model_type` 컬럼 필터링: `df[df['model_type'] == 'ALS']`

---

### 5.3 `pages/2_twiddler_compare.py` — Twiddler 전/후 비교

> **[제거 결정]** 아래 Bump Chart는 만들지 않기로 함. 실제로는 프로덕션 메인 화면(`app/main.py::_render_main_recommend`)이 Heavy 유저에게 Twiddler 적용 후 결과를 단일 뷰로 보여주면서, 카드 우상단에 적용 전 대비 순위 변동(▲/▼ + 변동폭)을 배지로만 표시한다(`get_rank_delta`). Before/After를 나란히 비교하는 화면 자체는 `SHOW_TWIDDLER_DEMO` 환경변수를 켰을 때만 나타나는 내부 검증용 화면(`_render_twiddler_demo`)으로 남아있고, 거기서도 Bump Chart 없이 동일한 배지 방식을 재사용한다.

**목적:** 페르소나 후처리(Twiddler) 적용 전/후 추천 순위 변화를 시각화

**레이아웃:**
```
┌──────────────────────────────────────────────────────────┐
│  Twiddler 전/후 비교 — User 001 | 가성비추구형            │
│  모델 선택: [ALS ▼]                                      │
├──────────────────────────────────────────────────────────┤
│  📊 순위 변화 Bump Chart (Plotly)                        │
│                                                          │
│  Before │              │ After                          │
│    1 ───┤  상품 A      ├─── 3   (▼ 2) ← 빨간 선        │
│    2 ───┤  상품 B      ├─── 1   (▲ 1) ← 초록 선        │
│    3 ───┤  상품 C      ├─── 2   (▲ 1) ← 초록 선        │
│    4 ───┤  상품 D      ├─── 4   (➡️)  ← 회색 선        │
│       ...                                               │
├──────────────────────────────────────────────────────────┤
│  🔼 순위 상승 상품   🔽 순위 하락 상품   ➡️ 유지 상품   │
│  [카드 그리드]       [카드 그리드]       [카드 그리드]  │
└──────────────────────────────────────────────────────────┘
```

**구현 포인트:**
- 상단 `st.selectbox`로 ALS / LightGCN 모델 선택
- Bump Chart: Plotly `go.Scatter`로 before → after 순위 연결
  - 순위 상승: 초록색 (`#22c55e`)
  - 순위 하락: 빨간색 (`#ef4444`)
  - 유지: 회색 (`#9ca3af`)
- y축은 순위이므로 **위가 1등이 되도록 y축 반전** (`autorange='reversed'`)
- 하단 카드는 순위 변화 기준 3그룹 분류 후 `st.columns(3)`으로 표시

---

### 5.4 `pages/3_detail_recommend.py` — 상세 페이지 연관 상품

> **[제거 결정]** 대체재(콘텐츠 기반) 추천은 만들지 않기로 함 — 상세 화면엔 보완재(Item-based CF)만 단일 컬럼으로 표시(`app/main.py::_render_detail_recommend`). `rec_type` 파라미터/스키마 필드 자체를 백엔드에서 삭제했다(`reports/BACKEND_INTEGRATION_PLAN.md` 참고).

**목적:** 특정 상품을 선택했을 때 대체재(콘텐츠 기반) + 보완재(Item-based CF) 표시

**레이아웃:**
```
┌──────────────────────────────────────────────────────────┐
│  상세 페이지 연관 상품 추천                               │
│  상품 선택: [드롭다운 ▼]                                 │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ 현재 상품                                        │   │
│  │  ╭──────╮  SSD MediumBlue 149                   │   │
│  │  │  🖥️  │  Electronics · $ 570.00               │   │
│  │  ╰──────╯                                        │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  🔄 대체재 (콘텐츠 기반)   │  🛒 보완재 (Item-based CF) │
│  [카드][카드][카드][카드]  │  [카드][카드][카드][카드]  │
└──────────────────────────────────────────────────────────┘
```

**구현 포인트:**
- 상품 선택 드롭다운: `products.csv` 전체 목록 (`카테고리 > 상품명` 형식 권장)
- 현재 상품은 가로형 카드로 상단에 크게 표시
- ~~대체재 / 보완재를 `st.columns(2)`로 분리~~ → (제거 결정) 보완재만 `st.columns(4)`로 카드 배치
- ~~`rec_type` 컬럼 필터링~~ → (제거 결정) `rec_type` 컬럼/파라미터 자체가 없음, 보완재 조회 함수 하나만 호출

---

## 6. 공통 컴포넌트 설계

### 6.1 `components/user_selector.py`
```python
def render_user_selector(demo_users_df: pd.DataFrame) -> int:
    """
    사이드바에 유저 드롭다운 + 유저 정보 카드 렌더링.
    반환값: 선택된 user_id (int)
    """
```

### 6.2 `components/product_card.py`
```python
def extract_color(name: str) -> str:
    """상품명 뒤에서 두 번째 단어를 CSS 색상명으로 추출"""

def extract_product_type(name: str) -> str:
    """상품명에서 종류(색상·번호 제외) 추출"""

def render_product_card(item: pd.Series, rank: int, badge: str = None):
    """
    원형 색상 아이콘 + 상품 정보 카드 렌더링.
    - item    : products.csv 1행 (name, category, price_usd 포함)
    - rank    : 추천 순위
    - badge   : '🔁 공통 추천', '🔼 순위 상승' 등 선택적 배지 (없으면 None)
    """
```

### 6.3 `components/metric_chart.py` — (제거 결정, 만들지 않음)
```python
def render_bump_chart(before_df: pd.DataFrame, after_df: pd.DataFrame) -> go.Figure:
    """
    Twiddler 전/후 순위 변화 Bump Chart 렌더링.
    - before_df / after_df : item_id, rank 컬럼 포함
    - 반환값: Plotly Figure (st.plotly_chart로 표시)
    """
```
실제로는 이 컴포넌트를 만들지 않고, `components/product_card.py::render_product_card`의 `rank_delta`/`plain_rank_badge` 인자로 카드별 순위 변동을 배지 하나로 대체했다(`app/utils/rank_delta.py::get_rank_delta` 참고).

---

## 7. 유틸리티 설계

### 7.1 `utils/data_loader.py`
```python
DATA_SOURCE = "csv"        # "csv" 또는 "sqlite" — 이 상수 하나만 바꾸면 전환됨
SQLITE_PATH = "data/recommend.db"

@st.cache_data
def load_recommendations() -> pd.DataFrame: ...

@st.cache_data
def load_detail_recommendations() -> pd.DataFrame: ...

@st.cache_data
def load_products() -> pd.DataFrame: ...

@st.cache_data
def load_persona_labels() -> pd.DataFrame: ...

@st.cache_data
def load_demo_users() -> pd.DataFrame: ...

def get_user_recommendations(
    rec_df: pd.DataFrame,
    user_id: int,
    model_type: str,   # 'ALS' | 'LightGCN'
    twiddler: str,     # 'before' | 'after'
    top_n: int = 10
) -> pd.DataFrame: ...
```

### 7.2 `utils/style_loader.py`
```python
def load_css(path: str = "app/static/style.css"):
    """CSS 파일을 읽어 st.markdown으로 주입. main.py에서 1회 호출."""
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
```

---

## 8. 대표 유저 선정 기준

| 기준 | 내용 |
|---|---|
| **페르소나 다양성** | 정의된 모든 페르소나 레이블에서 최소 1명씩 포함 |
| **Heavy 유저 위주** | 로그 수 상위 유저 중 선정 (ALS 추천 결과가 의미 있는 유저) |
| **추천 차이가 큰 유저** | ALS vs LightGCN Jaccard 유사도가 낮은 유저 우선 선정 (비교 효과 극대화) |

---

## 9. 실행 방법

```bash
pip install streamlit plotly pandas

# CSV 모드 실행
streamlit run app/main.py

# SQLite 모드 전환: utils/data_loader.py에서
# DATA_SOURCE = "sqlite" 로 변경 후 동일 명령 실행
```

---

## 10. 구현 순서

```
1단계: utils/data_loader.py        데이터 로드 · 캐싱 함수 구현
         ↓
2단계: utils/style_loader.py       CSS 주입 유틸 구현
         ↓
3단계: static/style.css            카드 · 배지 · 레이아웃 공통 스타일 작성
         ↓
4단계: components/product_card.py  원형 색상 아이콘 카드 컴포넌트 구현
         ↓
5단계: app/main.py 사이드바        유저 드롭다운 · session_state 설정
         ↓
6단계: pages/1_main_recommend.py   ALS vs LightGCN 카드 비교 구현
         ↓
7단계: pages/2_twiddler_compare.py (제거 결정) Bump Chart 대신 카드 배지로 순위 변화 표시
         ↓
8단계: pages/3_detail_recommend.py 연관 상품 추천 구현
         ↓
9단계: SQLite 연동 (선택)          DATA_SOURCE 전환 후 동일 로직 재사용
```

---

## 11. 주의사항 및 구현 팁

- **`st.session_state` 필수:** 유저 선택값이 페이지 이동 시 유지되어야 하므로 반드시 `session_state`에 저장
- **`@st.cache_data` 필수:** CSV/DB 로드는 반드시 캐싱하여 유저 변경 시 불필요한 재로드 방지
- **CSS 색상명 직접 활용:** 상품명의 색상 단어(`MediumBlue`, `DarkOrchid` 등)는 CSS Named Color와 동일하므로 별도 매핑 테이블 없이 바로 `background-color` 값으로 사용 가능
- **아이콘 이미지 fallback:** `<img onerror="this.style.display='none'">` 처리로 이미지 없을 때 원형 색상만 표시되도록 방어
- ~~**Bump Chart y축 반전 필수**~~ / ~~**Plotly 사용 권장**~~ → (제거 결정) Bump Chart를 만들지 않기로 해 두 항목 모두 해당 없음. 순위 변화는 카드 배지(▲/▼)로만 표시
- **SQLite 전환:** `data_loader.py`의 `DATA_SOURCE` 상수만 변경하면 나머지 코드 수정 불필요