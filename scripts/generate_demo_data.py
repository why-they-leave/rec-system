"""
데모용 유저 데이터(demo_users.csv) 생성 + data/dashboard/recommend.db(SQLite) 빌드 스크립트.
모든 파일은 data/dashboard/ 에 저장됩니다.

페르소나는 더 이상 가상의 라벨이 아니라 실제 KMeans 6-세그먼트 결과
(data/processed/customer_segments_labeled_train_only.csv — retail-clickstream-analysis
리포의 src/features/build_train_only_segments.py 산출물, ALS와 동일한 train cutoff
2025-08-01 기준)를 그대로 사용한다. heavy/cold 구분은 재학습된 ALS 아티팩트
(models/ALS/als_model.pkl)의 user_type_map을 그대로 써서, 여기서 뽑히는 유저는
실제로 추천 API가 정상 동작하는(=train 이벤트가 1건 이상 있는) 유저만 포함된다.

PRED_MAIN_RECOMMEND.csv / PRED_DETAIL_RECOMMEND.csv(과거 더미 추천 결과)는
백엔드 API 연결(reports/BACKEND_INTEGRATION_PLAN.md) 이후 더 이상 쓰이지 않아 삭제됨.

recommend.db는 app/utils/data_loader.py의 DATA_SOURCE="sqlite" 카탈로그 로더
(products/demo_users, 소규모 카탈로그성 데이터에만 적용 — als_events/segments/detail_cf
같은 대용량 파이프라인 산출물이나 als_model.pkl은 대상이 아님, reports/BACKEND_INTEGRATION_PLAN.md 참고)
가 읽는 SQLite 파일이다. CSV는 사람이 직접 열어보기 위한 용도로 계속 함께 생성한다.

실행:
    python scripts/generate_demo_data.py
"""

import pickle
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).parent.parent
DASHBOARD_DIR = ROOT_DIR / "data" / "dashboard"

N_HEAVY_PER_PERSONA = 3
N_COLD_PER_PERSONA = 2


# ── 1. products 로드 (이미 존재, 생성 불필요) ──────────────────────────────────
src = pd.read_csv(DASHBOARD_DIR / "products.csv")
products = src.rename(columns={"product_id": "item_id"})[
    ["item_id", "name", "category", "price_usd"]
]
print(f"products.csv (source): {len(products):,} rows")


# ── 2. demo_users.csv (실제 세그먼트 기반) ─────────────────────────────────
als_params = yaml.safe_load((ROOT_DIR / "configs" / "als" / "params.yaml").read_text(encoding="utf-8"))
split_date = als_params["split_date"]

with open(ROOT_DIR / "models" / "ALS" / "als_model.pkl", "rb") as f:
    user_type_map: dict = pickle.load(f)["user_type_map"]

als_events = pd.read_csv(ROOT_DIR / "data" / "interim" / "als_events.csv", parse_dates=["timestamp"])
train_log_counts = als_events.loc[als_events["timestamp"] < split_date].groupby("user_id").size()

segments = pd.read_csv(ROOT_DIR / "data" / "processed" / "customer_segments_labeled_train_only.csv")
segments = segments[["customer_id", "segment_name"]].rename(
    columns={"customer_id": "user_id", "segment_name": "persona_label"}
)
segments["user_type"] = segments["user_id"].map(user_type_map)
segments = segments.dropna(subset=["user_type"]).copy()  # train 이벤트 0건(ALS 서빙 불가) 유저 제외
segments["log_count"] = segments["user_id"].map(train_log_counts).fillna(0).astype(int)

demo_groups = []
for persona, group in segments.groupby("persona_label"):
    heavy = group[group["user_type"] == "heavy"]
    cold = group[group["user_type"] == "cold"]
    demo_groups.append(heavy.sample(n=min(N_HEAVY_PER_PERSONA, len(heavy)), random_state=42))
    if len(cold) > 0:
        demo_groups.append(cold.sample(n=min(N_COLD_PER_PERSONA, len(cold)), random_state=42))

demo_users = (
    pd.concat(demo_groups)[["user_id", "persona_label", "user_type", "log_count"]]
    .sort_values("user_id")
    .reset_index(drop=True)
)
demo_users.to_csv(DASHBOARD_DIR / "demo_users.csv", index=False)
print(f"demo_users.csv: {len(demo_users):,} rows (실제 세그먼트 기반, 페르소나 {demo_users['persona_label'].nunique()}종)")


# ── 3. recommend.db (SQLite) ────────────────────────────────────────────────
# app/utils/data_loader.py::load_products()가 기대하는 스키마와 맞추기 위해
# item_id로 변환하기 전의 원본 컬럼(product_id 포함) 그대로 저장한다.
db_path = DASHBOARD_DIR / "recommend.db"
with sqlite3.connect(db_path) as conn:
    src.to_sql("products", conn, if_exists="replace", index=False)
    demo_users.to_sql("demo_users", conn, if_exists="replace", index=False)
print(f"recommend.db: products({len(src):,}행)/demo_users({len(demo_users):,}행) 테이블 저장 → {db_path}")

print("\n[OK] demo_users.csv / recommend.db saved to:", DASHBOARD_DIR)
