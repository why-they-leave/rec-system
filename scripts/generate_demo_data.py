"""
데모용 샘플 데이터 생성 스크립트
실제 추천 결과가 준비되면 이 파일에서 생성된 CSV를 교체하면 됩니다.
모든 파일은 data/dashboard/ 에 저장됩니다.

실행:
    python scripts/generate_demo_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

ROOT_DIR = Path(__file__).parent.parent
DASHBOARD_DIR = ROOT_DIR / "data" / "dashboard"


# ── 1. products 로드 (이미 존재, 생성 불필요) ──────────────────────────────────
src = pd.read_csv(DASHBOARD_DIR / "products.csv")
products = src.rename(columns={"product_id": "item_id"})[
    ["item_id", "name", "category", "price_usd"]
]
print(f"products.csv (source): {len(products):,} rows")


# ── 2. demo_users.csv & persona_labels.csv ─────────────────────────────────────
PERSONAS = ["가성비추구형", "트렌드세터", "브랜드충성형", "충동구매형", "비교분석형"]

demo_users = pd.DataFrame(
    {
        "user_id": range(1, 11),
        "persona_label": [
            "가성비추구형", "트렌드세터", "브랜드충성형", "충동구매형", "비교분석형",
            "가성비추구형", "트렌드세터", "브랜드충성형", "충동구매형", "비교분석형",
        ],
        "user_type": ["heavy"] * 7 + ["cold"] * 3,
        "log_count": [285, 241, 198, 312, 176, 89, 143, 67, 23, 31],
    }
)
demo_users.to_csv(DASHBOARD_DIR / "demo_users.csv", index=False)
print(f"demo_users.csv: {len(demo_users):,} rows")

persona_labels = pd.DataFrame(
    {
        "user_id": range(1, 101),
        "persona_label": np.random.choice(PERSONAS, 100),
    }
)
persona_labels.to_csv(DASHBOARD_DIR / "persona_labels.csv", index=False)
print(f"persona_labels.csv: {len(persona_labels):,} rows")


# ── 3. PRED_MAIN_RECOMMEND.csv ─────────────────────────────────────────────────
item_ids = products["item_id"].tolist()

rows = []
for _, user in demo_users.iterrows():
    uid = int(user["user_id"])
    utype = user["user_type"]

    base_pool = np.random.choice(item_ids, 16, replace=False).tolist()

    model_items = {
        "ALS":      sorted(base_pool[:10], key=lambda _: np.random.random()),
        "LightGCN": sorted(base_pool[3:13], key=lambda _: np.random.random()),
    }

    for model_type, selected in model_items.items():
        scores_before = sorted(np.random.uniform(0.5, 1.0, 10), reverse=True)

        after_order = list(range(10))
        n_swaps = np.random.randint(3, 5)
        for _ in range(n_swaps):
            i, j = np.random.choice(10, 2, replace=False)
            after_order[i], after_order[j] = after_order[j], after_order[i]

        for rank, (item_id, score) in enumerate(zip(selected, scores_before), 1):
            rows.append(
                dict(
                    user_id=uid, item_id=item_id,
                    score=round(score, 4), rank=rank,
                    model_type=model_type, twiddler="before", user_type=utype,
                )
            )
        for new_rank, orig_idx in enumerate(after_order, 1):
            item_id = selected[orig_idx]
            score = scores_before[orig_idx] * round(np.random.uniform(0.88, 1.12), 3)
            rows.append(
                dict(
                    user_id=uid, item_id=item_id,
                    score=round(score, 4), rank=new_rank,
                    model_type=model_type, twiddler="after", user_type=utype,
                )
            )

pred_main = pd.DataFrame(rows)
pred_main.to_csv(DASHBOARD_DIR / "PRED_MAIN_RECOMMEND.csv", index=False)
print(f"PRED_MAIN_RECOMMEND.csv: {len(pred_main):,} rows")


# ── 4. PRED_DETAIL_RECOMMEND.csv ───────────────────────────────────────────────
rows = []
cat_map = products.groupby("category")["item_id"].apply(list).to_dict()

for _, row in products.iterrows():
    iid = int(row["item_id"])
    cat = row["category"]

    same_cat = [x for x in cat_map[cat] if x != iid]
    content = np.random.choice(same_cat, min(5, len(same_cat)), replace=False).tolist()

    other_ids = [x for c, ids in cat_map.items() if c != cat for x in ids]
    cf = np.random.choice(other_ids, min(5, len(other_ids)), replace=False).tolist()

    for rank, rec_id in enumerate(content, 1):
        rows.append(
            dict(item_id=iid, rec_item_id=int(rec_id),
                 score=round(np.random.uniform(0.65, 1.0), 4),
                 rank=rank, rec_type="content")
        )
    for rank, rec_id in enumerate(cf, 1):
        rows.append(
            dict(item_id=iid, rec_item_id=int(rec_id),
                 score=round(np.random.uniform(0.4, 0.75), 4),
                 rank=rank, rec_type="cf")
        )

pred_detail = pd.DataFrame(rows)
pred_detail.to_csv(DASHBOARD_DIR / "PRED_DETAIL_RECOMMEND.csv", index=False)
print(f"PRED_DETAIL_RECOMMEND.csv: {len(pred_detail):,} rows")

print("\n[OK] All files saved to:", DASHBOARD_DIR)
