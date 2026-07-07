"""
일회성 변환 스크립트: backend/TF-IDF/bowanjae.csv(상품명 기준 보완재 결과)를
products.csv의 상품명→item_id 매핑을 이용해 data/outputs/complementary/detail_cf.csv
(item_id 기준)로 변환한다.

전처리된 로그(data/processed/df_integrated_logs.csv)와 원본 상품 메타데이터
(data/raw/products.csv)가 아직 리포에 없어 scripts/run_bowanjae_pipeline.py를
바로 돌릴 수 없는 동안의 임시 조치 — 데이터가 확보되면 이 스크립트 대신
run_bowanjae_pipeline.py로 재생성한다. products.csv의 상품명이 모두 유일함을
전제로 한다(1197개 상품 각각 다른 name).

Usage:
    python scripts/convert_bowanjae_to_detail_cf.py
"""

import pandas as pd

PRODUCTS_PATH = "data/dashboard/products.csv"
BOWANJAE_PATH = "backend/TF-IDF/bowanjae.csv"
OUTPUT_PATH = "data/outputs/complementary/detail_cf.csv"


def main() -> None:
    products = pd.read_csv(PRODUCTS_PATH)
    if products["name"].duplicated().any():
        raise ValueError("products.csv에 중복된 상품명이 있어 이름 기준 매핑이 부정확합니다.")
    name_to_id = products.set_index("name")["product_id"].to_dict()

    bowanjae = pd.read_csv(BOWANJAE_PATH)
    result = pd.DataFrame({
        "item_id": bowanjae["prod_A_name"].map(name_to_id),
        "rec_item_id": bowanjae["prod_B_name"].map(name_to_id),
        "score": bowanjae["score"],
        "rank": bowanjae["rank"],
    })

    unmapped = result["item_id"].isna().sum() + result["rec_item_id"].isna().sum()
    if unmapped:
        raise ValueError(f"상품명 매핑 실패 {unmapped}건 — products.csv와 bowanjae.csv 상품명 불일치")
    result["item_id"] = result["item_id"].astype(int)
    result["rec_item_id"] = result["rec_item_id"].astype(int)

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"저장 완료: {OUTPUT_PATH} ({len(result):,}개 레코드)")


if __name__ == "__main__":
    main()
