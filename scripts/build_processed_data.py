"""
data/raw/{events,sessions,orders,order_items}.csv(원본 로그)로부터
data/processed/df_integrated_logs.csv를 생성한다 — scripts/run_bowanjae_pipeline.py(보완재 추천)의 입력.

ALS용 데이터마트(data/interim/als_events.csv)는 별도의 원본 데이터마트 생성 코드로
이미 만들어져 있으므로 이 스크립트에서는 다루지 않는다(src/modeling/als/model.py의
PATHS["full"]["mart"]가 그 파일을 직접 가리킨다).

이벤트 전처리 로직(특히 purchase 이벤트의 product_id 복원)은 기존
bowanjae_pipeline.py의 preprocess_data()를 그대로 재현한 것이다:
- events.csv의 'purchase'/'checkout' 이벤트는 product_id가 전부 NaN이다
  (세션/카트 단위 이벤트라 개별 상품과 직접 연결되지 않음).
- 'purchase'는 (customer_id, timestamp)==(customer_id, order_time)로 orders와
  매칭한 뒤 order_items로 실제 구매 상품(product_id)을 복원한다(주문 1건에
  상품이 여러 개면 그만큼 행이 늘어난다).
- 'checkout'은 어떤 주문과도 정확히 timestamp가 일치하지 않아(장바구니 이탈 등)
  이 방식으로 product_id를 복원할 수 없다 — bowanjae_pipeline.py 원본과 동일하게
  이번에도 제외한다(가정: checkout을 활용하려면 별도의 세션 내 next-order 매칭
  로직이 필요하며, 근거 없이 임의로 만들지 않았다). ALS 마트(data/interim/als_events.csv)는
  checkout을 포함하고 있어 이 부분만 두 파이프라인 간 처리 방식이 다르다.

날짜 필터는 두지 않는다 — data/interim/als_events.csv(실제 ALS 마트)가 원본 전체 기간
(~2025-11-01)을 담고 있는 것으로 확인되어, 이 스크립트도 동일하게 전체 기간을 사용한다.
특정 기간만 쓰고 싶다면 build_integrated_logs()의 events를 필터링해서 호출하면 된다.

Usage:
    python scripts/build_processed_data.py
"""

import os

import pandas as pd

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed"


def load_raw():
    events = pd.read_csv(os.path.join(RAW_DIR, "events.csv"), parse_dates=["timestamp"])
    sessions = pd.read_csv(os.path.join(RAW_DIR, "sessions.csv"), parse_dates=["start_time"])
    orders = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"), parse_dates=["order_time"])
    order_items = pd.read_csv(os.path.join(RAW_DIR, "order_items.csv"))
    return events, sessions, orders, order_items


def build_integrated_logs(events, sessions, orders, order_items):
    df_base_events = pd.merge(
        events[["session_id", "event_id", "timestamp", "event_type", "product_id"]],
        sessions[["session_id", "customer_id"]],
        on="session_id",
        how="left",
    )

    # add_to_cart / page_view — product_id 그대로 사용 (checkout은 제외)
    df_non_purchase = df_base_events[df_base_events["event_type"].isin(["add_to_cart", "page_view"])].copy()

    # purchase — orders/order_items로 실제 구매 상품 복원
    df_purchase_logs = df_base_events[df_base_events["event_type"] == "purchase"].copy()
    df_purchase_with_order_id = pd.merge(
        df_purchase_logs,
        orders[["order_id", "customer_id", "order_time"]],
        left_on=["customer_id", "timestamp"],
        right_on=["customer_id", "order_time"],
        how="inner",
    )
    df_purchase_with_items = pd.merge(
        df_purchase_with_order_id,
        order_items[["order_id", "product_id"]],
        on="order_id",
        how="left",
    )
    df_purchase_final = df_purchase_with_items.drop(columns=["product_id_x", "order_id", "order_time"])
    df_purchase_final = df_purchase_final.rename(columns={"product_id_y": "product_id"})

    common_columns = ["customer_id", "session_id", "event_id", "timestamp", "event_type", "product_id"]
    df_integrated_logs = pd.concat(
        [df_non_purchase[common_columns], df_purchase_final[common_columns]],
        axis=0,
        ignore_index=True,
    )
    df_integrated_logs["product_id"] = df_integrated_logs["product_id"].astype(int)
    df_integrated_logs = df_integrated_logs.sort_values(by=["customer_id", "timestamp"]).reset_index(drop=True)
    return df_integrated_logs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    events, sessions, orders, order_items = load_raw()
    print(f"[로드] events={len(events):,} / sessions={len(sessions):,} / orders={len(orders):,} / order_items={len(order_items):,}")

    df_integrated_logs = build_integrated_logs(events, sessions, orders, order_items)
    integrated_path = os.path.join(OUTPUT_DIR, "df_integrated_logs.csv")
    df_integrated_logs.to_csv(integrated_path, index=False)
    print(f"[저장] {integrated_path} ({len(df_integrated_logs):,}행, event_type={df_integrated_logs['event_type'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
