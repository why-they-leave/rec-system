"""
GDrive에 업로드된 데이터 번들을 내려받아 리포 루트에 압축 해제하는 공통 로직.

이 리포의 실제 추천 데이터(als_model.pkl, customer_segments_labeled_train_only.csv 등)는
용량이 커서(수십 MB) git에 커밋하지 않는다(.gitignore 참고) — 대신 GDrive에 올려두고 받는다.
두 곳에서 재사용된다:
    - scripts/download_data.py — 로컬 개발자가 clone 후 한 번 실행하는 CLI
    - app/main.py — Streamlit Community Cloud 콜드스타트 시 필요한 파일이 없으면 자동 실행
      (Cloud는 프로세스를 하나만 띄우므로 별도 설정 스크립트를 미리 실행해 둘 수 없다)

번들에 포함된 파일(= 앱을 바로 구동하는 데 필요한 최소 집합):
    data/dashboard/recommend.db                                   (app 카탈로그: products/demo_users)
    data/dashboard/products.csv                                   (backend catalog_service가 직접 읽음)
    data/dashboard/demo_users.csv                                 (CSV 폴백용)
    data/processed/customer_segments_labeled_train_only.csv       (persona_service)
    data/outputs/complementary/detail_cf.csv                      (complementary_service)
    models/ALS/als_model.pkl                                      (als_service)
    data/processed/tri_graph_uidx2tidx_train.json                 (graph_service — 유저→상품 서브그래프)
    data/processed/tri_graph_tidx2pidx.json                       (graph_service — 상품→세그먼트 lift)
    data/processed/tri_graph_uidx2pidx.json                       (graph_service — 유저 본인 세그먼트)
    data/processed/uidx_user_id_mapping.csv                       (graph_service — uidx↔user_id 매핑)
    data/processed/tidx_product_id_mapping.csv                    (graph_service — tidx↔product_id 매핑)
    data/processed/df_integrated_logs.csv                         (graph_service — cutoff 이전 구매 판정)
    data/outputs/eval/twiddler_accuracy.csv                       (load_twiddler_eval — 오프라인 성능 지표)
    data/outputs/eval/twiddler_diversity.csv                      (load_twiddler_eval — 오프라인 성능 지표)
    data/outputs/LightGCN/PRED_MAIN_RECOMMEND.csv                 (lightgcn_service — bipartite 추천, 유저별 top-100)
    data/outputs/LightGCN/lightgcn_test.csv                       (evaluate_twiddler — bipartite 정답셋)
    data/processed/segment_personas_train_only.json               (graph_service — 추천 근거 그래프 세그먼트 노드 이름)

data/processed/df_integrated_logs.csv(39MB)는 원본 로그 파생 파일치고 크지만, 다른 재학습
파이프라인에서도 재사용되는 원본 그대로를 유지하기 위해 컬럼을 축소하지 않고 그대로 포함한다.

data/processed/segment_personas_train_only.json은 graph_service.py가 없어도(즉 세그먼트
이름 없이 "세그먼트 N"으로만) 그래프 자체는 동작하도록 선택적으로 읽는 파일이지만, 이름이
안 채워지는 문제(요청으로 발견)를 겪은 뒤로는 정상 배포 시 항상 채워지도록 필수 목록에 넣었다.

재학습/재현(원본 로그 → 파이프라인 재실행)까지 필요하면 data/raw/, data/interim/als_events.csv도
별도로 받아야 한다 — 이 모듈의 범위 밖이다(reports/BACKEND_INTEGRATION_PLAN.md 참고).
"""

from __future__ import annotations

import uuid
import zipfile
from pathlib import Path

import gdown

GDRIVE_FILE_ID_ENV = "REC_SYSTEM_DATA_FILE_ID"

ROOT_DIR = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    "data/dashboard/recommend.db",
    "data/dashboard/products.csv",
    "data/dashboard/demo_users.csv",
    "data/processed/customer_segments_labeled_train_only.csv",
    "data/outputs/complementary/detail_cf.csv",
    "models/ALS/als_model.pkl",
    "data/processed/tri_graph_uidx2tidx_train.json",
    "data/processed/tri_graph_tidx2pidx.json",
    "data/processed/tri_graph_uidx2pidx.json",
    "data/processed/uidx_user_id_mapping.csv",
    "data/processed/tidx_product_id_mapping.csv",
    "data/processed/df_integrated_logs.csv",
    "data/outputs/eval/twiddler_accuracy.csv",
    "data/outputs/eval/twiddler_diversity.csv",
    "data/outputs/LightGCN/PRED_MAIN_RECOMMEND.csv",
    "data/outputs/LightGCN/lightgcn_test.csv",
    "data/processed/segment_personas_train_only.json",
]


def is_data_ready() -> bool:
    """앱 구동에 필요한 파일이 전부 있으면 True."""
    return all((ROOT_DIR / f).exists() for f in REQUIRED_FILES)


def ensure_data_downloaded(file_id: str) -> None:
    """필요한 파일이 이미 있으면 아무 것도 하지 않고, 없으면 GDrive에서 받아 리포 루트에 압축 해제한다.

    Streamlit Community Cloud는 여러 세션이 같은 프로세스/파일시스템을 공유해, 콜드스타트
    직후 두 세션이 동시에 진입하면 고정된 zip 경로를 두고 경쟁하는 문제(하나가 지운 파일을
    다른 하나가 또 지우려 해 FileNotFoundError)가 있었다(요청으로 재현). 세션마다 고유한
    임시 zip 경로를 써서 애초에 경쟁이 발생하지 않게 한다 — 동시에 두 세션이 진입하면
    다운로드가 중복될 뿐, 서로의 파일을 건드리지 않는다.
    """
    if is_data_ready():
        return

    zip_path = ROOT_DIR / f"rec-system-data-required-{uuid.uuid4().hex}.zip"
    try:
        gdown.download(id=file_id, output=str(zip_path), quiet=False)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(ROOT_DIR)
    finally:
        zip_path.unlink(missing_ok=True)
