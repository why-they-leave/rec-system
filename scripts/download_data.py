"""
GDrive에 업로드된 데이터 번들을 내려받아 리포 루트에 압축 해제한다.

이 리포의 실제 추천 데이터(als_model.pkl, customer_segments_labeled_train_only.csv 등)는
용량이 커서(수십 MB) git에 커밋하지 않는다(.gitignore 참고) — 대신 GDrive에 올려두고
이 스크립트로 받는다. 리포를 새로 clone한 사람은 이 스크립트만 실행하면
`streamlit run app/main.py` + `uvicorn backend.main:app`을 바로 돌릴 수 있다.

파일 ID는 코드에 하드코딩하지 않고 환경변수 REC_SYSTEM_DATA_FILE_ID로만 받는다 —
GDrive 공유 링크(https://drive.google.com/file/d/<파일ID>/view)에서 <파일ID> 부분을 그대로 넘기면 된다.

번들에 포함된 파일(= 앱을 바로 구동하는 데 필요한 최소 집합):
    data/dashboard/recommend.db                                   (app 카탈로그: products/demo_users)
    data/dashboard/products.csv                                   (backend catalog_service가 직접 읽음)
    data/dashboard/demo_users.csv                                 (CSV 폴백용)
    data/processed/customer_segments_labeled_train_only.csv       (persona_service)
    data/outputs/complementary/detail_cf.csv                      (complementary_service)
    models/ALS/als_model.pkl                                      (als_service)

재학습/재현(원본 로그 → 파이프라인 재실행)까지 필요하면 data/raw/, data/interim/als_events.csv,
data/processed/df_integrated_logs.csv도 별도로 받아야 한다 — 이 스크립트의 범위 밖이다
(reports/BACKEND_INTEGRATION_PLAN.md 참고).

Usage:
    REC_SYSTEM_DATA_FILE_ID=xxxxxxxx python scripts/download_data.py
"""

import os
import zipfile
from pathlib import Path

import gdown

GDRIVE_FILE_ID_ENV = "REC_SYSTEM_DATA_FILE_ID"

ROOT_DIR = Path(__file__).parent.parent
ZIP_PATH = ROOT_DIR / "rec-system-data-required.zip"

_EXPECTED_FILES = [
    "data/dashboard/recommend.db",
    "data/dashboard/products.csv",
    "data/dashboard/demo_users.csv",
    "data/processed/customer_segments_labeled_train_only.csv",
    "data/outputs/complementary/detail_cf.csv",
    "models/ALS/als_model.pkl",
]


def main() -> None:
    file_id = os.environ.get(GDRIVE_FILE_ID_ENV)
    if not file_id:
        raise SystemExit(
            f"{GDRIVE_FILE_ID_ENV} 환경변수가 비어 있다. GDrive 공유 링크"
            f"(https://drive.google.com/file/d/<파일ID>/view)의 파일 ID를 넘겨라. "
            f"예: {GDRIVE_FILE_ID_ENV}=xxxx python scripts/download_data.py"
        )

    gdown.download(id=file_id, output=str(ZIP_PATH), quiet=False)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(ROOT_DIR)
    ZIP_PATH.unlink()

    print("\n[OK] 압축 해제 완료:")
    for rel_path in _EXPECTED_FILES:
        full_path = ROOT_DIR / rel_path
        status = f"{full_path.stat().st_size / 1e6:.1f} MB" if full_path.exists() else "누락!"
        print(f"  {rel_path}  ({status})")


if __name__ == "__main__":
    main()
