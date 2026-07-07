"""
GDrive에 업로드된 데이터 번들을 내려받는 로컬 개발용 CLI.

실제 로직은 app/utils/data_bootstrap.py에 있다 — Streamlit Community Cloud 콜드스타트에서도
그 모듈을 그대로 재사용한다(app/main.py 참고). 리포를 새로 clone한 사람은 이 스크립트만
실행하면 `streamlit run app/main.py`를 바로 돌릴 수 있다.

파일 ID는 코드에 하드코딩하지 않고 환경변수 REC_SYSTEM_DATA_FILE_ID로만 받는다 —
GDrive 공유 링크(https://drive.google.com/file/d/<파일ID>/view)에서 <파일ID> 부분을 그대로 넘기면 된다.

Usage:
    REC_SYSTEM_DATA_FILE_ID=xxxxxxxx python scripts/download_data.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from utils.data_bootstrap import GDRIVE_FILE_ID_ENV, REQUIRED_FILES, ROOT_DIR, ensure_data_downloaded


def main() -> None:
    file_id = os.environ.get(GDRIVE_FILE_ID_ENV)
    if not file_id:
        raise SystemExit(
            f"{GDRIVE_FILE_ID_ENV} 환경변수가 비어 있다. GDrive 공유 링크"
            f"(https://drive.google.com/file/d/<파일ID>/view)의 파일 ID를 넘겨라. "
            f"예: {GDRIVE_FILE_ID_ENV}=xxxx python scripts/download_data.py"
        )

    ensure_data_downloaded(file_id)

    print("\n[OK] 압축 해제 완료:")
    for rel_path in REQUIRED_FILES:
        full_path = ROOT_DIR / rel_path
        status = f"{full_path.stat().st_size / 1e6:.1f} MB" if full_path.exists() else "누락!"
        print(f"  {rel_path}  ({status})")


if __name__ == "__main__":
    main()
