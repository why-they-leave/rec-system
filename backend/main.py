"""
추천 시스템 서빙 API — 외부 클라이언트(로컬 개발 시 curl/Postman, 또는 다른 서비스)를 위한 HTTP 진입점.
Streamlit 앱(app/)은 이 서버를 거치지 않고 backend/api/core.py의 함수를 Streamlit 프로세스
안에서 직접 호출한다(app/utils/data_loader.py 참고) — Streamlit Community Cloud처럼 프로세스를
하나만 띄울 수 있는 환경에서도 앱이 동작하게 하기 위함(reports/BACKEND_INTEGRATION_PLAN.md 참고).
모델 학습 코드는 src/에 있고, 여기(backend/)에는 아티팩트를 로드해 응답하는
API 코드만 둔다.

Usage:
    uvicorn backend.main:app --reload
"""

from fastapi import FastAPI

from backend.api.routers import recommend_detail, recommend_main

app = FastAPI(title="추천 시스템 서빙 API")

app.include_router(recommend_main.router)
app.include_router(recommend_detail.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
