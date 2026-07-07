"""
추천 시스템 서빙 API — Streamlit 앱(app/)이 호출하는 유일한 진입점.
모델 학습 코드는 src/에 있고, 여기(backend/)에는 아티팩트를 로드해 응답하는
API 코드만 둔다 (reports/BACKEND_INTEGRATION_PLAN.md 참고).

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
