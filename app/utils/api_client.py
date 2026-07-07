"""
백엔드 추천 서빙 API(backend/main.py, FastAPI) 호출 래퍼.

BACKEND_API_URL 환경변수로 API 주소를 바꿀 수 있다(기본값: 로컬 uvicorn 개발 서버).
연결 실패/타임아웃은 BackendUnavailableError로 통일해, app/main.py가 한 곳에서 처리할 수 있게 한다.
"""

import os

import requests

BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://127.0.0.1:8000")
_TIMEOUT_SECONDS = 5


class BackendUnavailableError(Exception):
    """백엔드 API에 연결할 수 없거나 응답이 지연될 때 발생."""


def _get(path: str, params: dict) -> dict:
    try:
        response = requests.get(f"{BACKEND_API_URL}{path}", params=params, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise BackendUnavailableError(f"백엔드 API 호출 실패: {path} ({e})") from e


def get_main_recommendations(
    user_id: int,
    model_type: str,
    twiddler: str = "before",
    top_n: int = 10,
    graph_type: str = "tripartite",
) -> dict:
    """GET /recommend/main — {"status", "message", "items": [...]} 반환.

    graph_type은 model_type="LightGCN"일 때만 의미를 가진다("bipartite" | "tripartite").
    """
    params = {"user_id": user_id, "model_type": model_type, "twiddler": twiddler, "top_n": top_n}
    if model_type == "LightGCN":
        params["graph_type"] = graph_type
    return _get("/recommend/main", params)


def get_detail_recommendations(
    item_id: int,
    top_n: int = 8,
    user_id: int | None = None,
    twiddler: str = "before",
) -> dict:
    """GET /recommend/detail(보완재) — {"status", "message", "items": [...]} 반환."""
    params = {"item_id": item_id, "top_n": top_n, "twiddler": twiddler}
    if user_id is not None:
        params["user_id"] = user_id
    return _get("/recommend/detail", params)
