"""
Twiddler(페르소나 가중치 재랭킹) — 순수 재랭킹 함수 모음.

이 모듈은 I/O를 하지 않는다(페르소나 조회, 카테고리 매핑, 노출 이력 조회는
backend/api/services/*가 담당하고, 그 결과를 이미 계산된 값으로 이 함수들에 주입한다).
단, 하이퍼파라미터는 configs/twiddler/params.yaml에서 로드해 모듈 상수로 노출한다
(코드 수정 없이 값만 바꿔 재실험할 수 있도록 — configs/als/params.yaml과 동일한 패턴).

파이프라인: 원본 점수 -> Rule 1(유저 개인화 카테고리 가중치) -> Rule 2(유저 개인화 노출 이력 패널티)
         -> 재정렬 -> top_k 절단

Rule 1/2의 affinity/alpha/decay는 세그먼트 평균이 아니라 유저 단위 개인화 값을 쓴다
(backend/api/services/persona_service.py::get_user_affinity/get_user_alpha/get_user_decay).

과거 Rule 3(저노출 상품 최소 노출 보장)은 제거했다 — 유저 페르소나와 무관한 아이템 공급측
형평성 장치라 연구질문과 무관했고, 실측 결과 실효성도 낮았다(configs/twiddler/params.yaml 참고).
"""

from __future__ import annotations

from pathlib import Path

import yaml

_PARAMS_PATH = Path(__file__).resolve().parents[3] / "configs" / "twiddler" / "params.yaml"


def _load_params(path: Path = _PARAMS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_params = _load_params()

NUM_CATEGORIES = _params["num_categories"]  # data/dashboard/products.csv 기준 카테고리 종류 수 (균등분포 baseline用)

BASE_ALPHA = _params["base_alpha"]
MULTIPLIER_FLOOR = _params["multiplier_floor"]
MULTIPLIER_CEILING = _params["multiplier_ceiling"]
EXPOSURE_DECAY = _params["exposure_decay"]
EXPLORATION_DECAY_MIN = _params["exploration_decay_min"]
EXPLORATION_DECAY_MAX = _params["exploration_decay_max"]
POOL_MULTIPLIER = _params["pool_multiplier"]


def apply_persona_weight(
    items: list[dict],
    id_key: str,
    category_map: dict[int, str],
    affinity: dict[str, float],
    alpha: float,
) -> list[dict]:
    """카테고리가 페르소나 선호와 맞는 아이템의 점수를 가중치만큼 곱해 조정한다.

    배율은 [MULTIPLIER_FLOOR, MULTIPLIER_CEILING] 범위로 clip한다 — 상한이 없으면
    로열티가 매우 높은 유저의 alpha가 세그먼트 평균보다 훨씬 강하게 한 카테고리로
    쏠려 다양성을 해친다(검증됨, notebooks/20260708_ML_twiddler_final_design.ipynb).
    """
    for item in items:
        category = category_map.get(item[id_key])
        deviation = affinity.get(category, 0.0)
        multiplier = max(MULTIPLIER_FLOOR, min(MULTIPLIER_CEILING, 1 + alpha * deviation))
        item["score"] = item["score"] * multiplier
    return items


def apply_exposure_penalty(
    items: list[dict],
    id_key: str,
    exposure_counts: dict[int, float],
    decay: float,
) -> list[dict]:
    """최근 자주 노출된 아이템일수록 점수를 감쇠시켜 새로고침 시 다양성을 준다.

    decay는 유저 개인화 값(persona_service.get_user_decay)을 그대로 받는다 —
    전역 EXPOSURE_DECAY를 여기서 직접 참조하지 않는다.
    """
    for item in items:
        count = exposure_counts.get(item[id_key], 0.0)
        item["score"] = item["score"] * (decay ** count)
    return items


def rerank(
    items: list[dict],
    *,
    id_key: str = "item_id",
    category_map: dict[int, str],
    affinity: dict[str, float],
    alpha: float,
    exposure_counts: dict[int, float] | None = None,
    decay: float = EXPOSURE_DECAY,
    top_k: int,
) -> list[dict]:
    """유저 개인화 affinity/alpha/decay 기준으로 재랭킹한 뒤 rank를 재부여하고 top_k로 절단해 반환한다."""
    items = apply_persona_weight(items, id_key, category_map, affinity, alpha)

    if exposure_counts:
        items = apply_exposure_penalty(items, id_key, exposure_counts, decay)

    items = sorted(items, key=lambda it: it["score"], reverse=True)
    items = items[:top_k]
    for rank, item in enumerate(items, start=1):
        item["rank"] = rank
    return items
