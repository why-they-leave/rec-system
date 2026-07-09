"""상품 종류별(41종) 라인아트 아이콘 경로/색상 유틸.

app/static/images/products/의 PNG 파일명은 상품 타입 slug(snake_case, 예: "coffee_maker")와
동일하게 정리되어 있다(app/components/product_card.py, app/components/user_graph.py 공용).
41장 전부 동일한 파란 톤(짙은 남색 outline + 옅은 하늘색 채움) 팔레트로 생성돼 있어,
CSS filter: hue-rotate()로 그 베이스 색조를 상품 색상의 색조로 회전시키는 방식으로 재채색한다
(brightness(0) 계열로 단색 실루엣화하면 outline/채움의 명도 대비, 텍스트 줄 같은 디테일이
통째로 사라지는 문제가 있었다 — hue-rotate는 명도·채도를 건드리지 않아 디테일이 유지된다).
"""
from __future__ import annotations

import base64
import colorsys
from functools import lru_cache
from pathlib import Path

from matplotlib.colors import CSS4_COLORS, to_rgb

_ICONS_DIR = Path(__file__).resolve().parent.parent / "static" / "images" / "products"
ICONS_URL_DIR = "app/static/images/products"  # enableStaticServing 기준 상대 URL(main.py의 로고 경로 관례와 동일)

# 아이콘 세트의 공통 베이스 색조(hue, degree). 5개 표본 아이콘(ssd/lamp/t-shirt/vacuum/cookware)의
# 채도 0.2 이상 픽셀 평균 hue ≈ 204°(표준편차 ~13°)로, 41장이 사실상 단일 색조라 하나의
# 기준값으로 고정해도 충분하다(재생성 스크립트가 아니라 하드코딩 상수로 관리).
_BASE_HUE_DEG = 204.0


def icon_color_filter(color_name: str) -> str:
    """상품 색상명(CSS 색상 키워드, 예: "DarkMagenta")을 아이콘에 입힐 CSS filter 값으로 변환.

    hue-rotate만 적용해 원본 라인아트의 명도/디테일은 그대로 두고 색조만 목표 색상 쪽으로
    돌린다. CSS4_COLORS에 없는 이름이면 빈 문자열(필터 없음 → 원본 파란 톤 유지)을 반환한다.
    """
    rgb_hex = CSS4_COLORS.get(color_name.strip().lower())
    if rgb_hex is None:
        return ""
    h, _l, _s = colorsys.rgb_to_hls(*to_rgb(rgb_hex))
    delta = (h * 360 - _BASE_HUE_DEG) % 360
    return f"hue-rotate({delta:.0f}deg)"


@lru_cache(maxsize=1)
def _available_slugs() -> frozenset[str]:
    return frozenset(p.stem for p in _ICONS_DIR.glob("*.png"))


def to_icon_slug(product_type: str) -> str:
    """'Coffee Maker' → 'coffee_maker' (파일명 규칙과 동일한 slug로 정규화)."""
    return product_type.strip().lower().replace(" ", "_")


def icon_slug_for(product_type: str) -> str | None:
    """상품 타입에 대응하는 아이콘 파일이 있으면 slug, 없으면 None(이모지 폴백 필요)."""
    slug = to_icon_slug(product_type)
    return slug if slug in _available_slugs() else None


def icon_url(slug: str) -> str:
    """<img src="">에 쓸 정적 서빙 URL(상대 경로)."""
    return f"{ICONS_URL_DIR}/{slug}.png"


@lru_cache(maxsize=64)
def icon_data_uri(slug: str) -> str:
    """SVG <image>에 임베드할 base64 PNG data URI(pyvis처럼 정적 URL을 못 쓰는 곳용)."""
    data = (_ICONS_DIR / f"{slug}.png").read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"
