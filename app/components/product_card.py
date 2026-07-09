"""
상품 카드 컴포넌트.
아이콘: 상품 타입별 라인아트 PNG(app/static/images/products/, 41종)를 상품 색상(CSS filter:
hue-rotate)으로 물들여 표시 — 원형 배지 없이 아이콘 자체가 상품명의 색상을 반영한다.
이미지가 없는 타입(신규 상품 등)은 이모지(🏷️)로 폴백한다.
HTML은 <img> 한 줄뿐이라 st.markdown 파싱 문제 없음.
"""

from collections.abc import Callable

import pandas as pd
import streamlit as st
from utils.category_emoji import extract_color
from utils.product_icons import icon_color_filter, icon_slug_for, icon_url

_FALLBACK_EMOJI = "🏷️"


def extract_product_type(name: str) -> str:
    """상품명에서 종류 추출 — 색상·번호 제외."""
    parts = name.split()
    return " ".join(parts[:-2]) if len(parts) >= 3 else name


def _badge_widget(badge: str | None) -> None:
    """배지 내용에 따라 native Streamlit 위젯으로 표시.
    badge가 None이면 동일 높이의 투명 플레이스홀더를 렌더링해 카드 높이를 고정.
    """
    if badge is None:
        st.markdown('<div style="height:38px"></div>', unsafe_allow_html=True)
    elif "공통" in badge:
        st.success(badge, icon=None)
    elif "▲" in badge:
        st.success(badge, icon=None)
    elif "▼" in badge:
        st.error(badge, icon=None)
    elif "➡" in badge:
        st.info(badge, icon=None)
    else:
        st.success(badge, icon=None)


def _icon_img_html(color: str, product_type: str, size: int) -> str:
    """상품 타입 아이콘 <img> 태그(또는 폴백 이모지 <div>) HTML 문자열만 반환(마크다운 출력 X).

    filter: hue-rotate()로 아이콘의 파란 베이스 색조를 상품 색상 쪽으로 회전시킨다 — 명도·
    채도는 건드리지 않아 outline/채움 디테일이 그대로 유지된다(product_icons.icon_color_filter 참고).
    """
    slug = icon_slug_for(product_type)
    if slug:
        filter_css = icon_color_filter(color)
        style = f"width:{size}px;height:{size}px;object-fit:contain;display:block;"
        if filter_css:
            style += f"filter:{filter_css};"
        return f'<img src="{icon_url(slug)}" alt="{product_type}" style="{style}" />'
    return (
        f'<div style="width:{size}px;height:{size}px;display:flex;align-items:center;'
        f'justify-content:center;font-size:{size // 2 - 2}px;">{_FALLBACK_EMOJI}</div>'
    )


def product_icon_html(name: str, size: int = 40) -> str:
    """상품명만으로 카드와 동일한 색상 반영 아이콘 <img> HTML을 반환(요청 반영: 순위
    요약 리스트에서도 카드와 같은 아이콘을 쓰고 싶어 카드 내부 헬퍼를 공개 함수로 노출).
    """
    return _icon_img_html(extract_color(name), extract_product_type(name), size)


def _circle(color: str, product_type: str, size: int = 96) -> None:
    """단일 아이콘만 중앙에 표시(원형 배지 없음) — 상세 페이지 가로형 카드(render_current_product_card) 전용."""
    st.markdown(
        f'<div style="margin:0 auto;width:{size}px;">{_icon_img_html(color, product_type, size)}</div>',
        unsafe_allow_html=True,
    )


def _photo_panel(color: str, product_type: str, category: str, size: int = 160) -> None:
    """사진 패널 — 아이콘을 중앙에 크게 둔다.

    좌상단 카테고리 아이콘 배지는 16px로 축소하면 실루엣이 뭉개져 알아볼 수 없다는
    이유로 제거했다(요청 반영) — 카테고리 정보는 바로 아래 브랜드 라인 텍스트로 이미
    표시되므로 정보 손실은 없다.
    """
    st.markdown(
        f'<div class="product-photo-panel">{_icon_img_html(color, product_type, size)}</div>',
        unsafe_allow_html=True,
    )


# ── Twiddler 순위 변동 배지 (카드 우상단) ────────────────────────────────────────
# "new": 비교 대상(rank_before_map)은 있었지만 그 안에 없던 상품 — 직전 대비 새로 진입.
# 특히 새로고침 시뮬레이션에서 직전 라운드 top-10 밖에 있던 상품이 새로 올라오는 경우가
# 정상 시나리오라 자주 나타난다(요청으로 발견 — 이전엔 빈 배지로 표시돼 혼란스러웠음).
_DIRECTION_ICON: dict[str, str] = {"up": "▲", "down": "▼", "same": "–", "new": "🆕"}


def render_product_card(
    item: pd.Series,
    rank: int,
    badge: str = None,
    score: float = None,
    rank_delta: dict | None = None,
    plain_rank_badge: int | None = None,
    rank_before: int | None = None,
    footer: Callable[[], None] | None = None,
) -> None:
    """상품 카드 렌더링 — 커머스 앱(오늘의집 등) 카드 참고 레이아웃(요청 반영).

    사진 패널(카테고리 아이콘 배지) → 브랜드 라인(카테고리·타입) → 제목 → 매칭율/가격 →
    순위 → 순위변동 배지 순으로, 참고 이미지의 "할인율·별점·리뷰·배송·쿠폰" 자리를 전부
    우리가 실제로 가진 데이터(추천 점수→매칭율, 순위, 순위변동)로만 채운다 — 없는 데이터는
    만들어내지 않는다.

    score: 추천 점수(0~1) — "매칭 N%"로 변환해 가격 옆에 표시.
    rank_delta: get_rank_delta()가 반환한 {"direction","label"} — 순위변동 배지로 표시
                (프로덕션 및 데모 "적용 후" 상태용).
    plain_rank_badge: 값이 있으면 순위변동 계산 없이 단순 "N위"만 표시(데모 "적용 전" 상태용).
    rank_before: Twiddler 적용 전 순위. 있으면 순위 줄에 "전 순위 N위"를 함께 표시.
    """
    color = extract_color(item["name"])
    product_type = extract_product_type(item["name"])

    with st.container(border=True):
        _photo_panel(color, product_type, item["category"])

        st.markdown(
            f'<div class="product-brand-line">{item["category"]} · {product_type}</div>'
            f'<div class="product-title-line">{item["name"]}</div>',
            unsafe_allow_html=True,
        )

        match_html = (
            f'<span class="product-match-pill">매칭 {score * 100:.0f}%</span>'
            if score is not None
            else ""
        )
        st.markdown(
            f'<div class="product-match-price-row">{match_html}'
            f'<span class="product-price-line">$ {float(item["price_usd"]):.2f}</span></div>',
            unsafe_allow_html=True,
        )

        current_rank = plain_rank_badge if plain_rank_badge is not None else rank
        rank_text = f"🏆 {current_rank}위"
        if rank_before is not None:
            rank_text += f" · 전 순위 {rank_before}위"
        st.markdown(f'<div class="product-rating-row">{rank_text}</div>', unsafe_allow_html=True)

        if rank_delta is not None:
            st.markdown(
                f'<span class="badge badge-{rank_delta["direction"]}">'
                f'{_DIRECTION_ICON.get(rank_delta["direction"], "")} {rank_delta["label"]}</span>',
                unsafe_allow_html=True,
            )
        _badge_widget(badge)  # 공통 추천 배지 — None이면 동일 높이 플레이스홀더
        if footer:
            # "연관 상품 보기" 버튼을 카드 바깥(별도 st.button)이 아니라 카드 테두리
            # 안쪽에 넣어달라는 요청 반영 — 호출부(main.py)가 버튼 렌더링+클릭 처리를
            # 콜백으로 넘기면 여기서 container(border=True) 블록이 닫히기 전에 호출한다.
            footer()


def render_current_product_card(item: pd.Series) -> None:
    """상세 페이지용 현재 상품 가로형 강조 카드."""
    color = extract_color(item["name"])
    product_type = extract_product_type(item["name"])

    with st.container(border=True):
        col_icon, col_text = st.columns([1, 4])
        with col_icon:
            _circle(color, product_type, size=88)
        with col_text:
            st.write(f"**{item['name']}**")
            st.caption(f"{item['category']}  ·  $ {float(item['price_usd']):.2f}")
