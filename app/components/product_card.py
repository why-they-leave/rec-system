"""
상품 카드 컴포넌트.
아이콘: 상품 타입별 라인아트 PNG(app/static/images/products/, 41종)를 상품 색상(CSS filter:
hue-rotate)으로 물들여 표시 — 원형 배지 없이 아이콘 자체가 상품명의 색상을 반영한다.
이미지가 없는 타입(신규 상품 등)은 이모지(🏷️)로 폴백한다.
HTML은 <img> 한 줄뿐이라 st.markdown 파싱 문제 없음.
"""
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


def _circle(color: str, product_type: str, size: int = 60) -> None:
    """상품 타입 아이콘을 상품 색상으로 물들여 표시(원형 배지 없음). 단일 라인 <div> →
    코드블록 파싱 문제 없음.

    filter: hue-rotate()로 아이콘의 파란 베이스 색조를 상품 색상 쪽으로 회전시킨다 — 명도·
    채도는 건드리지 않아 outline/채움 디테일이 그대로 유지된다(요청 반영: 원형 배지에 가두지
    않고 아이콘 자체가 상품명의 색상을 반영하도록 변경. product_icons.icon_color_filter 참고).
    """
    slug = icon_slug_for(product_type)
    if slug:
        filter_css = icon_color_filter(color)
        style = f"width:{size}px;height:{size}px;object-fit:contain;display:block;margin:0 auto 6px auto;"
        if filter_css:
            style += f"filter:{filter_css};"
        icon_html = f'<img src="{icon_url(slug)}" alt="{product_type}" style="{style}" />'
    else:
        icon_html = (
            f'<div style="width:{size}px;height:{size}px;display:flex;align-items:center;'
            f'justify-content:center;font-size:{size // 2 - 2}px;margin:0 auto 6px auto;">'
            f'{_FALLBACK_EMOJI}</div>'
        )
    st.markdown(icon_html, unsafe_allow_html=True)


# ── Twiddler 순위 변동 배지 (카드 우상단) ────────────────────────────────────────
# "new": 비교 대상(rank_before_map)은 있었지만 그 안에 없던 상품 — 직전 대비 새로 진입.
# 특히 새로고침 시뮬레이션에서 직전 라운드 top-10 밖에 있던 상품이 새로 올라오는 경우가
# 정상 시나리오라 자주 나타난다(요청으로 발견 — 이전엔 빈 배지로 표시돼 혼란스러웠음).
_DIRECTION_ICON: dict[str, str] = {"up": "▲", "down": "▼", "same": "–", "new": "🆕"}


def _corner_badge(direction: str | None, label: str | None, icon: str | None = "auto") -> None:
    """카드 우상단 순위 변동 배지. style.css의 .badge-up/.badge-down/.badge-same 재사용.

    icon="auto"면 direction에 맞는 화살표/마이너스 아이콘을 붙이고, None이면 아이콘 없이
    라벨만 표시(데모 "적용 전" 상태 — 방향 계산 없는 단순 순번 표기용).
    direction이 None이면 배지 없이 동일한 마크업(높이만 동일, 내용은 숨김)을 렌더링해,
    순위 변동 배지가 있는 카드와 없는 카드 간 원(circle) 시작 위치가 어긋나지 않게 한다.
    """
    if direction is None:
        st.markdown(
            '<div style="display:flex;justify-content:flex-end;margin-bottom:4px;">'
            '<span class="badge" style="visibility:hidden;">placeholder</span></div>',
            unsafe_allow_html=True,
        )
        return
    prefix = f"{_DIRECTION_ICON.get(direction, '')} " if icon == "auto" else ""
    st.markdown(
        f'<div style="display:flex;justify-content:flex-end;margin-bottom:4px;">'
        f'<span class="badge badge-{direction}">{prefix}{label}</span></div>',
        unsafe_allow_html=True,
    )


def render_product_card(
    item: pd.Series,
    rank: int,
    badge: str = None,
    score: float = None,
    rank_delta: dict | None = None,
    plain_rank_badge: int | None = None,
    rank_before: int | None = None,
) -> None:
    """상품 카드 렌더링.

    score: 추천 점수(0~1 확률 등).
    rank_delta: get_rank_delta()가 반환한 {"direction","label"} — 우상단에 방향 배지로 표시
                (프로덕션 및 데모 "적용 후" 상태용).
    plain_rank_badge: 값이 있으면 우상단에 방향 계산 없는 회색 "N위" 배지만 표시
                      (데모 "적용 전" 상태용). rank_delta보다 우선한다.
    rank_before: Twiddler 적용 전 순위. score와 함께 주어지면 "전 순위 N위 · 추천 점수 X.XXX"
                 서브텍스트로 표시.
    """
    color = extract_color(item["name"])
    product_type = extract_product_type(item["name"])

    with st.container(border=True):
        if plain_rank_badge is not None:
            _corner_badge("same", f"{plain_rank_badge}위", icon=None)
        elif rank_delta is not None:
            _corner_badge(rank_delta["direction"], rank_delta["label"])
        else:
            _corner_badge(None, None)  # 배지 없는 카드도 동일 높이 확보 → 원(circle) 위치 정렬

        _circle(color, product_type)
        st.write(f"**{item['name']}**")
        st.caption(item['category'])
        st.write(f"**$ {float(item['price_usd']):.2f}**")

        if rank_before is not None and score is not None:
            st.markdown(
                f'<div style="font-size:11px;color:var(--text-muted);margin-top:2px;">'
                f'전 순위 {rank_before}위 · 추천 점수 {score:.3f}</div>',
                unsafe_allow_html=True,
            )
        elif score is not None:
            st.caption(f"추천 점수: {score:.3f}")
        elif rank_delta is None and plain_rank_badge is None:
            st.caption(f"★ rank: {rank}")
        _badge_widget(badge)  # 항상 호출 — None이면 동일 높이 플레이스홀더


def render_current_product_card(item: pd.Series) -> None:
    """상세 페이지용 현재 상품 가로형 강조 카드."""
    color = extract_color(item["name"])
    product_type = extract_product_type(item["name"])

    with st.container(border=True):
        col_icon, col_text = st.columns([1, 4])
        with col_icon:
            _circle(color, product_type, size=56)
        with col_text:
            st.write(f"**{item['name']}**")
            st.caption(f"{item['category']}  ·  $ {float(item['price_usd']):.2f}")
