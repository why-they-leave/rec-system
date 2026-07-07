"""
상품 카드 컴포넌트.
아이콘: 상품 타입별 이모지 + st.container(border=True) 네이티브 렌더링.
HTML/SVG를 사용하지 않아 st.markdown 파싱 문제 없음.
"""
import pandas as pd
import streamlit as st


# ── 상품 타입 → 이모지 매핑 ────────────────────────────────────────────────────
_PRODUCT_EMOJI: dict[str, str] = {
    # Electronics
    "ssd": "💾", "keyboard": "⌨️", "headphones": "🎧", "smartwatch": "⌚",
    "mouse": "🖱️", "speaker": "🔊", "monitor": "🖥️", "webcam": "📷",
    # Home & Kitchen
    "lamp": "💡", "coffee_maker": "☕", "vacuum": "🌀", "toaster": "🍞",
    "air_fryer": "💨", "blender": "🥤", "cookware": "🍳",
    # Beauty
    "lipstick": "💄", "conditioner": "🧴", "serum": "💧", "shampoo": "🧴",
    "sunscreen": "☀️", "moisturizer": "✨",
    # Sports
    "cycling_helmet": "⛑️", "tennis_racket": "🎾", "dumbbell": "🏋️",
    "water_bottle": "🥤", "yoga_mat": "🧘",
    # Fashion
    "t-shirt": "👕", "jeans": "👖", "socks": "🧦", "sneakers": "👟",
    "dress": "👗", "hoodie": "🧥", "jacket": "🧥",
    # Books
    "paperback": "📖", "hardcover": "📕", "e-book": "📱",
    # Toys
    "board_game": "♟️", "puzzle": "🧩", "building_blocks": "🧱",
    "action_figure": "⚡", "doll": "🧸",
}


def extract_color(name: str) -> str:
    """상품명 뒤에서 두 번째 단어를 CSS 색상명으로 추출."""
    parts = name.split()
    return parts[-2] if len(parts) >= 2 else "gray"


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


def _circle(color: str, emoji: str, size: int = 60) -> None:
    """색상 원형 배경 + 이모지. 단일 라인 <div> → 코드블록 파싱 문제 없음."""
    st.markdown(
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;background-color:{color};'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:{size // 2 - 2}px;margin:0 auto 6px auto;">{emoji}</div>',
        unsafe_allow_html=True,
    )


# ── Twiddler 순위 변동 배지 (카드 우상단) ────────────────────────────────────────
_DIRECTION_ICON: dict[str, str] = {"up": "▲", "down": "▼", "same": "–"}


def _corner_badge(direction: str, label: str, icon: str | None = "auto") -> None:
    """카드 우상단 순위 변동 배지. style.css의 .badge-up/.badge-down/.badge-same 재사용.

    icon="auto"면 direction에 맞는 화살표/마이너스 아이콘을 붙이고, None이면 아이콘 없이
    라벨만 표시(데모 "적용 전" 상태 — 방향 계산 없는 단순 순번 표기용).
    """
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
    emoji = _PRODUCT_EMOJI.get(product_type.lower().replace(" ", "_"), "🏷️")

    with st.container(border=True):
        if plain_rank_badge is not None:
            _corner_badge("same", f"{plain_rank_badge}위", icon=None)
        elif rank_delta is not None:
            _corner_badge(rank_delta["direction"], rank_delta["label"])

        _circle(color, emoji)
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
    emoji = _PRODUCT_EMOJI.get(product_type.lower().replace(" ", "_"), "🏷️")

    with st.container(border=True):
        col_icon, col_text = st.columns([1, 4])
        with col_icon:
            _circle(color, emoji, size=56)
        with col_text:
            st.write(f"**{item['name']}**")
            st.caption(f"{item['category']}  ·  $ {float(item['price_usd']):.2f}")
