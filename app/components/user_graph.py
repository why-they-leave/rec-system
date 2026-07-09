"""
유저 중심 추천 근거 서브그래프(유저→상품→세그먼트) 시각화 컴포넌트.

데이터 로딩/조합은 전부 app/utils/data_loader.py(→backend.api.core→graph_service)를 거치고,
여기서는 pyvis 스타일링과 Streamlit 렌더링만 담당한다(CLAUDE.md 역할 분리: 백엔드=데이터
로딩, 컴포넌트=순수 렌더링). 상품 이름/카테고리 라벨은 이미 캐시된 load_products()를 재사용해
그래프 서비스가 카탈로그 파일을 중복으로 알 필요가 없게 한다.
"""

from __future__ import annotations

import base64

import pandas as pd
import streamlit as st
from components.user_selector import PERSONA_KO
from pyvis.network import Network
from utils.category_emoji import extract_color, get_product_emoji, product_type_from_name
from utils.data_loader import get_user_subgraph, load_products
from utils.product_icons import icon_color_filter, icon_data_uri, icon_slug_for

# ── 레이아웃 상수 ────────────────────────────────────────────────────────
_GRAPH_HEIGHT_PX = 600
# pyvis 템플릿은 빈 <h1>(heading 미사용) 이 두 번 중복 렌더돼 카드 위에 여백을 만들고,
# 이 여백만큼 남는 <body> 영역이 #mynetwork 테두리 바깥으로 흰 배경만 튀어나와 보였다
# (요청으로 발견 — _patch_iframe_background에서 그 <h1>들을 숨기므로 더 이상 여유값이
# 필요 없다. 스크롤바 방지용 최소값만 남긴다).
_IFRAME_HEIGHT_PX = _GRAPH_HEIGHT_PX + 4

# ── 색상 상수 ────────────────────────────────────────────────────────────
# 그래프 자체 배경은 흰색 유지 — 페이지 배경(#f8fafc)과 맞추면 노드/엣지 색상 대비가
# 떨어져 가독성이 나빠진다(요청으로 확인). 대신 iframe 안쪽(카드 wrapper/body)만
# 이 값으로 통일해 카드 내부에서 흰색 얼룩이 지지 않게 한다(_patch_iframe_background 참고).
_BG_COLOR = "#ffffff"
_FONT_COLOR = "#1a1a1a"
# config.toml primaryColor(#6366f1)는 원 배경 위에 이모지를 얹으면 배경도 이모지(👤 실루엣은
# 짙은 회색/검정)도 둘 다 어두워 잘 안 보였다(요청으로 발견) — 같은 색조 계열에서 한 단계
# 밝은 톤(indigo-400)으로 바꿔 대비를 확보했다.
_COLOR_USER = "#818cf8"
_COLOR_PRODUCT_PURCHASED = "#22c55e"
_COLOR_PRODUCT_VIEWED = "#94a3b8"
_COLOR_PRODUCT_HOP2 = "#cbd5e1"  # 2홉 확장 상품(유저 본인 행동 아님) — 더 옅은 색
_COLOR_SEGMENT = "#f59e0b"
_COLOR_SEGMENT_OWN = "#ef4444"  # 유저 본인 세그먼트 강조

# 🧑(사람, 기본 노란빛 피부톤)로 — 👤(bust in silhouette)는 대부분의 이모지 폰트에서
# 검정/짙은 회색 실루엣이라 배경이 밝아져도 아이콘 자체가 여전히 어두워 대비가 부족했다.
_USER_EMOJI = "🧑"

# ── 크기/두께 상수 ───────────────────────────────────────────────────────
_SIZE_USER = 32
_SIZE_PRODUCT_HOP1 = 22
_SIZE_PRODUCT_HOP2 = 14
_SIZE_SEGMENT = 26
_EDGE_WIDTH_PURCHASED = 3
_EDGE_WIDTH_VIEWED = 1
_LIFT_WIDTH_MIN = 1.0
_LIFT_WIDTH_MAX = 8.0
_LIFT_WIDTH_SCALE = 1.5

# 상품 노드는 항상 보이는 라벨(상품명 텍스트)을 없애고 title(hover 툴팁)로만 노출한다 —
# 노드 수가 많아(hop1 최대 12개 + hop2 확장분) 상품명을 전부 상시 표시하면 라벨끼리, 또는
# 라벨과 엣지/화살촉이 겹쳐 그래프가 읽기 어려워졌다(요청으로 발견). pyvis의 add_node는
# falsy label(빈 문자열)을 노드 id로 폴백시키므로 공백 한 칸으로 사실상 라벨을 비운다.
_HIDDEN_LABEL = " "

# 이모지를 색상 원 위에 그린 SVG를 data URI로 미리 렌더링해 pyvis shape="circularImage"에 쓴다.
# circularImage는 "라벨 바깥" 계열(dot/diamond 등과 동일)이라 size 그대로 균일한 크기를 유지하고
# (라벨 길이에 흔들리지 않음), label(상품명)은 이미지 아래 별도로 렌더진다 — shape="circle"(라벨
# 안쪽 계열)은 상품명 길이에 따라 노드 크기가 제각각이 되는 문제가 있어서(전 요청에서 발견)
# 이 방식으로 대체했다.
_EMOJI_IMAGE_SIZE_PX = 96


def _emoji_circle_image(emoji: str, bg_color: str) -> str:
    """이모지 + 색상 원 배경을 SVG로 그려 base64 data URI로 반환.

    사람 노드(🧑)처럼 상품 아이콘 이미지가 없는 노드, 또는 아이콘이 없는 상품 타입의 폴백용.
    """
    r = _EMOJI_IMAGE_SIZE_PX / 2
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_EMOJI_IMAGE_SIZE_PX}" '
        f'height="{_EMOJI_IMAGE_SIZE_PX}">'
        f'<circle cx="{r}" cy="{r}" r="{r - 2}" fill="{bg_color}"/>'
        f'<text x="{r}" y="{r}" font-size="{r * 1.1:.0f}" text-anchor="middle" '
        f'dominant-baseline="central">{emoji}</text>'
        "</svg>"
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _icon_circle_image(icon_slug: str, bg_color: str, product_color: str | None = None) -> str:
    """상품 타입 아이콘(PNG) + 색상 원 배경을 SVG로 그려 base64 data URI로 반환.

    바깥 원(bg_color)은 그대로 유지 — 상품 자체 색상이 아니라 구매/조회/hop2 같은 노드
    상태를 나타낸다(app/components/user_graph.py의 _COLOR_PRODUCT_* 참고). 아이콘 자체는
    product_color가 주어지면 filter: hue-rotate()로 상품명의 색상을 반영한다(요청 반영 —
    원은 상태색 그대로, 아이콘만 상품 색상). hue-rotate는 명도/채도를 안 건드려 아이콘
    내부 디테일(outline/채움 대비)이 그대로 유지된다(app/components/product_card.py의
    _circle()에서 brightness(0) 계열 필터 시도 시 디테일 소실 확인, 그래서 hue-rotate 채택).
    """
    r = _EMOJI_IMAGE_SIZE_PX / 2
    plate_r = r * 0.86
    icon_r = r * 0.8
    icon_size = icon_r * 2
    icon_offset = r - icon_r
    filter_css = icon_color_filter(product_color) if product_color else ""
    filter_attr = f' style="filter:{filter_css};"' if filter_css else ""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_EMOJI_IMAGE_SIZE_PX}" '
        f'height="{_EMOJI_IMAGE_SIZE_PX}">'
        f'<circle cx="{r}" cy="{r}" r="{r - 2}" fill="{bg_color}"/>'
        f'<circle cx="{r}" cy="{r}" r="{plate_r:.1f}" fill="#ffffff"/>'
        f'<image href="{icon_data_uri(icon_slug)}" x="{icon_offset:.1f}" y="{icon_offset:.1f}" '
        f'width="{icon_size:.1f}" height="{icon_size:.1f}" preserveAspectRatio="xMidYMid meet"{filter_attr}/>'
        "</svg>"
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


# ── 범례 — 그래프 우측에 고정 배치되는 세로형 박스(요청 반영: 하단 텍스트 한 줄은 눈에
# 안 띄어서 색상 스와치 + 박스 형태로 변경). render_user_graph()에서 st.columns로 그래프
# 옆(우측)에 배치한다. 폰트는 이전(0.85rem)보다 키운 1.1rem 유지.
_LEGEND_ITEMS: list[tuple[str, str, str]] = [
    # (스와치 문자, 색상, 라벨)
    ("●", _COLOR_USER, "유저 (중심)"),
    ("●", _COLOR_PRODUCT_PURCHASED, "구매 상품"),
    ("●", _COLOR_PRODUCT_VIEWED, "조회/장바구니 상품"),
    ("●", _COLOR_PRODUCT_HOP2, "세그먼트 연관 인기 상품 (2홉)"),
    ("◆", _COLOR_SEGMENT, "연관 세그먼트"),
    ("◆", _COLOR_SEGMENT_OWN, "유저 본인 세그먼트"),
]
_LEGEND_ROWS_HTML = "".join(
    f'<div style="margin-bottom:0.35rem;">'
    f'<span style="color:{color}; font-size:1.3em; vertical-align:middle;">{swatch}</span> '
    f'<span style="vertical-align:middle;">{label}</span></div>'
    for swatch, color, label in _LEGEND_ITEMS
)
_LEGEND_HTML = f"""
<div style="font-size:1.1rem; line-height:1.5; border:1px solid #e2e8f0; border-radius:10px;
            padding:0.9rem 1rem; background:{_BG_COLOR};">
  <div style="font-weight:600; margin-bottom:0.5rem;">범례</div>
  {_LEGEND_ROWS_HTML}
  <hr style="margin:0.6rem 0; border-color:#e2e8f0;">
  <div style="font-size:0.9em; color:#475569; line-height:1.6;">
    실선(굵음) = 구매<br>점선 = 조회 등<br>세그먼트 엣지 굵기/라벨(lift) = 연관 강도(클수록 강함)
  </div>
</div>
"""


def _lift_edge_width(lift: float) -> float:
    """lift 값을 pyvis 엣지 두께로 변환(값이 클수록 굵게, 상하한 clip)."""
    width = _LIFT_WIDTH_MIN + lift * _LIFT_WIDTH_SCALE
    return max(_LIFT_WIDTH_MIN, min(_LIFT_WIDTH_MAX, width))


def _build_network(graph: dict, products_df: pd.DataFrame) -> Network:
    """그래프 dict(노드/엣지)를 pyvis Network로 변환하고 노드/엣지 스타일을 입힌다."""
    product_lookup = products_df.set_index("item_id")[["name", "category", "price_usd"]].to_dict(
        "index"
    )

    net = Network(
        height=f"{_GRAPH_HEIGHT_PX}px",
        width="100%",
        bgcolor=_BG_COLOR,
        font_color=_FONT_COLOR,
        directed=True,
        notebook=False,
        cdn_resources="in_line",  # HTML에 vis-network 리소스를 인라인 임베드
        # → lib/ 폴더를 디스크에 따로 쓰지 않음(읽기전용에 가까운
        # 배포 환경에서도 안전, 임시파일 불필요).
    )

    for node in graph["nodes"]:
        if node["node_type"] == "user":
            # 상품 노드와 동일하게 색상 원 위에 아이콘(사람 이모지)을 그린 circularImage로 —
            # 유저 노드도 한눈에 "이게 중심 유저"라고 알아볼 수 있게 한다(요청 반영).
            net.add_node(
                node["node_id"],
                label=node["label"],
                title=node["label"],
                shape="circularImage",
                image=_emoji_circle_image(_USER_EMOJI, _COLOR_USER),
                color=_COLOR_USER,
                size=_SIZE_USER,
                physics=False,
                x=0,
                y=0,  # 중심 고정 — 레이아웃이 흔들려도 유저 노드는 항상 중앙
            )
        elif node["node_type"] == "product":
            info = product_lookup.get(node["ref_id"], {})
            name = info.get("name", f"상품 {node['ref_id']}")
            category = info.get("category", "-")
            price = info.get("price_usd")
            icon_slug = icon_slug_for(product_type_from_name(name))
            is_hop2 = node["hop"] == 2
            if node.get("purchased") is True:
                color, size = _COLOR_PRODUCT_PURCHASED, _SIZE_PRODUCT_HOP1
            elif is_hop2:
                color, size = _COLOR_PRODUCT_HOP2, _SIZE_PRODUCT_HOP2
            else:
                color, size = _COLOR_PRODUCT_VIEWED, _SIZE_PRODUCT_HOP1
            # 아이콘(PNG 있으면 그걸로, 없으면 이모지 폴백)은 색상 원과 함께 미리 구운 SVG
            # 이미지(circularImage)로 노드 "안에" 표시하고, 상품명은 hover 시 title 툴팁으로만
            # 보여준다(상시 라벨 제거 — 위 _HIDDEN_LABEL 참고).
            title = f"{name}<br>{category}" + (f"<br>$ {price:.2f}" if price is not None else "")
            if icon_slug:
                image = _icon_circle_image(icon_slug, color, extract_color(name))
            else:
                image = _emoji_circle_image(get_product_emoji(name, category), color)
            net.add_node(
                node["node_id"],
                label=_HIDDEN_LABEL,
                title=title,
                shape="circularImage",
                image=image,
                color=color,
                size=size,
            )
        else:  # segment
            color = _COLOR_SEGMENT_OWN if node.get("is_own_segment") else _COLOR_SEGMENT
            segment_name = node.get("segment_name")
            if segment_name:
                # 다이아몬드 노드가 작아 영문 원문(segment_name)을 그대로 쓰면 길게 겹쳐 보인다
                # — 유저 선택 드롭다운과 같은 짧은 한글 태그(PERSONA_KO)를 라벨로, 원문은
                # title(hover 툴팁)로만 노출한다(요청 반영: "세그먼트 3" 대신 실제 이름).
                label = PERSONA_KO.get(segment_name, segment_name)
                title = f"{segment_name} ({label})" if label != segment_name else segment_name
            else:
                label = f"세그먼트 {node['ref_id']}"
                title = label
            net.add_node(
                node["node_id"],
                label=label,
                title=title,
                color=color,
                size=_SIZE_SEGMENT,
                shape="diamond",
            )

    for edge in graph["edges"]:
        edge_type = edge["edge_type"]
        if edge_type == "purchased":
            net.add_edge(
                edge["source"],
                edge["target"],
                color=_COLOR_PRODUCT_PURCHASED,
                width=_EDGE_WIDTH_PURCHASED,
                dashes=False,
                title="구매",
            )
        elif edge_type == "viewed":
            net.add_edge(
                edge["source"],
                edge["target"],
                color=_COLOR_PRODUCT_VIEWED,
                width=_EDGE_WIDTH_VIEWED,
                dashes=True,
                title="조회 등",
            )
        elif edge_type == "own_segment":
            # label(상시 표시)이 도착 세그먼트 노드의 라벨과 겹쳐 보였다(요청으로 발견) —
            # title(hover 툴팁)로만 노출한다.
            net.add_edge(
                edge["source"],
                edge["target"],
                color=_COLOR_SEGMENT_OWN,
                width=_EDGE_WIDTH_VIEWED,
                dashes=True,
                title="소속 세그먼트",
            )
        else:  # "lift"
            lift = edge.get("lift") or 0.0
            net.add_edge(
                edge["source"],
                edge["target"],
                color=_COLOR_SEGMENT,
                width=_lift_edge_width(lift),
                label=f"lift {lift:.2f}",
                title=f"lift {lift:.2f}",
            )

    # 화살촉이 라벨을 가리는 문제 + 노드/라벨 밀집으로 인한 혼동(요청으로 발견) — 화살촉을
    # 더 작게, 반발력(gravitationalConstant)과 스프링 길이를 한 번 더 키워 노드 간 간격을
    # 넓힌다. 상품 라벨은 이제 hover 전용이라(위 _HIDDEN_LABEL) 상시 겹침은 대부분 사라졌고,
    # 이 옵션은 남은 세그먼트 라벨/엣지 라벨 간 여백을 더 확보하기 위함이다.
    net.set_options("""
    {
      "physics": {"barnesHut": {"gravitationalConstant": -15000, "springLength": 220, "springConstant": 0.03}},
      "edges": {"arrows": {"to": {"scaleFactor": 0.35}}},
      "interaction": {"hover": true, "tooltipDelay": 100, "zoomView": true, "dragView": true}
    }
    """)
    return net


def _patch_iframe_background(html: str) -> str:
    """pyvis 템플릿의 <body>/Bootstrap `.card`는 bgcolor 인자가 안 닿는 하드코딩된 CSS라
    #mynetwork(그래프 본체, bgcolor 적용됨)와 그 바깥 여백의 색이 미묘하게 어긋나 보였다
    (요청으로 발견). generate_html() 결과에 오버라이드 스타일을 주입해 iframe 안쪽을
    _BG_COLOR로 통일하고, 옅은 테두리(#e2e8f0)로 페이지 배경과 자연스럽게 구분한다.
    """
    override = f"""
    <style>
      html, body {{ background-color: {_BG_COLOR} !important; margin: 0; }}
      /* pyvis 템플릿이 중복 삽입하는 빈 heading — 실사용 안 하지만 여백은 차지해 카드
         바깥으로 배경이 튀어나오는 원인이었다 */
      center, h1 {{ display: none !important; margin: 0 !important; }}
      div.card {{
        background-color: {_BG_COLOR} !important;
        border: none !important;
        box-shadow: none !important;
        margin: 0 !important;
      }}
      #mynetwork {{ border: 1px solid #e2e8f0 !important; }}
    </style>
    """
    return html.replace("</head>", f"{override}</head>")


def render_user_graph(user_id: int) -> None:
    """유저 중심 추천 근거 서브그래프(유저→상품→세그먼트)를 pyvis로 렌더링."""
    st.markdown("#### 추천 근거 그래프")

    show_hop2 = st.toggle(
        "2홉까지 보기 (같은 세그먼트의 다른 인기 상품 포함)",
        key=f"graph_hops_toggle_{user_id}",
    )
    hops = 2 if show_hop2 else 1

    graph, status, message = get_user_subgraph(user_id, hops)

    if status == "not_implemented":
        st.info(f"🚧 {message}" if message else "🚧 그래프 데이터가 아직 준비되지 않았습니다.")
        return

    if not graph["edges"]:
        st.info(message or "이 유저는 상호작용 데이터가 없어 그래프를 표시할 수 없습니다.")
        return

    products_df = load_products()
    net = _build_network(graph, products_df)
    html = net.generate_html(notebook=False)
    html = _patch_iframe_background(html)

    # 범례를 그래프 우측에 고정 배치(요청 반영 — 이전엔 하단 텍스트 한 줄이라 눈에 안 띄었음).
    graph_col, legend_col = st.columns([5, 1.3])
    with graph_col:
        st.iframe(html, height=_IFRAME_HEIGHT_PX)
    with legend_col:
        st.markdown(_LEGEND_HTML, unsafe_allow_html=True)

    if message:
        st.caption(f"ℹ️ {message}")
