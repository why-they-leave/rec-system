"""카테고리/상품 → 이모지·아이콘 매핑 (app/main.py 카테고리 필터, app/components/user_graph.py 노드 아이콘 공용)."""

import re

CATEGORY_EMOJI: dict[str, str] = {
    "Electronics": "💻",
    "Home & Kitchen": "🏠",
    "Beauty": "💄",
    "Sports": "⚽",
    "Fashion": "👗",
    "Books": "📚",
    "Toys": "🎮",
}

# app/static/images/categories/의 카테고리 아이콘 PNG(7종) — 파일명은 카테고리명을
# lower_snake_case로 정규화한 slug(예: "Home & Kitchen" → "home_kitchen")와 동일하다.
CATEGORY_ICONS_URL_DIR = "app/static/images/categories"  # enableStaticServing 기준 상대 URL

_CATEGORY_ICON_SLUG: dict[str, str] = {
    "Electronics": "electronics",
    "Home & Kitchen": "home_kitchen",
    "Beauty": "beauty",
    "Sports": "sports",
    "Fashion": "fashion",
    "Books": "books",
    "Toys": "toys",
}


def category_icon_url(category: str) -> str | None:
    """카테고리명을 아이콘 정적 URL로 변환. 매핑에 없으면 None(호출부에서 CATEGORY_EMOJI 폴백)."""
    slug = _CATEGORY_ICON_SLUG.get(category)
    return f"{CATEGORY_ICONS_URL_DIR}/{slug}.png" if slug else None


# data/raw/products.csv의 상품명은 "{타입} {색상} {번호}"(예: "Headphones Snow 402") 형식으로
# 고정돼 있어, 타입 토큰(43종)을 전수 조사해 이모지를 정확히 매핑할 수 있다(카테고리보다
# 세밀한 상품별 아이콘). 카테고리 자체가 타입인 경우는 없어 카테고리 이모지와 별개로 관리한다.
PRODUCT_TYPE_EMOJI: dict[str, str] = {
    # Beauty
    "Conditioner": "🧴",
    "Lipstick": "💄",
    "Moisturizer": "🧴",
    "Serum": "💧",
    "Shampoo": "🧴",
    "Sunscreen": "🧴",
    # Books
    "E-book": "📱",
    "Hardcover": "📕",
    "Paperback": "📘",
    # Electronics
    "Headphones": "🎧",
    "Keyboard": "⌨️",
    "Monitor": "🖥️",
    "Mouse": "🖱️",
    "SSD": "💾",
    "Smartwatch": "⌚",
    "Speaker": "🔊",
    "Webcam": "📷",
    # Fashion
    "Dress": "👗",
    "Hoodie": "👕",
    "Jacket": "🧥",
    "Jeans": "👖",
    "Sneakers": "👟",
    "Socks": "🧦",
    "T-shirt": "👕",
    # Home & Kitchen
    "Air Fryer": "🍟",
    "Blender": "🥤",
    "Coffee Maker": "☕",
    "Cookware": "🍳",
    "Lamp": "💡",
    "Toaster": "🍞",
    "Vacuum": "🧹",
    # Sports
    "Cycling Helmet": "🚴",
    "Dumbbell": "🏋️",
    "Tennis Racket": "🎾",
    "Water Bottle": "🥤",
    "Yoga Mat": "🧘",
    # Toys
    "Action Figure": "🦸",
    "Board Game": "🎲",
    "Building Blocks": "🧱",
    "Doll": "🪆",
    "Puzzle": "🧩",
}

# 카테고리 → 하위 상품 타입(서브카테고리) 목록 — PRODUCT_TYPE_EMOJI와 동일한 카테고리
# 그룹핑을 그대로 재사용(products.csv 실측 41종). 사이드바 카테고리 아코디언에서 사용.
CATEGORY_SUBTYPES: dict[str, list[str]] = {
    "Electronics": [
        "Headphones",
        "Keyboard",
        "Monitor",
        "Mouse",
        "SSD",
        "Smartwatch",
        "Speaker",
        "Webcam",
    ],
    "Home & Kitchen": [
        "Air Fryer",
        "Blender",
        "Coffee Maker",
        "Cookware",
        "Lamp",
        "Toaster",
        "Vacuum",
    ],
    "Beauty": ["Conditioner", "Lipstick", "Moisturizer", "Serum", "Shampoo", "Sunscreen"],
    "Sports": ["Cycling Helmet", "Dumbbell", "Tennis Racket", "Water Bottle", "Yoga Mat"],
    "Fashion": ["Dress", "Hoodie", "Jacket", "Jeans", "Sneakers", "Socks", "T-shirt"],
    "Books": ["E-book", "Hardcover", "Paperback"],
    "Toys": ["Action Figure", "Board Game", "Building Blocks", "Doll", "Puzzle"],
}

_TRAILING_COLOR_NUMBER = re.compile(r"\s+[A-Za-z]+\s+\d+$")


def product_type_from_name(name: str) -> str:
    """상품명("Headphones Snow 402")에서 색상·번호를 제거해 타입만 추출(app/components/user_graph.py 공용)."""
    return _TRAILING_COLOR_NUMBER.sub("", name).strip()


def extract_color(name: str) -> str:
    """상품명 뒤에서 두 번째 단어를 CSS 색상명으로 추출(예: "Headphones Snow 402" → "Snow").
    app/components/product_card.py, app/components/user_graph.py 공용.
    """
    parts = name.split()
    return parts[-2] if len(parts) >= 2 else "gray"


def get_product_emoji(name: str, category: str | None = None) -> str:
    """상품명에서 타입 토큰을 추출해 PRODUCT_TYPE_EMOJI로 조회하고, 없으면 카테고리 이모지로 폴백한다.

    상품명 형식이 다르거나(타입 추출 실패) 새 타입이 추가돼 매핑에 없는 경우를 대비한 폴백.
    """
    product_type = product_type_from_name(name)
    if product_type in PRODUCT_TYPE_EMOJI:
        return PRODUCT_TYPE_EMOJI[product_type]
    if category and category in CATEGORY_EMOJI:
        return CATEGORY_EMOJI[category]
    return "🛍️"
