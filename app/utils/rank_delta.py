"""Twiddler 적용 전/후 순위 변화를 배지용 데이터로 변환하는 공통 유틸.

프로덕션 화면(RelatedProductsRecommend 격)과 내부 데모 화면(TwiddlerComparisonDemo 격)
양쪽에서 동일하게 사용해 배지 로직이 어긋나지 않도록 한다.
"""

from __future__ import annotations


def get_rank_delta(before: int, after: int) -> dict:
    """순위 변화를 {"direction": "up"|"down"|"same", "label": "+2"|"-1"|"유지"}로 변환."""
    delta = before - after  # 양수 = 순위 상승 (숫자가 낮을수록 좋은 순위)
    if delta > 0:
        return {"direction": "up", "label": f"+{delta}"}
    if delta < 0:
        return {"direction": "down", "label": f"{delta}"}
    return {"direction": "same", "label": "유지"}
