"""lightgcn_service의 순수 로직(추천 결과 DataFrame -> dict 리스트 변환) 테스트 (Issue #11).

CSV 로딩(I/O)은 제외하고, 이미 로드된 DataFrame에서 유저별 top_n을 뽑는 부분만 테스트한다.
"""

import pandas as pd

from backend.api.services.lightgcn_service import _recommendations_from_df


class TestRecommendationsFromDf:
    def test_filters_by_user_id_and_sorts_by_rank(self):
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2],
                "item_id": [10, 20, 30, 40],
                "score": [0.5, 0.9, 0.1, 0.7],
                "rank": [2, 1, 3, 1],
            }
        )
        result = _recommendations_from_df(df, user_id=1, top_n=10)

        assert [r["item_id"] for r in result] == [20, 10, 30]
        assert [r["rank"] for r in result] == [1, 2, 3]

    def test_limits_to_top_n(self):
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 1],
                "item_id": [10, 20, 30],
                "score": [0.9, 0.8, 0.7],
                "rank": [1, 2, 3],
            }
        )
        result = _recommendations_from_df(df, user_id=1, top_n=2)
        assert len(result) == 2

    def test_unknown_user_returns_empty_list(self):
        df = pd.DataFrame({"user_id": [1], "item_id": [10], "score": [0.5], "rank": [1]})
        result = _recommendations_from_df(df, user_id=999, top_n=10)
        assert result == []

    def test_item_fields_have_correct_types(self):
        df = pd.DataFrame({"user_id": [1], "item_id": [10], "score": [0.5], "rank": [1]})
        result = _recommendations_from_df(df, user_id=1, top_n=10)
        item = result[0]
        assert item["item_id"] == 10
        assert isinstance(item["item_id"], int)
        assert isinstance(item["score"], float)
        assert isinstance(item["rank"], int)
