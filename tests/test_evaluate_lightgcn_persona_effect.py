"""evaluate_lightgcn_persona_effect.py 단위 테스트.

bi-graph(페르소나 없음) vs tri-graph(페르소나 있음) 추천 결과를 비교하는 순수 계산
함수만 테스트한다(CSV I/O는 통합 테스트 범위 밖 — evaluate_twiddler.py 테스트 관례와 동일).
"""

import pandas as pd

from src.evaluation.evaluate_lightgcn_persona_effect import (
    compute_accuracy_rows,
    compute_category_share,
    compute_rank_shift,
)


class TestComputeAccuracyRows:
    def test_hr_recall_ndcg_computed_per_k_with_condition_label(self):
        recs_by_user = {1: [{"item_id": 10, "rank": 1}, {"item_id": 11, "rank": 2}]}
        ground_truth = {1: {10}}

        rows = compute_accuracy_rows(recs_by_user, ground_truth, "tripartite", [1, 2])

        assert rows[0] == {
            "condition": "tripartite",
            "k": 1,
            "HR": 1.0,
            "Recall": 1.0,
            "NDCG": 1.0,
            "eval_users": 1,
        }
        assert rows[1]["k"] == 2

    def test_user_missing_from_recs_is_skipped(self):
        recs_by_user = {1: [{"item_id": 10, "rank": 1}]}
        ground_truth = {1: {10}, 2: {20}}

        rows = compute_accuracy_rows(recs_by_user, ground_truth, "bipartite", [1])

        assert rows[0]["eval_users"] == 1


class TestComputeRankShift:
    def test_overlapping_item_rank_difference_is_averaged(self):
        bi_recs = {1: [{"item_id": 10, "rank": 1}, {"item_id": 11, "rank": 2}]}
        tri_recs = {1: [{"item_id": 10, "rank": 3}, {"item_id": 12, "rank": 1}]}

        result = compute_rank_shift(bi_recs, tri_recs, k=3)

        assert result["mean_abs_rank_shift"] == 2.0  # |1-3|
        assert result["mean_overlap_ratio"] == round(1 / 3, 4)  # 1개 겹침 / k=3
        assert result["n_users"] == 1

    def test_no_overlap_gives_zero_shift_but_counts_user(self):
        bi_recs = {1: [{"item_id": 10, "rank": 1}]}
        tri_recs = {1: [{"item_id": 99, "rank": 1}]}

        result = compute_rank_shift(bi_recs, tri_recs, k=1)

        assert result["mean_abs_rank_shift"] == 0.0
        assert result["mean_overlap_ratio"] == 0.0
        assert result["n_users"] == 1

    def test_user_missing_from_tri_recs_is_excluded(self):
        bi_recs = {1: [{"item_id": 10, "rank": 1}], 2: [{"item_id": 20, "rank": 1}]}
        tri_recs = {1: [{"item_id": 10, "rank": 1}]}

        result = compute_rank_shift(bi_recs, tri_recs, k=1)

        assert result["n_users"] == 1


class TestComputeCategoryShare:
    def test_share_sums_to_one(self):
        recs_by_user = {
            1: [{"item_id": 10, "rank": 1}, {"item_id": 11, "rank": 2}],
            2: [{"item_id": 12, "rank": 1}],
        }
        category_map = {10: "A", 11: "B", 12: "A"}

        df = compute_category_share(
            recs_by_user, k=2, category_map=category_map, condition="bipartite"
        )

        assert set(df["condition"]) == {"bipartite"}
        assert pytest_approx_sum(df["share"]) == 1.0
        a_share = df[df["category"] == "A"]["share"].iloc[0]
        assert a_share == round(2 / 3, 4)

    def test_items_beyond_k_are_excluded(self):
        recs_by_user = {1: [{"item_id": 10, "rank": 1}, {"item_id": 11, "rank": 5}]}
        category_map = {10: "A", 11: "B"}

        df = compute_category_share(
            recs_by_user, k=1, category_map=category_map, condition="tripartite"
        )

        assert list(df["category"]) == ["A"]

    def test_unknown_category_is_excluded(self):
        recs_by_user = {1: [{"item_id": 999, "rank": 1}]}

        df = compute_category_share(recs_by_user, k=1, category_map={}, condition="bipartite")

        assert df.empty


def pytest_approx_sum(series: pd.Series) -> float:
    return round(float(series.sum()), 6)
