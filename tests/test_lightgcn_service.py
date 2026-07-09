"""lightgcn_service.py 단위 테스트.

retail-clickstream-analysis #41에서 LightGCN 추천 CSV 파일명을 graph_mode별로
(PRED_MAIN_RECOMMEND_bipartite.csv / _tripartite.csv) 분리했다. 이 레포의 bipartite
데이터는 그 이전에 고정 파일명(PRED_MAIN_RECOMMEND.csv)으로 이미 복사돼 GDrive 배포
번들(app/utils/data_bootstrap.py REQUIRED_FILES)에도 그 이름으로 들어가 있으므로,
새 이름을 우선 찾고 없으면 예전 이름으로 폴백해야 기존 배포를 깨지 않는다.
"""

import pandas as pd
import pytest

from backend.api.services import lightgcn_service


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    """모듈 전역 캐시가 테스트 간에 새지 않도록 매 테스트마다 초기화한다."""
    monkeypatch.setattr(lightgcn_service, "_recs_cache", {})


def _write_rec_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


class TestResolveArtifactPath:
    def test_prefers_new_filename_over_legacy_bipartite(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lightgcn_service, "_ARTIFACT_DIR", tmp_path)
        (tmp_path / "PRED_MAIN_RECOMMEND_bipartite.csv").write_text("x")
        (tmp_path / "PRED_MAIN_RECOMMEND.csv").write_text("legacy")

        resolved = lightgcn_service._resolve_artifact_path("bipartite")

        assert resolved == tmp_path / "PRED_MAIN_RECOMMEND_bipartite.csv"

    def test_falls_back_to_legacy_filename_for_bipartite(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lightgcn_service, "_ARTIFACT_DIR", tmp_path)
        (tmp_path / "PRED_MAIN_RECOMMEND.csv").write_text("legacy")

        resolved = lightgcn_service._resolve_artifact_path("bipartite")

        assert resolved == tmp_path / "PRED_MAIN_RECOMMEND.csv"

    def test_tripartite_has_no_legacy_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lightgcn_service, "_ARTIFACT_DIR", tmp_path)
        (tmp_path / "PRED_MAIN_RECOMMEND.csv").write_text("legacy")

        resolved = lightgcn_service._resolve_artifact_path("tripartite")

        assert resolved is None

    def test_returns_none_when_no_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lightgcn_service, "_ARTIFACT_DIR", tmp_path)

        assert lightgcn_service._resolve_artifact_path("bipartite") is None


class TestGetRecommendations:
    def test_not_implemented_when_artifact_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lightgcn_service, "_ARTIFACT_DIR", tmp_path)

        items, status, message = lightgcn_service.get_recommendations(1, 10, "tripartite")

        assert status == "not_implemented"
        assert items == []
        assert message

    def test_ok_when_artifact_exists_and_user_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lightgcn_service, "_ARTIFACT_DIR", tmp_path)
        _write_rec_csv(
            tmp_path / "PRED_MAIN_RECOMMEND_tripartite.csv",
            [
                {"user_id": 1, "item_id": 10, "score": 0.9, "rank": 1},
                {"user_id": 1, "item_id": 11, "score": 0.5, "rank": 2},
            ],
        )

        items, status, message = lightgcn_service.get_recommendations(1, 10, "tripartite")

        assert status == "ok"
        assert message is None
        assert items == [
            {"item_id": 10, "score": 0.9, "rank": 1, "user_type": "all"},
            {"item_id": 11, "score": 0.5, "rank": 2, "user_type": "all"},
        ]

    def test_not_implemented_when_user_not_in_recs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lightgcn_service, "_ARTIFACT_DIR", tmp_path)
        _write_rec_csv(
            tmp_path / "PRED_MAIN_RECOMMEND_bipartite.csv",
            [{"user_id": 1, "item_id": 10, "score": 0.9, "rank": 1}],
        )

        items, status, message = lightgcn_service.get_recommendations(999, 10, "bipartite")

        assert status == "not_implemented"
        assert items == []

    def test_unknown_graph_type_falls_back_to_tripartite(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lightgcn_service, "_ARTIFACT_DIR", tmp_path)

        items, status, message = lightgcn_service.get_recommendations(1, 10, "unknown")

        assert status == "not_implemented"

    def test_top_n_limits_result_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lightgcn_service, "_ARTIFACT_DIR", tmp_path)
        _write_rec_csv(
            tmp_path / "PRED_MAIN_RECOMMEND_tripartite.csv",
            [{"user_id": 1, "item_id": i, "score": 1.0 - i * 0.01, "rank": i} for i in range(1, 6)],
        )

        items, status, _ = lightgcn_service.get_recommendations(1, 2, "tripartite")

        assert status == "ok"
        assert len(items) == 2
