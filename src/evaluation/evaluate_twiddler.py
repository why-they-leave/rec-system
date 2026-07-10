"""
Twiddler(페르소나 재랭킹) 오프라인 성능 평가 — 메인(ALS) / 상세(보완재) 두 컨텍스트.

Usage:
    python -m src.evaluation.evaluate_twiddler

로직은 새로 만들지 않고 notebooks/20260708_ML_als_vs_als_twiddler_offline.ipynb,
notebooks/20260708_ML_complementary_vs_twiddler_offline.ipynb의 셀 코드를 그대로 옮긴 것이다 —
두 노트북 모두 실서빙 코드(persona_service.py, rerank.py, complementary/model.py,
src/modeling/als/evaluate.py)를 그대로 import해서 계산했으므로, 이 스크립트도 동일한 모듈을
그대로 재사용한다(재구현 없음). reports/UI_TAB_RESTRUCTURE_PLAN.md §Tab1 참고.

출력 (data/outputs/eval/):
    twiddler_accuracy.csv  — context(main/detail), condition(baseline/twiddler), k, segment,
                              HR, Recall, NDCG, eval_users
    twiddler_diversity.csv — context, condition, k, segment, repetition_rate, unique_item_ratio,
                              categories_first, categories_cumulative, n_users

segment 컬럼: "ALL"(population 전체 평균, 기존과 동일) 또는 세그먼트명(persona_service.get_persona
반환값과 동일 문자열) — 유저 1명씩 계산한 지표를 "ALL"과 해당 유저의 세그먼트 버킷 양쪽에 동시에
적립해 한 번의 순회로 population/세그먼트 breakdown을 모두 만든다(재계산 없음).
reports/UI_TAB_RESTRUCTURE_PLAN.md §Tab1 3단 구성(population 지표 + 세그먼트 breakdown + 선택
유저 케이스) 참고 — "선택 유저 케이스"는 population 시뮬레이션과 무관하게 가벼워 여기서 다루지
않고 backend/api/core.py::get_user_twiddler_case가 라이브로 계산한다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.api.services import catalog_service, persona_service
from src.modeling.als.evaluate import hit_rate_at_k, ndcg_at_k, recall_at_k
from src.modeling.complementary.model import run_modeling
from src.modeling.twiddler import rerank as rerank_mod

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
ALS_OUTPUT_DIR = ROOT_DIR / "data" / "outputs" / "ALS"
LIGHTGCN_OUTPUT_DIR = ROOT_DIR / "data" / "outputs" / "LightGCN"
EVAL_OUTPUT_DIR = ROOT_DIR / "data" / "outputs" / "eval"

MAIN_CONTEXT_LABEL = "main"
LIGHTGCN_BIPARTITE_CONTEXT_LABEL = "main_lightgcn_bipartite"

K_LIST_MAIN = [5, 10, 20]
K_LIST_DETAIL = [1, 3, 5]
T_ROUNDS = 5  # 반복 새로고침(메인)/재방문(상세) 시뮬레이션 횟수 — 두 컨텍스트 공통

POOL_MULTIPLIER = rerank_mod.POOL_MULTIPLIER

ALL_SEGMENTS_LABEL = "ALL"  # population 전체 평균 버킷(기존 aggregate 표와 동일)
_NO_PERSONA_LABEL = (
    "페르소나 없음"  # get_persona가 None을 반환하는 유저용 버킷(현재 데이터엔 거의 없음)
)


def _segment_label(user_or_cust_id: int) -> str:
    return persona_service.get_persona(user_or_cust_id) or _NO_PERSONA_LABEL


def _decay_and_record(exposure_counts: dict, shown_ids: list, decay: float) -> None:
    """backend/api/services/exposure_service.record_exposure와 동일한 노출 이력 갱신 로직."""
    for iid in list(exposure_counts.keys()):
        decayed = exposure_counts[iid] * decay
        if decayed < 0.01:
            del exposure_counts[iid]
        else:
            exposure_counts[iid] = decayed
    for iid in shown_ids:
        exposure_counts[iid] = exposure_counts.get(iid, 0.0) + 1.0


def _diversity_metrics(rounds: list[list[int]], k: int, category_map: dict) -> dict:
    """첫 회차 대비 반복 중복률과 누적 카테고리 커버리지를 계산한다(메인/상세 공통)."""
    first = set(rounds[0])
    overlaps = [len(first & set(r)) / k for r in rounds[1:]]
    all_items = [it for r in rounds for it in r]
    all_cats = {category_map.get(it) for it in all_items} - {None}
    first_cats = {category_map.get(it) for it in rounds[0]} - {None}
    return {
        "repetition_rate": np.mean(overlaps) if overlaps else 1.0,
        "unique_item_ratio": len(set(all_items)) / (k * len(rounds)),
        "categories_first": len(first_cats),
        "categories_cumulative": len(all_cats),
    }


# ── 메인(ALS) ────────────────────────────────────────────────────────────────


def _load_main_context(
    output_dir: Path = ALS_OUTPUT_DIR,
    rec_filename: str = "PRED_MAIN_RECOMMEND.csv",
    test_filename: str = "als_test.csv",
) -> dict:
    """PRED_MAIN_RECOMMEND.csv/정답셋 기반 평가 컨텍스트 로드.

    ALS(기본값)와 LightGCN-bipartite 양쪽에서 재사용한다 — 두 모델 다 [user_id, item_id,
    score, rank] 스키마의 precomputed 추천 결과 + [user_id, item_id] 정답셋이라는 동일한
    형태를 공유하므로 파일 경로만 바꿔 호출하면 된다.
    """
    recs_df = pd.read_csv(output_dir / rec_filename)
    test_df = pd.read_csv(output_dir / test_filename)
    ground_truth = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    eval_users = list(ground_truth.keys())
    category_map = catalog_service.get_category_map()
    recs_by_user = {
        uid: g.sort_values("rank")[["item_id", "score"]].to_dict("records")
        for uid, g in recs_df[recs_df["user_id"].isin(eval_users)].groupby("user_id")
    }
    logger.info("[메인] 평가 대상 유저 수: %s명", len(eval_users))
    return {
        "ground_truth": ground_truth,
        "eval_users": eval_users,
        "category_map": category_map,
        "recs_by_user": recs_by_user,
    }


def _apply_current_twiddler_main(
    candidates: list[dict], uid: int, k: int, category_map: dict
) -> list[int]:
    """실서빙과 동일한 순서로 호출: get_persona → get_user_affinity/alpha → rerank."""
    persona_label = persona_service.get_persona(uid)
    if persona_label is None:
        return [it["item_id"] for it in candidates[:k]]
    affinity = persona_service.get_user_affinity(uid)
    alpha = persona_service.get_user_alpha(uid)
    reranked = rerank_mod.rerank(
        candidates,
        id_key="item_id",
        category_map=category_map,
        affinity=affinity,
        alpha=alpha,
        exposure_counts=None,
        top_k=k,
    )
    return [it["item_id"] for it in reranked]


def _main_accuracy_rows(ctx: dict, context_label: str = MAIN_CONTEXT_LABEL) -> list[dict]:
    """단일 세션(1회 추천) 정확도: baseline(모델 only) vs twiddler(모델+Twiddler).

    유저별 지표를 "ALL"과 그 유저의 세그먼트 버킷 양쪽에 동시 적립(세그먼트 breakdown).
    context_label로 ALS("main")/LightGCN-bipartite("main_lightgcn_bipartite") 결과를 구분한다.
    """
    rows = []
    for k in K_LIST_MAIN:
        pool_n = k * POOL_MULTIPLIER
        for condition in ["baseline", "twiddler"]:
            buckets: dict[str, dict[str, list]] = {}
            for uid in ctx["eval_users"]:
                if uid not in ctx["recs_by_user"]:
                    continue
                candidates = [dict(it) for it in ctx["recs_by_user"][uid][:pool_n]]
                if condition == "baseline":
                    recommended = [it["item_id"] for it in candidates[:k]]
                else:
                    recommended = _apply_current_twiddler_main(
                        candidates, uid, k, ctx["category_map"]
                    )
                true_items = ctx["ground_truth"][uid]
                hr = hit_rate_at_k(recommended, true_items)
                recall = recall_at_k(recommended, true_items)
                ndcg = ndcg_at_k(recommended, true_items)
                for segment in (ALL_SEGMENTS_LABEL, _segment_label(uid)):
                    b = buckets.setdefault(segment, {"HR": [], "Recall": [], "NDCG": []})
                    b["HR"].append(hr)
                    b["Recall"].append(recall)
                    b["NDCG"].append(ndcg)
            for segment, b in buckets.items():
                rows.append(
                    {
                        "context": context_label,
                        "condition": condition,
                        "k": k,
                        "segment": segment,
                        "HR": round(np.mean(b["HR"]), 4),
                        "Recall": round(np.mean(b["Recall"]), 4),
                        "NDCG": round(np.mean(b["NDCG"]), 4),
                        "eval_users": len(b["HR"]),
                    }
                )
    logger.info(
        "[메인:%s] 정확도 계산 완료 (%s행, 세그먼트 breakdown 포함)", context_label, len(rows)
    )
    return rows


def _simulate_main_sessions(
    candidates_base: list[dict], uid: int, k: int, condition: str, category_map: dict
) -> list[list[int]]:
    """T_ROUNDS회 반복 새로고침을 시뮬레이션한다(노출 이력 패널티는 twiddler 조건에서만 갱신)."""
    if condition == "baseline":
        return [[it["item_id"] for it in candidates_base[:k]] for _ in range(T_ROUNDS)]
    persona_label = persona_service.get_persona(uid)
    if persona_label is None:
        return [[it["item_id"] for it in candidates_base[:k]] for _ in range(T_ROUNDS)]
    affinity = persona_service.get_user_affinity(uid)
    alpha = persona_service.get_user_alpha(uid)
    decay = persona_service.get_user_decay(uid)
    exposure_counts: dict = {}
    sessions = []
    for _ in range(T_ROUNDS):
        candidates = [dict(it) for it in candidates_base]
        exposure_arg = exposure_counts if exposure_counts else None
        reranked = rerank_mod.rerank(
            candidates,
            id_key="item_id",
            category_map=category_map,
            affinity=affinity,
            alpha=alpha,
            exposure_counts=exposure_arg,
            decay=decay,
            top_k=k,
        )
        slate = [it["item_id"] for it in reranked]
        sessions.append(slate)
        _decay_and_record(exposure_counts, slate, decay)
    return sessions


def _main_diversity_rows(ctx: dict, context_label: str = MAIN_CONTEXT_LABEL) -> list[dict]:
    """반복 새로고침(T_ROUNDS회) 다양성: baseline vs twiddler (세그먼트 breakdown 포함)."""
    rows = []
    for k in K_LIST_MAIN:
        pool_n = k * POOL_MULTIPLIER
        for condition in ["baseline", "twiddler"]:
            buckets: dict[str, dict[str, list]] = {}
            for uid in ctx["eval_users"]:
                if uid not in ctx["recs_by_user"]:
                    continue
                candidates_base = ctx["recs_by_user"][uid][:pool_n]
                sessions = _simulate_main_sessions(
                    candidates_base, uid, k, condition, ctx["category_map"]
                )
                m = _diversity_metrics(sessions, k, ctx["category_map"])
                for segment in (ALL_SEGMENTS_LABEL, _segment_label(uid)):
                    b = buckets.setdefault(
                        segment,
                        {
                            "repetition_rate": [],
                            "unique_item_ratio": [],
                            "categories_first": [],
                            "categories_cumulative": [],
                        },
                    )
                    for key in b:
                        b[key].append(m[key])
            for segment, b in buckets.items():
                rows.append(
                    {
                        "context": context_label,
                        "condition": condition,
                        "k": k,
                        "segment": segment,
                        "repetition_rate": round(np.mean(b["repetition_rate"]), 4),
                        "unique_item_ratio": round(np.mean(b["unique_item_ratio"]), 4),
                        "categories_first": round(np.mean(b["categories_first"]), 2),
                        "categories_cumulative": round(np.mean(b["categories_cumulative"]), 2),
                        "n_users": len(b["repetition_rate"]),
                    }
                )
    logger.info(
        "[메인:%s] 다양성 계산 완료 (%s행, 세그먼트 breakdown 포함)", context_label, len(rows)
    )
    return rows


# ── 상세(보완재) ─────────────────────────────────────────────────────────────


def _load_detail_context() -> dict:
    """run_modeling을 재실행해 보완재 후보 풀과 세션 기반 평가 케이스를 구성한다.

    저장된 detail_cf.csv를 그대로 읽지 않고 동일 로직(run_modeling)을 다시 실행하는 이유는
    train/test 세션 분리(random_state=42)까지 재현해 정답(ground_truth) 세션을 얻기 위함이다
    — detail_cf.csv에는 train 결과만 저장돼 있고 test_df는 없다.
    """
    df_logs = pd.read_csv(ROOT_DIR / "data" / "processed" / "df_integrated_logs.csv")
    products = pd.read_csv(ROOT_DIR / "data" / "raw" / "products.csv")
    top_n_recs, _df_recs, _train_df, test_df = run_modeling(df_logs, products)
    category_map = catalog_service.get_category_map()

    repro = top_n_recs.rename(columns={"prod_A": "item_id", "prod_B": "rec_item_id"})[
        ["item_id", "rec_item_id", "score", "rank"]
    ]
    recs_dict = {
        item_id: g.sort_values("rank")[["rec_item_id", "score"]].to_dict("records")
        for item_id, g in repro.groupby("item_id")
    }

    cases = []  # (customer_id, target_prod, ground_truth_set)
    for _sid, g in test_df.groupby("session_id"):
        prods = list(set(g["product_id"].tolist()))
        if len(prods) < 2:
            continue
        cust = int(g["customer_id"].iloc[0])
        for i, target_prod in enumerate(prods):
            if target_prod not in recs_dict:
                continue
            ground_truth = set(prods[:i] + prods[i + 1 :])
            cases.append((cust, target_prod, ground_truth))

    unique_custs = {c[0] for c in cases}
    persona_cache = {}
    for cust in unique_custs:
        label = persona_service.get_persona(cust)
        persona_cache[cust] = {
            "persona": label,
            "affinity": persona_service.get_user_affinity(cust) if label is not None else {},
            "alpha": persona_service.get_user_alpha(cust) if label is not None else 0.0,
            "decay": persona_service.get_user_decay(cust)
            if label is not None
            else rerank_mod.EXPOSURE_DECAY,
        }
    logger.info("[상세] 평가 케이스 수: %s건, 고유 고객 수: %s명", len(cases), len(unique_custs))
    return {
        "cases": cases,
        "recs_dict": recs_dict,
        "persona_cache": persona_cache,
        "category_map": category_map,
    }


def _detail_accuracy_rows(ctx: dict) -> list[dict]:
    """단일 조회 정확도: baseline(보완재 only) vs twiddler(보완재+Twiddler) (세그먼트 breakdown 포함)."""
    rows = []
    for k in K_LIST_DETAIL:
        for condition in ["baseline", "twiddler"]:
            buckets: dict[str, dict[str, list]] = {}
            for cust, target_prod, ground_truth in ctx["cases"]:
                candidates = [dict(it) for it in ctx["recs_dict"][target_prod]]
                pf = ctx["persona_cache"][cust]
                if condition == "baseline" or pf["persona"] is None:
                    recommended = [it["rec_item_id"] for it in candidates[:k]]
                else:
                    reranked = rerank_mod.rerank(
                        candidates,
                        id_key="rec_item_id",
                        category_map=ctx["category_map"],
                        affinity=pf["affinity"],
                        alpha=pf["alpha"],
                        exposure_counts=None,
                        top_k=k,
                    )
                    recommended = [it["rec_item_id"] for it in reranked]
                hr = hit_rate_at_k(recommended, ground_truth)
                recall = recall_at_k(recommended, ground_truth)
                ndcg = ndcg_at_k(recommended, ground_truth)
                segment = pf["persona"] or _NO_PERSONA_LABEL
                for key in (ALL_SEGMENTS_LABEL, segment):
                    b = buckets.setdefault(key, {"HR": [], "Recall": [], "NDCG": []})
                    b["HR"].append(hr)
                    b["Recall"].append(recall)
                    b["NDCG"].append(ndcg)
            for segment, b in buckets.items():
                rows.append(
                    {
                        "context": "detail",
                        "condition": condition,
                        "k": k,
                        "segment": segment,
                        "HR": round(np.mean(b["HR"]), 4),
                        "Recall": round(np.mean(b["Recall"]), 4),
                        "NDCG": round(np.mean(b["NDCG"]), 4),
                        "eval_users": len(b["HR"]),
                    }
                )
    logger.info("[상세] 정확도 계산 완료 (%s행, 세그먼트 breakdown 포함)", len(rows))
    return rows


def _simulate_detail_views(
    candidates_base: list[dict], pf: dict, k: int, condition: str, category_map: dict
) -> list[list[int]]:
    """T_ROUNDS회 반복 재방문을 시뮬레이션한다(후보 풀이 5개로 고정돼 K=5에서는 효과 없음)."""
    if condition == "baseline" or pf["persona"] is None:
        return [[it["rec_item_id"] for it in candidates_base[:k]] for _ in range(T_ROUNDS)]
    exposure_counts: dict = {}
    views = []
    for _ in range(T_ROUNDS):
        candidates = [dict(it) for it in candidates_base]
        exposure_arg = exposure_counts if exposure_counts else None
        reranked = rerank_mod.rerank(
            candidates,
            id_key="rec_item_id",
            category_map=category_map,
            affinity=pf["affinity"],
            alpha=pf["alpha"],
            exposure_counts=exposure_arg,
            decay=pf["decay"],
            top_k=k,
        )
        slate = [it["rec_item_id"] for it in reranked]
        views.append(slate)
        _decay_and_record(exposure_counts, slate, pf["decay"])
    return views


def _detail_diversity_rows(ctx: dict) -> list[dict]:
    """반복 재방문(T_ROUNDS회) 다양성: baseline vs twiddler (세그먼트 breakdown 포함)."""
    unique_item_cust = list({(t, c) for c, t, _ in ctx["cases"]})
    rows = []
    for k in K_LIST_DETAIL:
        for condition in ["baseline", "twiddler"]:
            buckets: dict[str, dict[str, list]] = {}
            for target_prod, cust in unique_item_cust:
                candidates_base = ctx["recs_dict"][target_prod]
                pf = ctx["persona_cache"][cust]
                views = _simulate_detail_views(
                    candidates_base, pf, k, condition, ctx["category_map"]
                )
                m = _diversity_metrics(views, k, ctx["category_map"])
                segment = pf["persona"] or _NO_PERSONA_LABEL
                for key in (ALL_SEGMENTS_LABEL, segment):
                    b = buckets.setdefault(
                        key,
                        {
                            "repetition_rate": [],
                            "unique_item_ratio": [],
                            "categories_first": [],
                            "categories_cumulative": [],
                        },
                    )
                    for mk in b:
                        b[mk].append(m[mk])
            for segment, b in buckets.items():
                rows.append(
                    {
                        "context": "detail",
                        "condition": condition,
                        "k": k,
                        "segment": segment,
                        "repetition_rate": round(np.mean(b["repetition_rate"]), 4),
                        "unique_item_ratio": round(np.mean(b["unique_item_ratio"]), 4),
                        "categories_first": round(np.mean(b["categories_first"]), 2),
                        "categories_cumulative": round(np.mean(b["categories_cumulative"]), 2),
                        "n_users": len(b["repetition_rate"]),
                    }
                )
    logger.info("[상세] 다양성 계산 완료 (%s행, 세그먼트 breakdown 포함)", len(rows))
    return rows


def _existing_rows_for_context(existing_path: Path, context_label: str) -> list[dict]:
    """이미 저장된 CSV에서 특정 context 행만 골라낸다(재계산 불가능한 컨텍스트 보존용)."""
    if not existing_path.exists():
        return []
    df = pd.read_csv(existing_path)
    return df[df["context"] == context_label].to_dict("records")


def main() -> None:
    """ALS/보완재/LightGCN-bipartite 세 컨텍스트를 계산해 population/세그먼트 breakdown CSV로 저장한다.

    ALS·보완재 원본 데이터(data/outputs/ALS/*, df_integrated_logs.csv 등)는 gitignore
    대상이라 로컬에 없을 수 있다 — 그 경우 새로 계산하지 않고 기존에 저장된 CSV의 해당
    context 행을 그대로 보존한다(재계산 가능한 LightGCN-bipartite만 갱신).
    """
    np.random.seed(42)
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    accuracy_path = EVAL_OUTPUT_DIR / "twiddler_accuracy.csv"
    diversity_path = EVAL_OUTPUT_DIR / "twiddler_diversity.csv"

    logger.info("===== Twiddler 오프라인 평가 시작 =====")

    accuracy_rows: list[dict] = []
    diversity_rows: list[dict] = []

    try:
        main_ctx = _load_main_context()
        accuracy_rows += _main_accuracy_rows(main_ctx, MAIN_CONTEXT_LABEL)
        diversity_rows += _main_diversity_rows(main_ctx, MAIN_CONTEXT_LABEL)
    except FileNotFoundError as e:
        logger.warning("[메인:ALS] 원본 파일 없음(%s) — 기존 CSV의 'main' 행을 보존한다.", e)
        accuracy_rows += _existing_rows_for_context(accuracy_path, MAIN_CONTEXT_LABEL)
        diversity_rows += _existing_rows_for_context(diversity_path, MAIN_CONTEXT_LABEL)

    try:
        detail_ctx = _load_detail_context()
        accuracy_rows += _detail_accuracy_rows(detail_ctx)
        diversity_rows += _detail_diversity_rows(detail_ctx)
    except FileNotFoundError as e:
        logger.warning("[상세] 원본 파일 없음(%s) — 기존 CSV의 'detail' 행을 보존한다.", e)
        accuracy_rows += _existing_rows_for_context(accuracy_path, "detail")
        diversity_rows += _existing_rows_for_context(diversity_path, "detail")

    lightgcn_ctx = _load_main_context(
        output_dir=LIGHTGCN_OUTPUT_DIR,
        rec_filename="PRED_MAIN_RECOMMEND.csv",
        test_filename="lightgcn_test.csv",
    )
    accuracy_rows += _main_accuracy_rows(lightgcn_ctx, LIGHTGCN_BIPARTITE_CONTEXT_LABEL)
    diversity_rows += _main_diversity_rows(lightgcn_ctx, LIGHTGCN_BIPARTITE_CONTEXT_LABEL)

    pd.DataFrame(accuracy_rows).to_csv(accuracy_path, index=False)
    pd.DataFrame(diversity_rows).to_csv(diversity_path, index=False)
    logger.info("[저장] %s", accuracy_path)
    logger.info("[저장] %s", diversity_path)
    logger.info("===== Twiddler 오프라인 평가 완료 =====")


if __name__ == "__main__":
    main()
