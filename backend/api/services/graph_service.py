"""
유저 → 상품 → 세그먼트 추천 근거 서브그래프 조회 서비스.

data/processed/tri_graph_uidx2tidx_train.json, tri_graph_tidx2pidx.json,
tri_graph_uidx2pidx.json, uidx_user_id_mapping.csv, tidx_product_id_mapping.csv,
segment_personas_train_only.json을 첫 호출 시 1회 로드해 메모리에 캐시한다(als_service.py와
동일 패턴). tri-graph JSON은 그 자체로 인접 리스트(딕셔너리)이므로 uidx/tidx로 바로 O(1)
조회 가능하며, networkx 등으로 전체 그래프를 메모리에 올릴 필요가 없다(유저 약 20,000 x
상품 1,197 규모).

구매 여부 판정에는 tri_graph_uidx2tidx_valid.json을 쓰지 않는다 — 이 파일은
cutoff(CUTOFF_DATE) 이후(미래) 구매만 담은 모델 평가용 정답 라벨이라, "구매 여부" 판정에
쓰면 아직 일어나지 않은 미래 구매를 과거 행동처럼 보여주는 데이터 누수가 된다(검증됨,
reports/USER_GRAPH_VIZ_PLAN.md 참고). 대신 이미 rec-system에 있는
data/processed/df_integrated_logs.csv(customer_id, event_type, product_id, timestamp)에서
cutoff 이전 & event_type=="purchase" 로 직접 판정한다 — user_id/item_id를 그대로 쓰므로
uidx/tidx 매핑도 필요 없다.

파일이 아직 준비되지 않았으면 status="not_implemented"를, 유저가 그래프에 없거나(uidx
매핑 없음) 상호작용이 없으면(콜드 유저) status="ok" + 빈 엣지를 반환한다
(complementary_service.py와 동일한 graceful 폴백 관례).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_ROOT_DIR = Path(__file__).resolve().parents[3]
_DATA_DIR = _ROOT_DIR / "data" / "processed"

_UIDX_TIDX_TRAIN_PATH = _DATA_DIR / "tri_graph_uidx2tidx_train.json"
_TIDX_PIDX_PATH = _DATA_DIR / "tri_graph_tidx2pidx.json"
_UIDX_PIDX_PATH = _DATA_DIR / "tri_graph_uidx2pidx.json"
_UIDX_USER_ID_MAP_PATH = _DATA_DIR / "uidx_user_id_mapping.csv"
_TIDX_PRODUCT_ID_MAP_PATH = _DATA_DIR / "tidx_product_id_mapping.csv"
_SEGMENT_PERSONAS_PATH = _DATA_DIR / "segment_personas_train_only.json"
_INTEGRATED_LOGS_PATH = _DATA_DIR / "df_integrated_logs.csv"

_REQUIRED_PATHS = [
    _UIDX_TIDX_TRAIN_PATH, _TIDX_PIDX_PATH, _UIDX_PIDX_PATH,
    _UIDX_USER_ID_MAP_PATH, _TIDX_PRODUCT_ID_MAP_PATH, _INTEGRATED_LOGS_PATH,
]

# ALS(configs/als/params.yaml)·tri-graph 파이프라인과 동일한 train/valid 분리 기준일.
CUTOFF_DATE = pd.Timestamp("2025-08-01")

# ── 노드 수 폭발 방지 상수 (매직넘버 금지 규칙) ─────────────────────────
MAX_PRODUCTS_HOP1 = 12             # 유저-상품 엣지 상한(구매 우선, 그다음 tidx 오름차순)
MAX_SEGMENTS_PER_PRODUCT = 3       # 상품 1개당 보여줄 lift 상위 세그먼트 수(세그먼트 총량은 6개뿐)
HOP2_MAX_EXPANDED_SEGMENTS = 2     # 2홉에서 확장할 세그먼트 수(유저 본인 세그먼트 우선)
HOP2_MAX_PRODUCTS_PER_SEGMENT = 5  # 세그먼트당 추가로 보여줄 lift 상위 인기 상품 수
MAX_TOTAL_NODES = 60               # 안전판 — hop2 확장 후에도 초과 시 hop2 상품부터 컷

_EMPTY_GRAPH: dict = {"nodes": [], "edges": []}

_loaded = False
_files_available = True
_uidx2tidx_train: dict[int, list[int]] = {}
_tidx2pidx: dict[int, list[list]] = {}
_pidx2tidx: dict[int, list[tuple[int, float]]] = {}   # segment_id -> [(tidx, lift), ...] desc (hop2 역인덱스)
_uidx2pidx: dict[int, int] = {}                         # uidx -> 본인 segment_id(단일)
_user_id_to_uidx: dict[int, int] = {}
_tidx_to_product_id: dict[int, int] = {}
_segment_names: dict[int, str] = {}
_purchased_by_user: dict[int, set[int]] = {}            # user_id -> cutoff 이전 구매 product_id 집합


def _load_all() -> None:
    """6개 아티팩트를 첫 호출 시 1회 로드. 필수 파일이 없으면 조용히 not-available 표시만 남긴다."""
    global _loaded, _files_available
    global _uidx2tidx_train, _tidx2pidx, _pidx2tidx
    global _uidx2pidx, _user_id_to_uidx, _tidx_to_product_id, _segment_names, _purchased_by_user
    if _loaded:
        return
    _loaded = True

    if not all(p.exists() for p in _REQUIRED_PATHS):
        _files_available = False
        return

    with open(_UIDX_TIDX_TRAIN_PATH, encoding="utf-8") as f:
        _uidx2tidx_train = {int(k): v for k, v in json.load(f).items()}

    with open(_TIDX_PIDX_PATH, encoding="utf-8") as f:
        raw_tidx2pidx = {int(k): v for k, v in json.load(f).items()}
    _tidx2pidx = raw_tidx2pidx

    pidx2tidx: dict[int, list[tuple[int, float]]] = {}
    for tidx, seg_lift_list in raw_tidx2pidx.items():
        for segment_id, lift in seg_lift_list:
            pidx2tidx.setdefault(int(segment_id), []).append((tidx, float(lift)))
    for pairs in pidx2tidx.values():
        pairs.sort(key=lambda pair: pair[1], reverse=True)
    _pidx2tidx = pidx2tidx

    with open(_UIDX_PIDX_PATH, encoding="utf-8") as f:
        raw_uidx2pidx = json.load(f)
    # 스펙상 값은 [segment_id] 단일 원소 리스트지만, 방어적으로 첫 값만 취한다.
    _uidx2pidx = {int(k): int(v[0]) for k, v in raw_uidx2pidx.items() if v}

    user_map_df = pd.read_csv(_UIDX_USER_ID_MAP_PATH)
    _user_id_to_uidx = dict(zip(user_map_df["user_id"], user_map_df["uidx"]))

    product_map_df = pd.read_csv(_TIDX_PRODUCT_ID_MAP_PATH)
    _tidx_to_product_id = dict(zip(product_map_df["tidx"], product_map_df["product_id"]))

    _segment_names = {}
    if _SEGMENT_PERSONAS_PATH.exists():
        with open(_SEGMENT_PERSONAS_PATH, encoding="utf-8") as f:
            personas = json.load(f)
        _segment_names = {int(p["segment_id"]): p["segment_name"] for p in personas}

    logs_df = pd.read_csv(_INTEGRATED_LOGS_PATH, usecols=["customer_id", "event_type", "product_id", "timestamp"])
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
    purchases = logs_df[(logs_df["event_type"] == "purchase") & (logs_df["timestamp"] < CUTOFF_DATE)]
    _purchased_by_user = {
        int(uid): set(g["product_id"].astype(int)) for uid, g in purchases.groupby("customer_id")
    }


def _user_node_id(user_id: int) -> str:
    return f"user:{user_id}"


def _product_node_id(product_id: int) -> str:
    return f"product:{product_id}"


def _segment_node_id(segment_id: int) -> str:
    return f"segment:{segment_id}"


def _pick_expanded_segments(own_segment_id: int | None, segment_best_lift: dict[int, float]) -> list[int]:
    """2홉 확장 대상 세그먼트 선택 — 유저 본인 세그먼트 우선, 그다음 lift 높은 순."""
    ordered = sorted(segment_best_lift, key=lambda s: segment_best_lift[s], reverse=True)
    picked: list[int] = []
    if own_segment_id is not None:
        picked.append(own_segment_id)
    for segment_id in ordered:
        if len(picked) >= HOP2_MAX_EXPANDED_SEGMENTS:
            break
        if segment_id not in picked:
            picked.append(segment_id)
    return picked[:HOP2_MAX_EXPANDED_SEGMENTS]


def _apply_total_node_cap(nodes: dict[str, dict], edges: list[dict]) -> tuple[list[dict], list[dict], bool]:
    """MAX_TOTAL_NODES 초과 시 hop=2 상품 노드부터 잘라내는 안전판."""
    if len(nodes) <= MAX_TOTAL_NODES:
        return list(nodes.values()), edges, False

    keep_ids = {nid for nid, n in nodes.items() if n["hop"] != 2}
    hop2_ids = [nid for nid, n in nodes.items() if n["hop"] == 2]
    budget = max(MAX_TOTAL_NODES - len(keep_ids), 0)
    keep_ids |= set(hop2_ids[:budget])

    trimmed_nodes = [n for nid, n in nodes.items() if nid in keep_ids]
    trimmed_edges = [e for e in edges if e["source"] in keep_ids and e["target"] in keep_ids]
    return trimmed_nodes, trimmed_edges, True


def get_user_subgraph(user_id: int, hops: int = 1) -> tuple[dict, str, str | None]:
    """유저 중심 1~2홉 추천 근거 서브그래프(유저→상품→세그먼트)를 조회한다.

    반환: (graph, status, message). graph = {"nodes": [...], "edges": [...]}
    (pyvis 등 렌더러가 바로 소비할 수 있는 평문 dict 리스트 — 스타일링은 컴포넌트 몫).
    status: "ok"(edges가 비어있을 수 있음) | "not_implemented"(그래프 원본 파일 미존재).
    """
    _load_all()
    if not _files_available:
        return _EMPTY_GRAPH, "not_implemented", "그래프 데이터 파일이 아직 준비되지 않았습니다."

    hops = 2 if hops >= 2 else 1
    user_node_id = _user_node_id(user_id)
    nodes: dict[str, dict] = {
        user_node_id: {
            "node_id": user_node_id, "node_type": "user", "ref_id": user_id,
            "label": f"User {user_id:05d}", "hop": 0,
        }
    }
    edges: list[dict] = []

    uidx = _user_id_to_uidx.get(user_id)
    if uidx is None:
        return {"nodes": list(nodes.values()), "edges": edges}, "ok", (
            "이 유저는 그래프 데이터에 없습니다(신규 유저이거나 학습 시점 이후 유입된 유저일 수 있습니다)."
        )

    all_tidx = set(_uidx2tidx_train.get(uidx, []))
    if not all_tidx:
        return {"nodes": list(nodes.values()), "edges": edges}, "ok", (
            "이 유저는 상호작용 기록이 없어 그래프를 표시할 수 없습니다(콜드 유저)."
        )

    purchased_product_ids = _purchased_by_user.get(user_id, set())
    ordered_tidx = sorted(
        all_tidx,
        key=lambda t: (_tidx_to_product_id.get(t) not in purchased_product_ids, t),
    )[:MAX_PRODUCTS_HOP1]
    truncated = len(all_tidx) > MAX_PRODUCTS_HOP1

    own_segment_id = _uidx2pidx.get(uidx)
    segment_best_lift: dict[int, float] = {}

    for tidx in ordered_tidx:
        product_id = _tidx_to_product_id.get(tidx)
        if product_id is None:
            continue
        product_id = int(product_id)
        product_node_id = _product_node_id(product_id)
        purchased = product_id in purchased_product_ids
        nodes[product_node_id] = {
            "node_id": product_node_id, "node_type": "product", "ref_id": product_id,
            "purchased": purchased, "hop": 1,
        }
        edges.append({
            "source": user_node_id, "target": product_node_id,
            "edge_type": "purchased" if purchased else "viewed", "hop": 1,
        })

        top_segments = sorted(_tidx2pidx.get(tidx, []), key=lambda p: p[1], reverse=True)
        for segment_id, lift in top_segments[:MAX_SEGMENTS_PER_PRODUCT]:
            segment_id = int(segment_id)
            segment_node_id = _segment_node_id(segment_id)
            is_own = segment_id == own_segment_id
            prev_own = nodes.get(segment_node_id, {}).get("is_own_segment", False)
            nodes[segment_node_id] = {
                "node_id": segment_node_id, "node_type": "segment", "ref_id": segment_id,
                "segment_name": _segment_names.get(segment_id, f"세그먼트 {segment_id}"),
                "is_own_segment": is_own or prev_own, "hop": 1,
            }
            edges.append({
                "source": product_node_id, "target": segment_node_id,
                "edge_type": "lift", "lift": float(lift), "hop": 1,
            })
            segment_best_lift[segment_id] = max(segment_best_lift.get(segment_id, 0.0), float(lift))

    if own_segment_id is not None:
        segment_node_id = _segment_node_id(own_segment_id)
        if segment_node_id not in nodes:
            nodes[segment_node_id] = {
                "node_id": segment_node_id, "node_type": "segment", "ref_id": own_segment_id,
                "segment_name": _segment_names.get(own_segment_id, f"세그먼트 {own_segment_id}"),
                "is_own_segment": True, "hop": 1,
            }
        else:
            nodes[segment_node_id]["is_own_segment"] = True
        edges.append({
            "source": user_node_id, "target": segment_node_id,
            "edge_type": "own_segment", "hop": 1,
        })

    if hops == 2:
        shown_tidx = set(ordered_tidx)
        for segment_id in _pick_expanded_segments(own_segment_id, segment_best_lift):
            segment_node_id = _segment_node_id(segment_id)
            added = 0
            for tidx, lift in _pidx2tidx.get(segment_id, []):
                if added >= HOP2_MAX_PRODUCTS_PER_SEGMENT:
                    break
                if tidx in shown_tidx:
                    continue
                product_id = _tidx_to_product_id.get(tidx)
                if product_id is None:
                    continue
                product_id = int(product_id)
                product_node_id = _product_node_id(product_id)
                if product_node_id not in nodes:
                    nodes[product_node_id] = {
                        "node_id": product_node_id, "node_type": "product", "ref_id": product_id,
                        "purchased": None, "hop": 2,  # 유저 본인의 상호작용이 아닌 세그먼트 연관 인기 상품
                    }
                edges.append({
                    "source": segment_node_id, "target": product_node_id,
                    "edge_type": "lift", "lift": float(lift), "hop": 2,
                })
                shown_tidx.add(tidx)
                added += 1

    node_list, edge_list, was_capped = _apply_total_node_cap(nodes, edges)
    message = "상호작용/연관 상품이 많아 일부만 표시했습니다." if (truncated or was_capped) else None
    return {"nodes": node_list, "edges": edge_list}, "ok", message
