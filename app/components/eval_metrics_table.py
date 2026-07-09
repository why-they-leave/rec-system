from textwrap import dedent

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from utils.data_loader import get_user_twiddler_case, load_twiddler_eval

# src/evaluation/evaluate_twiddler.py::ALL_SEGMENTS_LABEL과 동일한 값 — population 전체 평균 버킷.
_ALL_SEGMENTS_LABEL = "ALL"

# context별 condition 라벨 — reports/UI_TAB_RESTRUCTURE_PLAN.md §Tab1 표기 그대로.
_CONDITION_LABELS: dict[str, dict[str, str]] = {
    "main": {"baseline": "ALS only", "twiddler": "ALS+Twiddler"},
    "detail": {"baseline": "보완재 only", "twiddler": "보완재+Twiddler"},
    "main_lightgcn_bipartite": {
        "baseline": "LightGCN bipartite only",
        "twiddler": "LightGCN bipartite+Twiddler",
    },
}
# baseline/twiddler 2계열 고정 색상 — seaborn "colorblind" 팔레트 첫 두 색(파랑/주황)과
# 동일한 CVD-safe 조합(오늘 노트북들이 쓴 PALETTE와 통일, 카테고리 색은 절대 순환하지 않음).
_CONDITION_COLORS: dict[str, str] = {"baseline": "#0173B2", "twiddler": "#DE8F05"}

_ACCURACY_COLS = ["condition", "k", "HR", "Recall", "NDCG", "eval_users"]
_DIVERSITY_COLS = [
    "condition",
    "k",
    "repetition_rate",
    "unique_item_ratio",
    "categories_first",
    "categories_cumulative",
    "n_users",
]
_DIVERSITY_RENAME = {
    "repetition_rate": "반복률(중복)",
    "unique_item_ratio": "고유 아이템 비율",
    "categories_first": "1회차 카테고리 수",
    "categories_cumulative": "누적 카테고리 수",
    "eval_users": "평가 건수",
    "n_users": "평가 건수",
}

# 그래프에 표시할 지표 — (컬럼명, 서브플롯 제목). 한 축에 하나의 지표만 두어(dual-axis 금지)
# 지표 개수만큼 서브플롯을 나란히 배치한다.
_ACCURACY_METRICS = [("HR", "HR@K"), ("Recall", "Recall@K"), ("NDCG", "NDCG@K")]
_DIVERSITY_METRICS = [
    ("repetition_rate", "반복률(중복, 낮을수록 다양)"),
    ("categories_cumulative", "누적 카테고리 수(높을수록 다양)"),
]


def _localize(df: pd.DataFrame, context: str, cols: list[str]) -> pd.DataFrame:
    """condition 값을 context별 한국어 라벨로 바꾸고 필요한 컬럼만 남긴다(표 뷰용)."""
    labels = _CONDITION_LABELS[context]
    out = df[cols].copy()
    out["condition"] = out["condition"].map(labels).fillna(out["condition"])
    return out.rename(columns=_DIVERSITY_RENAME)


def _grouped_bar_figure(
    df: pd.DataFrame, context: str, metrics: list[tuple[str, str]]
) -> go.Figure:
    """condition(baseline/twiddler) 그룹 막대그래프 — 지표별 서브플롯, K가 x축.

    범례는 첫 서브플롯에만 표시하고 나머지는 legendgroup으로 묶어 중복 노출을 막는다
    (지표가 2~3개뿐이라 범례 반복이 오히려 산만하다).
    """
    labels = _CONDITION_LABELS[context]
    k_order = sorted(df["k"].unique())
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=[title for _, title in metrics])

    for col, (metric_key, _title) in enumerate(metrics, start=1):
        for condition in ("baseline", "twiddler"):
            sub = df[df["condition"] == condition].set_index("k").reindex(k_order)
            fig.add_trace(
                go.Bar(
                    x=[f"K={k}" for k in k_order],
                    y=sub[metric_key],
                    name=labels[condition],
                    marker_color=_CONDITION_COLORS[condition],
                    text=sub[metric_key].round(4),
                    textposition="outside",
                    legendgroup=condition,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        barmode="group",
        height=300,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.15, "xanchor": "center", "x": 0.5},
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def _interpret_headline(
    df: pd.DataFrame,
    context: str,
    metric_key: str,
    metric_label: str,
    higher_is_better: bool,
) -> str:
    """baseline→twiddler 변화율을 K별로 계산해 한 줄 해석 문구를 만든다.

    K마다 방향이 엇갈릴 수 있어(예: HR@5 개선, HR@10 하락) "전반적으로 개선됐다"처럼
    단정하지 않고, 실제 계산된 숫자를 K별로 그대로 보여준 뒤 방향이 갈리면 그 사실 자체를
    알린다 — 없는 경향을 있는 것처럼 보여주지 않기 위함(하드코딩 문구 대신 CSV 재계산 때마다
    자동 갱신되도록 설계, #10 이슈의 "숫자-표시 불일치" 재발 방지).
    """
    pivot = df.pivot(index="k", columns="condition", values=metric_key).sort_index()
    if "baseline" not in pivot.columns or "twiddler" not in pivot.columns:
        return ""

    parts, directions = [], []
    for k, row in pivot.iterrows():
        base, twid = row["baseline"], row["twiddler"]
        if pd.isna(base) or pd.isna(twid) or base == 0:
            continue
        delta_pct = (twid - base) / base * 100
        is_better = (delta_pct > 0) if higher_is_better else (delta_pct < 0)
        directions.append(is_better)
        arrow = "▲" if delta_pct > 0 else ("▼" if delta_pct < 0 else "–")
        parts.append(f"K={int(k)} {arrow}{abs(delta_pct):.1f}%")

    if not parts:
        return ""

    if all(directions):
        verdict = "전반적으로 개선"
    elif not any(directions):
        verdict = "전반적으로 악화"
    else:
        verdict = "K에 따라 효과가 엇갈림 — 특정 K에 최적화하면 다른 K가 희생될 수 있음"

    return f"💡 {metric_label} 변화 ({_CONDITION_LABELS[context]['twiddler']} vs baseline): {', '.join(parts)} — {verdict}"


def _render_metric_section(
    df: pd.DataFrame,
    context: str,
    metrics: list[tuple[str, str]],
    cols: list[str],
    key: str,
    headline: tuple[str, str, bool] | None = None,
) -> None:
    """그래프를 기본으로 보여주고, 원본 표는 expander에 접어둔다(접근성: 표 뷰는 항상 존재).

    headline: (metric_key, metric_label, higher_is_better) — 주어지면 그래프 바로 아래에
    자동 계산된 해석 캡션을 붙인다(표를 펼쳐보기 전에 "뭐가 달라졌는지"부터 보이게).
    """
    if df.empty:
        st.caption("데이터가 없습니다.")
        return
    st.plotly_chart(
        _grouped_bar_figure(df, context, metrics),
        width="stretch",
        key=key,
        config={"displayModeBar": False},  # 카메라/줌 등 Plotly 기본 툴바 숨김(요청 반영)
    )
    if headline:
        metric_key, metric_label, higher_is_better = headline
        interpretation = _interpret_headline(
            df, context, metric_key, metric_label, higher_is_better
        )
        if interpretation:
            st.caption(interpretation)
    with st.expander("표로 보기"):
        st.dataframe(_localize(df, context, cols), hide_index=True, width="stretch")


def render_eval_metrics(context: str, persona_label: str | None = None) -> None:
    """사전계산된 Twiddler 정확도/다양성 지표를 population 전체 + 선택 페르소나 breakdown 2단으로 렌더링.

    context: "main"(ALS) 또는 "detail"(보완재) — src/evaluation/evaluate_twiddler.py가
    생성한 data/outputs/eval/twiddler_{accuracy,diversity}.csv를 읽는다.
    persona_label: 현재 선택된 유저의 페르소나(segment_name). 주어지면 population 표
    바로 아래에 "이 페르소나만" 필터링한 breakdown을 추가로 보여준다(둘 다 population
    aggregate이지 유저 1명 지표가 아니다 — HR/NDCG는 표본이 많아야 의미가 있어서 개인
    단위로는 보여주지 않는다, reports/UI_TAB_RESTRUCTURE_PLAN.md §Tab1 참고).
    """
    try:
        accuracy_df, diversity_df = load_twiddler_eval()
    except FileNotFoundError:
        st.info(
            "🚧 아직 계산되지 않았습니다. `python -m src.evaluation.evaluate_twiddler` 실행 후 "
            "`data/outputs/eval/` 에 생성된 CSV를 읽어옵니다."
        )
        return

    acc_ctx = accuracy_df[accuracy_df["context"] == context]
    div_ctx = diversity_df[diversity_df["context"] == context]

    st.markdown("**① 전체 정확도 (단일 세션/조회 기준, population 평균)**")
    _render_metric_section(
        acc_ctx[acc_ctx["segment"] == _ALL_SEGMENTS_LABEL],
        context,
        _ACCURACY_METRICS,
        _ACCURACY_COLS,
        key=f"acc_all_{context}",
        headline=("HR", "HR@K", True),
    )

    st.markdown("**② 전체 다양성 (반복 새로고침/재방문 기준, population 평균)**")
    _render_metric_section(
        div_ctx[div_ctx["segment"] == _ALL_SEGMENTS_LABEL],
        context,
        _DIVERSITY_METRICS,
        _DIVERSITY_COLS,
        key=f"div_all_{context}",
        headline=("repetition_rate", "반복률", False),
    )

    if not persona_label:
        return

    seg_acc = acc_ctx[acc_ctx["segment"] == persona_label]
    seg_div = div_ctx[div_ctx["segment"] == persona_label]
    st.markdown(f"**③ 선택된 페르소나 breakdown ({persona_label})**")
    if seg_acc.empty and seg_div.empty:
        st.caption("이 페르소나는 평가 데이터가 부족해 breakdown을 계산하지 못했습니다.")
        return
    _render_metric_section(
        seg_acc,
        context,
        _ACCURACY_METRICS,
        _ACCURACY_COLS,
        key=f"acc_seg_{context}",
        headline=("HR", "HR@K", True),
    )
    _render_metric_section(
        seg_div,
        context,
        _DIVERSITY_METRICS,
        _DIVERSITY_COLS,
        key=f"div_seg_{context}",
        headline=("repetition_rate", "반복률", False),
    )


def render_user_twiddler_case(user_id: int, heading: str = "Twiddler 재랭킹 근거") -> None:
    """선택된 유저 1명의 실제 Twiddler 재랭킹 근거(alpha/decay/선호 카테고리)를 보여준다.

    population 평균 지표(HR/NDCG)와 달리 유저 1명 기준으로는 HR/NDCG가 0 또는 1의 노이즈성
    값이라 "정확도"로 보여주지 않는다 — 대신 실제로 계산된 재랭킹 파라미터 자체를 보여줘
    "이 유저에게 왜 이렇게 재정렬됐는지"를 설명한다(아래 before/after 카드 비교의 근거 숫자).

    heading: 호출부에서 번호(예: "2. Twiddler 재랭킹 근거")를 붙이고 싶을 때 넘긴다.
    """
    # 유저 번호/페르소나는 위쪽 페르소나·유저 카드에 이미 표시되므로 여기서는 반복하지
    # 않고 섹션 라벨만 짧게 둔다(요청 반영: "반복 헤딩 없이").
    st.markdown(f'<div class="section-label">{heading}</div>', unsafe_allow_html=True)
    case = get_user_twiddler_case(user_id)
    if case is None:
        st.caption(
            "이 유저는 페르소나 데이터가 없어 Twiddler가 적용되지 않습니다(인기도 기반 추천)."
        )
        return

    # st.metric 3개를 회색 테두리 상자에 넣은 버전은 글씨가 작고 서로 구분이 안 돼
    # 밋밋하다는 피드백(요청 반영: "디자인적으로 확 안들어와") — 아이콘 + 컬러 포인트
    # + 훨씬 큰 숫자로 된 커스텀 카드로 바꾼다. 기술 용어(alpha/decay)는 여전히 괄호로
    # 병기해 일반 사용자는 의미로, 개발자는 파라미터명으로 읽을 수 있게 한다.
    st.markdown(
        dedent(
            f"""
            <div class="twiddler-case-grid">
                <div class="twiddler-case-card">
                    <div class="twiddler-case-label">관심 카테고리 반영 강도 (alpha)</div>
                    <div class="twiddler-case-value">{case["alpha"]:.3f}</div>
                </div>
                <div class="twiddler-case-card">
                    <div class="twiddler-case-label">중복 노출 감소율 (decay)</div>
                    <div class="twiddler-case-value">{case["decay"]:.3f}</div>
                </div>
                <div class="twiddler-case-card">
                    <div class="twiddler-case-label">선호 카테고리</div>
                    <div class="twiddler-case-value">{case["top_category"] or "-"}</div>
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )
    st.caption(
        f"카테고리 편차 {case['top_category_deviation']:.3f} · 편차가 높을수록 해당 카테고리 점수를 높임"
    )
