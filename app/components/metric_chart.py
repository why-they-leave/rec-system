import pandas as pd
import plotly.graph_objects as go

COLOR_UP   = "#22c55e"
COLOR_DOWN = "#ef4444"
COLOR_SAME = "#9ca3af"


def _rank_color(rank_before: float, rank_after: float) -> str:
    if pd.isna(rank_before) or pd.isna(rank_after):
        return COLOR_SAME
    if rank_after < rank_before:
        return COLOR_UP
    if rank_after > rank_before:
        return COLOR_DOWN
    return COLOR_SAME


def render_bump_chart(before_df: pd.DataFrame, after_df: pd.DataFrame) -> go.Figure:
    """Twiddler 전/후 순위 변화 Bump Chart.
    before_df / after_df: item_id, rank 컬럼 필수.
    반환값: Plotly Figure (st.plotly_chart로 표시).
    """
    merged = (
        before_df[["item_id", "rank"]]
        .rename(columns={"rank": "rank_before"})
        .merge(
            after_df[["item_id", "rank"]].rename(columns={"rank": "rank_after"}),
            on="item_id",
            how="outer",
        )
    )

    fig = go.Figure()

    for _, row in merged.iterrows():
        rb = row["rank_before"]
        ra = row["rank_after"]
        color = _rank_color(rb, ra)

        rb_label = f"#{int(rb)}" if not pd.isna(rb) else "—"
        ra_label = f"#{int(ra)}" if not pd.isna(ra) else "—"

        fig.add_trace(
            go.Scatter(
                x=["Before", "After"],
                y=[rb, ra],
                mode="lines+markers+text",
                line=dict(color=color, width=2.5),
                marker=dict(size=9, color=color),
                text=[rb_label, ra_label],
                textposition=["middle left", "middle right"],
                textfont=dict(size=11),
                name=f"item {int(row['item_id'])}",
                showlegend=False,
                hovertemplate=(
                    f"<b>item_id: {int(row['item_id'])}</b><br>"
                    f"Before: {rb_label}<br>"
                    f"After:  {ra_label}<extra></extra>"
                ),
            )
        )

    valid_ranks = pd.concat(
        [merged["rank_before"].dropna(), merged["rank_after"].dropna()]
    )
    max_rank = int(valid_ranks.max()) if not valid_ranks.empty else 10

    fig.update_layout(
        xaxis=dict(
            tickvals=["Before", "After"],
            tickfont=dict(size=14, color="#1a1a1a"),
            fixedrange=True,
        ),
        yaxis=dict(
            autorange="reversed",
            tickvals=list(range(1, max_rank + 1)),
            title="순위",
            title_font=dict(size=12),
        ),
        margin=dict(l=60, r=60, t=20, b=20),
        height=440,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)

    return fig
