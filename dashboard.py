import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

EVAL_LOG = Path("logs/evaluations.jsonl")
TEST_LOG = Path("logs/test_results.jsonl")
REFRESH_SECONDS = 10


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


@st.fragment(run_every=REFRESH_SECONDS)
def live_section():
    train = load_jsonl(EVAL_LOG)
    test = load_jsonl(TEST_LOG)

    # --- Chart ---
    fig = go.Figure()

    if train:
        # Group training rows by session
        sessions: dict[int, list] = {}
        for row in train:
            s = row.get("session", 1)
            sessions.setdefault(s, []).append(row)

        # Find last iteration per session (for aligning test results)
        session_last_iter: dict[int, int] = {}
        for s, rows in sessions.items():
            session_last_iter[s] = max(r["iteration"] for r in rows)

        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        for idx, (s, rows) in enumerate(sorted(sessions.items())):
            rows_sorted = sorted(rows, key=lambda r: r["iteration"])
            x = [r["iteration"] for r in rows_sorted]
            y = [r["accuracy"] for r in rows_sorted]
            color = colors[idx % len(colors)]

            # Shaded ±5pp noise band
            y_upper = [min(v + 0.05, 1.0) for v in y]
            y_lower = [max(v - 0.05, 0.0) for v in y]
            fig.add_trace(go.Scatter(
                x=x + x[::-1],
                y=y_upper + y_lower[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"Train session {s}",
                line=dict(color=color),
                marker=dict(size=5),
                hovertemplate="Iter %{x}<br>Accuracy: %{y:.1%}<extra>Session " + str(s) + "</extra>",
            ))

    # Test results as red squares aligned to the last training iteration of each session
    if test:
        test_x = []
        test_y = []
        test_text = []
        for i, row in enumerate(test):
            session_index = i + 1
            # Align to last iteration of that session, or fall back to session index
            x_val = session_last_iter.get(session_index, session_index) if train else session_index
            test_x.append(x_val)
            test_y.append(row["accuracy"])
            test_text.append(
                f"Session {session_index}<br>Test: {row['accuracy']:.1%} ({row['num_correct']}/{row['num_problems']})"
            )

        fig.add_trace(go.Scatter(
            x=test_x,
            y=test_y,
            mode="markers",
            name="Test accuracy",
            marker=dict(symbol="square", size=12, color="red"),
            text=test_text,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Accuracy",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Prompt inspector ---
    if train:
        options = {
            f"Iter {r['iteration']} (sess {r.get('session', '?')}) — {r['accuracy']:.1%}": r
            for r in reversed(train)
        }
        selected_label = st.selectbox("Inspect prompt", list(options.keys()))
        row = options[selected_label]
        st.code(row["prompt"], language=None)
        st.caption(f"{row['num_correct']}/{row['num_problems']} correct  |  Session {row.get('session', '?')}")
    else:
        st.info("No training data yet — start the optimizer with `uv run maths-prompt`.")


st.title("maths-prompt dashboard")
live_section()
