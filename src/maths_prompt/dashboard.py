"""Streamlit dashboard for monitoring prompt optimisation progress."""

import json

import pandas as pd
import streamlit as st

from maths_prompt.config import EVAL_LOG_PATH, SESSION_LOG_PATH, TEST_LOG_PATH

st.set_page_config(page_title="maths-prompt dashboard", layout="wide")
st.title("maths-prompt optimisation dashboard")


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        rows = []
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip partial lines mid-write
        return rows


train_logs = load_jsonl(EVAL_LOG_PATH)
test_logs = load_jsonl(TEST_LOG_PATH)
session_logs = load_jsonl(SESSION_LOG_PATH)

if not train_logs:
    st.info("No evaluation logs yet. Run `uv run maths-prompt start` to begin.")
    st.stop()

train_df = pd.DataFrame(
    [
        {
            "iteration": r["iteration"],
            "session": r.get("session", 1),
            "accuracy": r["accuracy"],
            "num_correct": r["num_correct"],
            "num_problems": r["num_problems"],
            "timestamp": r.get("timestamp", ""),
            "prompt": r["prompt"],
        }
        for r in train_logs
    ]
)

# Summary metrics
best_idx = train_df["accuracy"].idxmax()
best = train_df.loc[best_idx]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Best training accuracy", f"{best['accuracy']:.1%}")
col2.metric("Total evaluations", len(train_df))
col3.metric("Sessions", int(train_df["session"].max()))
if test_logs:
    latest_test = test_logs[-1]
    col4.metric("Latest test accuracy", f"{latest_test['accuracy']:.1%}")
else:
    col4.metric("Latest test accuracy", "—")

st.divider()

# Accuracy over iterations
st.subheader("Training accuracy over iterations")
st.line_chart(train_df.set_index("iteration")["accuracy"])

# Test accuracy over time
if test_logs:
    st.subheader("Test accuracy over sessions")
    test_df = pd.DataFrame(
        [{"session": i + 1, "test_accuracy": r["accuracy"]} for i, r in enumerate(test_logs)]
    )
    st.line_chart(test_df.set_index("session")["test_accuracy"])

st.divider()

# Best prompt
st.subheader("Best prompt found")
st.code(best["prompt"], language=None)

st.divider()

# Per-category breakdown for best eval
st.subheader("Per-category accuracy (best evaluation)")
best_log = train_logs[best_idx]
if "problems" in best_log:
    cat_df = pd.DataFrame(best_log["problems"])
    summary = cat_df.groupby("category")["correct"].agg(["sum", "count"])
    summary["accuracy"] = summary["sum"] / summary["count"]
    summary.columns = ["correct", "total", "accuracy"]
    st.dataframe(summary.style.format({"accuracy": "{:.1%}"}))

st.divider()

# API usage & cost
if session_logs:
    st.subheader("API usage per session")
    sess_df = pd.DataFrame(session_logs)

    total_cost = sess_df["estimated_cost_usd"].sum()
    total_tool_calls = sess_df["tool_calls_made"].sum()
    total_input = sess_df["input_tokens"].sum()
    total_cache_read = sess_df["cache_read_tokens"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated total cost", f"${total_cost:.3f}")
    c2.metric("Total tool calls", int(total_tool_calls))
    c3.metric("Total input tokens", f"{total_input:,}")
    c4.metric("Cache read tokens", f"{total_cache_read:,}")

    # Cost per session chart
    if len(sess_df) > 1:
        st.line_chart(sess_df.set_index("session")["estimated_cost_usd"])

    # Sessions table
    display_cols = ["session", "tool_calls_made", "input_tokens", "output_tokens",
                    "cache_creation_tokens", "cache_read_tokens", "estimated_cost_usd",
                    "test_accuracy", "success"]
    available = [c for c in display_cols if c in sess_df.columns]
    fmt = {}
    if "estimated_cost_usd" in available:
        fmt["estimated_cost_usd"] = "${:.4f}"
    if "test_accuracy" in available:
        fmt["test_accuracy"] = lambda x: f"{x:.1%}" if x is not None else "—"
    st.dataframe(sess_df[available].style.format(fmt), use_container_width=True)

    st.divider()

# Full history table
st.subheader("Evaluation history")


@st.fragment
def eval_history_table():
    display_df = train_df[["iteration", "session", "accuracy", "num_correct", "num_problems", "prompt"]].copy()
    display_df["accuracy"] = display_df["accuracy"].map("{:.1%}".format)
    display_df["prompt"] = display_df["prompt"].str[:80] + "..."
    event = st.dataframe(
        display_df,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )
    if event.selection and event.selection.rows:
        selected_row = event.selection.rows[0]
        full_prompt = train_df.iloc[selected_row]["prompt"]
        iter_num = int(train_df.iloc[selected_row]["iteration"])
        acc = train_df.iloc[selected_row]["accuracy"]
        st.caption(f"Iteration {iter_num} | Accuracy {acc:.1%}")
        st.code(full_prompt, language=None)


eval_history_table()

# CSV download with full prompts
export_df = train_df[["iteration", "session", "accuracy", "num_correct", "num_problems", "prompt"]].copy()
st.download_button("Download full CSV", export_df.to_csv(index=False), "evaluations.csv", "text/csv")

st.button("Refresh", on_click=st.rerun)
