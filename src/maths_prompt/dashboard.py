"""Streamlit dashboard for monitoring prompt optimisation progress."""

import json

import pandas as pd
import streamlit as st

from maths_prompt.config import EVAL_LOG_PATH, TEST_LOG_PATH

st.set_page_config(page_title="maths-prompt dashboard", layout="wide")
st.title("maths-prompt optimisation dashboard")


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


train_logs = load_jsonl(EVAL_LOG_PATH)
test_logs = load_jsonl(TEST_LOG_PATH)

if not train_logs:
    st.info("No evaluation logs yet. Run `uv run maths-prompt` to start.")
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

# Full history table
st.subheader("Evaluation history")
display_df = train_df[["iteration", "session", "accuracy", "num_correct", "num_problems", "prompt"]].copy()
display_df["accuracy"] = display_df["accuracy"].map("{:.1%}".format)
display_df["prompt"] = display_df["prompt"].str[:80] + "..."
st.dataframe(display_df, use_container_width=True)
