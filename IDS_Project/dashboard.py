"""Beginner-friendly Streamlit dashboard for the IDS demo."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit import st_autorefresh

from ai_agent import load_agent


DATA_PATH = Path("data/sample.csv")
MODEL_PATH = Path("model/lstm_model.h5")
ALERTS_LOG = Path("alerts.log")


def tail_log(log_path: Path, max_lines: int = 10) -> str:
    if not log_path.exists():
        return "No alerts yet."
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-max_lines:]) if lines else "No alerts yet."


@st.cache_resource
def get_agent(feature_columns: list[str]):
    return load_agent(MODEL_PATH, feature_columns)


st.set_page_config(page_title="AI IDS Dashboard", page_icon="🛡️", layout="wide")

st.title("AI-Based Intrusion Detection Dashboard")
st.caption("Academic prototype: simple, readable, and focused on core IDS concepts.")

if "running" not in st.session_state:
    st.session_state.running = False
if "row_index" not in st.session_state:
    st.session_state.row_index = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "latest" not in st.session_state:
    st.session_state.latest = None

raw_data = pd.read_csv(DATA_PATH)
feature_columns = list(raw_data.columns)
agent = get_agent(feature_columns)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Start Monitoring", use_container_width=True):
        st.session_state.running = True
with col2:
    if st.button("Stop Monitoring", use_container_width=True):
        st.session_state.running = False
with col3:
    if st.button("Reset", use_container_width=True):
        st.session_state.running = False
        st.session_state.row_index = 0
        st.session_state.history = []
        st.session_state.latest = None

status = "Running" if st.session_state.running else "Idle"
st.info(f"System Status: **{status}**")

if st.session_state.running:
    st_autorefresh(interval=1000, key="ids_refresh")
    row = raw_data.iloc[st.session_state.row_index]
    result = agent.predict(row.values)
    st.session_state.latest = result
    st.session_state.history.append(result.label)

    if result.label == "ATTACK":
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with ALERTS_LOG.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamp} - ALERT - Model flagged traffic as ATTACK\n")

    st.session_state.row_index = (st.session_state.row_index + 1) % len(raw_data)

latest = st.session_state.latest

left, middle, right = st.columns(3)
with left:
    st.metric("Latest Prediction", latest.label if latest else "--")
with middle:
    st.metric("Confidence", f"{latest.confidence:.2f}" if latest else "--")
with right:
    total_scans = len(st.session_state.history)
    st.metric("Rows Processed", total_scans)

st.subheader("Traffic Statistics")
if st.session_state.history:
    history_df = pd.DataFrame({"prediction": st.session_state.history})
    counts = history_df["prediction"].value_counts().reindex(["NORMAL", "ATTACK"], fill_value=0)
    st.bar_chart(counts)
else:
    st.write("No traffic processed yet. Click **Start Monitoring** to begin.")

st.subheader("Recent Alerts")
st.text_area("alerts.log", tail_log(ALERTS_LOG), height=200)

st.caption("Tip: The dashboard processes one CSV row per refresh to simulate real-time detection.")
