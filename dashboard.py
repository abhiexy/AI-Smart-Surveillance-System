
import streamlit as st
import pandas as pd
import os
import json
import time
from PIL import Image

st.set_page_config(
    page_title="Smart Surveillance Dashboard",
    page_icon="📹",
    layout="wide"
)

LOG_FILE = "outputs/behaviour_history.csv"
FRAME_FILE = "outputs/live_frame.jpg"
STATUS_FILE = "outputs/live_status.json"

st.title("📹 Smart Surveillance Dashboard")
st.caption("Real-Time Human Behaviour Monitoring")

# Sidebar controls
st.sidebar.header("Dashboard Settings")

auto_refresh = st.sidebar.checkbox(
    "Live auto refresh",
    value=True
)

refresh_interval = st.sidebar.selectbox(
    "Refresh interval",
    [1, 2, 3, 5],
    index=2
)

# Read live status
status = {
    "running": False,
    "people": []
}

if os.path.exists(STATUS_FILE):
    try:
        with open(STATUS_FILE, "r") as file:
            status = json.load(file)
    except Exception:
        pass

# Live monitoring
st.subheader("🔴 Live Monitoring")

if status.get("running", False):
    st.success("Detection engine is running")
else:
    st.warning("Detection engine is stopped or not publishing")

live_col, people_col = st.columns([2, 1])

with live_col:
    st.markdown("**Live Webcam Feed**")

    if os.path.exists(FRAME_FILE):
        try:
            image = Image.open(FRAME_FILE)
            st.image(
                image,
                caption="Live Human Behaviour Detection",
                use_container_width=True
            )
        except Exception:
            st.warning("Waiting for the next webcam frame...")
    else:
        st.info("Waiting for webcam frame...")

with people_col:
    st.markdown("**Currently Detected People**")

    people = status.get("people", [])

    if people:
        for person in people:
            st.info(
                f"Person ID: {person['id']}\n\n"
                f"Behaviour: {person['behaviour']}\n\n"
                f"Confidence: {person['confidence']}"
            )
    else:
        st.write("No people currently detected.")

    st.metric("People in Current Frame", len(people))

st.divider()

# Behaviour counts
st.subheader("📊 Behaviour Overview")

behaviour_counts = {
    "Standing": 0,
    "Sitting": 0,
    "Walking": 0,
    "Running": 0
}

for person in people:
    behaviour = person.get("behaviour")

    if behaviour in behaviour_counts:
        behaviour_counts[behaviour] += 1

cols = st.columns(4)

for col, (behaviour, count) in zip(
    cols, behaviour_counts.items()
):
    col.metric(behaviour, count)

st.divider()

# Historical records
st.subheader("📚 Behaviour History")

if os.path.exists(LOG_FILE):
    try:
        df = pd.read_csv(LOG_FILE)

        if not df.empty:
            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"],
                errors="coerce"
            )

            col1, col2 = st.columns(2)

            col1.metric("Total Events", len(df))
            col2.metric(
                "Unique Tracking IDs",
                df["Person ID"].nunique()
            )

            st.markdown("**Behaviour Distribution**")
            st.bar_chart(df["Behaviour"].value_counts())

            st.markdown("**Person-wise Activity**")
            person_counts = pd.crosstab(
                df["Person ID"],
                df["Behaviour"]
            )
            st.dataframe(
                person_counts,
                use_container_width=True
            )

            st.markdown("**Recent Events**")
            st.dataframe(
                df.sort_values(
                    "Timestamp",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Behaviour History",
                df.to_csv(index=False).encode("utf-8"),
                "behaviour_history.csv",
                "text/csv"
            )
        else:
            st.info("No behaviour events recorded yet.")

    except Exception as error:
        st.error(f"Unable to read history: {error}")
else:
    st.info("Behaviour history file not found.")

# Refresh dashboard
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
