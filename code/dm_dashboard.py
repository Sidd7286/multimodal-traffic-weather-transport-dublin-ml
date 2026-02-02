import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import streamlit as st
import plotly.express as px
from wordcloud import WordCloud

# ---------------------------
# Paths & data loading
# ---------------------------
base_path = Path(r"../cleaned dataset")
 # ✅ update if needed

traffic = pd.read_csv(base_path / "merged_traffic_weather_main_dataset_clean.csv", low_memory=False)

# ---------- flexible loaders (Member 3 + 4) ----------
def load_first_existing(base_path: Path, candidates: list[str], **read_csv_kwargs):
    for f in candidates:
        p = base_path / f
        if p.exists():
            return pd.read_csv(p, **read_csv_kwargs), f
    return None, None

member3_df, member3_file = load_first_existing(
    base_path,
    [
        "bus_trip_updates_cleaned_member3.csv",
        "bus_trip_updates_raw_member3.csv",
        "bus_daily_cleaned_member3.csv",
    ],
    low_memory=False
)

member4_df, member4_file = load_first_existing(
    base_path,
    [
        "bus_text_enriched_member4.csv",
        "bus_gtfs_text_member4.csv",
    ],
    low_memory=False
)

# Parse dates where possible
if "date" in traffic.columns:
    traffic["date"] = pd.to_datetime(traffic["date"], errors="coerce")

if member3_df is not None and "date" in member3_df.columns:
    member3_df["date"] = pd.to_datetime(member3_df["date"], errors="coerce")

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("DM Project Dashboard")
menu = st.sidebar.radio(
    "Select Section",
    ["Traffic + Weather", "Daily Bus Delays", "Text Delay Analytics", "Project Summary"]
)

# ======================================================
# 1) Traffic + Weather
# ======================================================
if menu == "Traffic + Weather":
    st.title("Traffic & Weather Impact Analysis")

    st.subheader("Daily Traffic Trend")
    fig = px.line(traffic, x="date", y="daily_flow_total")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Temperature vs Traffic")
    fig = px.scatter(
        traffic,
        x="tmean",
        y="daily_flow_total",
        trendline="ols",
        labels={"tmean": "Mean Temperature (°C)", "daily_flow_total": "Daily Traffic Flow"}
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Rainfall vs Traffic")
    fig = px.scatter(
        traffic,
        x="rain_mm",
        y="daily_flow_total",
        trendline="ols",
        labels={"rain_mm": "Rainfall (mm)", "daily_flow_total": "Daily Traffic Flow"}
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Correlation Heatmap (Traffic & Weather)")
    numeric_traffic = traffic.select_dtypes(include="number")
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_traffic.corr(), cmap="coolwarm", annot=False)
    st.pyplot(plt)

# ======================================================
# 2) Member 3 — Bus Delays
# ======================================================
elif menu == "Daily Bus Delays":
    st.title("Member 3 — Bus Delay Dataset")

    if member3_df is None:
        st.error("Member 3 dataset file not found in this folder.")
        st.stop()

    st.caption(f"Loaded file: {member3_file} | Shape: {member3_df.shape}")
    st.dataframe(member3_df.head(20), width="stretch")

    # detect delay column
    delay_candidates = [
    "arrival_delay_sec",
    "delay_minutes",
    "delay_min",
    "delay",
    "bus_delay_mean_min"
]
    delay_col = next((c for c in delay_candidates if c in member3_df.columns), None)

    trips_candidates = ["bus_trips_count", "trip_count", "n_trips", "total_trips"]
    trips_col = next((c for c in trips_candidates if c in member3_df.columns), None)

    date_col = "date" if "date" in member3_df.columns else None

    st.subheader("Quick Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(member3_df):,}")
    c2.metric("Columns", f"{len(member3_df.columns):,}")
    c3.metric("Delay column", delay_col if delay_col else "Not found")

    if date_col and delay_col:
        st.subheader("Delay Trend")
        fig = px.line(member3_df.sort_values(date_col), x=date_col, y=delay_col)
        st.plotly_chart(fig, width="stretch")

    if delay_col:
        st.subheader("Delay Distribution")
        fig = px.histogram(member3_df, x=delay_col, nbins=50)
        st.plotly_chart(fig, width="stretch")

    if date_col and trips_col:
        st.subheader("Trips Over Time")
        fig = px.bar(member3_df.sort_values(date_col), x=date_col, y=trips_col)
        st.plotly_chart(fig, width="stretch")

# ======================================================
# 3) Member 4 — Text Delay Analytics
# ======================================================
elif menu == "Text Delay Analytics":
    st.title("Member 4 — Text Delay Analytics")

    if member4_df is None:
        st.error("Member 4 dataset file not found in this folder.")
        st.stop()

    st.caption(f"Loaded file: {member4_file} | Shape: {member4_df.shape}")
    st.dataframe(member4_df.head(20), width="stretch")

    # detect text + label columns
    text_candidates = ["text_clean", "clean_text_joined", "delay_text", "message", "text"]
    text_col = next((c for c in text_candidates if c in member4_df.columns), None)

    label_candidates = ["label", "delay_class", "class"]
    label_col = next((c for c in label_candidates if c in member4_df.columns), None)

    route_candidates = ["route_id", "route", "line_id"]
    route_col = next((c for c in route_candidates if c in member4_df.columns), None)

    st.subheader("Label / Class Distribution")
    if label_col:
        class_counts = (
            member4_df[label_col].astype(str)
            .value_counts()
            .reset_index(name="count")
            .rename(columns={"index": "class"})
        )
        fig = px.bar(class_counts, x="label", y="count")


        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No label column found (expected: label / delay_class).")

    st.subheader("Message Length (Word Count)")
    if text_col:
        member4_df["_word_count"] = member4_df[text_col].astype(str).apply(lambda x: len(x.split()))
        fig = px.histogram(member4_df, x="_word_count", nbins=60)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No text column found (text_clean / clean_text_joined / delay_text).")

    st.subheader("Route-wise Message Count")
    if route_col:
        rc = member4_df.groupby(route_col).size().reset_index(name="count").sort_values("count", ascending=False).head(30)
        fig = px.bar(rc, x=route_col, y="count")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No route column found (route_id / route).")

    st.subheader("Word Cloud")
    if text_col:
        text_blob = " ".join(member4_df[text_col].astype(str).tolist()[:50000])
        wc = WordCloud(width=900, height=400, background_color="white").generate(text_blob)
        plt.figure(figsize=(10, 4))
        plt.imshow(wc)
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.info("Word cloud needs a text column.")

# ======================================================
# 4) Project Summary
# ======================================================
elif menu == "Project Summary":
    st.title("Project Summary")

    st.markdown("""
### Member 1 + 2 (Merged Traffic + Weather)
- File: `merged_traffic_weather_main_dataset_clean.csv`
- Target: `daily_flow_total`
- Models: Linear Regression, Random Forest, Decision Tree, Gradient Boosting

### Member 3 (Bus Delay Dataset)
- Loaded: shown in dashboard (file auto-detected)
- Includes bus delay metrics + trip counts (depending on file)

### Member 4 (Text Delay Dataset)
- Loaded: shown in dashboard (file auto-detected)
- Text analytics + class distribution + word cloud
""")
