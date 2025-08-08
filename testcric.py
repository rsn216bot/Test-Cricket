
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Data Source
# =========================
GOOGLE_SHEET_ID = "17DwxDTomoofFTb8NdTo-XQNQ4diGP53f-ZYd0Iln1HM"
DATA_URL = "https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv".format(GOOGLE_SHEET_ID=GOOGLE_SHEET_ID)

@st.cache_data
def load_data(url: str):
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.warning("Couldn't load from Google Sheets URL. Falling back to local CSV 'All Test Matches Till 2025.csv'.\nError: {}".format(e))
        df = pd.read_csv("All Test Matches Till 2025.csv")
    # Clean and enrich
    if "ballDateTime" in df.columns:
        df["ballDateTime"] = pd.to_datetime(df["ballDateTime"], errors="coerce")
        df["year"] = df["ballDateTime"].dt.year
    else:
        # If no datetime column present, try inferring year from a 'matchDate' column if exists
        if "matchDate" in df.columns:
            df["ballDateTime"] = pd.to_datetime(df["matchDate"], errors="coerce")
            df["year"] = df["ballDateTime"].dt.year
        else:
            df["year"] = np.nan

    # Normalise connection
    if "battingConnectionId" in df.columns:
        df["conn_clean"] = (
            df["battingConnectionId"]
            .fillna("nan").astype(str).str.strip().str.lower()
        )
    else:
        df["conn_clean"] = "nan"

    # Normalise booleans
    def to_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        s = str(x).strip().lower()
        return s in ("true", "1", "yes")

    if "isWicket" in df.columns:
        df["isWicket_bool"] = df["isWicket"].apply(to_bool)
    else:
        df["isWicket_bool"] = False

    # Numeric safety
    for c in ["runsScored", "shotAngle", "shotMagnitude", "overNumber"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

df = load_data(DATA_URL)

# =========================
# Sidebar Filters (Batting-first focus)
# =========================
st.sidebar.header("Filters — Batting")

# Guard against missing columns with safe get
def safe_unique(col):
    return sorted(df[col].dropna().unique()) if col in df.columns else []

batter = st.sidebar.selectbox("Batter", safe_unique("battingPlayer"))
opposition = st.sidebar.multiselect("Opposition (bowlingTeam)", safe_unique("bowlingTeam"))
bowler = st.sidebar.multiselect("Bowler", safe_unique("bowlerPlayer"))
bowling_type = st.sidebar.multiselect("Bowling Type", safe_unique("bowlingTypeId"))
years = st.sidebar.multiselect("Year", sorted(df["year"].dropna().unique()) if "year" in df.columns else [])

max_over = int(np.nanmax(df["overNumber"])) if "overNumber" in df.columns and not df["overNumber"].isna().all() else 0
over_range = st.sidebar.slider("Over range", 0, max_over, (0, max_over))

# Apply filters
filt = df.copy()
if batter:
    filt = filt[filt.get("battingPlayer", "").eq(batter)]
if opposition:
    filt = filt[filt.get("bowlingTeam", "").isin(opposition)]
if bowler:
    filt = filt[filt.get("bowlerPlayer", "").isin(bowler)]
if bowling_type:
    filt = filt[filt.get("bowlingTypeId", "").isin(bowling_type)]
if years:
    filt = filt[filt.get("year", "").isin(years)]
if "overNumber" in filt.columns:
    filt = filt[(filt["overNumber"] >= over_range[0]) & (filt["overNumber"] <= over_range[1])]

st.title("Test Batting Dashboard (Mid-2023 to 2025)")
st.caption("Data loaded from Google Sheets. Change the sheet ID in code if needed.")

# =========================
# Metric helpers
# =========================
SAFE_CONN = {"welltimed", "middled", "left", "blank", "nan"}

def false_shot_pct(df_in: pd.DataFrame) -> float:
    if df_in.empty or "conn_clean" not in df_in.columns:
        return 0.0
    conn = df_in["conn_clean"]
    false_shots = conn[~conn.isin(SAFE_CONN)]
    pct = (len(false_shots) / len(conn)) * 100 if len(conn) else 0
    return round(pct, 2)

def agg_block(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df_in.empty or group_col not in df_in.columns:
        return pd.DataFrame(columns=[group_col, "Runs", "Balls", "Dismissals", "False Shot %", "Average", "Strike Rate"])
    g = df_in.groupby(group_col, dropna=False).agg(
        Runs=("runsScored", "sum") if "runsScored" in df_in.columns else ("conn_clean", "count"),
        Balls=("runsScored", "count") if "runsScored" in df_in.columns else ("conn_clean", "count"),
        Dismissals=("isWicket_bool", "sum") if "isWicket_bool" in df_in.columns else ("conn_clean", "size"),
    ).reset_index()
    # False Shot % per group
    fs_list = []
    for val in g[group_col]:
        subset = df_in[df_in[group_col] == val]
        fs_list.append(false_shot_pct(subset))
    g["False Shot %"] = fs_list
    # Average & SR
    def safe_avg(r):
        dism = r["Dismissals"] if pd.notna(r["Dismissals"]) else 0
        runs = r["Runs"] if pd.notna(r["Runs"]) else 0
        return round(runs / dism, 2) if dism else round(runs, 2)
    def safe_sr(r):
        runs = r["Runs"] if pd.notna(r["Runs"]) else 0
        balls = r["Balls"] if pd.notna(r["Balls"]) else 0
        return round((runs / balls) * 100, 2) if balls else 0.0
    g["Average"] = g.apply(safe_avg, axis=1)
    g["Strike Rate"] = g.apply(safe_sr, axis=1)
    g = g.sort_values(["Runs", "Strike Rate"], ascending=[False, False])
    return g

# =========================
# Batting-related tabs
# =========================
bat_tab1, bat_tab2 = st.tabs(["Batting Feet Stats", "Batting Shot Type Stats"])

with bat_tab1:
    st.subheader("Batting Feet Stats")
    st.dataframe(agg_block(filt, "battingFeetId"))

with bat_tab2:
    st.subheader("Batting Shot Type Stats")
    st.dataframe(agg_block(filt, "battingShotTypeId"))

# =========================
# Bowling-related tabs (impact on the selected batter)
# =========================
bowl_tab1, bowl_tab2, bowl_tab3 = st.tabs(["Bowling Hand Stats", "Bowling Detail Stats", "Bowler Stats"])

with bowl_tab1:
    st.subheader("Bowling Hand Stats (vs selected batter)")
    st.dataframe(agg_block(filt, "bowlingHandId"))

with bowl_tab2:
    st.subheader("Bowling Detail Stats (vs selected batter)")
    st.dataframe(agg_block(filt, "bowlingDetailId"))

with bowl_tab3:
    st.subheader("Bowler Stats (vs selected batter)")
    st.dataframe(agg_block(filt, "bowlerPlayer"))

# =========================
# Wagon Wheel — Shot Areas & Hot Zones
# =========================
ww_tab1, ww_tab2 = st.tabs(["Wagon Wheel (Shot Areas)", "Hot Zones (Runs by Angle)"])

def plot_wagon_wheel(df_in: pd.DataFrame, title: str):
    if df_in.empty or "shotAngle" not in df_in.columns or "shotMagnitude" not in df_in.columns:
        st.info("No shot tracking data (shotAngle/shotMagnitude) available for current filter.")
        return
    data = df_in[["shotAngle", "shotMagnitude", "runsScored"]].dropna()
    if data.empty:
        st.info("No valid shotAngle/shotMagnitude rows after filtering.")
        return
    theta = np.deg2rad(data["shotAngle"].astype(float).values % 360)
    mag = data["shotMagnitude"].astype(float).values.copy()
    is_boundary = data["runsScored"].isin([4, 6]).values if "runsScored" in data.columns else np.zeros_like(mag, dtype=bool)
    mag = np.where(is_boundary, mag * 1.2, mag)  # simple bump for boundaries

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.scatter(theta, mag, alpha=0.6)
    ax.set_title(title)
    st.pyplot(fig)

with ww_tab1:
    st.subheader("Wagon Wheel — Shot Areas & Magnitude")
    plot_wagon_wheel(filt, f"Shots for {batter}")

with ww_tab2:
    st.subheader("Hot Zones — Runs density by angle (binned)")
    if not filt.empty and "shotAngle" in filt.columns and filt["shotAngle"].notna().any():
        angle = (filt["shotAngle"] % 360).dropna().astype(float)
        bins = np.arange(0, 361, 30)
        labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
        df_angles = pd.cut(angle, bins=bins, labels=labels, include_lowest=True)
        tmp = filt.loc[angle.index, ["runsScored"]].copy() if "runsScored" in filt.columns else pd.DataFrame(index=angle.index, data={"runsScored": 0})
        tmp["angleBin"] = df_angles.values
        hot = tmp.groupby("angleBin")["runsScored"].sum().reindex(labels, fill_value=0).reset_index()
        st.bar_chart(hot.set_index("angleBin"))
    else:
        st.info("No shotAngle data available for current filter.")

# =========================
# Line–Length Matrix (False Shot % and Average)
# =========================
st.subheader("Line × Length Matrix — False Shot % and Average")
if filt.empty or "lengthTypeId" not in filt.columns or "lineTypeId" not in filt.columns:
    st.info("No data for current filters or missing line/length columns.")
else:
    grp = filt.groupby(["lengthTypeId", "lineTypeId"], dropna=False)
    rows = []
    for (length, line), d in grp:
        fs = false_shot_pct(d)
        dismissals = d["isWicket_bool"].sum() if "isWicket_bool" in d.columns else 0
        runs = d["runsScored"].sum() if "runsScored" in d.columns else 0
        avg = round(runs / dismissals, 2) if dismissals > 0 else round(runs, 2)
        rows.append({"lengthTypeId": length, "lineTypeId": line, "False Shot %": fs, "Average": avg})
    mat = pd.DataFrame(rows)
    if not mat.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("False Shot %")
            p1 = mat.pivot(index="lengthTypeId", columns="lineTypeId", values="False Shot %")
            st.dataframe(p1.style.background_gradient(cmap="Reds", axis=None))
        with c2:
            st.caption("Average")
            p2 = mat.pivot(index="lengthTypeId", columns="lineTypeId", values="Average")
            st.dataframe(p2.style.background_gradient(cmap="Greens", axis=None))
