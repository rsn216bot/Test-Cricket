
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge

st.set_page_config(layout="wide")

# =========================
# Data Source (Google Sheets CSV)
# =========================
GOOGLE_SHEET_ID = "17DwxDTomoofFTb8NdTo-XQNQ4diGP53f-ZYd0Iln1HM"
DATA_URL = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv"

@st.cache_data
def load_data(url: str):
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error("Couldn't load from Google Sheets URL.\n"
                 "Share the sheet as 'Anyone with the link (Viewer)'.\n"
                 f"Error: {e}")
        return pd.DataFrame()

    # Clean & enrich
    if "ballDateTime" in df.columns:
        df["ballDateTime"] = pd.to_datetime(df["ballDateTime"], errors="coerce")
        df["year"] = df["ballDateTime"].dt.year
    else:
        df["year"] = np.nan

    # Booleans
    def to_bool(x):
        if isinstance(x, bool): return x
        if isinstance(x, (int, float)): return bool(x)
        s = str(x).strip().lower()
        return s in ("true", "1", "yes")
    df["isWicket_bool"] = df.get("isWicket", False)
    df["isWicket_bool"] = df["isWicket_bool"].apply(to_bool)

    # Numerics
    for c in ["runsScored", "overNumber", "ballNumber", "shotAngle", "shotMagnitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Connection (for False Shot %)
    if "battingConnectionId" in df.columns:
        df["conn_clean"] = df["battingConnectionId"].fillna("nan").astype(str).strip().str.lower()
    else:
        df["conn_clean"] = "nan"

    return df

df = load_data(DATA_URL)

# =========================
# Sidebar Filters — Batting (default batter = Virat Kohli)
# =========================
st.sidebar.header("Filters — Batting")

def safe_unique(col):
    return sorted(df[col].dropna().unique()) if col in df.columns else []

batters = safe_unique("battingPlayer")
default_batter = "Virat Kohli" if "Virat Kohli" in batters else (batters[0] if batters else None)
batter = st.sidebar.selectbox("Batter", batters, index=(batters.index(default_batter) if default_batter in batters else 0) if batters else 0)

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

st.title("Batting Dashboard")

# =========================
# Metric helpers (for tables & matrices)
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
    g["False Shot %"] = [false_shot_pct(df_in[df_in[group_col] == val]) for val in g[group_col]]
    g["Average"] = g.apply(lambda r: np.inf if (pd.notna(r["Dismissals"]) and r["Dismissals"] == 0) else round((r["Runs"] or 0) / (r["Dismissals"] or 1), 2), axis=1)
    g["Strike Rate"] = g.apply(lambda r: round(((r["Runs"] or 0) / (r["Balls"] or 1)) * 100, 2) if r["Balls"] else 0.0, axis=1)
    g = g.sort_values(["Runs", "Strike Rate"], ascending=[False, False])
    # format Average: keep np.inf as ∞ later in presentation
    return g

# =========================
# Line–Length Matrix — two tabs, scrollable, 2dp, colored
# =========================
st.subheader("Line × Length — Metrics")
if filt.empty or "lengthTypeId" not in filt.columns or "lineTypeId" not in filt.columns:
    st.info("No data for current filters or missing line/length columns.")
else:
    grp = filt.groupby(["lengthTypeId", "lineTypeId"], dropna=False)
    rows = []
    for (length, line), d in grp:
        fs = false_shot_pct(d)
        dismissals = d["isWicket_bool"].sum() if "isWicket_bool" in d.columns else 0
        runs = d["runsScored"].sum() if "runsScored" in d.columns else 0
        avg = np.inf if dismissals == 0 else (runs / dismissals if dismissals else 0.0)
        rows.append({"lengthTypeId": length, "lineTypeId": line, "False Shot %": fs, "Average": avg})
    mat = pd.DataFrame(rows)

    tab_fs, tab_avg = st.tabs(["False Shot %", "Average"])
    if not mat.empty:
        with tab_fs:
            p1 = mat.pivot(index="lengthTypeId", columns="lineTypeId", values="False Shot %")
            # round to 2 dp and color: low -> green, high -> red (reverse RdYlGn)
            sty = (p1.round(2)
                     .style.background_gradient(cmap="RdYlGn_r", axis=None)
                     .format("{:.2f}"))
            st.dataframe(sty, use_container_width=True, height=520)

        with tab_avg:
            p2 = mat.pivot(index="lengthTypeId", columns="lineTypeId", values="Average")
            # Styler on numeric with inf allowed; color high -> green
            def fmt_avg(v):
                return "∞" if np.isinf(v) else f"{v:.2f}"
            sty2 = (p2.style
                      .background_gradient(cmap="RdYlGn", axis=None)
                      .format(fmt_avg))
            st.dataframe(sty2, use_container_width=True, height=520)
