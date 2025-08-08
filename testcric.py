
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        st.warning("Couldn't load from Google Sheets URL. Falling back to local CSV 'All Test Matches Till 2025.csv'.\\nError: {}".format(e))
        df = pd.read_csv("All Test Matches Till 2025.csv")
    # Clean and enrich
    if "ballDateTime" in df.columns:
        df["ballDateTime"] = pd.to_datetime(df["ballDateTime"], errors="coerce")
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

    # Booleans
    def to_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        s = str(x).strip().lower()
        return s in ("true", "1", "yes")
    df["isWicket_bool"] = df.get("isWicket", False)
    df["isWicket_bool"] = df["isWicket_bool"].apply(to_bool)

    # Numerics
    for c in ["runsScored", "shotAngle", "shotMagnitude", "overNumber", "ballNumber"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data(DATA_URL)

# =========================
# Sidebar Filters — Batting
# =========================
st.sidebar.header("Filters — Batting")

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

st.title("Test Batting Dashboard (Mid-2023 to 2025) — Batting Focus")
st.caption("Now with pitch-based wagon wheel lines, hot zones heatmap, and split Line–Length matrix tabs.")

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
        if dism == 0:
            return "∞"  # explicitly infinity when no dismissals
        return round(runs / dism, 2)
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
# Helpers to draw a simple cricket pitch (Cartesian), and convert polar -> cartesian
# =========================
def draw_pitch(ax, radius=100):
    # Outfield circle (for context)
    circ = plt.Circle((0, 0), radius, fill=False, linewidth=1, alpha=0.4)
    ax.add_artist(circ)
    # Pitch strip (rect around y-axis)
    strip_w = 30   # width of the pitch strip (x-direction)
    strip_h = 200  # length (y-direction)
    rect = plt.Rectangle((-strip_w/2, 0), strip_w, strip_h, fill=False, linewidth=1.2)
    ax.add_patch(rect)
    # Crease near batter (close to origin)
    crease_y = 0
    ax.plot([-strip_w/2, strip_w/2], [crease_y, crease_y], lw=1)
    # Stumps position marker
    ax.plot([0], [0], marker="o", markersize=4, color="black")
    # Style
    ax.set_aspect("equal")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-10, radius)  # show only front of batter
    ax.axis("off")

def polar_to_xy(angle_deg, r):
    # 0° is straight ahead (positive y), clockwise positive
    theta = np.deg2rad((360 - angle_deg) % 360)  # convert to standard math rotation
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return x, y

# =========================
# Wagon Wheel — Lines on a pitch
# =========================
ww_tab1, ww_tab2 = st.tabs(["Wagon Wheel (Lines)", "Hot Zones on Pitch"])

with ww_tab1:
    st.subheader("Wagon Wheel — Lines by Angle & Magnitude")
    if filt.empty or "shotAngle" not in filt.columns or "shotMagnitude" not in filt.columns:
        st.info("No shot tracking data (shotAngle/shotMagnitude) available for current filter.")
    else:
        data = filt[["shotAngle", "shotMagnitude", "runsScored"]].dropna()
        if data.empty:
            st.info("No valid shotAngle/shotMagnitude rows after filtering.")
        else:
            # scale magnitudes to reasonable visual units
            mag = data["shotMagnitude"].astype(float).clip(lower=0)
            # give extra length to boundaries 4/6
            extra = np.where(data["runsScored"].isin([4, 6]), 1.25, 1.0)
            r = (mag / (mag.max() if mag.max() else 1.0)) * 100 * extra
            angles = data["shotAngle"].astype(float).values

            fig, ax = plt.subplots(figsize=(7, 7))
            draw_pitch(ax, radius=100)
            for ang, rr, rs in zip(angles, r, data["runsScored"].values):
                x, y = polar_to_xy(ang, rr)
                ax.plot([0, x], [0, y], linewidth=1.2)
            ax.set_title(f"Wagon Wheel — {batter}")
            st.pyplot(fig)

# =========================
# Hot Zones — Heatmap on the pitch (run density)
# =========================
with ww_tab2:
    st.subheader("Hot Zones — Run Density on Pitch")
    if filt.empty or "shotAngle" not in filt.columns or "shotMagnitude" not in filt.columns:
        st.info("No shot tracking data (shotAngle/shotMagnitude) available for current filter.")
    else:
        data = filt[["shotAngle", "shotMagnitude", "runsScored"]].dropna()
        if data.empty:
            st.info("No valid shotAngle/shotMagnitude rows after filtering.")
        else:
            mag = data["shotMagnitude"].astype(float).clip(lower=0)
            # map to pitch radius
            r = (mag / (mag.max() if mag.max() else 1.0)) * 100
            angles = data["shotAngle"].astype(float).values
            # convert to cartesian endpoints and weight by runs
            xs, ys, ws = [], [], []
            for ang, rr, run in zip(angles, r, data["runsScored"].values):
                x, y = polar_to_xy(ang, rr)
                xs.append(x); ys.append(y); ws.append(run if not np.isnan(run) else 0)
            xs = np.array(xs); ys = np.array(ys); ws = np.array(ws)

            fig, ax = plt.subplots(figsize=(7, 7))
            draw_pitch(ax, radius=100)
            # hexbin heatmap weighted by runs
            hb = ax.hexbin(xs, ys, C=ws, reduce_C_function=np.sum, gridsize=20, extent=[-100, 100, -10, 100])
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("Runs density")
            ax.set_title(f"Hot Zones — {batter}")
            st.pyplot(fig)

# =========================
# Line–Length Matrix — split tabs
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
        avg = "∞" if dismissals == 0 else round(runs / dismissals, 2)
        rows.append({"lengthTypeId": length, "lineTypeId": line, "False Shot %": fs, "Average": avg})
    mat = pd.DataFrame(rows)
    if not mat.empty:
        tab_fs, tab_avg = st.tabs(["False Shot %", "Average"])
        with tab_fs:
            p1 = mat.pivot(index="lengthTypeId", columns="lineTypeId", values="False Shot %")
            st.dataframe(p1.style.background_gradient(cmap="Reds", axis=None))
        with tab_avg:
            p2 = mat.pivot(index="lengthTypeId", columns="lineTypeId", values="Average")
            # For Average, we can't use numeric gradient if ∞ strings exist. Show as table.
            st.dataframe(p2)
