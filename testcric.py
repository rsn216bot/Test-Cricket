
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

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
        st.warning("Couldn't load from Google Sheets URL. Falling back to local CSV 'All Test Matches Till 2025.csv'.\nError: {}".format(e))
        df = pd.read_csv("All Test Matches Till 2025.csv")
    # Clean & enrich
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
        if isinstance(x, bool): return x
        if isinstance(x, (int, float)): return bool(x)
        s = str(x).strip().lower()
        return s in ("true", "1", "yes")
    df["isWicket_bool"] = df.get("isWicket", False)
    df["isWicket_bool"] = df["isWicket_bool"].apply(to_bool)

    # Numerics
    for c in ["runsScored", "shotAngle", "shotMagnitude", "overNumber", "ballNumber"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Handedness if available
    if "battingHandId" in df.columns:
        df["battingHandId"] = df["battingHandId"].fillna("Right").astype(str)
    else:
        df["battingHandId"] = "Right"

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

# Determine batting hand (from data if present, else allow override)
default_hand = "Right"
if "battingHandId" in filt.columns and not filt["battingHandId"].dropna().empty:
    # use the most frequent hand for selected batter
    default_hand = filt["battingHandId"].mode().iat[0] if not filt["battingHandId"].mode().empty else "Right"
hand = st.sidebar.radio("Batting hand", ["Right", "Left"], index=0 if str(default_hand).lower().startswith("r") else 1)

st.title("Test Batting Dashboard — Batting Focus")
st.caption("Wagon wheel uses **full-circle** ground, 0° = wicketkeeper (behind stumps), 90° = point. OFF/LEG labelled by batting hand.")

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
        return "∞" if dism == 0 else round(runs / dism, 2)
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
# Pitch drawing helpers (full-circle) & angle mapping
# =========================
RADIUS = 100  # field radius for drawing

def draw_full_pitch(ax, radius=RADIUS):
    # Outer boundary circle
    circle = Circle((0, 0), radius, fill=False, linewidth=1.5)
    ax.add_patch(circle)
    # Pitch strip rectangle centered at origin, vertical
    strip_w = 10
    strip_h = 44
    pitch = Rectangle((-strip_w/2, -strip_h/2), strip_w, strip_h, fill=False, linewidth=1.2)
    ax.add_patch(pitch)
    # Creases (short ticks at each end)
    crease_y = strip_h/2
    ax.plot([-strip_w/2, strip_w/2], [crease_y, crease_y], lw=0.8)
    ax.plot([-strip_w/2, strip_w/2], [-crease_y, -crease_y], lw=0.8)
    # Stumps: small dot at origin (batter facing up by convention)
    ax.plot([0], [0], marker="o", markersize=3, color="black")
    ax.set_aspect("equal")
    ax.set_xlim(-radius*1.1, radius*1.1)
    ax.set_ylim(-radius*1.1, radius*1.1)
    ax.axis("off")

def polar_to_xy(angle_deg, r):
    # Convention: 0° = wicketkeeper (negative y), 90° = point (positive x),
    # 180° = bowler (positive y), 270° = fine leg (negative x).
    theta = np.deg2rad(angle_deg)
    x = r * np.sin(theta)
    y = -r * np.cos(theta)
    return x, y

def off_leg_labels(ax, hand: str, radius=RADIUS):
    # OFF side = +x for Right-hand; -x for Left-hand
    off_x = radius * 0.65 if hand.lower().startswith("r") else -radius * 0.65
    leg_x = -off_x
    ax.text(off_x, 0, "OFF", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(leg_x, 0, "LEG", ha="center", va="center", fontsize=11, weight="bold")

# =========================
# Wagon Wheel — Full circle lines with run filters
# =========================
ww_tab1, ww_tab2 = st.tabs(["Wagon Wheel (Lines)", "Hot Zones on Full Ground"])

with ww_tab1:
    st.subheader("Wagon Wheel — Full Ground (Lines)")
    if filt.empty or "shotAngle" not in filt.columns or "shotMagnitude" not in filt.columns:
        st.info("No shot tracking data (shotAngle/shotMagnitude) available for current filter.")
    else:
        # Run filters
        if "runsScored" in filt.columns and filt["runsScored"].notna().any():
            avail_runs = sorted(filt["runsScored"].dropna().astype(int).unique().tolist())
        else:
            avail_runs = [0,1,2,3,4,6]
        default_runs = [r for r in [1,2,3,4,6] if r in avail_runs] or avail_runs
        runs_sel = st.multiselect("Show runs", options=avail_runs, default=default_runs)
        data = filt[filt.get("runsScored").isin(runs_sel)][["shotAngle","shotMagnitude","runsScored"]].dropna()
        if data.empty:
            st.info("No shots match selected run filters.")
        else:
            # scale magnitudes (0..1) then to radius
            mag = data["shotMagnitude"].astype(float).clip(lower=0)
            if mag.max() > 0:
                base = (mag / mag.max()) * (RADIUS * 0.95)
            else:
                base = np.full(len(mag), RADIUS * 0.6)
            runs = data["runsScored"].astype(int).values
            # Push boundaries beyond rope (4 slightly, 6 more)
            end_r = np.where(runs==6, RADIUS*1.15,
                     np.where(runs==4, RADIUS*1.03, base))
            angles = data["shotAngle"].astype(float).values

            # Color map by runs
            color_map = {0:"grey",1:"#6f42c1",2:"#17a2b8",3:"#20c997",4:"#fd7e14",6:"#dc3545"}
            colors = [color_map.get(int(r), "black") for r in runs]

            fig, ax = plt.subplots(figsize=(7,7))
            draw_full_pitch(ax, RADIUS)
            off_leg_labels(ax, hand, RADIUS)

            # Draw each shot as a line
            for ang, rr, col in zip(angles, end_r, colors):
                x, y = polar_to_xy(ang, rr)
                ax.plot([0, x], [0, y], linewidth=1.4, color=col, alpha=0.9)

            # Legend-like hint
            st.pyplot(fig)

# =========================
# Hot Zones — Full-circle heatmap weighted by runs
# =========================
with ww_tab2:
    st.subheader("Hot Zones — Full Ground (Run-weighted)")
    if filt.empty or "shotAngle" not in filt.columns or "shotMagnitude" not in filt.columns:
        st.info("No shot tracking data (shotAngle/shotMagnitude) available for current filter.")
    else:
        data = filt[["shotAngle","shotMagnitude","runsScored"]].dropna()
        if data.empty:
            st.info("No valid shotAngle/shotMagnitude rows after filtering.")
        else:
            mag = data["shotMagnitude"].astype(float).clip(lower=0)
            r = (mag / (mag.max() if mag.max() else 1.0)) * (RADIUS * 0.95)
            angles = data["shotAngle"].astype(float).values
            xs, ys, ws = [], [], []
            for ang, rr, run in zip(angles, r, data["runsScored"].values):
                x, y = polar_to_xy(ang, rr)
                xs.append(x); ys.append(y); ws.append(0 if pd.isna(run) else run)
            xs = np.array(xs); ys = np.array(ys); ws = np.array(ws)

            fig, ax = plt.subplots(figsize=(7,7))
            draw_full_pitch(ax, RADIUS)
            off_leg_labels(ax, hand, RADIUS)

            hb = ax.hexbin(xs, ys, C=ws, reduce_C_function=np.sum, gridsize=28,
                           extent=[-RADIUS, RADIUS, -RADIUS, RADIUS])
            # Clip hexbin to circle
            circ = Circle((0,0), RADIUS, transform=ax.transData)
            for coll in ax.collections:
                coll.set_clip_path(circ)

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
            st.dataframe(p2)

