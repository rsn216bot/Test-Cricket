
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge

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

    # Normalise booleans
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

    # Hand
    if "battingHandId" in df.columns:
        df["battingHandId"] = df["battingHandId"].fillna("Right").astype(str)
    else:
        df["battingHandId"] = "Right"

    # Connection (for False Shot % in other tables)
    if "battingConnectionId" in df.columns:
        df["conn_clean"] = df["battingConnectionId"].fillna("nan").astype(str).str.strip().str.lower()
    else:
        df["conn_clean"] = "nan"

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

# Batting hand selector
default_hand = "Right"
if "battingHandId" in filt.columns and not filt["battingHandId"].dropna().empty:
    mode_series = filt["battingHandId"].mode()
    if not mode_series.empty:
        default_hand = mode_series.iat[0]
hand = st.sidebar.radio("Batting hand", ["Right", "Left"], index=0 if str(default_hand).lower().startswith("r") else 1)

st.title("Batting Dashboard — Full Ground Wagon Wheel + Stats")
st.caption("Wagon wheel now has 8 marked zones. OFF/LEG labels sit outside the circle so they don't overlap numbers.")

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
    fs_list = []
    for val in g[group_col]:
        subset = df_in[df_in[group_col] == val]
        fs_list.append(false_shot_pct(subset))
    g["False Shot %"] = fs_list
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
# Batting-related tabs (tables)
# =========================
st.header("Batting & Bowling Tables")
bat_tab1, bat_tab2 = st.tabs(["Batting Feet Stats", "Batting Shot Type Stats"])
with bat_tab1:
    st.subheader("Batting Feet Stats")
    st.dataframe(agg_block(filt, "battingFeetId"))

with bat_tab2:
    st.subheader("Batting Shot Type Stats")
    st.dataframe(agg_block(filt, "battingShotTypeId"))

bowl_tab1, bowl_tab2, bowl_tab3 = st.tabs(["Bowling Hand Stats (vs batter)", "Bowling Detail Stats (vs batter)", "Bowler Stats (vs batter)"])
with bowl_tab1:
    st.dataframe(agg_block(filt, "bowlingHandId"))
with bowl_tab2:
    st.dataframe(agg_block(filt, "bowlingDetailId"))
with bowl_tab3:
    st.dataframe(agg_block(filt, "bowlerPlayer"))

# =========================
# Zone mapping from fielding positions -> 8 wedges
# =========================
ZONE_LABELS = ["OFF Behind Sq", "OFF Square", "OFF Cover", "OFF Straight",
               "LEG Straight", "LEG Midwicket", "LEG Square", "LEG Behind Sq"]

ZONE_KEYWORDS = {
    "OFF Behind Sq": ["third", "third man", "deep third", "gully", "slip", "backward point", "deep back point", "deep back"],
    "OFF Square": ["point", "deep point", "square", "backward sq", "backward square", "cover point"],
    "OFF Cover": ["extra cover", "cover", "deep cover", "cover drive", "deep extra"],
    "OFF Straight": ["mid-off", "long off", "straight off", "off drive"],
    "LEG Straight": ["mid-on", "long on", "on drive", "straight on"],
    "LEG Midwicket": ["mid-wicket", "midwicket", "deep mid", "cow corner"],
    "LEG Square": ["square leg", "deep square", "backward square leg", "backward sq leg"],
    "LEG Behind Sq": ["fine leg", "short fine", "leg slip", "leg gully", "deep fine", "short leg"]
}

def pick_fieldpos_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "fieldingPosition", "fielding_position", "fielderPosition", "fielder_position",
        "fielderPos", "field_position", "fielderPositionName", "fielderRole"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "position" in c.lower() or ("field" in c.lower() and "pos" in c.lower()):
            return c
    return None

FIELD_POS_COL = pick_fieldpos_col(filt)

def map_pos_to_zone(val: str) -> str:
    s = str(val).strip().lower()
    for zone, kws in ZONE_KEYWORDS.items():
        for k in kws:
            if k in s:
                return zone
    if "point" in s: return "OFF Square"
    if "cover" in s: return "OFF Cover"
    if "off" in s: return "OFF Straight"
    if "on" in s: return "LEG Straight"
    if "mid" in s and "off" in s: return "OFF Straight"
    if "mid" in s and "on" in s: return "LEG Straight"
    if "wicket" in s: return "LEG Midwicket" if "mid" in s else "LEG Square"
    if "leg" in s: return "LEG Square"
    if "slip" in s or "gully" in s: return "OFF Behind Sq"
    if "keeper" in s: return "OFF Behind Sq"
    return "Unknown"

# =========================
# Run filter for wheel/zone plots
# =========================
if "runsScored" in filt.columns and filt["runsScored"].notna().any():
    avail_runs = sorted(filt["runsScored"].dropna().astype(int).unique().tolist())
else:
    avail_runs = [0,1,2,3,4,6]
default_runs = [r for r in [1,2,3,4,6] if r in avail_runs] or avail_runs
runs_sel = st.multiselect("Show runs in wheel/zone views", options=avail_runs, default=default_runs)

filt_runs = filt[filt.get("runsScored").isin(runs_sel)] if "runsScored" in filt.columns else filt

# =========================
# Drawing helpers (full-circle pitch & zone guides)
# =========================
RADIUS = 100

def draw_full_pitch(ax, radius=RADIUS):
    circle = Circle((0, 0), radius, fill=False, linewidth=1.5)
    ax.add_patch(circle)
    strip_w = 10
    strip_h = 44
    pitch = Rectangle((-strip_w/2, -strip_h/2), strip_w, strip_h, fill=False, linewidth=1.2)
    ax.add_patch(pitch)
    crease_y = strip_h/2
    ax.plot([-strip_w/2, strip_w/2], [crease_y, crease_y], lw=0.8)
    ax.plot([-strip_w/2, strip_w/2], [-crease_y, -crease_y], lw=0.8)
    ax.plot([0],[0], marker="o", markersize=3, color="black")
    ax.set_aspect("equal")
    ax.set_xlim(-radius*1.15, radius*1.15)  # extra margin for OFF/LEG labels outside
    ax.set_ylim(-radius*1.15, radius*1.15)
    ax.axis("off")

def off_leg_labels(ax, hand: str, radius=RADIUS):
    # Put labels just outside the boundary to avoid overlapping numbers
    off_x = radius * 1.05 if hand.lower().startswith("r") else -radius * 1.05
    leg_x = -off_x
    ax.text(off_x, 0, "OFF", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(leg_x, 0, "LEG", ha="center", va="center", fontsize=11, weight="bold")

def draw_zone_guides(ax, radius=RADIUS):
    # 8 zones: draw faint radial separators at 22.5 + 45*k degrees
    starts = np.linspace(22.5, 360+22.5, 9)[:-1]
    for ang in starts:
        x = radius * np.sin(np.deg2rad(ang))
        y = -radius * np.cos(np.deg2rad(ang))
        ax.plot([0, x], [0, y], linestyle="--", linewidth=0.6, alpha=0.4, color="gray")

def polar_to_xy(angle_deg, r):
    # 0° = keeper (down), 90° = point (right), 180° = bowler (up), 270° = fine leg (left)
    theta = np.deg2rad(angle_deg)
    x = r * np.sin(theta)
    y = -r * np.cos(theta)
    return x, y

# =========================
# Wagon Wheel — Lines with zone guides
# =========================
ww_tab1, ww_tab2 = st.tabs(["Wagon Wheel (Lines + Zones)", "8‑Zone Runs & Hot Map"])

with ww_tab1:
    st.subheader("Wagon Wheel — Full Ground with Zones Marked")
    if filt_runs.empty or "shotAngle" not in filt_runs.columns or "shotMagnitude" not in filt_runs.columns:
        st.info("No shot tracking data (shotAngle/shotMagnitude) available for current filter.")
    else:
        data = filt_runs[["shotAngle","shotMagnitude","runsScored"]].dropna()
        if data.empty:
            st.info("No valid shotAngle/shotMagnitude rows after filtering.")
        else:
            mag = data["shotMagnitude"].astype(float).clip(lower=0)
            if mag.max() > 0:
                base = (mag / mag.max()) * (RADIUS * 0.95)
            else:
                base = np.full(len(mag), RADIUS * 0.6)
            runs = data["runsScored"].astype(int).values
            end_r = np.where(runs==6, RADIUS*1.15,
                     np.where(runs==4, RADIUS*1.03, base))
            angles = data["shotAngle"].astype(float).values

            color_map = {0:"grey",1:"#6f42c1",2:"#17a2b8",3:"#20c997",4:"#fd7e14",6:"#dc3545"}
            colors = [color_map.get(int(r), "black") for r in runs]

            fig, ax = plt.subplots(figsize=(7,7))
            draw_full_pitch(ax, RADIUS)
            draw_zone_guides(ax, RADIUS)
            off_leg_labels(ax, hand, RADIUS)

            for ang, rr, col in zip(angles, end_r, colors):
                x, y = polar_to_xy(ang, rr)
                ax.plot([0, x], [0, y], linewidth=1.2, color=col, alpha=0.9)

            st.pyplot(fig)

# =========================
# 8‑Zone aggregation from fielding positions
# =========================
def pick_fieldpos_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "fieldingPosition", "fielding_position", "fielderPosition", "fielder_position",
        "fielderPos", "field_position", "fielderPositionName", "fielderRole"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "position" in c.lower() or ("field" in c.lower() and "pos" in c.lower()):
            return c
    return None

FIELD_POS_COL = pick_fieldpos_col(filt_runs)

def map_pos_to_zone(val: str) -> str:
    s = str(val).strip().lower()
    for zone, kws in ZONE_KEYWORDS.items():
        for k in kws:
            if k in s:
                return zone
    if "point" in s: return "OFF Square"
    if "cover" in s: return "OFF Cover"
    if "off" in s: return "OFF Straight"
    if "on" in s: return "LEG Straight"
    if "mid" in s and "off" in s: return "OFF Straight"
    if "mid" in s and "on" in s: return "LEG Straight"
    if "wicket" in s: return "LEG Midwicket" if "mid" in s else "LEG Square"
    if "leg" in s: return "LEG Square"
    if "slip" in s or "gully" in s: return "OFF Behind Sq"
    if "keeper" in s: return "OFF Behind Sq"
    return "Unknown"

with ww_tab2:
    if FIELD_POS_COL is None:
        st.error("Couldn't find a fielding position column in your data. Please confirm the column name (e.g., 'fieldingPosition').")
    else:
        zdf = filt_runs[[FIELD_POS_COL, "runsScored"]].copy()
        zdf[FIELD_POS_COL] = zdf[FIELD_POS_COL].astype(str)
        zdf["zone"] = zdf[FIELD_POS_COL].apply(map_pos_to_zone)
        zone_runs = zdf.groupby("zone")["runsScored"].sum().reindex(ZONE_LABELS, fill_value=0).reset_index()

        # Zone Runs (numbers) and Hot Map (fills)
        sub1, sub2 = st.tabs(["Zone Runs (8 wedges)", "Zone Hot Map"])

        start_angles = np.linspace(22.5, 360+22.5, 9)  # 8 sectors of 45°

        with sub1:
            fig, ax = plt.subplots(figsize=(7,7))
            draw_full_pitch(ax, RADIUS)
            off_leg_labels(ax, hand, RADIUS)
            # faint wedge separators
            for ang in start_angles[:-1]:
                x = RADIUS * np.sin(np.deg2rad(ang))
                y = -RADIUS * np.cos(np.deg2rad(ang))
                ax.plot([0, x], [0, y], linestyle="--", linewidth=0.6, alpha=0.4, color="gray")

            for i, label in enumerate(ZONE_LABELS):
                start = start_angles[i]
                wedge = Wedge(center=(0,0), r=RADIUS, theta1=start, theta2=start+45, width=RADIUS*0.35, alpha=0.12)
                ax.add_patch(wedge)
                mid = start + 22.5
                x = (RADIUS*0.6) * np.sin(np.deg2rad(mid))
                y = -(RADIUS*0.6) * np.cos(np.deg2rad(mid))
                runs = int(zone_runs.loc[zone_runs["zone"]==label, "runsScored"].values[0]) if (zone_runs["zone"]==label).any() else 0
                ax.text(x, y, str(runs), ha="center", va="center", fontsize=12, weight="bold" )
            st.pyplot(fig)

        with sub2:
            fig2, ax2 = plt.subplots(figsize=(7,7))
            draw_full_pitch(ax2, RADIUS)
            off_leg_labels(ax2, hand, RADIUS)
            for ang in start_angles[:-1]:
                x = RADIUS * np.sin(np.deg2rad(ang))
                y = -RADIUS * np.cos(np.deg2rad(ang))
                ax2.plot([0, x], [0, y], linestyle="--", linewidth=0.6, alpha=0.4, color="gray")

            total = zone_runs["runsScored"].max()
            scale = total if total > 0 else 1
            for i, label in enumerate(ZONE_LABELS):
                start = start_angles[i]
                val = float(zone_runs.loc[zone_runs["zone"]==label, "runsScored"].values[0]) if (zone_runs["zone"]==label).any() else 0.0
                intensity = val / scale
                wedge = Wedge(center=(0,0), r=RADIUS, theta1=start, theta2=start+45, width=RADIUS*0.9,
                              facecolor=(1-intensity, 1-intensity, 1), edgecolor="black", alpha=0.5)
                ax2.add_patch(wedge)
                mid = start + 22.5
                x = (RADIUS*0.55) * np.sin(np.deg2rad(mid))
                y = -(RADIUS*0.55) * np.cos(np.deg2rad(mid))
                ax2.text(x, y, str(int(val)), ha="center", va="center", fontsize=11, weight="bold" )
            st.pyplot(fig2)

# =========================
# Line–Length Matrix — split tabs
# =========================
st.header("Line × Length — Metrics")
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
