
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

    # Booleans
    def to_bool(x):
        if isinstance(x, bool): return x
        if isinstance(x, (int, float)): return bool(x)
        s = str(x).strip().lower()
        return s in ("true", "1", "yes")
    df["isWicket_bool"] = df.get("isWicket", False)
    df["isWicket_bool"] = df["isWicket_bool"].apply(to_bool)

    # Numerics
    for c in ["runsScored", "overNumber", "ballNumber"]:
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

st.title("Batting Dashboard — Zones-only View (Center Origin)")
st.caption("8 lavender wedges from the **center**. Divider lines are drawn **outside** the pitch only.")

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
# Top-level tabs: Tables | Shot Areas | Line–Length
# =========================
tab_tables, tab_zones, tab_matrix = st.tabs(["📋 Tables", "🧭 Shot Areas", "🧮 Line–Length"])

# ---------- TABLES ----------
with tab_tables:
    st.subheader("Batting & Bowling Tables")
    bat_tabs = st.tabs(["Batting Feet", "Batting Shot Type"])
    with bat_tabs[0]:
        st.dataframe(agg_block(filt, "battingFeetId"))
    with bat_tabs[1]:
        st.dataframe(agg_block(filt, "battingShotTypeId"))

    bowl_tabs = st.tabs(["Bowling Hand (vs batter)", "Bowling Detail (vs batter)", "Bowler (vs batter)"])
    with bowl_tabs[0]:
        st.dataframe(agg_block(filt, "bowlingHandId"))
    with bowl_tabs[1]:
        st.dataframe(agg_block(filt, "bowlingDetailId"))
    with bowl_tabs[2]:
        st.dataframe(agg_block(filt, "bowlerPlayer"))

# ---------- ZONES (Single tab: Shot Areas)
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

def pick_fieldpos_col(df_in: pd.DataFrame) -> str | None:
    candidates = [
        "fieldingPosition", "fielding_position", "fielderPosition", "fielder_position",
        "fielderPos", "field_position", "fielderPositionName", "fielderRole"
    ]
    for c in candidates:
        if c in df_in.columns:
            return c
    for c in df_in.columns:
        if "position" in c.lower() or ("field" in c.lower() and "pos" in c.lower()):
            return c
    return None

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

with tab_zones:
    # Run filter (zones reflect only selected runs)
    if "runsScored" in filt.columns and filt["runsScored"].notna().any():
        avail_runs = sorted(filt["runsScored"].dropna().astype(int).unique().tolist())
    else:
        avail_runs = [0,1,2,3,4,6]
    default_runs = [r for r in [1,2,3,4,6] if r in avail_runs] or avail_runs
    runs_sel = st.multiselect("Runs to include", options=avail_runs, default=default_runs)

    filt_runs = filt[filt.get("runsScored").isin(runs_sel)] if "runsScored" in filt.columns else filt

    FIELD_POS_COL = pick_fieldpos_col(filt_runs)
    if FIELD_POS_COL is None:
        st.error("Couldn't find a fielding position column in your data. Please confirm the column name (e.g., 'fieldingPosition').")
    else:
        zdf = filt_runs[[FIELD_POS_COL, "runsScored"]].copy()
        zdf[FIELD_POS_COL] = zdf[FIELD_POS_COL].astype(str)
        zdf["zone"] = zdf[FIELD_POS_COL].apply(map_pos_to_zone)
        zone_runs = zdf.groupby("zone")["runsScored"].sum().reindex(ZONE_LABELS, fill_value=0).reset_index()

        # Drawing params
        RADIUS = 100
        STRIP_H = 44
        STRIP_W = 10
        LAVENDER = (230/255, 230/255, 250/255)  # #E6E6FA

        # Wedge geometry: origin at CENTER, but don't draw inside pitch -> inner radius = STRIP_H/2
        r_inner = STRIP_H/2 + 1.0  # +1px gap so lines don't touch crease
        start_angles = np.linspace(22.5, 360+22.5, 9)  # 8 slices of 45°

        def draw_ground(ax):
            # outer circle
            circ = Circle((0,0), RADIUS, fill=False, linewidth=1.5)
            ax.add_patch(circ)
            # pitch strip centered
            pitch = Rectangle((-STRIP_W/2, -STRIP_H/2), STRIP_W, STRIP_H, fill=False, linewidth=1.2)
            ax.add_patch(pitch)
            # creases
            ax.plot([-STRIP_W/2, STRIP_W/2], [STRIP_H/2, STRIP_H/2], lw=0.8)
            ax.plot([-STRIP_W/2, STRIP_W/2], [-STRIP_H/2, -STRIP_H/2], lw=0.8)
            # origin marker (center)
            ax.plot([0],[0], marker="o", markersize=3, color="black")
            ax.set_aspect("equal")
            ax.set_xlim(-RADIUS*1.15, RADIUS*1.15)
            ax.set_ylim(-RADIUS*1.15, RADIUS*1.15)
            ax.axis("off")

        st.subheader("Shot Areas (8 Zones, Lavender Fill) — Center Origin")
        fig, ax = plt.subplots(figsize=(7,7))
        draw_ground(ax)

        # OFF/LEG labels outside boundary
        off_x = RADIUS * 1.05 if hand.lower().startswith("r") else -RADIUS * 1.05
        leg_x = -off_x
        ax.text(off_x, 0, "OFF", ha="center", va="center", fontsize=11, weight="bold")
        ax.text(leg_x, 0, "LEG", ha="center", va="center", fontsize=11, weight="bold")

        # Draw separators ONLY outside pitch (i.e., from r_inner to RADIUS)
        for ang in start_angles[:-1]:
            x0 = r_inner * np.sin(np.deg2rad(ang))
            y0 = -r_inner * np.cos(np.deg2rad(ang))
            x1 = RADIUS * np.sin(np.deg2rad(ang))
            y1 = -RADIUS * np.cos(np.deg2rad(ang))
            ax.plot([x0, x1], [y0, y1], linestyle="--", linewidth=0.6, alpha=0.4, color="gray")

        # Draw uniform lavender wedges (no intensity scaling)
        for i, label in enumerate(ZONE_LABELS):
            start = start_angles[i]
            wedge = Wedge(center=(0,0), r=RADIUS, theta1=start, theta2=start+45,
                          width=RADIUS - r_inner, facecolor=LAVENDER, edgecolor="black", alpha=0.9)
            ax.add_patch(wedge)
            # place runs text
            mid = start + 22.5
            x = ((r_inner + (RADIUS - r_inner)*0.6)) * np.sin(np.deg2rad(mid))
            y = -((r_inner + (RADIUS - r_inner)*0.6)) * np.cos(np.deg2rad(mid))
            runs = int(zone_runs.loc[zone_runs["zone"]==label, "runsScored"].values[0]) if (zone_runs["zone"]==label).any() else 0
            ax.text(x, y, str(runs), ha="center", va="center", fontsize=12, weight="bold", color="black")

        st.pyplot(fig)

# ---------- LINE–LENGTH ----------
with tab_matrix:
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
            t1, t2 = st.tabs(["False Shot %", "Average"])
            with t1:
                p1 = mat.pivot(index="lengthTypeId", columns="lineTypeId", values="False Shot %")
                st.dataframe(p1.style.background_gradient(cmap="Reds", axis=None))
            with t2:
                p2 = mat.pivot(index="lengthTypeId", columns="lineTypeId", values="Average")
                st.dataframe(p2)
