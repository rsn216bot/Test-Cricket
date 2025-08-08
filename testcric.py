
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge

st.set_page_config(page_title="Test Cricket Dashboard", layout="wide")

# ===== Data =====
GOOGLE_SHEET_ID = "17DwxDTomoofFTb8NdTo-XQNQ4diGP53f-ZYd0Iln1HM"
DATA_URL = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv"

@st.cache_data
def load_data(url: str):
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error("Couldn't load from Google Sheets.\nMake sure sharing is 'Anyone with link'.\n\nError: {}".format(e))
        return pd.DataFrame()
    # Clean
    if "ballDateTime" in df.columns:
        df["ballDateTime"] = pd.to_datetime(df["ballDateTime"], errors="coerce")
        df["year"] = df["ballDateTime"].dt.year
    else:
        df["year"] = np.nan

    # booleans
    def to_bool(x):
        if isinstance(x, bool): return x
        if isinstance(x, (int, float)): return bool(x)
        return str(x).strip().lower() in ("true","1","yes")
    df["isWicket_bool"] = df.get("isWicket", False).apply(to_bool)

    # numerics
    for c in ["runsScored","overNumber","ballNumber"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # connection for false shots
    if "battingConnectionId" in df.columns:
        df["conn_clean"] = df["battingConnectionId"].fillna("nan").astype(str).str.strip().str.lower()
    else:
        df["conn_clean"] = "nan"

    return df

df = load_data(DATA_URL)

# ===== Sidebar Filters =====
st.sidebar.header("Filters — Batting")

def safe_unique(col):
    return sorted(df[col].dropna().unique()) if col in df.columns else []

# Batter select with default "Virat Kohli" if present
batters = safe_unique("battingPlayer")
default_index = batters.index("Virat Kohli") if "Virat Kohli" in batters else 0
batter = st.sidebar.selectbox("Batter", batters, index=default_index if batters else 0)

opposition = st.sidebar.multiselect("Opposition (bowlingTeam)", safe_unique("bowlingTeam"))
bowler = st.sidebar.multiselect("Bowler", safe_unique("bowlerPlayer"))
bowling_type = st.sidebar.multiselect("Bowling Type", safe_unique("bowlingTypeId"))
years = st.sidebar.multiselect("Year", sorted(df["year"].dropna().unique()) if "year" in df.columns else [])
max_over = int(np.nanmax(df["overNumber"])) if "overNumber" in df.columns and not df["overNumber"].isna().all() else 0
over_range = st.sidebar.slider("Over range", 0, max_over, (0, max_over))

# Apply filters
filt = df.copy()
if batter:
    filt = filt[filt.get("battingPlayer","") == batter]
if opposition:
    filt = filt[filt.get("bowlingTeam","").isin(opposition)]
if bowler:
    filt = filt[filt.get("bowlerPlayer","").isin(bowler)]
if bowling_type:
    filt = filt[filt.get("bowlingTypeId","").isin(bowling_type)]
if years:
    filt = filt[filt.get("year","").isin(years)]
if "overNumber" in filt.columns:
    filt = filt[(filt["overNumber"] >= over_range[0]) & (filt["overNumber"] <= over_range[1])]

st.title("Test Cricket — Batting Dashboard")

# ===== Helpers for metrics =====
SAFE_CONN = {"welltimed","middled","left","blank","nan"}

def false_shot_pct(frame: pd.DataFrame) -> float:
    if frame.empty or "conn_clean" not in frame.columns:
        return 0.0
    conn = frame["conn_clean"]
    bad = conn[~conn.isin(SAFE_CONN)]
    pct = (len(bad) / len(conn)) * 100 if len(conn) else 0.0
    return round(pct, 2)

def agg_block(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    if frame.empty or col not in frame.columns:
        return pd.DataFrame(columns=[col,"Runs","Balls","Dismissals","False Shot %","Average","Strike Rate"])
    g = frame.groupby(col, dropna=False).agg(
        Runs=("runsScored","sum"),
        Balls=("runsScored","count"),
        Dismissals=("isWicket_bool","sum")
    ).reset_index()
    g["False Shot %"] = [false_shot_pct(frame[frame[col]==v]) for v in g[col]]
    g["Average"] = g.apply(lambda r: "∞" if r["Dismissals"]==0 else round(r["Runs"]/r["Dismissals"],2), axis=1)
    g["Strike Rate"] = g.apply(lambda r: round((r["Runs"]/r["Balls"])*100,2) if r["Balls"] else 0.0, axis=1)
    return g.sort_values(["Runs","Strike Rate"], ascending=[False,False])

# ===== Tabs =====
tab_tables, tab_zones, tab_matrix = st.tabs(["📋 Tables","🧭 Shot Areas","🧮 Line–Length"])

# --- Tables ---
with tab_tables:
    st.subheader("Batting & Bowling Tables")
    b1, b2 = st.tabs(["Batting Feet","Batting Shot Type"])
    with b1:
        st.dataframe(agg_block(filt,"battingFeetId"), use_container_width=True, height=420)
    with b2:
        st.dataframe(agg_block(filt,"battingShotTypeId"), use_container_width=True, height=420)

    bow1, bow2, bow3 = st.tabs(["Bowling Hand (vs batter)","Bowling Detail (vs batter)","Bowler (vs batter)"])
    with bow1:
        st.dataframe(agg_block(filt,"bowlingHandId"), use_container_width=True, height=380)
    with bow2:
        st.dataframe(agg_block(filt,"bowlingDetailId"), use_container_width=True, height=380)
    with bow3:
        st.dataframe(agg_block(filt,"bowlerPlayer"), use_container_width=True, height=380)

# --- Zones (center origin, lavender wedges, dividers outside pitch) ---
ZONE_LABELS = ["OFF Behind Sq","OFF Square","OFF Cover","OFF Straight",
               "LEG Straight","LEG Midwicket","LEG Square","LEG Behind Sq"]

ZONE_KEYWORDS = {
    "OFF Behind Sq": ["third","third man","deep third","gully","slip","backward point","deep back point","deep back"],
    "OFF Square": ["point","deep point","square","backward sq","backward square","cover point"],
    "OFF Cover": ["extra cover","cover","deep cover","cover drive","deep extra"],
    "OFF Straight": ["mid-off","long off","straight off","off drive"],
    "LEG Straight": ["mid-on","long on","on drive","straight on"],
    "LEG Midwicket": ["mid-wicket","midwicket","deep mid","cow corner"],
    "LEG Square": ["square leg","deep square","backward square leg","backward sq leg"],
    "LEG Behind Sq": ["fine leg","short fine","leg slip","leg gully","deep fine","short leg"]
}

def pick_fieldpos_col(frame: pd.DataFrame):
    cands = ["fieldingPosition","fielding_position","fielderPosition","fielder_position",
             "fielderPos","field_position","fielderPositionName","fielderRole"]
    for c in cands:
        if c in frame.columns: return c
    for c in frame.columns:
        if "position" in c.lower() or ("field" in c.lower() and "pos" in c.lower()):
            return c
    return None

def map_pos_to_zone(s: str) -> str:
    s = str(s).strip().lower()
    for zone, kws in ZONE_KEYWORDS.items():
        for k in kws:
            if k in s: return zone
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
    # run filter for zones
    if "runsScored" in filt.columns and filt["runsScored"].notna().any():
        avail_runs = sorted(filt["runsScored"].dropna().astype(int).unique().tolist())
    else:
        avail_runs = [0,1,2,3,4,6]
    default_runs = [r for r in [1,2,3,4,6] if r in avail_runs] or avail_runs
    runs_sel = st.multiselect("Runs to include", options=avail_runs, default=default_runs)

    fr = filt[filt.get("runsScored").isin(runs_sel)] if "runsScored" in filt.columns else filt

    pos_col = pick_fieldpos_col(fr)
    if pos_col is None:
        st.error("Couldn't find a fielding position column (e.g., 'fieldingPosition').")
    else:
        zdf = fr[[pos_col,"runsScored"]].copy()
        zdf[pos_col] = zdf[pos_col].astype(str)
        zdf["zone"] = zdf[pos_col].apply(map_pos_to_zone)
        zone_runs = zdf.groupby("zone")["runsScored"].sum().reindex(ZONE_LABELS, fill_value=0).reset_index()

        RADIUS = 100
        STRIP_H = 44
        STRIP_W = 10
        LAVENDER = (230/255,230/255,250/255)  # #E6E6FA
        r_inner = STRIP_H/2 + 1.0
        start_angles = np.linspace(22.5, 360+22.5, 9)

        def draw_ground(ax):
            ax.add_patch(Circle((0,0), RADIUS, fill=False, linewidth=1.5))
            ax.add_patch(Rectangle((-STRIP_W/2,-STRIP_H/2), STRIP_W, STRIP_H, fill=False, linewidth=1.2))
            ax.plot([-STRIP_W/2, STRIP_W/2],[STRIP_H/2, STRIP_H/2], lw=0.8)
            ax.plot([-STRIP_W/2, STRIP_W/2],[-STRIP_H/2,-STRIP_H/2], lw=0.8)
            ax.plot([0],[0], marker="o", markersize=3, color="black")
            ax.set_aspect("equal")
            ax.set_xlim(-RADIUS*1.15, RADIUS*1.15)
            ax.set_ylim(-RADIUS*1.15, RADIUS*1.15)
            ax.axis("off")

        st.subheader("Shot Areas (8 Zones, Lavender) — Center Origin")
        fig, ax = plt.subplots(figsize=(7,7))
        draw_ground(ax)

        # Labels outside boundary (fixed orientation)
        ax.text(RADIUS*1.05, 0, "OFF", ha="center", va="center", fontsize=11, weight="bold")
        ax.text(-RADIUS*1.05, 0, "LEG", ha="center", va="center", fontsize=11, weight="bold")

        # Dividers outside pitch only
        for ang in start_angles[:-1]:
            x0 = r_inner * np.sin(np.deg2rad(ang))
            y0 = -r_inner * np.cos(np.deg2rad(ang))
            x1 = RADIUS * np.sin(np.deg2rad(ang))
            y1 = -RADIUS * np.cos(np.deg2rad(ang))
            ax.plot([x0,x1],[y0,y1], linestyle="--", linewidth=0.6, alpha=0.4, color="gray")

        # Wedges uniform lavender
        for i, label in enumerate(ZONE_LABELS):
            start = start_angles[i]
            ax.add_patch(Wedge((0,0), RADIUS, start, start+45, width=RADIUS-r_inner,
                               facecolor=LAVENDER, edgecolor="black", alpha=0.9))
            mid = start + 22.5
            x = (r_inner + (RADIUS-r_inner)*0.6) * np.sin(np.deg2rad(mid))
            y = -(r_inner + (RADIUS-r_inner)*0.6) * np.cos(np.deg2rad(mid))
            val = int(zone_runs.loc[zone_runs["zone"]==label, "runsScored"].values[0]) if (zone_runs["zone"]==label).any() else 0
            ax.text(x, y, str(val), ha="center", va="center", fontsize=12, weight="bold")

        st.pyplot(fig)

# --- Line–Length (scrollable, rounded to 2 dp, two tabs) ---
with tab_matrix:
    st.subheader("Line × Length — Metrics")
    if filt.empty or "lengthTypeId" not in filt.columns or "lineTypeId" not in filt.columns:
        st.info("No data for current filters or missing line/length columns.")
    else:
        grp = filt.groupby(["lengthTypeId","lineTypeId"], dropna=False)
        rows = []
        for (length, line), d in grp:
            fs = false_shot_pct(d)
            dismissals = d["isWicket_bool"].sum() if "isWicket_bool" in d.columns else 0
            runs = d["runsScored"].sum() if "runsScored" in d.columns else 0
            avg = "∞" if dismissals == 0 else round(runs / dismissals, 2)
            rows.append({"lengthTypeId": length, "lineTypeId": line, "False Shot %": round(fs,2), "Average": avg})
        mat = pd.DataFrame(rows)
        if not mat.empty:
            t1, t2 = st.tabs(["False Shot %","Average"])
            with t1:
                p1 = mat.pivot(index="lengthTypeId", columns="lineTypeId", values="False Shot %")
                p1 = p1.round(2)
                st.dataframe(p1.style.background_gradient(cmap="Reds", axis=None), use_container_width=True, height=520)
            with t2:
                p2 = mat.pivot(index="lengthTypeId", columns="lineTypeId", values="Average")
                # stringify to keep "∞" and numeric with 2dp
                def fmt(v):
                    try:
                        return f"{float(v):.2f}"
                    except Exception:
                        return str(v)
                p2_display = p2.applymap(fmt)
                st.dataframe(p2_display, use_container_width=True, height=520)
