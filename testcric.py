
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
        st.error(
            "Could not load data from Google Sheets.\n\n"
            "Please ensure sharing is 'Anyone with the link'.\n"
            f"Error: {e}"
        )
        return pd.DataFrame()

    # Clean & enrich
    if "ballDateTime" in df.columns:
        df["ballDateTime"] = pd.to_datetime(df["ballDateTime"], errors="coerce")
        df["year"] = df["ballDateTime"].dt.year
    else:
        df["year"] = np.nan

    # Booleans
    def to_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float, np.integer, np.floating)):
            return bool(x)
        s = str(x).strip().lower()
        return s in ("true", "1", "yes")
    df["isWicket_bool"] = df.get("isWicket", False)
    df["isWicket_bool"] = df["isWicket_bool"].apply(to_bool)

    # Numerics
    for c in ["runsScored", "overNumber", "ballNumber"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Connection (for False Shot % in tables)
    if "battingConnectionId" in df.columns:
        df["conn_clean"] = (
            df["battingConnectionId"]
            .fillna("nan")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        df["conn_clean"] = "nan"

    return df

df = load_data(DATA_URL)

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("Filters")

def safe_unique(col):
    return sorted(df[col].dropna().unique()) if col in df.columns else []

# Batter filter with default = Virat Kohli
batters = safe_unique("battingPlayer")
default_idx = batters.index("Virat Kohli") if "Virat Kohli" in batters and len(batters) > 0 else 0
batter = st.sidebar.selectbox("Batter", batters, index=default_idx if batters else 0)

opposition = st.sidebar.multiselect("Opposition (bowlingTeam)", safe_unique("bowlingTeam"))
bowler = st.sidebar.multiselect("Bowler", safe_unique("bowlerPlayer"))
bowling_type = st.sidebar.multiselect("Bowling Type", safe_unique("bowlingTypeId"))
years = st.sidebar.multiselect("Year", sorted(df["year"].dropna().unique()) if "year" in df.columns else [])
max_over = int(np.nanmax(df["overNumber"])) if ("overNumber" in df.columns and not df["overNumber"].isna().all()) else 0
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
# Helpers for tables & matrices
# =========================
SAFE_CONN = {"welltimed", "middled", "left", "blank", "nan"}

def false_shot_pct(df_in: pd.DataFrame) -> float:
    if df_in.empty or "conn_clean" not in df_in.columns:
        return 0.0
    conn = df_in["conn_clean"]
    false_shots = conn[~conn.isin(SAFE_CONN)]
    pct = (len(false_shots) / len(conn)) * 100 if len(conn) else 0
    return float(np.round(pct, 2))

def agg_block(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df_in.empty or group_col not in df_in.columns:
        return pd.DataFrame(columns=[group_col, "Runs", "Balls", "Dismissals", "False Shot %", "Average", "Strike Rate"])
    g = df_in.groupby(group_col, dropna=False).agg(
        Runs=("runsScored", "sum") if "runsScored" in df_in.columns else ("conn_clean", "count"),
        Balls=("runsScored", "count") if "runsScored" in df_in.columns else ("conn_clean", "count"),
        Dismissals=("isWicket_bool", "sum") if "isWicket_bool" in df_in.columns else ("conn_clean", "size"),
    ).reset_index()

    # Drop blank labels and groups with zero balls
    g = g[g[group_col].notna()]
    bad = {"", "nan", "none", "unknown"}
    g = g[~g[group_col].astype(str).str.strip().str.lower().isin(bad)]
    g = g[g["Balls"] > 0]

    g["False Shot %"] = g.apply(lambda r: false_shot_pct(df_in[df_in[group_col] == r[group_col]]), axis=1)

    # Average & SR
    g["Average"] = g.apply(lambda r: (np.inf if r["Dismissals"] == 0 else round(r["Runs"] / r["Dismissals"], 2)), axis=1)
    g["Strike Rate"] = g.apply(lambda r: (round((r["Runs"] / r["Balls"]) * 100, 2) if r["Balls"] else 0.0), axis=1)
    g = g.sort_values(["Runs", "Strike Rate"], ascending=[False, False])
    return g

# =========================
# Tabs
# =========================
tab_tables, tab_zones, tab_matrix = st.tabs(["Tables", "Shot Areas", "Line × Length"])

# ---------- TABLES ----------
with tab_tables:
    st.subheader("Batting")
    bt1, bt2 = st.tabs(["Feet", "Shot Type"])
    with bt1:
        st.dataframe(agg_block(filt, "battingFeetId"), use_container_width=True, height=420)
    with bt2:
        st.dataframe(agg_block(filt, "battingShotTypeId"), use_container_width=True, height=420)

    st.subheader("Bowling vs Batter")
    bw1, bw2, bw3 = st.tabs(["Hand", "Detail", "Bowler"])
    with bw1:
        st.dataframe(agg_block(filt, "bowlingHandId"), use_container_width=True, height=420)
    with bw2:
        st.dataframe(agg_block(filt, "bowlingDetailId"), use_container_width=True, height=420)
    with bw3:
        st.dataframe(agg_block(filt, "bowlerPlayer"), use_container_width=True, height=420)

# ---------- SHOT AREAS (8 zones, violet) ----------
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
    # Runs filter (what contributes to zone totals)
    if "runsScored" in filt.columns and filt["runsScored"].notna().any():
        avail_runs = sorted(filt["runsScored"].dropna().astype(int).unique().tolist())
    else:
        avail_runs = [0,1,2,3,4,6]
    default_runs = [r for r in [1,2,3,4,6] if r in avail_runs] or avail_runs
    runs_sel = st.multiselect("Runs to include", options=avail_runs, default=default_runs)

    filt_runs = filt[filt.get("runsScored").isin(runs_sel)] if "runsScored" in filt.columns else filt

    FIELD_POS_COL = pick_fieldpos_col(filt_runs)
    if FIELD_POS_COL is None:
        st.error("Couldn't find a fielding position column in your data (e.g., 'fieldingPosition').")
    else:
        zdf = filt_runs[[FIELD_POS_COL, "runsScored"]].copy()
        zdf[FIELD_POS_COL] = zdf[FIELD_POS_COL].astype(str)
        zdf["zone"] = zdf[FIELD_POS_COL].apply(map_pos_to_zone)
        zone_runs = zdf.groupby("zone")["runsScored"].sum().reindex(ZONE_LABELS, fill_value=0).reset_index()

        # Drawing (smaller)
        RADIUS = 52
        STRIP_H = 40
        STRIP_W = 9
        VIOLET = (127/255, 0/255, 255/255)  # #7F00FF

        r_inner = STRIP_H/2 + 1.0      # keep dividers outside pitch
        start_angles = np.linspace(22.5, 360+22.5, 9)  # 8 slices

        def draw_ground(ax):
            circ = Circle((0,0), RADIUS, fill=False, linewidth=1.2)
            ax.add_patch(circ)
            pitch = Rectangle((-STRIP_W/2, -STRIP_H/2), STRIP_W, STRIP_H, fill=False, linewidth=1.0)
            ax.add_patch(pitch)
            ax.plot([-STRIP_W/2, STRIP_W/2], [STRIP_H/2, STRIP_H/2], lw=0.7)
            ax.plot([-STRIP_W/2, STRIP_W/2], [-STRIP_H/2, -STRIP_H/2], lw=0.7)
            ax.plot([0],[0], marker="o", markersize=2.5, color="black")
            ax.set_aspect("equal")
            ax.set_xlim(-RADIUS*1.02, RADIUS*1.02)
            ax.set_ylim(-RADIUS*1.02, RADIUS*1.02)
            ax.axis("off")

        st.subheader("Shot Areas")
        fig, ax = plt.subplots(figsize=(3.0,3.0))  # much smaller
        draw_ground(ax)

        # OFF / LEG text (outside circle)
        ax.text(RADIUS * 1.02, 0, "OFF", ha="center", va="center", fontsize=9, weight="bold")
        ax.text(-RADIUS * 1.02, 0, "LEG", ha="center", va="center", fontsize=9, weight="bold")

        # Dividers only outside pitch
        for ang in start_angles[:-1]:
            x0 = r_inner * np.sin(np.deg2rad(ang))
            y0 = -r_inner * np.cos(np.deg2rad(ang))
            x1 = RADIUS * np.sin(np.deg2rad(ang))
            y1 = -RADIUS * np.cos(np.deg2rad(ang))
            ax.plot([x0, x1], [y0, y1], linestyle="--", linewidth=0.45, alpha=0.5, color="gray")

        # Uniform violet wedges + numbers
        for i, label in enumerate(ZONE_LABELS):
            start = start_angles[i]
            wedge = Wedge(center=(0,0), r=RADIUS, theta1=start, theta2=start+45,
                          width=RADIUS - r_inner, facecolor=VIOLET, edgecolor="black", alpha=0.35)
            ax.add_patch(wedge)
            mid = start + 22.5
            x = ((r_inner + (RADIUS - r_inner)*0.6)) * np.sin(np.deg2rad(mid))
            y = -((r_inner + (RADIUS - r_inner)*0.6)) * np.cos(np.deg2rad(mid))
            runs = int(zone_runs.loc[zone_runs["zone"]==label, "runsScored"].values[0]) if (zone_runs["zone"]==label).any() else 0
            ax.text(x, y, str(runs), ha="center", va="center", fontsize=10, weight="bold", color="black")

        st.pyplot(fig, use_container_width=False)

# ---------- LINE × LENGTH ----------
with tab_matrix:
    st.subheader("Line × Length")
    if filt.empty or "lengthTypeId" not in filt.columns or "lineTypeId" not in filt.columns:
        st.info("No data for current filters or missing line/length columns.")
    else:
        grp = filt.groupby(["lengthTypeId", "lineTypeId"], dropna=False)
        rows = []
        for (length, line), d in grp:
            fs = false_shot_pct(d)
            dismissals = d["isWicket_bool"].sum() if "isWicket_bool" in d.columns else 0
            runs = d["runsScored"].sum() if "runsScored" in d.columns else 0
            avg = np.inf if dismissals == 0 else round(runs / dismissals, 2)
            rows.append({"lengthTypeId": length, "lineTypeId": line, "False Shot %": fs, "Average": avg})
        mat = pd.DataFrame(rows)

        # Enforce custom ordering
        length_order = [
            "Full Toss",
            "Yorker",
            "Half volley",
            "Length",
            "Back of a length",
            "half tracker",
            "Short",
        ]
        line_order = [
            "wide outside off stump",
            "outside off stump",
            "off stump",
            "middle stump",
            "leg stump",
            "wide down leg",
        ]

        if not mat.empty:
            t1, t2 = st.tabs(["False Shot %", "Average"])

            with t1:
                p1 = mat.pivot(index="lengthTypeId", columns="lineTypeId", values="False Shot %")
                p1 = p1.reindex(index=length_order)
                p1 = p1.reindex(columns=line_order)
                styler1 = p1.style.format("{:.2f}")
                st.dataframe(styler1, use_container_width=True, height=520)

            with t2:
                p2 = mat.pivot(index="lengthTypeId", columns="lineTypeId", values="Average")
                p2 = p2.reindex(index=length_order)
                p2 = p2.reindex(columns=line_order)

                def fmt_avg(v):
                    if pd.isna(v):
                        return ""
                    if np.isinf(v):
                        return "∞"
                    return f"{v:.2f}"

                styler2 = p2.style.format(fmt_avg)
                st.dataframe(styler2, use_container_width=True, height=520)
