
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
    import pandas as pd, numpy as np
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.warning(\"Couldn't load from Google Sheets URL. Falling back to local CSV 'All Test Matches Till 2025.csv'.\\nError: {}\".format(e))
        df = pd.read_csv(\"All Test Matches Till 2025.csv\")
    # Clean & enrich
    if \"ballDateTime\" in df.columns:
        df[\"ballDateTime\"] = pd.to_datetime(df[\"ballDateTime\"], errors=\"coerce\")
        df[\"year\"] = df[\"ballDateTime\"].dt.year
    else:
        df[\"year\"] = np.nan

    # Normalise booleans
    def to_bool(x):
        if isinstance(x, bool): return x
        if isinstance(x, (int, float)): return bool(x)
        s = str(x).strip().lower()
        return s in (\"true\", \"1\", \"yes\")
    df[\"isWicket_bool\"] = df.get(\"isWicket\", False)
    df[\"isWicket_bool\"] = df[\"isWicket_bool\"].apply(to_bool)

    # Numerics
    for c in [\"runsScored\", \"overNumber\", \"ballNumber\"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors=\"coerce\")

    # Hand
    if \"battingHandId\" in df.columns:
        df[\"battingHandId\"] = df[\"battingHandId\"].fillna(\"Right\").astype(str)
    else:
        df[\"battingHandId\"] = \"Right\"

    # Connection (for False Shot % in other tables)
    if \"battingConnectionId\" in df.columns:
        df[\"conn_clean\"] = df[\"battingConnectionId\"].fillna(\"nan\").astype(str).str.strip().str.lower()
    else:
        df[\"conn_clean\"] = \"nan\"

    return df

df = load_data(DATA_URL)

# =========================
# Sidebar Filters — Batting
# =========================
st.sidebar.header(\"Filters — Batting (8-zone wheel)\")

def safe_unique(col):
    return sorted(df[col].dropna().unique()) if col in df.columns else []

batter = st.sidebar.selectbox(\"Batter\", safe_unique(\"battingPlayer\"))
opposition = st.sidebar.multiselect(\"Opposition (bowlingTeam)\", safe_unique(\"bowlingTeam\"))
bowler = st.sidebar.multiselect(\"Bowler\", safe_unique(\"bowlerPlayer\"))
bowling_type = st.sidebar.multiselect(\"Bowling Type\", safe_unique(\"bowlingTypeId\"))
years = st.sidebar.multiselect(\"Year\", sorted(df[\"year\"].dropna().unique()) if \"year\" in df.columns else [])
max_over = int(np.nanmax(df[\"overNumber\"])) if \"overNumber\" in df.columns and not df[\"overNumber\"].isna().all() else 0
over_range = st.sidebar.slider(\"Over range\", 0, max_over, (0, max_over))

# Apply filters
filt = df.copy()
if batter:
    filt = filt[filt.get(\"battingPlayer\", \"\").eq(batter)]
if opposition:
    filt = filt[filt.get(\"bowlingTeam\", \"\").isin(opposition)]
if bowler:
    filt = filt[filt.get(\"bowlerPlayer\", \"\").isin(bowler)]
if bowling_type:
    filt = filt[filt.get(\"bowlingTypeId\", \"\").isin(bowling_type)]
if years:
    filt = filt[filt.get(\"year\", \"\").isin(years)]
if \"overNumber\" in filt.columns:
    filt = filt[(filt[\"overNumber\"] >= over_range[0]) & (filt[\"overNumber\"] <= over_range[1])]

# Batting hand selector (defaults from data mode)
default_hand = \"Right\"
if \"battingHandId\" in filt.columns and not filt[\"battingHandId\"].dropna().empty:
    mode_series = filt[\"battingHandId\"].mode()
    if not mode_series.empty:
        default_hand = mode_series.iat[0]
hand = st.sidebar.radio(\"Batting hand\", [\"Right\", \"Left\"], index=0 if str(default_hand).lower().startswith(\"r\") else 1)

st.title(\"8‑Zone Wagon Wheel — Full Ground (No shot angle)\")
st.caption(\"Zones computed from fielding positions. OFF/LEG labels adjust with batting hand.\")

# =========================
# Zone mapping from fielding positions -> 8 wedges
# =========================
ZONE_LABELS = [\"OFF Behind Sq\", \"OFF Square\", \"OFF Cover\", \"OFF Straight\",
               \"LEG Straight\", \"LEG Midwicket\", \"LEG Square\", \"LEG Behind Sq\"]

# Keywords per zone (lowercased)
ZONE_KEYWORDS = {
    \"OFF Behind Sq\": [\"third\", \"third man\", \"deep third\", \"gully\", \"slip\", \"backward point\", \"deep back point\", \"deep back\"],
    \"OFF Square\": [\"point\", \"deep point\", \"square\", \"backward sq\", \"backward square\", \"cover point\"],
    \"OFF Cover\": [\"extra cover\", \"cover\", \"deep cover\", \"cover drive\", \"deep extra\"],
    \"OFF Straight\": [\"mid-off\", \"long off\", \"straight off\", \"off drive\"],
    \"LEG Straight\": [\"mid-on\", \"long on\", \"on drive\", \"straight on\"],
    \"LEG Midwicket\": [\"mid-wicket\", \"midwicket\", \"deep mid\", \"cow corner\"],
    \"LEG Square\": [\"square leg\", \"deep square\", \"backward square leg\", \"backward sq leg\"],
    \"LEG Behind Sq\": [\"fine leg\", \"short fine\", \"leg slip\", \"leg gully\", \"deep fine\", \"short leg\"]
}

def pick_fieldpos_col(df: pd.DataFrame) -> str | None:
    candidates = [
        \"fieldingPosition\", \"fielding_position\", \"fielderPosition\", \"fielder_position\",
        \"fielderPos\", \"field_position\", \"fielderPositionName\", \"fielderRole\"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Last resort: if dismissal/caught info carries position name
    for c in df.columns:
        if \"position\" in c.lower() or (\"field\" in c.lower() and \"pos\" in c.lower()):
            return c
    return None

FIELD_POS_COL = pick_fieldpos_col(filt)

def map_pos_to_zone(val: str) -> str:
    s = str(val).strip().lower()
    for zone, kws in ZONE_KEYWORDS.items():
        for k in kws:
            if k in s:
                return zone
    # generic fallbacks
    if \"point\" in s: return \"OFF Square\"
    if \"cover\" in s: return \"OFF Cover\"
    if \"off\" in s: return \"OFF Straight\"
    if \"on\" in s: return \"LEG Straight\"
    if \"mid\" in s and \"off\" in s: return \"OFF Straight\"
    if \"mid\" in s and \"on\" in s: return \"LEG Straight\"
    if \"wicket\" in s: return \"LEG Midwicket\" if \"mid\" in s else \"LEG Square\"
    if \"leg\" in s: return \"LEG Square\"
    if \"slip\" in s or \"gully\" in s: return \"OFF Behind Sq\"
    if \"keeper\" in s: return \"OFF Behind Sq\"
    return \"Unknown\"

def zone_ordinal(zone: str) -> int:
    # We order clockwise starting at OFF Behind Sq (top-right quadrant) for RHB.
    try:
        return ZONE_LABELS.index(zone)
    except ValueError:
        return -1

# =========================
# Run filter
# =========================
if \"runsScored\" in filt.columns and filt[\"runsScored\"].notna().any():
    avail_runs = sorted(filt[\"runsScored\"].dropna().astype(int).unique().tolist())
else:
    avail_runs = [0,1,2,3,4,6]
default_runs = [r for r in [1,2,3,4,6] if r in avail_runs] or avail_runs
runs_sel = st.multiselect(\"Show runs\", options=avail_runs, default=default_runs)

filt_runs = filt[filt.get(\"runsScored\").isin(runs_sel)] if \"runsScored\" in filt.columns else filt

# =========================
# Zone aggregation
# =========================
if FIELD_POS_COL is None:
    st.error(\"Couldn't find a fielding position column in your data. Please confirm the column name (e.g., 'fieldingPosition').\")
else:
    zdf = filt_runs[[FIELD_POS_COL, \"runsScored\"]].copy()
    zdf[FIELD_POS_COL] = zdf[FIELD_POS_COL].astype(str)
    zdf[\"zone\"] = zdf[FIELD_POS_COL].apply(map_pos_to_zone)
    zone_runs = zdf.groupby(\"zone\")[\"runsScored\"].sum().reindex(ZONE_LABELS, fill_value=0).reset_index()

    # =========================
    # Draw full-circle pitch with 8 zones (wedge fills) and totals
    # =========================
    RADIUS = 100

    def draw_full_pitch(ax, radius=RADIUS):
        circle = Circle((0, 0), radius, fill=False, linewidth=1.5)
        ax.add_patch(circle)
        strip_w = 10
        strip_h = 44
        pitch = Rectangle((-strip_w/2, -strip_h/2), strip_w, strip_h, fill=False, linewidth=1.2)
        ax.add_patch(pitch)
        # Creases
        crease_y = strip_h/2
        ax.plot([-strip_w/2, strip_w/2], [crease_y, crease_y], lw=0.8)
        ax.plot([-strip_w/2, strip_w/2], [-crease_y, -crease_y], lw=0.8)
        ax.plot([0],[0], marker=\"o\", markersize=3, color=\"black\")
        ax.set_aspect(\"equal\")
        ax.set_xlim(-radius*1.1, radius*1.1)
        ax.set_ylim(-radius*1.1, radius*1.1)
        ax.axis(\"off\")

    def off_leg_labels(ax, hand: str, radius=RADIUS):
        off_x = radius * 0.7 if hand.lower().startswith(\"r\") else -radius * 0.7
        leg_x = -off_x
        ax.text(off_x, 0, \"OFF\", ha=\"center\", va=\"center\", fontsize=11, weight=\"bold\")
        ax.text(leg_x, 0, \"LEG\", ha=\"center\", va=\"center\", fontsize=11, weight=\"bold\")

    st.subheader(\"Zone Runs (8 wedges)\")
    fig, ax = plt.subplots(figsize=(7,7))
    draw_full_pitch(ax, RADIUS)
    off_leg_labels(ax, hand, RADIUS)

    # Compute wedge angles (8 equal slices, starting at 45°, clockwise)
    # We’ll lay them OFF Behind Sq -> OFF Square -> OFF Cover -> OFF Straight ->
    # LEG Straight -> LEG Midwicket -> LEG Square -> LEG Behind Sq
    # and flip labels visually with OFF/LEG text (zones themselves are relative to batter, so names stay).
    start_angles = np.linspace(22.5, 360+22.5, 9)  # 8 sectors of 45°
    # Map zone label -> wedge index
    for i, label in enumerate(ZONE_LABELS):
        start = start_angles[i]
        wedge = Wedge(center=(0,0), r=RADIUS, theta1=start, theta2=start+45, width=RADIUS*0.35, alpha=0.15)
        ax.add_patch(wedge)
        # Text at sector center
        mid = start + 22.5
        x = (RADIUS*0.6) * np.sin(np.deg2rad(mid))
        y = -(RADIUS*0.6) * np.cos(np.deg2rad(mid))
        runs = int(zone_runs.loc[zone_runs[\"zone\"]==label, \"runsScored\"].values[0]) if (zone_runs[\"zone\"]==label).any() else 0
        ax.text(x, y, str(runs), ha=\"center\", va=\"center\", fontsize=12, weight=\"bold\" )

    st.pyplot(fig)

    # =========================
    # Heat “map” (wedge color intensity by runs)
    # =========================
    st.subheader(\"Zone Hot Map (run-weighted fill)\")
    fig2, ax2 = plt.subplots(figsize=(7,7))
    draw_full_pitch(ax2, RADIUS)
    off_leg_labels(ax2, hand, RADIUS)

    total = zone_runs[\"runsScored\"].max()
    # Avoid division by 0
    scale = total if total > 0 else 1

    for i, label in enumerate(ZONE_LABELS):
        start = start_angles[i]
        # Normalize color intensity 0..1
        val = float(zone_runs.loc[zone_runs[\"zone\"]==label, \"runsScored\"].values[0]) if (zone_runs[\"zone\"]==label).any() else 0.0
        intensity = val / scale
        wedge = Wedge(center=(0,0), r=RADIUS, theta1=start, theta2=start+45, width=RADIUS*0.9,
                      facecolor=(1-intensity, 1-intensity, 1), edgecolor=\"black\", alpha=0.5)
        ax2.add_patch(wedge)
        mid = start + 22.5
        x = (RADIUS*0.55) * np.sin(np.deg2rad(mid))
        y = -(RADIUS*0.55) * np.cos(np.deg2rad(mid))
        ax2.text(x, y, str(int(val)), ha=\"center\", va=\"center\", fontsize=11, weight=\"bold\" )

    st.pyplot(fig2)
