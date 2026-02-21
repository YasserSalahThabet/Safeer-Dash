import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# ----------------------------
# Branding assets
# ----------------------------
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
HERO_IMG = ASSETS / "hero.jpg"       # hero/banner image
LOGO_IMG = ASSETS / "logo.png"       # logo for sidebar
FAVICON_IMG = ASSETS / "favicon.png" # optional

st.set_page_config(
    page_title="Safeer Dash",
    page_icon=str(FAVICON_IMG) if FAVICON_IMG.exists() else "🟢",
    layout="wide",
)

# Optional: small CSS polish for a cleaner look
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }

      div[data-testid="stMetric"] { 
          background: rgba(255,255,255,0.02); 
          padding: 8px; 
          border-radius: 12px; 
      }

      div[data-testid="stMetricValue"] {
          font-size: 19px !important;
      }

      div[data-testid="stMetricLabel"] {
          font-size: 11px !important;
          opacity: 0.8;
      }

      .safeer-subtitle { 
          margin-top: -10px; 
          opacity: 0.85; 
      }
    </style>
    """,
    unsafe_allow_html=True
)
# ----------------------------
# Hero/Header
# ----------------------------
if HERO_IMG.exists():
    st.image(str(HERO_IMG), use_container_width=True)

st.markdown("# Safeer Dash")
st.markdown('<div class="safeer-subtitle">Driver performance dashboard • low performers highlighted first</div>', unsafe_allow_html=True)
st.divider()

# ----------------------------
# Header detection + mapping (Arabic template-friendly)
# ----------------------------
AR_HEADER_MARKERS = ["معرف السائق", "اسم السائق", "معدل التوصيل", "معدل الإلغاء"]

COLS = {
    "driver_id": ["معرف السائق", "Driver_ID", "driver_id", "id"],
    "user_name": ["اسم اليوزر", "User", "Username", "user_name"],
    "driver_name": ["اسم السائق", "Driver_Name", "driver_name", "السائق"],
    "completed": ["طلب مكتمل", "Completed", "completed_orders"],
    "cancelled": ["طلب ملغي", "Cancelled", "orders_cancelled"],
    "driver_reject": ["رفض السائق", "Driver Rejections", "driver_rejections"],
    "auto_reject": ["رفض تلقائي", "Auto Rejection", "auto_reject"],
    "cancel_rate": ["معدل الإلغاء", "Cancel_Rate", "cancel_rate"],
    "delivery_rate": ["معدل التوصيل", "Delivery_Rate", "delivery_rate"],
    "daily_deficit": ["العجزاليومي", "Daily Deficit", "daily_deficit"],
    "monthly_deficit": ["العجز الشهري", "Monthly Deficit", "monthly_deficit"],
    "invalid": ["غير صالح", "Invalid", "invalid"],
    "work_days": ["أيام العمل", "Work Days", "work_days"],
}

CANCEL_RED_THRESHOLD = 0.002  # 0.20% (because rates are stored 0..1)

def normalize(x) -> str:
    return str(x).strip()

def find_header_row(df_preview: pd.DataFrame, max_rows: int = 40) -> int | None:
    for r in range(min(max_rows, len(df_preview))):
        row_vals = [normalize(v) for v in df_preview.iloc[r].tolist()]
        hits = sum(1 for m in AR_HEADER_MARKERS if m in row_vals)
        if hits >= 2:
            return r
    return None

def pick_col(df_cols, candidates):
    cols = [normalize(c) for c in df_cols]
    for cand in candidates:
        cand = normalize(cand)
        if cand in cols:
            return cand
    return None

@st.cache_data(show_spinner=False)
def load_excel(uploaded_file) -> pd.DataFrame:
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = xls.sheet_names[0]

    preview = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, nrows=50)
    header_row = find_header_row(preview)
    if header_row is None:
        header_row = 1  # fallback (common in your reports)

    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)
    df.columns = [normalize(c) for c in df.columns]
    return df

def to_report(df_raw: pd.DataFrame) -> pd.DataFrame:
    mapped = {key: pick_col(df_raw.columns, cands) for key, cands in COLS.items()}

    required = ["driver_id", "driver_name", "delivery_rate", "cancel_rate", "completed", "cancelled"]
    missing = [k for k in required if mapped.get(k) is None]
    if missing:
        st.error(
            "Missing required columns in your raw Excel:\n"
            + ", ".join(missing)
            + "\n\nFound columns:\n"
            + ", ".join(df_raw.columns)
        )
        st.stop()

    out = pd.DataFrame({
        "Driver_ID": pd.to_numeric(df_raw[mapped["driver_id"]], errors="coerce"),
        "Driver_Name": df_raw[mapped["driver_name"]].astype(str).str.strip(),
        "User_Name": df_raw[mapped["user_name"]].astype(str).str.strip() if mapped["user_name"] else "",
        "Delivery_Rate": pd.to_numeric(df_raw[mapped["delivery_rate"]], errors="coerce"),
        "Cancel_Rate": pd.to_numeric(df_raw[mapped["cancel_rate"]], errors="coerce"),
        "Orders_Completed": pd.to_numeric(df_raw[mapped["completed"]], errors="coerce"),
        "Orders_Cancelled": pd.to_numeric(df_raw[mapped["cancelled"]], errors="coerce"),
        "Driver_Rejections": pd.to_numeric(df_raw[mapped["driver_reject"]], errors="coerce") if mapped["driver_reject"] else 0,
        "Auto_Rejections": pd.to_numeric(df_raw[mapped["auto_reject"]], errors="coerce") if mapped["auto_reject"] else 0,
        "Daily_Deficit": pd.to_numeric(df_raw[mapped["daily_deficit"]], errors="coerce") if mapped["daily_deficit"] else 0,
        "Monthly_Deficit": pd.to_numeric(df_raw[mapped["monthly_deficit"]], errors="coerce") if mapped["monthly_deficit"] else 0,
        "Invalid": pd.to_numeric(df_raw[mapped["invalid"]], errors="coerce") if mapped["invalid"] else 0,
        "Work_Days": pd.to_numeric(df_raw[mapped["work_days"]], errors="coerce") if mapped["work_days"] else 0,
    })

    # Clean / types
    out = out.dropna(subset=["Driver_ID", "Driver_Name"], how="all")
    out["Driver_ID"] = pd.to_numeric(out["Driver_ID"], errors="coerce").astype("Int64")
    out["Driver_Name"] = out["Driver_Name"].fillna("").astype(str).str.strip()
    out["User_Name"] = out["User_Name"].fillna("").astype(str).str.strip()

    num_cols = [
        "Delivery_Rate", "Cancel_Rate", "Orders_Completed", "Orders_Cancelled",
        "Driver_Rejections", "Auto_Rejections", "Daily_Deficit", "Monthly_Deficit",
        "Invalid", "Work_Days"
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # ----------------------------
    # Performance Score as TRUE percentage (0..1)
    # ----------------------------
    score_0_100 = (
        (out["Delivery_Rate"] * 100) * 0.55
        + ((1 - out["Cancel_Rate"]) * 100) * 0.25
        + (100 - (out["Driver_Rejections"] * 10)) * 0.10
        + (100 - (out["Orders_Cancelled"] * 2)) * 0.10
    )
    out["Performance_Score"] = (score_0_100 / 100).clip(lower=0, upper=1)

    # Low performers first
    out["Attention_Rank"] = out["Performance_Score"].rank(ascending=True, method="dense").astype(int)
    out = out.sort_values(["Performance_Score", "Cancel_Rate", "Orders_Cancelled"], ascending=[True, False, False])

    return out

# ----------------------------
# Styling rules (your requirements)
# ----------------------------
def style_attention(df: pd.DataFrame):
    def red_if_reject(x):
        try:
            return "color: red; font-weight: 700;" if float(x) > 0 else ""
        except Exception:
            return ""

    def red_if_cancel_high(x):
        try:
            return "color: red; font-weight: 700;" if float(x) >= CANCEL_RED_THRESHOLD else ""
        except Exception:
            return ""

    def red_if_score_not_100(x):
        # score is 0..1, 100% == 1.0
        try:
            return "color: red; font-weight: 700;" if round(float(x), 6) != 1.0 else ""
        except Exception:
            return ""

    sty = df.style

    # Force percent formatting
    fmt = {}
    for c in ["Performance_Score", "Delivery_Rate", "Cancel_Rate"]:
        if c in df.columns:
            fmt[c] = "{:.2%}"
    if fmt:
        sty = sty.format(fmt)

    # Apply red rules
    if "Driver_Rejections" in df.columns:
        sty = sty.applymap(red_if_reject, subset=["Driver_Rejections"])
    if "Cancel_Rate" in df.columns:
        sty = sty.applymap(red_if_cancel_high, subset=["Cancel_Rate"])
    if "Performance_Score" in df.columns:
        sty = sty.applymap(red_if_score_not_100, subset=["Performance_Score"])

    return sty

# ----------------------------
# Sidebar (logo + upload + filters)
# ----------------------------
with st.sidebar:
    if LOGO_IMG.exists():
        st.image(str(LOGO_IMG), use_container_width=True)

    st.markdown("### Upload report")
    uploaded = st.file_uploader("Upload raw Excel template (.xlsx)", type=["xlsx"])

    st.divider()
    st.markdown("### Filters")
    search = st.text_input("Search (ID / Driver / User)", "")
    min_delivery = st.slider("Min Delivery Rate", 0.0, 1.0, 0.0, 0.01)
    max_cancel = st.slider("Max Cancel Rate", 0.0, 1.0, 1.0, 0.01)

    st.caption("Rules")
    st.caption("• Rejects > 0 = red")
    st.caption("• Cancel ≥ 0.20% = red")
    st.caption("• Score ≠ 100% = red")

if not uploaded:
    st.info("Upload the raw Excel file to start.")
    st.stop()

# ----------------------------
# Load -> transform -> filter
# ----------------------------
raw = load_excel(uploaded)
report = to_report(raw)

f = report.copy()
if search.strip():
    s = search.strip().lower()
    f = f[
        f["Driver_Name"].str.lower().str.contains(s, na=False)
        | f["User_Name"].str.lower().str.contains(s, na=False)
        | f["Driver_ID"].astype(str).str.contains(s, na=False)
    ]

f = f[(f["Delivery_Rate"] >= min_delivery) & (f["Cancel_Rate"] <= max_cancel)]

# ----------------------------
# KPIs
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Drivers (filtered)", f"{len(f):,}")
k2.metric("Avg Delivery Rate", f"{f['Delivery_Rate'].mean():.2%}" if len(f) else "—")
k3.metric("Avg Cancel Rate", f"{f['Cancel_Rate'].mean():.2%}" if len(f) else "—")
k4.metric("Avg Score", f"{f['Performance_Score'].mean():.2%}" if len(f) else "—")

st.divider()

# ----------------------------
# Needs attention (low performers first)
# ----------------------------
st.subheader("🚨 Needs Attention (Low performers first)")
attention_view = f[[
    "Attention_Rank", "Driver_ID", "Driver_Name", "User_Name",
    "Performance_Score", "Delivery_Rate", "Cancel_Rate",
    "Orders_Completed", "Orders_Cancelled", "Driver_Rejections"
]].head(30)

st.dataframe(style_attention(attention_view), use_container_width=True, hide_index=True)

# ----------------------------
# Driver dropdown (lookup)
# ----------------------------
st.divider()
st.subheader("🔎 Driver Lookup")

driver_list = f["Driver_Name"].dropna().unique().tolist()
selected = st.selectbox("Select driver", ["(choose)"] + driver_list)

if selected != "(choose)":
    d = f[f["Driver_Name"] == selected].head(1)

    if len(d):
        d = d.iloc[0]

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Driver ID", str(d["Driver_ID"]))
        c2.metric("Delivery Rate", f"{d['Delivery_Rate']:.2%}")
        c3.metric("Orders Completed", f"{int(d['Orders_Completed']):,}")
        c4.metric("Driver Rejections", f"{int(d['Driver_Rejections']):,}")
        c5.metric("Score", f"{d['Performance_Score']:.2%}")

# ----------------------------
# Charts
# ----------------------------
st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Lowest 20 Scores")
    worst = f.sort_values("Performance_Score", ascending=True).head(20)
    fig1 = px.bar(
        worst,
        x="Performance_Score",
        y="Driver_Name",
        orientation="h",
        hover_data=["Driver_ID", "Delivery_Rate", "Cancel_Rate", "Orders_Cancelled", "Driver_Rejections"],
    )
    fig1.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("Cancel Rate vs Delivery Rate")
    fig2 = px.scatter(
        f,
        x="Cancel_Rate",
        y="Delivery_Rate",
        size="Orders_Cancelled",
        hover_data=["Driver_Name", "Driver_ID", "Performance_Score", "Driver_Rejections"],
    )
    fig2.update_xaxes(tickformat=".0%")
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Full table + export
# ----------------------------
st.divider()
st.subheader("All Drivers (Filtered)")
st.dataframe(style_attention(f), use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download filtered CSV",
    data=f.to_csv(index=False, encoding="utf-8-sig"),
    file_name="drivers_filtered.csv",
    mime="text/csv",
)





