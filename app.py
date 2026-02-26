import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# =========================
# Branding assets
# =========================
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
HERO_IMG = ASSETS / "hero.jpg"
LOGO_IMG = ASSETS / "logo.png"
FAVICON_IMG = ASSETS / "favicon.png"

st.set_page_config(
    page_title="Safeer Dash",
    page_icon=str(FAVICON_IMG) if FAVICON_IMG.exists() else "🟢",
    layout="wide",
)

# =========================
# CSS (smaller KPI text so numbers fit)
# =========================
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }

      div[data-testid="stMetric"] {
          background: rgba(255,255,255,0.02);
          padding: 8px;
          border-radius: 12px;
      }

      div[data-testid="stMetricValue"] { font-size: 19px !important; }
      div[data-testid="stMetricLabel"] { font-size: 11px !important; opacity: 0.8; }

      .safeer-subtitle { margin-top: -10px; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Header
# =========================
if HERO_IMG.exists():
    st.image(str(HERO_IMG), use_container_width=True)

st.markdown("# Safeer Dash")
st.markdown('<div class="safeer-subtitle">Multi-file driver performance dashboard • alerts prioritized</div>', unsafe_allow_html=True)
st.divider()

# =========================
# Helpers
# =========================
def normalize_col(c: str) -> str:
    return str(c).strip()

@st.cache_data(show_spinner=False)
def read_first_sheet_excel(uploaded_file) -> pd.DataFrame:
    xls = pd.ExcelFile(uploaded_file)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(uploaded_file, sheet_name=sheet, header=0)
    df.columns = [normalize_col(c) for c in df.columns]
    return df

def safe_to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def pick(df_cols, candidates):
    cols = [normalize_col(c) for c in df_cols]
    for cand in candidates:
        if cand in cols:
            return cand
    return None

CANCEL_RED_THRESHOLD = 0.002  # 0.20%

# =========================
# PERFORMANCE FILE MAPPING (your template)
# =========================
PERF_COLS = {
    "driver_id": ["معرّف السائق", "معرف السائق", "Driver_ID", "driver_id", "id"],
    "first_name": ["اسم السائق", "First Name", "first_name"],
    "last_name": ["اسم السائق.1", "Last Name", "last_name"],

    # Key metrics (as you said)
    "cancel_rate": ["معدل الإلغاء بسبب مشاكل التوصيل", "معدل الإلغاء", "Cancel_Rate", "cancel_rate"],
    "driver_reject": ["المهام المرفوضة (السائق)", "رفض السائق", "Driver Rejections", "driver_rejections"],
    "orders_delivered": ["المهام التي تم تسليمها", "الطلبات المسلمة", "طلبات مكتملة", "طلب مكتمل",
                         "Orders_Delivered", "orders_delivered"],

    # Delivery rate / completion metric in your report
    "delivery_rate": ["معدل اكتمال الطلبات (غير متعلق بالتوصيل)", "معدل اكتمال الطلبات",
                      "معدل التوصيل", "Delivery_Rate", "delivery_rate"],
}

def build_performance_report(df_raw: pd.DataFrame) -> pd.DataFrame:
    mapped = {k: pick(df_raw.columns, v) for k, v in PERF_COLS.items()}

    required = ["driver_id", "first_name", "last_name", "delivery_rate", "cancel_rate", "orders_delivered", "driver_reject"]
    missing = [k for k in required if not mapped.get(k)]
    if missing:
        st.error("Performance file is missing required columns: " + ", ".join(missing))
        st.write("Columns found:")
        st.write(list(df_raw.columns))
        st.stop()

    driver_name = (
        df_raw[mapped["first_name"]].astype(str).str.strip().fillna("")
        + " "
        + df_raw[mapped["last_name"]].astype(str).str.strip().fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    out = pd.DataFrame({
        "Driver_ID": safe_to_numeric(df_raw[mapped["driver_id"]]),
        "Driver_Name": driver_name,

        "Delivery_Rate": safe_to_numeric(df_raw[mapped["delivery_rate"]]),
        "Cancel_Rate": safe_to_numeric(df_raw[mapped["cancel_rate"]]),
        "Orders_Delivered": safe_to_numeric(df_raw[mapped["orders_delivered"]]),
        "Driver_Rejections": safe_to_numeric(df_raw[mapped["driver_reject"]]),
    })

    out["Driver_ID"] = pd.to_numeric(out["Driver_ID"], errors="coerce").astype("Int64")
    out["Delivery_Rate"] = out["Delivery_Rate"].fillna(0).clip(0, 1)
    out["Cancel_Rate"] = out["Cancel_Rate"].fillna(0).clip(0, 1)
    out["Orders_Delivered"] = out["Orders_Delivered"].fillna(0)
    out["Driver_Rejections"] = out["Driver_Rejections"].fillna(0)

    # Performance score (0..1)
    score = (
        (out["Delivery_Rate"] * 0.65)
        + ((1 - out["Cancel_Rate"]) * 0.30)
        + (1 - (out["Driver_Rejections"].clip(lower=0) / 10)).clip(0, 1) * 0.05
    )
    out["Performance_Score"] = score.clip(0, 1)

    return out

# =========================
# Sidebar uploads
# =========================
with st.sidebar:
    if LOGO_IMG.exists():
        st.image(str(LOGO_IMG), use_container_width=True)

    st.markdown("### Upload files")
    perf_file = st.file_uploader("1) Performance report (required)", type=["xlsx"], key="perf")
    face_file = st.file_uploader("2) Face recognition file (optional)", type=["xlsx"], key="face")
    extra_file = st.file_uploader("3) Extra file (optional)", type=["xlsx"], key="extra")

    st.divider()
    st.markdown("### Filters")
    search = st.text_input("Search (ID / Name)", "")
    min_delivery = st.slider("Min Delivery Rate", 0.0, 1.0, 0.0, 0.01)
    max_cancel = st.slider("Max Cancel Rate", 0.0, 1.0, 1.0, 0.01)

    st.divider()
    st.caption("Alert rules")
    st.caption("• Driver Rejections > 0 = red")
    st.caption("• Cancel Rate ≥ 0.20% = red")
    st.caption("• Performance Score ≠ 100% = red")

if not perf_file:
    st.info("Upload the Performance report to start.")
    st.stop()

# =========================
# Load + Build Performance
# =========================
perf_raw = read_first_sheet_excel(perf_file)
perf = build_performance_report(perf_raw)

# =========================
# Optional: merge Face Recognition file
# =========================
face_df = None
if face_file:
    face_raw = read_first_sheet_excel(face_file)

    st.sidebar.markdown("### Face file mapping")
    face_id_col = st.sidebar.selectbox("Face file: Driver ID column", options=list(face_raw.columns), key="face_id")
    face_value_col = st.sidebar.selectbox("Face file: Face Recognition column", options=list(face_raw.columns), key="face_val")

    face_df = face_raw[[face_id_col, face_value_col]].copy()
    face_df.columns = ["Driver_ID", "Face_Recognition"]
    face_df["Driver_ID"] = pd.to_numeric(face_df["Driver_ID"], errors="coerce").astype("Int64")
    face_df["Face_Recognition"] = safe_to_numeric(face_df["Face_Recognition"]).fillna(0)

# =========================
# Optional: merge Extra file (pick one metric)
# =========================
extra_df = None
if extra_file:
    extra_raw = read_first_sheet_excel(extra_file)

    st.sidebar.markdown("### Extra file mapping")
    extra_id_col = st.sidebar.selectbox("Extra file: Driver ID column", options=list(extra_raw.columns), key="extra_id")
    extra_value_col = st.sidebar.selectbox("Extra file: Metric column", options=list(extra_raw.columns), key="extra_val")

    extra_df = extra_raw[[extra_id_col, extra_value_col]].copy()
    extra_df.columns = ["Driver_ID", "Extra_Metric"]
    extra_df["Driver_ID"] = pd.to_numeric(extra_df["Driver_ID"], errors="coerce").astype("Int64")
    extra_df["Extra_Metric"] = safe_to_numeric(extra_df["Extra_Metric"]).fillna(0)

# =========================
# Merge everything into MASTER
# =========================
master = perf.copy()
if face_df is not None:
    master = master.merge(face_df, on="Driver_ID", how="left")
if extra_df is not None:
    master = master.merge(extra_df, on="Driver_ID", how="left")

if "Face_Recognition" in master.columns:
    master["Face_Recognition"] = master["Face_Recognition"].fillna(0)
if "Extra_Metric" in master.columns:
    master["Extra_Metric"] = master["Extra_Metric"].fillna(0)

# =========================
# Alert priority (THIS fixes your sorting complaint)
# =========================
master["Alert_Rejections"] = (master["Driver_Rejections"] > 0).astype(int)
master["Alert_Cancel"] = (master["Cancel_Rate"] >= CANCEL_RED_THRESHOLD).astype(int)
master["Alert_Score"] = ((master["Performance_Score"].round(6)) != 1.0).astype(int)

master["Alert_Priority"] = (
    master["Alert_Cancel"] * 3
    + master["Alert_Rejections"] * 2
    + master["Alert_Score"] * 1
)

# Sort: alerts first, then low score, then higher cancel, then more rejects
master = master.sort_values(
    ["Alert_Priority", "Performance_Score", "Cancel_Rate", "Driver_Rejections"],
    ascending=[False, True, False, False]
).reset_index(drop=True)

master["Attention_Rank"] = range(1, len(master) + 1)

# =========================
# Apply filters
# =========================
f = master.copy()
if search.strip():
    s = search.strip().lower()
    f = f[
        f["Driver_Name"].str.lower().str.contains(s, na=False)
        | f["Driver_ID"].astype(str).str.contains(s, na=False)
    ]

f = f[(f["Delivery_Rate"] >= min_delivery) & (f["Cancel_Rate"] <= max_cancel)]

# =========================
# KPIs
# =========================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Drivers (filtered)", f"{len(f):,}")
k2.metric("Avg Delivery Rate", f"{f['Delivery_Rate'].mean():.2%}" if len(f) else "—")
k3.metric("Avg Cancel Rate", f"{f['Cancel_Rate'].mean():.2%}" if len(f) else "—")
k4.metric("Avg Score", f"{f['Performance_Score'].mean():.2%}" if len(f) else "—")

st.divider()

# =========================
# Table styling
# =========================
def style_table(df):
    sty = df.style.format({
        "Performance_Score": "{:.2%}",
        "Delivery_Rate": "{:.2%}",
        "Cancel_Rate": "{:.2%}",
    })
    # Red rules
    if "Driver_Rejections" in df.columns:
        sty = sty.applymap(lambda x: "color:red;font-weight:700;" if float(x) > 0 else "", subset=["Driver_Rejections"])
    if "Cancel_Rate" in df.columns:
        sty = sty.applymap(lambda x: "color:red;font-weight:700;" if float(x) >= CANCEL_RED_THRESHOLD else "", subset=["Cancel_Rate"])
    if "Performance_Score" in df.columns:
        sty = sty.applymap(lambda x: "color:red;font-weight:700;" if round(float(x), 6) != 1.0 else "", subset=["Performance_Score"])
    return sty

# =========================
# Needs Attention
# =========================
st.subheader("🚨 Needs Attention (alerts prioritized)")
top_cols = [
    "Attention_Rank", "Driver_ID", "Driver_Name",
    "Alert_Priority", "Performance_Score",
    "Delivery_Rate", "Cancel_Rate",
    "Orders_Delivered", "Driver_Rejections",
]
if "Face_Recognition" in f.columns:
    top_cols.append("Face_Recognition")
if "Extra_Metric" in f.columns:
    top_cols.append("Extra_Metric")

attention_view = f[top_cols].head(40).copy()
st.dataframe(style_table(attention_view), use_container_width=True, hide_index=True)

# =========================
# Driver Lookup (UPDATED as you requested)
# Show: Delivery Rate, Orders Delivered, Cancellation Rate, Driver Rejections, Face Recognition
# =========================
st.divider()
st.subheader("🔎 Driver Lookup")

driver_list = f["Driver_Name"].dropna().unique().tolist()
selected_driver = st.selectbox("Select driver", ["(choose)"] + driver_list, key="driver_pick")

if selected_driver != "(choose)":
    row = f[f["Driver_Name"] == selected_driver].head(1)
    if len(row):
        row = row.iloc[0]

        # 5 KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Delivery Rate", f"{row['Delivery_Rate']:.2%}")
        c2.metric("Orders Delivered", f"{int(row['Orders_Delivered']):,}")
        c3.metric("Cancellation Rate", f"{row['Cancel_Rate']:.2%}")
        c4.metric("Driver Rejections", f"{int(row['Driver_Rejections']):,}")

        if "Face_Recognition" in f.columns:
            c5.metric("Face Recognition", f"{int(row['Face_Recognition']):,}")
        else:
            c5.metric("Face Recognition", "—")

        st.markdown("### Full driver record (all available fields)")
        st.json(row.dropna().to_dict())

# =========================
# Master table (all merged fields)
# =========================
st.divider()
st.subheader("📋 Master Table (all merged data)")

display_cols = [
    "Driver_ID", "Driver_Name",
    "Performance_Score", "Delivery_Rate", "Cancel_Rate",
    "Orders_Delivered", "Driver_Rejections"
]
if "Face_Recognition" in f.columns:
    display_cols.append("Face_Recognition")
if "Extra_Metric" in f.columns:
    display_cols.append("Extra_Metric")

display_df = f[display_cols].copy()
st.dataframe(style_table(display_df), use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download filtered master CSV",
    data=f.to_csv(index=False, encoding="utf-8-sig"),
    file_name="safeer_master_filtered.csv",
    mime="text/csv",
)

# =========================
# Charts
# =========================
st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Lowest 20 Performance Scores")
    worst = f.sort_values("Performance_Score", ascending=True).head(20)
    fig1 = px.bar(
        worst,
        x="Performance_Score",
        y="Driver_Name",
        orientation="h",
        hover_data=["Driver_ID", "Delivery_Rate", "Cancel_Rate", "Orders_Delivered", "Driver_Rejections"],
    )
    fig1.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("Cancel Rate vs Delivery Rate")
    fig2 = px.scatter(
        f,
        x="Cancel_Rate",
        y="Delivery_Rate",
        size="Orders_Delivered",
        hover_data=["Driver_Name", "Driver_ID", "Performance_Score", "Driver_Rejections"],
    )
    fig2.update_xaxes(tickformat=".0%")
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)
