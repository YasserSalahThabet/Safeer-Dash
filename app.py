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
    page_title="لوحة سفير - Safeer Dash",
    page_icon=str(FAVICON_IMG) if FAVICON_IMG.exists() else "🟢",
    layout="wide",
)

# =========================
# CSS (smaller KPI text)
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

st.markdown("# لوحة سفير - Safeer Dash")
st.markdown('<div class="safeer-subtitle">دمج 2–3 ملفات Excel • إبراز التنبيهات أولاً</div>', unsafe_allow_html=True)
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

# 0.20% = 0.002 (assuming rates are 0..1)
CANCEL_RED_THRESHOLD = 0.002

# =========================
# Performance mapping (your template)
# =========================
PERF_COLS = {
    "driver_id": ["معرّف السائق", "معرف السائق", "Driver_ID", "driver_id", "id"],
    "first_name": ["اسم السائق", "First Name", "first_name"],
    "last_name": ["اسم السائق.1", "Last Name", "last_name"],

    # Requested
    "cancel_rate": ["معدل الإلغاء بسبب مشاكل التوصيل", "معدل الإلغاء", "Cancel_Rate", "cancel_rate"],
    "driver_reject": ["المهام المرفوضة (السائق)", "رفض السائق", "Driver Rejections", "driver_rejections"],
    "orders_delivered": ["المهام التي تم تسليمها", "الطلبات المسلمة", "طلبات مكتملة", "طلب مكتمل",
                         "Orders_Delivered", "orders_delivered"],

    # Delivery metric in your report
    "delivery_rate": ["معدل اكتمال الطلبات (غير متعلق بالتوصيل)", "معدل اكتمال الطلبات",
                      "معدل التوصيل", "Delivery_Rate", "delivery_rate"],
}

def build_performance_report(df_raw: pd.DataFrame) -> pd.DataFrame:
    mapped = {k: pick(df_raw.columns, v) for k, v in PERF_COLS.items()}

    required = ["driver_id", "first_name", "last_name", "delivery_rate", "cancel_rate", "orders_delivered", "driver_reject"]
    missing = [k for k in required if not mapped.get(k)]
    if missing:
        st.error("❌ الأعمدة المطلوبة غير موجودة في ملف الأداء: " + ", ".join(missing))
        st.write("الأعمدة الموجودة في الملف:")
        st.write(list(df_raw.columns))
        st.stop()

    driver_name = (
        df_raw[mapped["first_name"]].astype(str).str.strip().fillna("")
        + " "
        + df_raw[mapped["last_name"]].astype(str).str.strip().fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    out = pd.DataFrame({
        "معرف_السائق": safe_to_numeric(df_raw[mapped["driver_id"]]),
        "اسم_السائق": driver_name,

        "معدل_التوصيل": safe_to_numeric(df_raw[mapped["delivery_rate"]]),
        "معدل_الإلغاء_مشاكل_التوصيل": safe_to_numeric(df_raw[mapped["cancel_rate"]]),
        "المهام_التي_تم_تسليمها": safe_to_numeric(df_raw[mapped["orders_delivered"]]),
        "المهام_المرفوضة_السائق": safe_to_numeric(df_raw[mapped["driver_reject"]]),
    })

    out["معرف_السائق"] = pd.to_numeric(out["معرف_السائق"], errors="coerce").astype("Int64")

    out["معدل_التوصيل"] = out["معدل_التوصيل"].fillna(0).clip(0, 1)
    out["معدل_الإلغاء_مشاكل_التوصيل"] = out["معدل_الإلغاء_مشاكل_التوصيل"].fillna(0).clip(0, 1)

    out["المهام_التي_تم_تسليمها"] = out["المهام_التي_تم_تسليمها"].fillna(0)
    out["المهام_المرفوضة_السائق"] = out["المهام_المرفوضة_السائق"].fillna(0)

    # Score 0..1
    score = (
        (out["معدل_التوصيل"] * 0.65)
        + ((1 - out["معدل_الإلغاء_مشاكل_التوصيل"]) * 0.30)
        + (1 - (out["المهام_المرفوضة_السائق"].clip(lower=0) / 10)).clip(0, 1) * 0.05
    )
    out["درجة_الأداء"] = score.clip(0, 1)

    return out

# =========================
# Sidebar: uploads + filters
# =========================
with st.sidebar:
    if LOGO_IMG.exists():
        st.image(str(LOGO_IMG), use_container_width=True)

    st.markdown("### رفع الملفات")
    perf_file = st.file_uploader("1) ملف الأداء (إلزامي)", type=["xlsx"], key="perf")
    face_file = st.file_uploader("2) ملف Face Recognition (اختياري)", type=["xlsx"], key="face")
    extra_file = st.file_uploader("3) ملف إضافي (اختياري)", type=["xlsx"], key="extra")

    st.divider()
    st.markdown("### فلاتر")
    search = st.text_input("بحث (المعرف / الاسم)", "")
    min_delivery = st.slider("أقل معدل توصيل", 0.0, 1.0, 0.0, 0.01)
    max_cancel = st.slider("أعلى معدل إلغاء", 0.0, 1.0, 1.0, 0.01)

    st.divider()
    st.caption("قواعد التنبيه")
    st.caption("• المهام المرفوضة (السائق) > 0 = أحمر")
    st.caption("• الإلغاء بسبب مشاكل التوصيل ≥ 0.20% = أحمر")
    st.caption("• درجة الأداء ≠ 100% = أحمر")

if not perf_file:
    st.info("ارفع ملف الأداء للبدء.")
    st.stop()

# =========================
# Load performance
# =========================
perf_raw = read_first_sheet_excel(perf_file)
perf = build_performance_report(perf_raw)

# =========================
# Optional: Face Recognition merge
# =========================
face_df = None
if face_file:
    face_raw = read_first_sheet_excel(face_file)

    st.sidebar.markdown("### إعدادات دمج Face Recognition")
    face_id_col = st.sidebar.selectbox("عمود معرف السائق (Face file)", options=list(face_raw.columns), key="face_id")
    face_value_col = st.sidebar.selectbox("عمود Face Recognition", options=list(face_raw.columns), key="face_val")

    face_df = face_raw[[face_id_col, face_value_col]].copy()
    face_df.columns = ["معرف_السائق", "Face_Recognition"]
    face_df["معرف_السائق"] = pd.to_numeric(face_df["معرف_السائق"], errors="coerce").astype("Int64")
    face_df["Face_Recognition"] = safe_to_numeric(face_df["Face_Recognition"]).fillna(0)

# =========================
# Optional: Extra merge
# =========================
extra_df = None
if extra_file:
    extra_raw = read_first_sheet_excel(extra_file)

    st.sidebar.markdown("### إعدادات دمج الملف الإضافي")
    extra_id_col = st.sidebar.selectbox("عمود معرف السائق (Extra file)", options=list(extra_raw.columns), key="extra_id")
    extra_value_col = st.sidebar.selectbox("العمود/المؤشر المراد إضافته", options=list(extra_raw.columns), key="extra_val")

    extra_df = extra_raw[[extra_id_col, extra_value_col]].copy()
    extra_df.columns = ["معرف_السائق", "مؤشر_إضافي"]
    extra_df["معرف_السائق"] = pd.to_numeric(extra_df["معرف_السائق"], errors="coerce").astype("Int64")
    extra_df["مؤشر_إضافي"] = safe_to_numeric(extra_df["مؤشر_إضافي"]).fillna(0)

# =========================
# MASTER merge
# =========================
master = perf.copy()
if face_df is not None:
    master = master.merge(face_df, on="معرف_السائق", how="left")
if extra_df is not None:
    master = master.merge(extra_df, on="معرف_السائق", how="left")

if "Face_Recognition" in master.columns:
    master["Face_Recognition"] = master["Face_Recognition"].fillna(0)
if "مؤشر_إضافي" in master.columns:
    master["مؤشر_إضافي"] = master["مؤشر_إضافي"].fillna(0)

# =========================
# Alert priority (fixes your precedence issue)
# =========================
master["تنبيه_رفض"] = (master["المهام_المرفوضة_السائق"] > 0).astype(int)
master["تنبيه_إلغاء"] = (master["معدل_الإلغاء_مشاكل_التوصيل"] >= CANCEL_RED_THRESHOLD).astype(int)
master["تنبيه_درجة"] = ((master["درجة_الأداء"].round(6)) != 1.0).astype(int)

master["أولوية_التنبيه"] = (
    master["تنبيه_إلغاء"] * 3
    + master["تنبيه_رفض"] * 2
    + master["تنبيه_درجة"] * 1
)

master = master.sort_values(
    ["أولوية_التنبيه", "درجة_الأداء", "معدل_الإلغاء_مشاكل_التوصيل", "المهام_المرفوضة_السائق"],
    ascending=[False, True, False, False]
).reset_index(drop=True)

master["ترتيب_المتابعة"] = range(1, len(master) + 1)

# =========================
# Filters
# =========================
f = master.copy()
if search.strip():
    s = search.strip().lower()
    f = f[
        f["اسم_السائق"].str.lower().str.contains(s, na=False)
        | f["معرف_السائق"].astype(str).str.contains(s, na=False)
    ]

f = f[(f["معدل_التوصيل"] >= min_delivery) & (f["معدل_الإلغاء_مشاكل_التوصيل"] <= max_cancel)]

# =========================
# Styling
# =========================
def style_table(df):
    sty = df.style.format({
        "درجة_الأداء": "{:.2%}",
        "معدل_التوصيل": "{:.2%}",
        "معدل_الإلغاء_مشاكل_التوصيل": "{:.2%}",
    })
    # red rules
    sty = sty.applymap(
        lambda x: "color:red;font-weight:700;" if float(x) > 0 else "",
        subset=["المهام_المرفوضة_السائق"]
    )
    sty = sty.applymap(
        lambda x: "color:red;font-weight:700;" if float(x) >= CANCEL_RED_THRESHOLD else "",
        subset=["معدل_الإلغاء_مشاكل_التوصيل"]
    )
    sty = sty.applymap(
        lambda x: "color:red;font-weight:700;" if round(float(x), 6) != 1.0 else "",
        subset=["درجة_الأداء"]
    )
    return sty

# =========================
# KPIs
# =========================
k1, k2, k3, k4 = st.columns(4)
k1.metric("عدد السائقين (بعد الفلترة)", f"{len(f):,}")
k2.metric("متوسط معدل التوصيل", f"{f['معدل_التوصيل'].mean():.2%}" if len(f) else "—")
k3.metric("متوسط معدل الإلغاء (مشاكل التوصيل)", f"{f['معدل_الإلغاء_مشاكل_التوصيل'].mean():.2%}" if len(f) else "—")
k4.metric("متوسط درجة الأداء", f"{f['درجة_الأداء'].mean():.2%}" if len(f) else "—")

st.divider()

# =========================
# Needs Attention
# =========================
st.subheader("🚨 سائقون يحتاجون متابعة (الأولوية أولاً)")
top_cols = [
    "ترتيب_المتابعة",
    "معرف_السائق",
    "اسم_السائق",
    "أولوية_التنبيه",
    "درجة_الأداء",
    "معدل_التوصيل",
    "معدل_الإلغاء_مشاكل_التوصيل",
    "المهام_التي_تم_تسليمها",
    "المهام_المرفوضة_السائق",
]
if "Face_Recognition" in f.columns:
    top_cols.append("Face_Recognition")
if "مؤشر_إضافي" in f.columns:
    top_cols.append("مؤشر_إضافي")

st.dataframe(style_table(f[top_cols].head(40)), use_container_width=True, hide_index=True)

# =========================
# Driver Lookup (FIXED: requested fields + no weird block)
# =========================
st.divider()
st.subheader("🔎 بحث سريع عن سائق (Driver Lookup)")

driver_list = f["اسم_السائق"].dropna().unique().tolist()
selected = st.selectbox("اختر السائق", ["(اختر)"] + driver_list)

if selected != "(اختر)":
    d = f[f["اسم_السائق"] == selected].head(1)
    if len(d):
        d = d.iloc[0]

        # Requested fields:
        # Delivery rate, Orders delivered, Cancellation rate, Driver rejections, Face recognition
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("معدل التوصيل", f"{d['معدل_التوصيل']:.2%}")
        c2.metric("المهام التي تم تسليمها", f"{int(d['المهام_التي_تم_تسليمها']):,}")
        c3.metric("معدل الإلغاء بسبب مشاكل التوصيل", f"{d['معدل_الإلغاء_مشاكل_التوصيل']:.2%}")
        c4.metric("المهام المرفوضة (السائق)", f"{int(d['المهام_المرفوضة_السائق']):,}")

        if "Face_Recognition" in f.columns:
            c5.metric("Face Recognition", f"{int(d['Face_Recognition']):,}")
        else:
            c5.metric("Face Recognition", "—")

        # Instead of showing big JSON block (the "weird column"),
        # we put full details inside an expander.
        with st.expander("عرض جميع بيانات السائق"):
            st.dataframe(pd.DataFrame(d).T, use_container_width=True)

# =========================
# Master Table (all merged fields)
# =========================
st.divider()
st.subheader("📋 الجدول النهائي (كل البيانات بعد الدمج)")

display_cols = [
    "معرف_السائق", "اسم_السائق",
    "درجة_الأداء", "معدل_التوصيل", "معدل_الإلغاء_مشاكل_التوصيل",
    "المهام_التي_تم_تسليمها", "المهام_المرفوضة_السائق",
]
if "Face_Recognition" in f.columns:
    display_cols.append("Face_Recognition")
if "مؤشر_إضافي" in f.columns:
    display_cols.append("مؤشر_إضافي")

st.dataframe(style_table(f[display_cols]), use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ تحميل النتائج CSV",
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
    st.subheader("أقل 20 سائق حسب درجة الأداء")
    worst = f.sort_values("درجة_الأداء", ascending=True).head(20)
    fig1 = px.bar(
        worst,
        x="درجة_الأداء",
        y="اسم_السائق",
        orientation="h",
        hover_data=[
            "معرف_السائق",
            "معدل_التوصيل",
            "معدل_الإلغاء_مشاكل_التوصيل",
            "المهام_التي_تم_تسليمها",
            "المهام_المرفوضة_السائق",
        ],
    )
    fig1.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("معدل الإلغاء مقابل معدل التوصيل")
    fig2 = px.scatter(
        f,
        x="معدل_الإلغاء_مشاكل_التوصيل",
        y="معدل_التوصيل",
        size="المهام_التي_تم_تسليمها",
        hover_data=["اسم_السائق", "معرف_السائق", "درجة_الأداء", "المهام_المرفوضة_السائق"],
    )
    fig2.update_xaxes(tickformat=".0%")
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)
