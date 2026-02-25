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
# CSS (تصغير KPI)
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

# =========================
# Hero/Header
# =========================
if HERO_IMG.exists():
    st.image(str(HERO_IMG), use_container_width=True)

st.markdown("# لوحة سفير - Safeer Dash")
st.markdown('<div class="safeer-subtitle">لوحة متابعة أداء السائقين • إبراز الأداء المنخفض أولاً</div>', unsafe_allow_html=True)
st.divider()

# =========================
# Column mapping (supports NEW template + OLD template)
# =========================
COLS = {
    "driver_id": ["معرّف السائق", "معرف السائق", "Driver_ID", "driver_id", "id"],

    # NEW template: first+last
    "driver_first": ["اسم السائق", "الاسم الأول", "First Name", "first_name"],
    "driver_last": ["اسم السائق.1", "اسم العائلة", "Last Name", "last_name"],

    # OLD template fallback: full name
    "driver_name": ["اسم السائق الكامل", "Driver_Name", "driver_name", "السائق"],

    # Delivery rate in NEW template (completion rate)
    "delivery_rate": [
        "معدل اكتمال الطلبات (غير متعلق بالتوصيل)",
        "معدل اكتمال الطلبات",
        "معدل التوصيل",
        "Delivery_Rate",
        "delivery_rate",
    ],

    # Cancellation rate in NEW template
    "cancel_rate": [
        "معدل الإلغاء بسبب مشاكل التوصيل",
        "معدل الإلغاء",
        "Cancel_Rate",
        "cancel_rate",
    ],

    # Orders delivered in NEW template
    "orders_delivered": [
        "المهام التي تم تسليمها",
        "الطلبات المسلمة",
        "طلبات مكتملة",
        "طلب مكتمل",
        "Orders_Completed",
        "orders_completed",
        "completed",
    ],

    # Driver rejections (optional)
    "driver_reject": [
        "المهام المرفوضة (السائق)",
        "رفض السائق",
        "Driver Rejections",
        "driver_rejections",
    ],

    # Face Recognition (optional — if not in file, we'll show —)
    "face_recognition": [
        "التعرف على الوجه",
        "Face Recognition",
        "Face_Recognition",
        "face_recognition",
    ],
}

CANCEL_RED_THRESHOLD = 0.002  # 0.20%

def normalize(x) -> str:
    return str(x).strip()

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
    sheet = xls.sheet_names[0]
    df = pd.read_excel(uploaded_file, sheet_name=sheet, header=0)
    df.columns = [normalize(c) for c in df.columns]
    return df

def to_report(df_raw: pd.DataFrame) -> pd.DataFrame:
    mapped = {k: pick_col(df_raw.columns, v) for k, v in COLS.items()}

    # Required for THIS dashboard
    required = ["driver_id", "delivery_rate", "cancel_rate", "orders_delivered"]
    missing = [k for k in required if not mapped.get(k)]
    if missing:
        st.error("❌ الأعمدة المطلوبة غير موجودة في الملف: " + ", ".join(missing))
        st.write("الأعمدة الموجودة في الملف:")
        st.write(list(df_raw.columns))
        st.stop()

    # Build driver name: first+last if available
    first_col = mapped.get("driver_first")
    last_col = mapped.get("driver_last")
    full_col = mapped.get("driver_name")

    if first_col and last_col:
        driver_name = (
            df_raw[first_col].astype(str).str.strip().fillna("")
            + " "
            + df_raw[last_col].astype(str).str.strip().fillna("")
        ).str.replace(r"\s+", " ", regex=True).str.strip()
    elif full_col:
        driver_name = df_raw[full_col].astype(str).str.strip()
    elif first_col:
        driver_name = df_raw[first_col].astype(str).str.strip()
    else:
        driver_name = ""

    out = pd.DataFrame({
        "معرف_السائق": pd.to_numeric(df_raw[mapped["driver_id"]], errors="coerce"),
        "اسم_السائق": driver_name,

        "معدل_التوصيل": pd.to_numeric(df_raw[mapped["delivery_rate"]], errors="coerce"),
        "معدل_الإلغاء": pd.to_numeric(df_raw[mapped["cancel_rate"]], errors="coerce"),
        "الطلبات_المسلمة": pd.to_numeric(df_raw[mapped["orders_delivered"]], errors="coerce"),

        "رفض_السائق": pd.to_numeric(df_raw[mapped["driver_reject"]], errors="coerce") if mapped.get("driver_reject") else 0,
        "التعرف_على_الوجه": pd.to_numeric(df_raw[mapped["face_recognition"]], errors="coerce") if mapped.get("face_recognition") else pd.NA,
    })

    out = out.dropna(subset=["معرف_السائق", "اسم_السائق"], how="all")
    out["معرف_السائق"] = pd.to_numeric(out["معرف_السائق"], errors="coerce").astype("Int64")
    out["اسم_السائق"] = out["اسم_السائق"].fillna("").astype(str).str.strip()

    for c in ["معدل_التوصيل", "معدل_الإلغاء", "الطلبات_المسلمة", "رفض_السائق"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # =========================
    # Score (0..1) - tuned for your new template
    # =========================
    # Delivery rate (completion) is main
    # Cancel rate penalizes
    # Driver rejects penalizes lightly
    score = (
        (out["معدل_التوصيل"].clip(0, 1) * 0.65) +
        ((1 - out["معدل_الإلغاء"].clip(0, 1)) * 0.30) +
        (1 - (out["رفض_السائق"].clip(lower=0) / 10)).clip(0, 1) * 0.05
    )
    out["درجة_الأداء"] = score.clip(0, 1)

    # Attention ranking: low score first, then higher cancel, then rejects
    out["ترتيب_المتابعة"] = out["درجة_الأداء"].rank(ascending=True, method="dense").astype(int)
    out = out.sort_values(["درجة_الأداء", "معدل_الإلغاء", "رفض_السائق"], ascending=[True, False, False])

    return out

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

    sty = df.style.format({
        "درجة_الأداء": "{:.2%}",
        "معدل_التوصيل": "{:.2%}",
        "معدل_الإلغاء": "{:.2%}",
    })

    if "رفض_السائق" in df.columns:
        sty = sty.applymap(red_if_reject, subset=["رفض_السائق"])
    if "معدل_الإلغاء" in df.columns:
        sty = sty.applymap(red_if_cancel_high, subset=["معدل_الإلغاء"])

    return sty

# =========================
# Sidebar (Arabic UI)
# =========================
with st.sidebar:
    if LOGO_IMG.exists():
        st.image(str(LOGO_IMG), use_container_width=True)

    st.markdown("### رفع التقرير")
    uploaded = st.file_uploader("ارفع ملف Excel (.xlsx)", type=["xlsx"])

    st.divider()
    st.markdown("### فلاتر")
    search = st.text_input("بحث (المعرف / الاسم)", "")
    min_delivery = st.slider("أقل معدل توصيل", 0.0, 1.0, 0.0, 0.01)
    max_cancel = st.slider("أعلى معدل إلغاء", 0.0, 1.0, 1.0, 0.01)

    st.divider()
    st.caption("قواعد التنبيه")
    st.caption("• رفض السائق > 0 = أحمر")
    st.caption("• الإلغاء ≥ 0.20% = أحمر")

if not uploaded:
    st.info("ارفع ملف Excel للبدء.")
    st.stop()

raw = load_excel(uploaded)
report = to_report(raw)

# Apply filters
f = report.copy()
if search.strip():
    s = search.strip().lower()
    f = f[
        f["اسم_السائق"].str.lower().str.contains(s, na=False)
        | f["معرف_السائق"].astype(str).str.contains(s, na=False)
    ]

f = f[(f["معدل_التوصيل"] >= min_delivery) & (f["معدل_الإلغاء"] <= max_cancel)]

# =========================
# KPIs
# =========================
k1, k2, k3, k4 = st.columns(4)
k1.metric("عدد السائقين", f"{len(f):,}")
k2.metric("متوسط معدل التوصيل", f"{f['معدل_التوصيل'].mean():.2%}" if len(f) else "—")
k3.metric("متوسط معدل الإلغاء", f"{f['معدل_الإلغاء'].mean():.2%}" if len(f) else "—")
k4.metric("متوسط درجة الأداء", f"{f['درجة_الأداء'].mean():.2%}" if len(f) else "—")

st.divider()

# =========================
# Needs attention
# =========================
st.subheader("🚨 سائقون يحتاجون متابعة (الأداء الأقل أولاً)")
attention_view = f[[
    "ترتيب_المتابعة",
    "معرف_السائق",
    "اسم_السائق",
    "درجة_الأداء",
    "معدل_التوصيل",
    "معدل_الإلغاء",
    "الطلبات_المسلمة",
    "رفض_السائق",
]].head(30)

st.dataframe(style_attention(attention_view), use_container_width=True, hide_index=True)

# =========================
# Driver Lookup (as requested)
# delivery rate, orders delivered, cancellation rate, face recognition
# =========================
st.divider()
st.subheader("🔎 بحث سريع عن سائق")

driver_list = f["اسم_السائق"].dropna().unique().tolist()
selected = st.selectbox("اختر السائق", ["(اختر)"] + driver_list)

if selected != "(اختر)":
    d = f[f["اسم_السائق"] == selected].head(1)
    if len(d):
        d = d.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("معدل التوصيل", f"{d['معدل_التوصيل']:.2%}")
        c2.metric("الطلبات المُسلّمة", f"{int(d['الطلبات_المسلمة']):,}")
        c3.metric("معدل الإلغاء", f"{d['معدل_الإلغاء']:.2%}")

        # Face recognition may not exist in file; show — if missing
        fr = d.get("التعرف_على_الوجه", pd.NA)
        if pd.isna(fr):
            c4.metric("التعرف على الوجه", "—")
        else:
            c4.metric("التعرف على الوجه", f"{int(fr):,}")

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
        hover_data=["معرف_السائق", "معدل_التوصيل", "معدل_الإلغاء", "الطلبات_المسلمة", "رفض_السائق"],
    )
    fig1.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("معدل الإلغاء مقابل معدل التوصيل")
    fig2 = px.scatter(
        f,
        x="معدل_الإلغاء",
        y="معدل_التوصيل",
        size="الطلبات_المسلمة",
        hover_data=["اسم_السائق", "معرف_السائق", "درجة_الأداء", "رفض_السائق"],
    )
    fig2.update_xaxes(tickformat=".0%")
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# Full table + export
# =========================
st.divider()
st.subheader("كل السائقين (بعد الفلترة)")
st.dataframe(style_attention(f), use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ تحميل النتائج CSV",
    data=f.to_csv(index=False, encoding="utf-8-sig"),
    file_name="safeer_dash_filtered.csv",
    mime="text/csv",
)
