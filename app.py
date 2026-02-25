import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# =========================
# إعدادات الهوية (Branding)
# =========================
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
HERO_IMG = ASSETS / "hero.jpg"
LOGO_IMG = ASSETS / "logo.png"
FAVICON_IMG = ASSETS / "favicon.png"

st.set_page_config(
    page_title="Safeer Dash | سفير",
    page_icon=str(FAVICON_IMG) if FAVICON_IMG.exists() else "🟢",
    layout="wide",
)

# =========================
# CSS (تصغير حجم أرقام الـ KPI)
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
          font-size: 19px !important;   /* أصغر عشان تظهر الأرقام كاملة */
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
# الهيرو / العنوان
# =========================
if HERO_IMG.exists():
    st.image(str(HERO_IMG), use_container_width=True)

st.markdown("# لوحة سفير - Safeer Dash")
st.markdown('<div class="safeer-subtitle">لوحة متابعة أداء السائقين • إبراز الأداء المنخفض أولاً</div>', unsafe_allow_html=True)
st.divider()

# =========================
# إعدادات الأعمدة + اكتشاف الهيدر
# =========================
AR_HEADER_MARKERS = ["معرّف السائق", "معرف السائق", "اسم السائق", "معدل التوصيل", "معدل الإلغاء"]

# دعم القالب الجديد: اسم أول + اسم أخير
COLS = {
    "driver_id": ["معرّف السائق", "معرف السائق", "Driver_ID", "driver_id", "id"],
    "driver_first": ["اسم السائق", "الاسم الأول", "First Name", "first_name"],
    "driver_last": ["اسم السائق.1", "اسم العائلة", "Last Name", "last_name"],
    "driver_name": ["اسم السائق الكامل", "Driver_Name", "driver_name", "السائق"],  # fallback لملفات قديمة

    "user_name": ["اسم اليوزر", "يوزر", "User", "Username", "user_name"],
    "completed": ["طلب مكتمل", "طلبات مكتملة", "Completed", "completed_orders"],
    "cancelled": ["طلب ملغي", "طلبات ملغاة", "Cancelled", "orders_cancelled"],
    "driver_reject": ["رفض السائق", "Driver Rejections", "driver_rejections"],
    "auto_reject": ["رفض تلقائي", "Auto Rejection", "auto_reject"],
    "cancel_rate": ["معدل الإلغاء", "Cancel_Rate", "cancel_rate"],
    "delivery_rate": ["معدل التوصيل", "Delivery_Rate", "delivery_rate"],
    "daily_deficit": ["العجزاليومي", "Daily Deficit", "daily_deficit"],
    "monthly_deficit": ["العجز الشهري", "Monthly Deficit", "monthly_deficit"],
    "invalid": ["غير صالح", "Invalid", "invalid"],
    "work_days": ["أيام العمل", "Work Days", "work_days"],
}

# 0.20% = 0.002 (لأن المعدلات 0..1)
CANCEL_RED_THRESHOLD = 0.002

def normalize(x) -> str:
    return str(x).strip()

def find_header_row(df_preview: pd.DataFrame, max_rows: int = 50) -> int | None:
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

    preview = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, nrows=60)
    header_row = find_header_row(preview)
    if header_row is None:
        header_row = 1

    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)
    df.columns = [normalize(c) for c in df.columns]
    return df

def to_report(df_raw: pd.DataFrame) -> pd.DataFrame:
    mapped = {key: pick_col(df_raw.columns, cands) for key, cands in COLS.items()}

    required = ["driver_id", "delivery_rate", "cancel_rate", "completed", "cancelled"]
    missing = [k for k in required if mapped.get(k) is None]
    if missing:
        st.error("❌ الأعمدة المطلوبة غير موجودة في الملف: " + ", ".join(missing))
        st.write("الأعمدة الموجودة في الملف:")
        st.write(list(df_raw.columns))
        st.stop()

    # دمج الاسم الأول + الاسم الأخير إذا توفروا
    first_col = mapped.get("driver_first")
    last_col = mapped.get("driver_last")
    full_col = mapped.get("driver_name")

    if first_col and last_col:
        driver_name_series = (
            df_raw[first_col].astype(str).str.strip().fillna("")
            + " "
            + df_raw[last_col].astype(str).str.strip().fillna("")
        ).str.replace(r"\s+", " ", regex=True).str.strip()
    elif full_col:
        driver_name_series = df_raw[full_col].astype(str).str.strip()
    elif first_col:
        driver_name_series = df_raw[first_col].astype(str).str.strip()
    else:
        driver_name_series = ""

    out = pd.DataFrame({
        "معرف_السائق": pd.to_numeric(df_raw[mapped["driver_id"]], errors="coerce"),
        "اسم_السائق": driver_name_series,
        "اسم_اليوزر": df_raw[mapped["user_name"]].astype(str).str.strip() if mapped.get("user_name") else "",

        "معدل_التوصيل": pd.to_numeric(df_raw[mapped["delivery_rate"]], errors="coerce"),
        "معدل_الإلغاء": pd.to_numeric(df_raw[mapped["cancel_rate"]], errors="coerce"),

        "طلبات_مكتملة": pd.to_numeric(df_raw[mapped["completed"]], errors="coerce"),
        "طلبات_ملغاة": pd.to_numeric(df_raw[mapped["cancelled"]], errors="coerce"),

        "رفض_السائق": pd.to_numeric(df_raw[mapped["driver_reject"]], errors="coerce") if mapped.get("driver_reject") else 0,
        "رفض_تلقائي": pd.to_numeric(df_raw[mapped["auto_reject"]], errors="coerce") if mapped.get("auto_reject") else 0,

        "العجز_اليومي": pd.to_numeric(df_raw[mapped["daily_deficit"]], errors="coerce") if mapped.get("daily_deficit") else 0,
        "العجز_الشهري": pd.to_numeric(df_raw[mapped["monthly_deficit"]], errors="coerce") if mapped.get("monthly_deficit") else 0,

        "غير_صالح": pd.to_numeric(df_raw[mapped["invalid"]], errors="coerce") if mapped.get("invalid") else 0,
        "أيام_العمل": pd.to_numeric(df_raw[mapped["work_days"]], errors="coerce") if mapped.get("work_days") else 0,
    })

    out = out.dropna(subset=["معرف_السائق", "اسم_السائق"], how="all")
    out["معرف_السائق"] = pd.to_numeric(out["معرف_السائق"], errors="coerce").astype("Int64")
    out["اسم_السائق"] = out["اسم_السائق"].fillna("").astype(str).str.strip()
    out["اسم_اليوزر"] = out["اسم_اليوزر"].fillna("").astype(str).str.strip()

    num_cols = [
        "معدل_التوصيل", "معدل_الإلغاء", "طلبات_مكتملة", "طلبات_ملغاة",
        "رفض_السائق", "رفض_تلقائي", "العجز_اليومي", "العجز_الشهري",
        "غير_صالح", "أيام_العمل"
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # =========================
    # حساب السكور كنسبة 0..1
    # =========================
    score_0_100 = (
        (out["معدل_التوصيل"] * 100) * 0.55
        + ((1 - out["معدل_الإلغاء"]) * 100) * 0.25
        + (100 - (out["رفض_السائق"] * 10)) * 0.10
        + (100 - (out["طلبات_ملغاة"] * 2)) * 0.10
    )
    out["درجة_الأداء"] = (score_0_100 / 100).clip(lower=0, upper=1)

    # ترتيب "يحتاج متابعة" (الأقل أداء أولاً)
    out["ترتيب_المتابعة"] = out["درجة_الأداء"].rank(ascending=True, method="dense").astype(int)
    out = out.sort_values(["درجة_الأداء", "معدل_الإلغاء", "طلبات_ملغاة"], ascending=[True, False, False])

    return out

# =========================
# التنسيق + التلوين
# =========================
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
        try:
            return "color: red; font-weight: 700;" if round(float(x), 6) != 1.0 else ""
        except Exception:
            return ""

    sty = df.style

    # عرض النسب كنسب مئوية دائماً
    fmt = {
        "درجة_الأداء": "{:.2%}",
        "معدل_التوصيل": "{:.2%}",
        "معدل_الإلغاء": "{:.2%}",
    }
    for k in list(fmt.keys()):
        if k not in df.columns:
            fmt.pop(k, None)
    if fmt:
        sty = sty.format(fmt)

    if "رفض_السائق" in df.columns:
        sty = sty.applymap(red_if_reject, subset=["رفض_السائق"])
    if "معدل_الإلغاء" in df.columns:
        sty = sty.applymap(red_if_cancel_high, subset=["معدل_الإلغاء"])
    if "درجة_الأداء" in df.columns:
        sty = sty.applymap(red_if_score_not_100, subset=["درجة_الأداء"])

    return sty

# =========================
# الشريط الجانبي (Arabic UI)
# =========================
with st.sidebar:
    if LOGO_IMG.exists():
        st.image(str(LOGO_IMG), use_container_width=True)

    st.markdown("### رفع التقرير")
    uploaded = st.file_uploader("ارفع ملف Excel الخام (.xlsx)", type=["xlsx"])

    st.divider()
    st.markdown("### فلاتر")
    search = st.text_input("بحث (المعرف / الاسم / اليوزر)", "")
    min_delivery = st.slider("أقل معدل توصيل", 0.0, 1.0, 0.0, 0.01)
    max_cancel = st.slider("أعلى معدل إلغاء", 0.0, 1.0, 1.0, 0.01)

    st.divider()
    st.caption("قواعد التنبيه")
    st.caption("• رفض السائق > 0 = أحمر")
    st.caption("• الإلغاء ≥ 0.20% = أحمر")
    st.caption("• درجة الأداء ≠ 100% = أحمر")

if not uploaded:
    st.info("ارفع ملف Excel للبدء.")
    st.stop()

raw = load_excel(uploaded)
report = to_report(raw)

# =========================
# تطبيق الفلاتر
# =========================
f = report.copy()
if search.strip():
    s = search.strip().lower()
    f = f[
        f["اسم_السائق"].str.lower().str.contains(s, na=False)
        | f["اسم_اليوزر"].str.lower().str.contains(s, na=False)
        | f["معرف_السائق"].astype(str).str.contains(s, na=False)
    ]

f = f[(f["معدل_التوصيل"] >= min_delivery) & (f["معدل_الإلغاء"] <= max_cancel)]

# =========================
# KPIs
# =========================
k1, k2, k3, k4 = st.columns(4)
k1.metric("عدد السائقين (بعد الفلترة)", f"{len(f):,}")
k2.metric("متوسط معدل التوصيل", f"{f['معدل_التوصيل'].mean():.2%}" if len(f) else "—")
k3.metric("متوسط معدل الإلغاء", f"{f['معدل_الإلغاء'].mean():.2%}" if len(f) else "—")
k4.metric("متوسط درجة الأداء", f"{f['درجة_الأداء'].mean():.2%}" if len(f) else "—")

st.divider()

# =========================
# جدول المتابعة (الأداء المنخفض أولاً)
# =========================
st.subheader("🚨 سائقون يحتاجون متابعة (الأقل أداء أولاً)")
attention_view = f[[
    "ترتيب_المتابعة", "معرف_السائق", "اسم_السائق", "اسم_اليوزر",
    "درجة_الأداء", "معدل_التوصيل", "معدل_الإلغاء",
    "طلبات_مكتملة", "طلبات_ملغاة", "رفض_السائق"
]].head(30)

st.dataframe(style_attention(attention_view), use_container_width=True, hide_index=True)

# =========================
# صفحة السائق (Dropdown) — طلبات مكتملة + رفض السائق بدل الإلغاء
# =========================
st.divider()
st.subheader("🔎 بحث سريع عن سائق")

driver_list = f["اسم_السائق"].dropna().unique().tolist()
selected = st.selectbox("اختر السائق", ["(اختر)"] + driver_list)

if selected != "(اختر)":
    d = f[f["اسم_السائق"] == selected].head(1)
    if len(d):
        d = d.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("معرف السائق", str(d["معرف_السائق"]))
        c2.metric("معدل التوصيل", f"{d['معدل_التوصيل']:.2%}")
        c3.metric("طلبات مكتملة", f"{int(d['طلبات_مكتملة']):,}")
        c4.metric("رفض السائق", f"{int(d['رفض_السائق']):,}")
        c5.metric("درجة الأداء", f"{d['درجة_الأداء']:.2%}")

# =========================
# الرسوم البيانية
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
        hover_data=["معرف_السائق", "معدل_التوصيل", "معدل_الإلغاء", "طلبات_ملغاة", "رفض_السائق"],
    )
    fig1.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("معدل الإلغاء مقابل معدل التوصيل")
    fig2 = px.scatter(
        f,
        x="معدل_الإلغاء",
        y="معدل_التوصيل",
        size="طلبات_ملغاة",
        hover_data=["اسم_السائق", "معرف_السائق", "درجة_الأداء", "رفض_السائق"],
    )
    fig2.update_xaxes(tickformat=".0%")
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# الجدول الكامل + تصدير
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
