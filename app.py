import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from io import BytesIO

# =========================
# Branding assets
# =========================
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"

# Put your two images here (whatever you currently use)
# If your filenames are different, just rename them here.
LEFT_IMG = ASSETS / "left.jpg"     # <-- your current left image file
RIGHT_IMG = ASSETS / "right.jpg"   # <-- your current right image file

LOGO_IMG = ASSETS / "logo.png"
FAVICON_IMG = ASSETS / "favicon.png"

st.set_page_config(
    page_title="لوحة سفير - Safeer Dash",
    page_icon=str(FAVICON_IMG) if FAVICON_IMG.exists() else "🟢",
    layout="wide",
)

# =========================
# CSS
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
# Header images (SWAPPED)
# =========================
# "Swap": show RIGHT image on the left column, and LEFT image on the right column
img_col1, img_col2 = st.columns([1, 1])

with img_col1:
    if RIGHT_IMG.exists():
        st.image(str(RIGHT_IMG), use_container_width=True)

with img_col2:
    if LEFT_IMG.exists():
        st.image(str(LEFT_IMG), use_container_width=True)

st.markdown("# لوحة سفير - Safeer Dash")
st.markdown('<div class="safeer-subtitle">صلاحيات متعددة • رفع متعدد الملفات • إبراز التنبيهات أولاً</div>', unsafe_allow_html=True)
st.divider()

# =========================
# Auth / Roles
# =========================
ROLES = {
    "التشغيل": "ops_password",
    "المارد البشرية": "hr_password",
    "الإشراف": "sup_password",
}

def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets["auth"][key]
    except Exception:
        return default

def require_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.role = None

    with st.sidebar:
        if LOGO_IMG.exists():
            st.image(str(LOGO_IMG), use_container_width=True)

        st.markdown("## تسجيل الدخول")
        role = st.selectbox("الدور", list(ROLES.keys()))
        pwd = st.text_input("كلمة المرور", type="password")
        c1, c2 = st.columns(2)
        login = c1.button("دخول")
        logout = c2.button("خروج")

        if logout:
            st.session_state.logged_in = False
            st.session_state.role = None
            st.rerun()

        if login:
            secret_key = ROLES[role]
            expected = get_secret(secret_key, "")
            if expected and pwd == expected:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.rerun()
            else:
                st.error("بيانات الدخول غير صحيحة أو لم يتم إعداد Secrets.")

    if not st.session_state.logged_in:
        st.info("الرجاء تسجيل الدخول من الشريط الجانبي.")
        st.stop()

require_login()
ROLE = st.session_state.role

# =========================
# Helpers
# =========================
def normalize_col(c: str) -> str:
    return str(c).strip()

@st.cache_data(show_spinner=False)
def read_first_sheet_excel_bytes(file_bytes: bytes) -> pd.DataFrame:
    bio = BytesIO(file_bytes)
    xls = pd.ExcelFile(bio)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(bio, sheet_name=sheet, header=0)
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

PERF_COLS = {
    "driver_id": ["معرّف السائق", "معرف السائق", "Driver_ID", "driver_id", "id"],
    "first_name": ["اسم السائق", "First Name", "first_name"],
    "last_name": ["اسم السائق.1", "Last Name", "last_name"],
    "cancel_rate": ["معدل الإلغاء بسبب مشاكل التوصيل", "معدل الإلغاء", "Cancel_Rate", "cancel_rate"],
    "driver_reject": ["المهام المرفوضة (السائق)", "رفض السائق", "Driver Rejections", "driver_rejections"],
    "orders_delivered": ["المهام التي تم تسليمها", "الطلبات المسلمة", "طلبات مكتملة", "طلب مكتمل",
                         "Orders_Delivered", "orders_delivered"],
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

    score = (
        (out["معدل_التوصيل"] * 0.65)
        + ((1 - out["معدل_الإلغاء_مشاكل_التوصيل"]) * 0.30)
        + (1 - (out["المهام_المرفوضة_السائق"].clip(lower=0) / 10)).clip(0, 1) * 0.05
    )
    out["درجة_الأداء"] = score.clip(0, 1)
    return out

def detect_file_type(cols: set[str]) -> str:
    perf_signals = {
        "معرّف السائق", "معرف السائق", "اسم السائق", "اسم السائق.1",
        "معدل الإلغاء بسبب مشاكل التوصيل",
        "المهام التي تم تسليمها",
        "المهام المرفوضة (السائق)",
    }
    face_signals = {"Face Recognition", "التعرف على الوجه", "Face_Recognition", "face_recognition"}
    perf_hits = len(perf_signals.intersection(cols))
    face_hits = len(face_signals.intersection(cols))
    if perf_hits >= 4:
        return "performance"
    if face_hits >= 1:
        return "face"
    return "unknown"

# =========================
# Sidebar: single multi-file uploader + filters
# =========================
with st.sidebar:
    st.markdown(f"### المستخدم: {ROLE}")

    st.markdown("### رفع الملفات")
    uploaded_files = st.file_uploader(
        "ارفع 2–3 ملفات Excel معًا",
        type=["xlsx"],
        accept_multiple_files=True
    )

    st.divider()
    st.markdown("### فلاتر")
    search = st.text_input("بحث (المعرف / الاسم)", "")
    min_delivery = st.slider("أقل معدل توصيل", 0.0, 1.0, 0.0, 0.01)
    max_cancel = st.slider("أعلى معدل إلغاء", 0.0, 1.0, 1.0, 0.01)

if not uploaded_files:
    st.info("ارفع ملفات Excel للبدء (على الأقل ملف الأداء).")
    st.stop()

# Read all files
file_items = []
for uf in uploaded_files:
    b = uf.getvalue()
    df = read_first_sheet_excel_bytes(b)
    cols = set(df.columns)
    kind = detect_file_type(cols)
    file_items.append({"name": uf.name, "df": df, "cols": cols, "kind_guess": kind})

perf_candidates = [x for x in file_items if x["kind_guess"] == "performance"]
face_candidates = [x for x in file_items if x["kind_guess"] == "face"]

def file_picker(label, options, key):
    names = [o["name"] for o in options]
    chosen = st.sidebar.selectbox(label, names, key=key)
    return next(o for o in options if o["name"] == chosen)

# Pick performance (required)
if len(perf_candidates) == 1:
    perf_item = perf_candidates[0]
elif len(perf_candidates) > 1:
    st.sidebar.warning("تم اكتشاف أكثر من ملف أداء. اختر الملف الصحيح:")
    perf_item = file_picker("اختر ملف الأداء", perf_candidates, "pick_perf")
else:
    st.sidebar.warning("لم يتم اكتشاف ملف الأداء تلقائيًا. اختر ملف الأداء يدويًا:")
    perf_item = file_picker("اختر ملف الأداء", file_items, "pick_perf_manual")

# Pick face (optional)
face_item = None
remaining = [x for x in file_items if x["name"] != perf_item["name"]]
if len(face_candidates) == 1:
    face_item = face_candidates[0]
elif len(face_candidates) > 1:
    st.sidebar.warning("تم اكتشاف أكثر من ملف Face Recognition. اختر الملف الصحيح:")
    face_item = file_picker("اختر ملف Face Recognition", face_candidates, "pick_face")
else:
    if remaining:
        face_choice = st.sidebar.selectbox("ملف Face Recognition (اختياري)", ["(بدون)"] + [x["name"] for x in remaining], key="face_optional")
        if face_choice != "(بدون)":
            face_item = next(x for x in remaining if x["name"] == face_choice)

# Pick extra (optional)
extra_item = None
remaining2 = [x for x in remaining if (face_item is None or x["name"] != face_item["name"])]
if remaining2:
    extra_choice = st.sidebar.selectbox("ملف إضافي (اختياري)", ["(بدون)"] + [x["name"] for x in remaining2], key="extra_optional")
    if extra_choice != "(بدون)":
        extra_item = next(x for x in remaining2 if x["name"] == extra_choice)

# Build performance
perf = build_performance_report(perf_item["df"])

# Face merge
face_df = None
if face_item:
    face_raw = face_item["df"]
    st.sidebar.markdown("### إعدادات دمج Face Recognition")
    face_id_col = st.sidebar.selectbox("عمود معرف السائق (Face)", options=list(face_raw.columns), key="face_id")
    face_value_col = st.sidebar.selectbox("عمود Face Recognition", options=list(face_raw.columns), key="face_val")

    face_df = face_raw[[face_id_col, face_value_col]].copy()
    face_df.columns = ["معرف_السائق", "Face_Recognition"]
    face_df["معرف_السائق"] = pd.to_numeric(face_df["معرف_السائق"], errors="coerce").astype("Int64")
    face_df["Face_Recognition"] = safe_to_numeric(face_df["Face_Recognition"]).fillna(0)

# Extra merge
extra_df = None
if extra_item:
    extra_raw = extra_item["df"]
    st.sidebar.markdown("### إعدادات دمج الملف الإضافي")
    extra_id_col = st.sidebar.selectbox("عمود معرف السائق (Extra)", options=list(extra_raw.columns), key="extra_id")
    extra_value_col = st.sidebar.selectbox("العمود/المؤشر المراد إضافته", options=list(extra_raw.columns), key="extra_val")

    extra_df = extra_raw[[extra_id_col, extra_value_col]].copy()
    extra_df.columns = ["معرف_السائق", "مؤشر_إضافي"]
    extra_df["معرف_السائق"] = pd.to_numeric(extra_df["معرف_السائق"], errors="coerce").astype("Int64")
    extra_df["مؤشر_إضافي"] = safe_to_numeric(extra_df["مؤشر_إضافي"]).fillna(0)

# MASTER
master = perf.copy()
if face_df is not None:
    master = master.merge(face_df, on="معرف_السائق", how="left")
if extra_df is not None:
    master = master.merge(extra_df, on="معرف_السائق", how="left")

if "Face_Recognition" in master.columns:
    master["Face_Recognition"] = master["Face_Recognition"].fillna(0)
if "مؤشر_إضافي" in master.columns:
    master["مؤشر_إضافي"] = master["مؤشر_إضافي"].fillna(0)

# Alerts + priority precedence
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

# Filters
f = master.copy()
if search.strip():
    s = search.strip().lower()
    f = f[
        f["اسم_السائق"].str.lower().str.contains(s, na=False)
        | f["معرف_السائق"].astype(str).str.contains(s, na=False)
    ]
f = f[(f["معدل_التوصيل"] >= min_delivery) & (f["معدل_الإلغاء_مشاكل_التوصيل"] <= max_cancel)]

# Styling
def style_table(df):
    sty = df.style.format({
        "درجة_الأداء": "{:.2%}",
        "معدل_التوصيل": "{:.2%}",
        "معدل_الإلغاء_مشاكل_التوصيل": "{:.2%}",
    })
    sty = sty.applymap(lambda x: "color:red;font-weight:700;" if float(x) > 0 else "", subset=["المهام_المرفوضة_السائق"])
    sty = sty.applymap(lambda x: "color:red;font-weight:700;" if float(x) >= CANCEL_RED_THRESHOLD else "", subset=["معدل_الإلغاء_مشاكل_التوصيل"])
    sty = sty.applymap(lambda x: "color:red;font-weight:700;" if round(float(x), 6) != 1.0 else "", subset=["درجة_الأداء"])
    return sty

# =========================
# Operations (التشغيل) = show EVERYTHING on one page
# =========================
if ROLE == "التشغيل":
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("عدد السائقين (بعد الفلترة)", f"{len(f):,}")
    k2.metric("متوسط معدل التوصيل", f"{f['معدل_التوصيل'].mean():.2%}" if len(f) else "—")
    k3.metric("متوسط معدل الإلغاء (مشاكل التوصيل)", f"{f['معدل_الإلغاء_مشاكل_التوصيل'].mean():.2%}" if len(f) else "—")
    k4.metric("متوسط درجة الأداء", f"{f['درجة_الأداء'].mean():.2%}" if len(f) else "—")

    st.divider()

    # Needs Attention
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

    st.dataframe(style_table(f[top_cols].head(50)), use_container_width=True, hide_index=True)

    st.divider()

    # Driver Lookup
    st.subheader("🔎 بحث سريع عن سائق (Driver Lookup)")
    driver_list = f["اسم_السائق"].dropna().unique().tolist()
    selected = st.selectbox("اختر السائق", ["(اختر)"] + driver_list, key="ops_driver_pick")

    if selected != "(اختر)":
        d = f[f["اسم_السائق"] == selected].head(1).iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("معدل التوصيل", f"{d['معدل_التوصيل']:.2%}")
        c2.metric("المهام التي تم تسليمها", f"{int(d['المهام_التي_تم_تسليمها']):,}")
        c3.metric("معدل الإلغاء بسبب مشاكل التوصيل", f"{d['معدل_الإلغاء_مشاكل_التوصيل']:.2%}")
        c4.metric("المهام المرفوضة (السائق)", f"{int(d['المهام_المرفوضة_السائق']):,}")
        if "Face_Recognition" in f.columns:
            c5.metric("Face Recognition", f"{int(d['Face_Recognition']):,}")
        else:
            c5.metric("Face Recognition", "—")

        with st.expander("عرض جميع بيانات السائق"):
            st.dataframe(pd.DataFrame(d).T, use_container_width=True)

    st.divider()

    # Master Table
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

    st.divider()

    # Charts
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

# =========================
# Other roles keep it simple (you can refine later)
# =========================
elif ROLE == "الإشراف":
    st.subheader("🚨 سائقون يحتاجون متابعة (الأولوية أولاً)")
    cols = [
        "ترتيب_المتابعة", "معرف_السائق", "اسم_السائق",
        "أولوية_التنبيه", "درجة_الأداء",
        "معدل_التوصيل", "معدل_الإلغاء_مشاكل_التوصيل",
        "المهام_التي_تم_تسليمها", "المهام_المرفوضة_السائق",
    ]
    st.dataframe(style_table(f[cols].head(60)), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔎 بحث سريع عن سائق")
    driver_list = f["اسم_السائق"].dropna().unique().tolist()
    selected = st.selectbox("اختر السائق", ["(اختر)"] + driver_list, key="sup_driver_pick")
    if selected != "(اختر)":
        d = f[f["اسم_السائق"] == selected].head(1).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("معدل التوصيل", f"{d['معدل_التوصيل']:.2%}")
        c2.metric("المهام التي تم تسليمها", f"{int(d['المهام_التي_تم_تسليمها']):,}")
        c3.metric("معدل الإلغاء", f"{d['معدل_الإلغاء_مشاكل_التوصيل']:.2%}")
        c4.metric("المهام المرفوضة", f"{int(d['المهام_المرفوضة_السائق']):,}")

else:  # HR
    st.subheader("🔎 بحث سريع عن سائق")
    driver_list = f["اسم_السائق"].dropna().unique().tolist()
    selected = st.selectbox("اختر السائق", ["(اختر)"] + driver_list, key="hr_driver_pick")
    if selected != "(اختر)":
        d = f[f["اسم_السائق"] == selected].head(1).iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("المعرف", str(d["معرف_السائق"]))
        c2.metric("المهام التي تم تسليمها", f"{int(d['المهام_التي_تم_تسليمها']):,}")
        c3.metric("المهام المرفوضة", f"{int(d['المهام_المرفوضة_السائق']):,}")

    st.divider()
    st.subheader("📋 الجدول النهائي")
    cols = ["معرف_السائق", "اسم_السائق", "المهام_التي_تم_تسليمها", "المهام_المرفوضة_السائق"]
    if "Face_Recognition" in f.columns:
        cols.append("Face_Recognition")
    st.dataframe(f[cols], use_container_width=True, hide_index=True)
