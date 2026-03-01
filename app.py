import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from io import BytesIO
from datetime import datetime

# =========================
# Paths / Assets
# =========================
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
DATA_DIR = ROOT / "data"
UPLOADS_DIR = ROOT / "uploads"

DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "safeer_hr.db"

LEFT_IMG = ASSETS / "left.jpg"     # change if your names differ
RIGHT_IMG = ASSETS / "right.jpg"   # change if your names differ
LOGO_IMG = ASSETS / "logo.png"
FAVICON_IMG = ASSETS / "favicon.png"

# =========================
# Page config
# =========================
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
      .small-note { font-size: 12px; opacity: 0.8; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* --- Make file uploader minimal --- */

    /* Hide instruction text like: "Drag and drop files here" */
    [data-testid="stFileUploaderDropzone"] *[data-testid="stMarkdownContainer"] {
        display: none !important;
    }

    /* Hide small helper text like size/limit notes */
    [data-testid="stFileUploaderDropzone"] small {
        display: none !important;
    }

    /* Hide any remaining text nodes inside the dropzone */
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] div[data-testid="stCaptionContainer"] {
        display: none !important;
    }

    /* Remove the big dashed upload area background */
    [data-testid="stFileUploaderDropzone"] {
        border: 0px !important;
        padding: 0px !important;
        background: transparent !important;
    }

    /* Keep only the browse button visible */
    [data-testid="stFileUploaderDropzone"] button {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# =========================
# Header images (swapped)
# =========================
cimg1, cimg2 = st.columns([1, 1])
with cimg1:
    if RIGHT_IMG.exists():
        st.image(str(RIGHT_IMG), use_container_width=True)
with cimg2:
    if LEFT_IMG.exists():
        st.image(str(LEFT_IMG), use_container_width=True)

st.markdown("# لوحة سفير - Safeer Dash")
st.markdown('<div class="safeer-subtitle">التشغيل / الموارد البشرية / الإشراف</div>', unsafe_allow_html=True)
st.divider()

# =========================
# Auth / Roles
# =========================
ROLES = {
    "التشغيل": "ops_password",
    "الموارد البشرية": "hr_password",
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
        col1, col2 = st.columns(2)
        login = col1.button("دخول")
        logout = col2.button("خروج")

        if logout:
            st.session_state.logged_in = False
            st.session_state.role = None
            st.rerun()

        if login:
            expected = get_secret(ROLES[role], "")
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
# SQLite (HR registry)
# =========================
def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = db_conn()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS drivers (
        driver_id INTEGER PRIMARY KEY,
        driver_name TEXT,
        user_id TEXT,
        start_date TEXT,
        status TEXT DEFAULT 'نشط',
        notes TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS warnings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        driver_id INTEGER,
        warning_date TEXT,
        warning_type TEXT,
        details TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        driver_id INTEGER,
        expense_date TEXT,
        amount REAL,
        expense_type TEXT,
        details TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        driver_id INTEGER,
        uploaded_at TEXT,
        filename TEXT,
        path TEXT,
        doc_type TEXT,
        notes TEXT
    )
    """)

    con.commit()
    con.close()

init_db()

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def upsert_driver(driver_id: int, driver_name: str = None, user_id: str = None,
                  start_date: str = None, status: str = None, notes: str = None):
    con = db_conn()
    cur = con.cursor()

    cur.execute("SELECT driver_id, driver_name, user_id, start_date, status, notes FROM drivers WHERE driver_id = ?", (int(driver_id),))
    row = cur.fetchone()

    if row is None:
        cur.execute("""
            INSERT INTO drivers (driver_id, driver_name, user_id, start_date, status, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(driver_id),
            driver_name or "",
            user_id or "",
            start_date or "",
            status or "نشط",
            notes or "",
            now_ts(),
            now_ts()
        ))
    else:
        existing = {
            "driver_name": row[1] or "",
            "user_id": row[2] or "",
            "start_date": row[3] or "",
            "status": row[4] or "نشط",
            "notes": row[5] or "",
        }
        new_driver_name = existing["driver_name"] if (driver_name is None or str(driver_name).strip() == "") else str(driver_name).strip()
        new_user_id = existing["user_id"] if (user_id is None) else str(user_id).strip()
        new_start_date = existing["start_date"] if (start_date is None) else str(start_date).strip()
        new_status = existing["status"] if (status is None) else str(status).strip()
        new_notes = existing["notes"] if (notes is None) else str(notes).strip()

        cur.execute("""
            UPDATE drivers
            SET driver_name=?, user_id=?, start_date=?, status=?, notes=?, updated_at=?
            WHERE driver_id=?
        """, (new_driver_name, new_user_id, new_start_date, new_status, new_notes, now_ts(), int(driver_id)))

    con.commit()
    con.close()

def get_hr_registry() -> pd.DataFrame:
    con = db_conn()
    df = pd.read_sql_query("""
        SELECT
            d.driver_id AS معرف_السائق,
            d.driver_name AS اسم_السائق,
            d.user_id AS رقم_المستخدم,
            d.start_date AS تاريخ_المباشرة,
            d.status AS الحالة,
            (SELECT COUNT(*) FROM warnings w WHERE w.driver_id = d.driver_id) AS عدد_التحذيرات,
            (SELECT COALESCE(SUM(amount),0) FROM expenses e WHERE e.driver_id = d.driver_id) AS مجموع_المصاريف
        FROM drivers d
        ORDER BY d.driver_id
    """, con)
    con.close()
    return df

def get_driver_full(driver_id: int):
    con = db_conn()
    d = pd.read_sql_query("SELECT * FROM drivers WHERE driver_id = ?", con, params=(int(driver_id),))
    w = pd.read_sql_query("SELECT * FROM warnings WHERE driver_id = ? ORDER BY warning_date DESC", con, params=(int(driver_id),))
    e = pd.read_sql_query("SELECT * FROM expenses WHERE driver_id = ? ORDER BY expense_date DESC", con, params=(int(driver_id),))
    docs = pd.read_sql_query("SELECT * FROM documents WHERE driver_id = ? ORDER BY uploaded_at DESC", con, params=(int(driver_id),))
    con.close()
    return d, w, e, docs

# =========================
# Excel helpers
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

# =========================
# RULES / TARGETS
# =========================
CANCEL_RED_THRESHOLD = 0.002        # 0.20%
ORDERS_TARGET_MONTH = 450

# =========================
# Column mapping (Performance)
# =========================
PERF_COLS = {
    "driver_id": ["معرّف السائق", "معرف السائق", "Driver_ID", "driver_id", "id"],
    "first_name": ["اسم السائق", "First Name", "first_name"],
    "last_name": ["اسم السائق.1", "Last Name", "last_name"],

    "delivery_rate": ["معدل اكتمال الطلبات (غير متعلق بالتوصيل)", "معدل التوصيل", "معدل توصيل", "Delivery_Rate", "delivery_rate"],
    "cancel_rate": ["معدل الإلغاء بسبب مشاكل التوصيل", "معدل الغاء", "معدل الإلغاء", "Cancel_Rate", "cancel_rate"],
    "orders_delivered": ["المهام التي تم تسليمها", "طلبات", "الطلبات", "الطلبات المسلمة", "طلبات مكتملة", "Orders_Delivered", "orders_delivered"],
    "reject_total": ["المهام المرفوضة", "المهام المرفوضة (السائق)", "رفض السائق", "Driver Rejections", "driver_rejections"],

    "work_days": ["اعدد ايام العمل", "عدد ايام العمل", "أيام العمل", "Work Days", "work_days", "days_worked"],

    "fr": ["FR", "Face Recognition", "Face_Recognition", "التعرف على الوجه", "face_recognition"],
    "vda": ["VDA", "vda", "مؤشر VDA", "مؤشر_إضافي"],

    "auto_reject": ["المهام المرفوضة تلقائيًا (تلقائياً)", "المهام المرفوضة تلقائيا", "Auto Reject", "auto_reject"],
    "ontime_d": ["نسبة الطلبات التي تم تسليمها في الوقت المحدد (D)", "نسبة التسليم في الوقت", "On-time (D)", "ontime_d"],
    "avg_delivery_time": ["متوسط مدة التوصيل لكل طلب مكتمل", "متوسط مدة التوصيل", "Avg Delivery Time", "avg_delivery_time"],
}

def build_performance_report(df_raw: pd.DataFrame) -> pd.DataFrame:
    mapped = {k: pick(df_raw.columns, v) for k, v in PERF_COLS.items()}

    required = ["driver_id", "first_name", "last_name", "delivery_rate", "cancel_rate", "orders_delivered", "reject_total"]
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
        "معرّف السائق": safe_to_numeric(df_raw[mapped["driver_id"]]),
        "اسم السائق": driver_name,

        "معدل توصيل": safe_to_numeric(df_raw[mapped["delivery_rate"]]),
        "معدل الغاء": safe_to_numeric(df_raw[mapped["cancel_rate"]]),
        "طلبات": safe_to_numeric(df_raw[mapped["orders_delivered"]]),
        "المهام المرفوضة": safe_to_numeric(df_raw[mapped["reject_total"]]),
    })

    out["معرّف السائق"] = pd.to_numeric(out["معرّف السائق"], errors="coerce").astype("Int64")
    out["معدل توصيل"] = out["معدل توصيل"].fillna(0).clip(0, 1)
    out["معدل الغاء"] = out["معدل الغاء"].fillna(0).clip(0, 1)
    out["طلبات"] = out["طلبات"].fillna(0)
    out["المهام المرفوضة"] = out["المهام المرفوضة"].fillna(0)

    if mapped.get("work_days"):
        out["اعدد ايام العمل"] = safe_to_numeric(df_raw[mapped["work_days"]]).fillna(0)
    else:
        out["اعدد ايام العمل"] = pd.NA

    if mapped.get("fr"):
        out["FR"] = safe_to_numeric(df_raw[mapped["fr"]]).fillna(0)
    else:
        out["FR"] = pd.NA

    if mapped.get("vda"):
        out["VDA"] = safe_to_numeric(df_raw[mapped["vda"]]).fillna(0)
    else:
        out["VDA"] = pd.NA

    # Keep expander raw fields stable
    out["اسم السائق (مكرر)"] = out["اسم السائق"]

    # Requested fields (populate from raw if exists, else map/NA)
    if "المهام التي تم تسليمها" in df_raw.columns:
        out["المهام التي تم تسليمها"] = df_raw["المهام التي تم تسليمها"]
    else:
        out["المهام التي تم تسليمها"] = df_raw[mapped["orders_delivered"]]

    if "المهام المرفوضة" in df_raw.columns:
        out["المهام المرفوضة"] = df_raw["المهام المرفوضة"]
    # already exists as metric

    if "المهام المرفوضة (السائق)" in df_raw.columns:
        out["المهام المرفوضة (السائق)"] = df_raw["المهام المرفوضة (السائق)"]
    else:
        out["المهام المرفوضة (السائق)"] = df_raw[mapped["reject_total"]]

    if mapped.get("auto_reject"):
        out["المهام المرفوضة تلقائيًا (تلقائياً)"] = df_raw[mapped["auto_reject"]]
    else:
        out["المهام المرفوضة تلقائيًا (تلقائياً)"] = pd.NA

    out["معدل الإلغاء بسبب مشاكل التوصيل"] = df_raw[mapped["cancel_rate"]]

    if mapped.get("ontime_d"):
        out["نسبة الطلبات التي تم تسليمها في الوقت المحدد (D)"] = df_raw[mapped["ontime_d"]]
    else:
        out["نسبة الطلبات التي تم تسليمها في الوقت المحدد (D)"] = pd.NA

    if mapped.get("avg_delivery_time"):
        out["متوسط مدة التوصيل لكل طلب مكتمل"] = df_raw[mapped["avg_delivery_time"]]
    else:
        out["متوسط مدة التوصيل لكل طلب مكتمل"] = pd.NA

    return out

def detect_file_type(cols: set[str]) -> str:
    perf_signals = {
        "معرّف السائق", "معرف السائق", "اسم السائق", "اسم السائق.1",
        "معدل الإلغاء بسبب مشاكل التوصيل", "معدل الغاء", "معدل توصيل",
        "المهام التي تم تسليمها", "طلبات", "المهام المرفوضة (السائق)", "المهام المرفوضة",
    }
    face_signals = {"Face Recognition", "Face_Recognition", "التعرف على الوجه", "FR"}
    vda_signals = {"VDA", "مؤشر VDA"}

    perf_hits = len(perf_signals.intersection(cols))
    face_hits = len(face_signals.intersection(cols))
    vda_hits = len(vda_signals.intersection(cols))

    if perf_hits >= 4:
        return "performance"
    if face_hits >= 1:
        return "face"
    if vda_hits >= 1:
        return "vda"
    return "unknown"

# =========================
# Styling (Attention table)
# =========================
def style_attention_table(df):
    sty = df.style.format({"معدل توصيل": "{:.2%}", "معدل الغاء": "{:.2%}"})
    sty = sty.applymap(lambda x: "color:red;font-weight:800;" if float(x) <= CANCEL_RED_THRESHOLD else "", subset=["معدل الغاء"])
    sty = sty.applymap(lambda x: "color:red;font-weight:800;" if float(x) < 1.0 else "", subset=["معدل توصيل"])
    sty = sty.applymap(lambda x: "color:red;font-weight:800;" if float(x) < ORDERS_TARGET_MONTH else "", subset=["طلبات"])
    sty = sty.applymap(lambda x: "color:red;font-weight:800;" if float(x) > 0 else "", subset=["المهام المرفوضة"])
    return sty

# =========================
# Sidebar uploader + filters (MINIMAL)
# =========================
with st.sidebar:
    st.markdown(f"### المستخدم الحالي: {ROLE}")

    st.markdown("### 📁")
    uploaded_files = st.file_uploader(
        "رفع ملفات",
        type=["xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.divider()
    search = st.text_input("بحث (المعرف / الاسم)", "")
    min_delivery = st.slider("أقل معدل توصيل", 0.0, 1.0, 0.0, 0.01)
    max_cancel = st.slider("أعلى/أقل معدل إلغاء", 0.0, 1.0, 1.0, 0.01)

# =========================
# HR Page
# =========================
def hr_page():
    st.subheader("👥 الموارد البشرية — سجل السائقين (دائم)")
    registry = get_hr_registry()
    st.dataframe(registry, use_container_width=True, hide_index=True)

# =========================
# Dashboard logic (Ops/Sup)
# =========================
def dashboard_logic():
    if not uploaded_files:
        st.info("ارفع ملف/ملفات للبدء.")
        st.stop()

    file_items = []
    for uf in uploaded_files:
        b = uf.getvalue()
        df = read_first_sheet_excel_bytes(b)
        cols = set(df.columns)
        kind = detect_file_type(cols)
        file_items.append({"name": uf.name, "df": df, "cols": cols, "kind_guess": kind})

    perf_candidates = [x for x in file_items if x["kind_guess"] == "performance"]

    def file_picker(options, key):
        names = [o["name"] for o in options]
        chosen = st.sidebar.selectbox("ملف الأداء", names, key=key)
        return next(o for o in options if o["name"] == chosen)

    if len(perf_candidates) == 1:
        perf_item = perf_candidates[0]
    elif len(perf_candidates) > 1:
        perf_item = file_picker(perf_candidates, "pick_perf")
    else:
        perf_item = file_picker(file_items, "pick_perf_manual")

    perf = build_performance_report(perf_item["df"])

    # Sync names into HR DB
    for _, r in perf.iterrows():
        did = r.get("معرّف السائق")
        name = r.get("اسم السائق")
        if pd.isna(did):
            continue
        upsert_driver(int(did), driver_name=str(name).strip())

    master = perf.copy()

    # Merge FR/VDA from other files
    for item in file_items:
        if item["name"] == perf_item["name"]:
            continue
        df = item["df"].copy()

        id_col = pick(df.columns, PERF_COLS["driver_id"])
        if not id_col:
            continue

        df_id = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
        fr_col = pick(df.columns, PERF_COLS["fr"])
        vda_col = pick(df.columns, PERF_COLS["vda"])

        temp = pd.DataFrame({"معرّف السائق": df_id})
        has_any = False
        if fr_col:
            temp["FR"] = safe_to_numeric(df[fr_col]).fillna(0); has_any = True
        if vda_col:
            temp["VDA"] = safe_to_numeric(df[vda_col]).fillna(0); has_any = True

        if has_any:
            master = master.merge(temp.drop_duplicates("معرّف السائق"), on="معرّف السائق", how="left")

    if "FR" in master.columns:
        master["FR"] = master["FR"].fillna(0)
    if "VDA" in master.columns:
        master["VDA"] = master["VDA"].fillna(0)

    f = master.copy()
    if search.strip():
        s = search.strip().lower()
        f = f[
            f["اسم السائق"].str.lower().str.contains(s, na=False)
            | f["معرّف السائق"].astype(str).str.contains(s, na=False)
        ]
    f = f[(f["معدل توصيل"] >= min_delivery)]
    f = f[(f["معدل الغاء"] <= max_cancel)]

    # Priority
    f["تنبيه الغاء"] = (f["معدل الغاء"] <= CANCEL_RED_THRESHOLD).astype(int)
    f["تنبيه توصيل"] = (f["معدل توصيل"] < 1.0).astype(int)
    f["تنبيه طلبات"] = (f["طلبات"] < ORDERS_TARGET_MONTH).astype(int)
    f["تنبيه رفض"] = (f["المهام المرفوضة"] > 0).astype(int)

    delivery_gap = (1.0 - f["معدل توصيل"]).clip(lower=0)
    orders_gap = (ORDERS_TARGET_MONTH - f["طلبات"]).clip(lower=0)

    f["أولوية"] = (
        f["تنبيه الغاء"] * 100000
        + delivery_gap * 10000
        + f["تنبيه طلبات"] * 2000
        + orders_gap * 2
        + f["المهام المرفوضة"] * 50
    )

    f = f.sort_values(
        ["أولوية", "تنبيه الغاء", "معدل توصيل", "طلبات", "المهام المرفوضة"],
        ascending=[False, False, True, True, False]
    ).reset_index(drop=True)

    f["ترتيب المتابعة"] = range(1, len(f) + 1)
    return f

# =========================
# Render
# =========================
if ROLE == "الموارد البشرية":
    hr_page()
    st.stop()

f = dashboard_logic()

# KPI row (updated)
k1, k2, k3, k4 = st.columns(4)
k1.metric("عدد السائقين", f"{len(f):,}")
k2.metric("متوسط معدل التوصيل", f"{f['معدل توصيل'].mean():.2%}" if len(f) else "—")
k3.metric("متوسط معدل الإلغاء", f"{f['معدل الغاء'].mean():.2%}" if len(f) else "—")
k4.metric("عدد الطلبات", f"{int(f['طلبات'].sum()):,}" if len(f) else "—")

st.divider()

# Attention table
st.subheader("🚨 سائقون يحتاجون متابعة (الأولوية أولاً)")
attention_cols = ["ترتيب المتابعة", "معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة"]
st.dataframe(style_attention_table(f[attention_cols].head(60)), use_container_width=True, hide_index=True)

st.divider()

# Driver lookup
st.subheader("🔎 بحث سريع عن سائق (Driver Lookup)")
driver_list = f["اسم السائق"].dropna().unique().tolist()
selected = st.selectbox("اختر السائق", ["(اختر)"] + driver_list, key="lookup_driver")

if selected != "(اختر)":
    d = f[f["اسم السائق"] == selected].head(1).iloc[0]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("معدل توصيل %", f"{float(d['معدل توصيل']):.2%}")
    c2.metric("طلبات", f"{int(d['طلبات']):,}")
    c3.metric("معدل الغاء %", f"{float(d['معدل الغاء']):.2%}")

    wd = d.get("اعدد ايام العمل")
    c4.metric("اعدد ايام العمل", "—" if pd.isna(wd) else f"{int(float(wd)):,}")

    fr = d.get("FR")
    c5.metric("FR", "—" if pd.isna(fr) else f"{int(float(fr)):,}")

    vda = d.get("VDA")
    c6.metric("VDA", "—" if pd.isna(vda) else f"{int(float(vda)):,}")

    with st.expander("عرض جميع بيانات السائق"):
        wanted = [
            "معرّف السائق",
            "اسم السائق",
            "اسم السائق (مكرر)",
            "المهام التي تم تسليمها",
            "المهام المرفوضة",
            "المهام المرفوضة (السائق)",
            "المهام المرفوضة تلقائيًا (تلقائياً)",
            "معدل الإلغاء بسبب مشاكل التوصيل",
            "نسبة الطلبات التي تم تسليمها في الوقت المحدد (D)",
            "متوسط مدة التوصيل لكل طلب مكتمل",
        ]
        row = pd.DataFrame([{col: d.get(col, pd.NA) for col in wanted}])

        # Coerce percent cols
        for pc in ["معدل الإلغاء بسبب مشاكل التوصيل", "نسبة الطلبات التي تم تسليمها في الوقت المحدد (D)"]:
            row[pc] = pd.to_numeric(row[pc], errors="coerce")

        st.dataframe(
            row.style.format({
                "معدل الإلغاء بسبب مشاكل التوصيل": "{:.2%}",
                "نسبة الطلبات التي تم تسليمها في الوقت المحدد (D)": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True
        )

st.divider()

# Bottom table
st.subheader("📋 الجدول النهائي (كامل البيانات)")
bottom_cols = ["معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة", "اعدد ايام العمل"]
if "FR" in f.columns:
    bottom_cols.append("FR")
if "VDA" in f.columns:
    bottom_cols.append("VDA")

st.dataframe(
    f[bottom_cols].style.format({"معدل توصيل": "{:.2%}", "معدل الغاء": "{:.2%}"}),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "⬇️ تحميل النتائج CSV",
    data=f.to_csv(index=False, encoding="utf-8-sig"),
    file_name="safeer_master_filtered.csv",
    mime="text/csv",
)


