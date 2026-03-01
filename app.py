import os
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
st.markdown('<div class="safeer-subtitle">التشغيل / الموارد البشرية / الإشراف • رفع متعدد الملفات • إبراز التنبيهات أولاً</div>', unsafe_allow_html=True)
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

def add_warning(driver_id: int, warning_date: str, warning_type: str, details: str):
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO warnings (driver_id, warning_date, warning_type, details, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (int(driver_id), warning_date, warning_type, details, now_ts()))
    con.commit()
    con.close()

def add_expense(driver_id: int, expense_date: str, amount: float, expense_type: str, details: str):
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO expenses (driver_id, expense_date, amount, expense_type, details, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (int(driver_id), expense_date, float(amount), expense_type, details, now_ts()))
    con.commit()
    con.close()

def add_document(driver_id: int, file_name: str, save_path: str, doc_type: str, notes: str):
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO documents (driver_id, uploaded_at, filename, path, doc_type, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (int(driver_id), now_ts(), file_name, save_path, doc_type, notes))
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
# PRIORITY RULES (your latest request)
# =========================
CANCEL_RED_THRESHOLD = 0.002        # 0.20%
ORDERS_TARGET_MONTH = 450

# =========================
# Performance mapping
# =========================
PERF_COLS = {
    "driver_id": ["معرّف السائق", "معرف السائق", "Driver_ID", "driver_id", "id"],
    "first_name": ["اسم السائق", "First Name", "first_name"],
    "last_name": ["اسم السائق.1", "Last Name", "last_name"],

    "cancel_rate": ["معدل الإلغاء بسبب مشاكل التوصيل", "معدل الغاء", "معدل الإلغاء", "Cancel_Rate", "cancel_rate"],
    "driver_reject": ["المهام المرفوضة (السائق)", "المهام المرفوضة", "رفض السائق", "Driver Rejections", "driver_rejections"],
    "orders_delivered": ["المهام التي تم تسليمها", "طلبات", "الطلبات", "الطلبات المسلمة", "طلبات مكتملة", "Orders_Delivered", "orders_delivered"],
    "delivery_rate": ["معدل اكتمال الطلبات (غير متعلق بالتوصيل)", "معدل التوصيل", "معدل توصيل", "Delivery_Rate", "delivery_rate"],

    # NEW: Work days for Driver Lookup (if exists)
    "work_days": ["اعدد ايام العمل", "عدد ايام العمل", "أيام العمل", "Work Days", "work_days", "days_worked"],

    # Optional: FR / VDA could exist in extra files too
    "fr": ["FR", "Face Recognition", "Face_Recognition", "التعرف على الوجه", "face_recognition"],
    "vda": ["VDA", "vda", "مؤشر VDA", "مؤشر_إضافي"],
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

        "معدل_توصيل": safe_to_numeric(df_raw[mapped["delivery_rate"]]),
        "معدل_الغاء": safe_to_numeric(df_raw[mapped["cancel_rate"]]),
        "طلبات": safe_to_numeric(df_raw[mapped["orders_delivered"]]),
        "المهام_المرفوضة": safe_to_numeric(df_raw[mapped["driver_reject"]]),
    })

    out["معرف_السائق"] = pd.to_numeric(out["معرف_السائق"], errors="coerce").astype("Int64")

    out["معدل_توصيل"] = out["معدل_توصيل"].fillna(0).clip(0, 1)
    out["معدل_الغاء"] = out["معدل_الغاء"].fillna(0).clip(0, 1)

    out["طلبات"] = out["طلبات"].fillna(0)
    out["المهام_المرفوضة"] = out["المهام_المرفوضة"].fillna(0)

    # optional: work days
    if mapped.get("work_days"):
        out["اعدد_ايام_العمل"] = safe_to_numeric(df_raw[mapped["work_days"]]).fillna(0)
    else:
        out["اعدد_ايام_العمل"] = pd.NA

    # optional: FR / VDA inside same file
    if mapped.get("fr"):
        out["FR"] = safe_to_numeric(df_raw[mapped["fr"]]).fillna(0)
    else:
        out["FR"] = pd.NA

    if mapped.get("vda"):
        out["VDA"] = safe_to_numeric(df_raw[mapped["vda"]]).fillna(0)
    else:
        out["VDA"] = pd.NA

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
# Styling (your thresholds)
# =========================
def style_attention_table(df):
    sty = df.style.format({
        "معدل_توصيل": "{:.2%}",
        "معدل_الغاء": "{:.2%}",
    })

    # Cancel in red if <= 0.20% (per your message)
    sty = sty.applymap(
        lambda x: "color:red;font-weight:800;" if float(x) <= CANCEL_RED_THRESHOLD else "",
        subset=["معدل_الغاء"]
    )

    # Delivery in red if < 100%
    sty = sty.applymap(
        lambda x: "color:red;font-weight:800;" if float(x) < 1.0 else "",
        subset=["معدل_توصيل"]
    )

    # Orders in red if < 450
    sty = sty.applymap(
        lambda x: "color:red;font-weight:800;" if float(x) < ORDERS_TARGET_MONTH else "",
        subset=["طلبات"]
    )

    # Rejections in red if > 0
    sty = sty.applymap(
        lambda x: "color:red;font-weight:800;" if float(x) > 0 else "",
        subset=["المهام_المرفوضة"]
    )

    return sty

# =========================
# Sidebar uploader + filters
# =========================
with st.sidebar:
    st.markdown(f"### المستخدم الحالي: {ROLE}")

    st.markdown("### رفع ملفات التشغيل/الأداء")
    uploaded_files = st.file_uploader(
        "ارفع 2–3 ملفات Excel معًا (للتشغيل/الأداء)",
        type=["xlsx"],
        accept_multiple_files=True
    )

    st.divider()
    st.markdown("### فلاتر التشغيل/الأداء")
    search = st.text_input("بحث (المعرف / الاسم)", "")
    min_delivery = st.slider("أقل معدل توصيل", 0.0, 1.0, 0.0, 0.01)
    max_cancel = st.slider("أعلى/أقل معدل إلغاء", 0.0, 1.0, 1.0, 0.01)  # kept for now

# =========================
# HR Page
# =========================
def hr_page():
    st.subheader("👥 الموارد البشرية — سجل السائقين (دائم)")
    st.markdown('<div class="small-note">السجل محفوظ في قاعدة بيانات SQLite داخل مجلد <b>data/</b>.</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📥 رفع قائمة سائقين (لإضافة الجدد فقط)")
    hr_list = st.file_uploader("ارفع ملف Excel يحتوي على معرف السائق واسم السائق (اختياري)", type=["xlsx"], key="hr_list")

    if hr_list:
        df_list = read_first_sheet_excel_bytes(hr_list.getvalue())
        st.success("تم قراءة الملف. اختر الأعمدة الصحيحة:")

        col_id = st.selectbox("عمود معرف السائق", options=list(df_list.columns), key="hr_col_id")
        col_name = st.selectbox("عمود اسم السائق", options=list(df_list.columns), key="hr_col_name")
        col_user = st.selectbox("عمود رقم المستخدم (إن وجد)", options=["(بدون)"] + list(df_list.columns), key="hr_col_user")

        if st.button("✅ إضافة/تحديث السائقين من الملف"):
            processed = 0
            for _, r in df_list.iterrows():
                did = safe_to_numeric(r.get(col_id))
                if pd.isna(did):
                    continue
                did = int(did)
                name = str(r.get(col_name, "")).strip()
                user_id = ""
                if col_user != "(بدون)":
                    user_id = str(r.get(col_user, "")).strip()
                upsert_driver(did, driver_name=name, user_id=user_id)
                processed += 1
            st.success(f"تمت المعالجة: {processed} صف (الجدد يتم إضافتهم تلقائياً).")

    st.divider()

    st.markdown("### ➕ إضافة/تحديث سائق")
    with st.form("hr_add_driver"):
        d_id = st.text_input("معرف السائق")
        d_name = st.text_input("اسم السائق")
        u_id = st.text_input("رقم المستخدم (User ID)")
        s_date = st.text_input("تاريخ المباشرة (مثال: 2026-02-28)")
        status = st.selectbox("الحالة", ["نشط", "موقوف", "منتهي", "إجازة"])
        notes = st.text_area("ملاحظات", "")
        submit = st.form_submit_button("حفظ")
        if submit:
            if d_id.strip() == "" or not d_id.strip().isdigit():
                st.error("أدخل معرف سائق صحيح (رقم).")
            else:
                upsert_driver(int(d_id), driver_name=d_name, user_id=u_id, start_date=s_date, status=status, notes=notes)
                st.success("تم حفظ السائق.")

    st.divider()

    st.markdown("### 📋 سجل السائقين")
    registry = get_hr_registry()
    st.dataframe(registry, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### 🔎 تفاصيل سائق")
    driver_ids = registry["معرف_السائق"].dropna().astype(int).tolist() if len(registry) else []
    if not driver_ids:
        st.info("لا يوجد سائقون بعد.")
        return

    selected_id = st.selectbox("اختر معرف السائق", driver_ids)
    d, w, e, docs = get_driver_full(int(selected_id))
    drow = d.iloc[0] if len(d) else None

    if drow is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("معرف السائق", str(drow.get("driver_id", "")))
        c2.metric("اسم السائق", str(drow.get("driver_name", "")))
        c3.metric("رقم المستخدم", str(drow.get("user_id", "")))
        c4.metric("تاريخ المباشرة", str(drow.get("start_date", "")))

        st.write(f"**الحالة:** {drow.get('status','')}")
        if str(drow.get("notes", "")).strip():
            st.write(f"**ملاحظات:** {drow.get('notes','')}")

    st.divider()

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### ⚠️ إضافة تحذير")
        with st.form("hr_add_warning"):
            w_date = st.text_input("تاريخ التحذير", value=datetime.now().strftime("%Y-%m-%d"))
            w_type = st.text_input("نوع التحذير", "")
            w_details = st.text_area("تفاصيل", "")
            w_submit = st.form_submit_button("إضافة تحذير")
            if w_submit:
                add_warning(int(selected_id), w_date, w_type, w_details)
                st.success("تم إضافة التحذير.")
                st.rerun()

    with colB:
        st.markdown("### 💰 إضافة مصروف")
        with st.form("hr_add_expense"):
            e_date = st.text_input("تاريخ المصروف", value=datetime.now().strftime("%Y-%m-%d"))
            amount = st.number_input("المبلغ", min_value=0.0, value=0.0, step=1.0)
            e_type = st.text_input("نوع المصروف", "")
            e_details = st.text_area("تفاصيل", "")
            e_submit = st.form_submit_button("إضافة مصروف")
            if e_submit:
                add_expense(int(selected_id), e_date, float(amount), e_type, e_details)
                st.success("تم إضافة المصروف.")
                st.rerun()

    st.divider()

    st.markdown("### 📎 رفع مستندات السائق")
    doc_file = st.file_uploader("ارفع مستند", type=["pdf", "png", "jpg", "jpeg", "doc", "docx"], key="hr_doc_file")
    doc_type = st.text_input("نوع المستند (مثال: هوية / رخصة / عقد)", key="hr_doc_type")
    doc_notes = st.text_input("ملاحظات المستند", key="hr_doc_notes")

    if doc_file and st.button("حفظ المستند"):
        driver_folder = UPLOADS_DIR / str(int(selected_id))
        driver_folder.mkdir(exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = doc_file.name.replace("/", "_").replace("\\", "_")
        save_path = driver_folder / f"{ts}_{safe_name}"

        with open(save_path, "wb") as f_out:
            f_out.write(doc_file.getvalue())

        add_document(int(selected_id), safe_name, str(save_path), doc_type, doc_notes)
        st.success("تم حفظ المستند.")
        st.rerun()

    st.markdown("### 📚 سجل التحذيرات")
    if len(w):
        show_w = w[["warning_date", "warning_type", "details"]].rename(columns={
            "warning_date": "التاريخ",
            "warning_type": "النوع",
            "details": "التفاصيل"
        })
        st.dataframe(show_w, use_container_width=True, hide_index=True)
    else:
        st.info("لا توجد تحذيرات.")

    st.markdown("### 💳 سجل المصاريف")
    if len(e):
        show_e = e[["expense_date", "amount", "expense_type", "details"]].rename(columns={
            "expense_date": "التاريخ",
            "amount": "المبلغ",
            "expense_type": "النوع",
            "details": "التفاصيل"
        })
        st.dataframe(show_e, use_container_width=True, hide_index=True)
    else:
        st.info("لا توجد مصاريف.")

    st.markdown("### 🗂️ المستندات")
    if len(docs):
        show_d = docs[["uploaded_at", "filename", "doc_type", "notes", "path"]].rename(columns={
            "uploaded_at": "تاريخ الرفع",
            "filename": "الملف",
            "doc_type": "نوع المستند",
            "notes": "ملاحظات",
            "path": "المسار"
        })
        st.dataframe(show_d, use_container_width=True, hide_index=True)
    else:
        st.info("لا توجد مستندات.")

# =========================
# Dashboard logic (Ops/Sup)
# =========================
def dashboard_logic():
    if not uploaded_files:
        st.info("ارفع ملفات Excel للبدء (على الأقل ملف الأداء).")
        st.stop()

    file_items = []
    for uf in uploaded_files:
        b = uf.getvalue()
        df = read_first_sheet_excel_bytes(b)
        cols = set(df.columns)
        kind = detect_file_type(cols)
        file_items.append({"name": uf.name, "df": df, "cols": cols, "kind_guess": kind})

    perf_candidates = [x for x in file_items if x["kind_guess"] == "performance"]

    def file_picker(label, options, key):
        names = [o["name"] for o in options]
        chosen = st.sidebar.selectbox(label, names, key=key)
        return next(o for o in options if o["name"] == chosen)

    if len(perf_candidates) == 1:
        perf_item = perf_candidates[0]
    elif len(perf_candidates) > 1:
        st.sidebar.warning("تم اكتشاف أكثر من ملف أداء. اختر الملف الصحيح:")
        perf_item = file_picker("اختر ملف الأداء", perf_candidates, "pick_perf")
    else:
        st.sidebar.warning("لم يتم اكتشاف ملف الأداء تلقائيًا. اختر ملف الأداء يدويًا:")
        perf_item = file_picker("اختر ملف الأداء", file_items, "pick_perf_manual")

    perf = build_performance_report(perf_item["df"])

    # Sync drivers into HR DB automatically
    for _, r in perf.iterrows():
        did = r.get("معرف_السائق")
        name = r.get("اسم_السائق")
        if pd.isna(did):
            continue
        upsert_driver(int(did), driver_name=str(name).strip())

    # Merge FR/VDA from other uploaded files if provided
    # We will try to find columns by name in any other file and merge by driver id
    master = perf.copy()

    for item in file_items:
        if item["name"] == perf_item["name"]:
            continue
        df = item["df"].copy()

        # detect driver id column in that file
        id_col = pick(df.columns, ["معرّف السائق", "معرف السائق", "Driver_ID", "driver_id", "id"])
        if not id_col:
            continue

        df_id = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")

        fr_col = pick(df.columns, PERF_COLS["fr"])
        vda_col = pick(df.columns, PERF_COLS["vda"])

        temp = pd.DataFrame({"معرف_السائق": df_id})

        if fr_col:
            temp["FR"] = safe_to_numeric(df[fr_col]).fillna(0)
        if vda_col:
            temp["VDA"] = safe_to_numeric(df[vda_col]).fillna(0)

        if ("FR" in temp.columns) or ("VDA" in temp.columns):
            master = master.merge(temp.drop_duplicates("معرف_السائق"), on="معرف_السائق", how="left")

    if "FR" in master.columns:
        master["FR"] = master["FR"].fillna(0)
    if "VDA" in master.columns:
        master["VDA"] = master["VDA"].fillna(0)

    # Filters
    f = master.copy()
    if search.strip():
        s = search.strip().lower()
        f = f[
            f["اسم_السائق"].str.lower().str.contains(s, na=False)
            | f["معرف_السائق"].astype(str).str.contains(s, na=False)
        ]

    f = f[(f["معدل_توصيل"] >= min_delivery)]
    # keep slider name but no longer used exactly like before; leaving as-is
    f = f[(f["معدل_الغاء"] <= max_cancel)]

    # PRIORITY score per your requirements
    f["تنبيه_الغاء"] = (f["معدل_الغاء"] <= CANCEL_RED_THRESHOLD).astype(int)
    f["تنبيه_توصيل"] = (f["معدل_توصيل"] < 1.0).astype(int)
    f["تنبيه_طلبات"] = (f["طلبات"] < ORDERS_TARGET_MONTH).astype(int)
    f["تنبيه_رفض"] = (f["المهام_المرفوضة"] > 0).astype(int)

    # Make a numeric priority score (higher = more urgent)
    # - Cancel alert strongest
    # - Delivery: the lower the delivery, the higher the priority
    # - Orders: the lower than 450, the higher
    # - Rejections: count contributes
    delivery_gap = (1.0 - f["معدل_توصيل"]).clip(lower=0)
    orders_gap = (ORDERS_TARGET_MONTH - f["طلبات"]).clip(lower=0)

    f["أولوية"] = (
        f["تنبيه_الغاء"] * 100000
        + delivery_gap * 10000
        + f["تنبيه_طلبات"] * 2000
        + orders_gap * 2
        + f["المهام_المرفوضة"] * 50
    )

    f = f.sort_values(
        ["أولوية", "تنبيه_الغاء", "معدل_توصيل", "طلبات", "المهام_المرفوضة"],
        ascending=[False, False, True, True, False]
    ).reset_index(drop=True)

    f["ترتيب_المتابعة"] = range(1, len(f) + 1)

    return f

# =========================
# Render by role
# =========================
if ROLE == "الموارد البشرية":
    hr_page()
    st.stop()

# Ops / Sup
f = dashboard_logic()

# KPIs (optional, simple)
k1, k2, k3, k4 = st.columns(4)
k1.metric("عدد السائقين (بعد الفلترة)", f"{len(f):,}")
k2.metric("متوسط معدل التوصيل", f"{f['معدل_توصيل'].mean():.2%}" if len(f) else "—")
k3.metric("متوسط معدل الإلغاء", f"{f['معدل_الغاء'].mean():.2%}" if len(f) else "—")
k4.metric("متوسط الطلبات", f"{int(f['طلبات'].mean()):,}" if len(f) else "—")

st.divider()

# =========================
# Needs attention (your exact columns)
# =========================
st.subheader("🚨 سائقون يحتاجون متابعة (الأولوية أولاً)")

attention_cols = [
    "ترتيب_المتابعة",
    "معرف_السائق",
    "اسم_السائق",
    "معدل_توصيل",
    "معدل_الغاء",
    "طلبات",
    "المهام_المرفوضة",
]
st.dataframe(style_attention_table(f[attention_cols].head(60)), use_container_width=True, hide_index=True)

st.divider()

# =========================
# Driver Lookup (updated)
# =========================
st.subheader("🔎 بحث سريع عن سائق (Driver Lookup)")

driver_list = f["اسم_السائق"].dropna().unique().tolist()
selected = st.selectbox("اختر السائق", ["(اختر)"] + driver_list, key="lookup_driver")

if selected != "(اختر)":
    d = f[f["اسم_السائق"] == selected].head(1).iloc[0]

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("معدل توصيل %", f"{float(d['معدل_توصيل']):.2%}")
    c2.metric("طلبات", f"{int(d['طلبات']):,}")
    c3.metric("معدل الغاء %", f"{float(d['معدل_الغاء']):.2%}")

    # Replace rejections with work days
    wd = d.get("اعدد_ايام_العمل")
    if pd.isna(wd):
        c4.metric("اعدد ايام العمل", "—")
    else:
        c4.metric("اعدد ايام العمل", f"{int(wd):,}")

    # FR
    fr = d.get("FR")
    if pd.isna(fr):
        c5.metric("FR", "—")
    else:
        c5.metric("FR", f"{int(fr):,}")

    # VDA
    vda = d.get("VDA")
    if pd.isna(vda):
        c6.metric("VDA", "—")
    else:
        c6.metric("VDA", f"{int(vda):,}")

    with st.expander("عرض جميع بيانات السائق"):
        st.dataframe(pd.DataFrame(d).T, use_container_width=True)

st.divider()

# =========================
# Bottom master table (all info)
# =========================
st.subheader("📋 الجدول النهائي (كامل البيانات)")

bottom_cols = [
    "معرف_السائق", "اسم_السائق",
    "معدل_توصيل", "معدل_الغاء",
    "طلبات", "المهام_المرفوضة",
    "اعدد_ايام_العمل"
]
if "FR" in f.columns:
    bottom_cols.append("FR")
if "VDA" in f.columns:
    bottom_cols.append("VDA")

# Format percentages nicely
bottom_df = f[bottom_cols].copy()
st.dataframe(bottom_df.style.format({"معدل_توصيل": "{:.2%}", "معدل_الغاء": "{:.2%}"}), use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ تحميل النتائج CSV",
    data=f.to_csv(index=False, encoding="utf-8-sig"),
    file_name="safeer_master_filtered.csv",
    mime="text/csv",
)
