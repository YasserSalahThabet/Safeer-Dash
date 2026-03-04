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

LEFT_IMG = ASSETS / "left.jpg"
RIGHT_IMG = ASSETS / "right.jpg"
LOGO_IMG = ASSETS / "logo.png"
FAVICON_IMG = ASSETS / "favicon.png"

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Safeer Dash",
    page_icon=str(FAVICON_IMG) if FAVICON_IMG.exists() else "🟢",
    layout="wide",
)

# =========================
# CSS
# =========================
st.markdown(
    """
    <style>
    /* ========== Streamlit 1.54 File Uploader: icon-only button ========== */
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] section div p,
    div[data-testid="stFileUploader"] section small,
    div[data-testid="stFileUploader"] section div span {
        display: none !important;
    }

    div[data-testid="stFileUploader"] section {
        border: 0 !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    div[data-testid="stFileUploader"] button {
        width: 100% !important;
        border-radius: 12px !important;
        padding: 0.75rem 0.9rem !important;
        font-weight: 800 !important;
        font-size: 16px !important;
        color: transparent !important;
        position: relative;
    }

    div[data-testid="stFileUploader"] button::after {
        content: "📁 تحميل ملفات";
        color: white;
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        white-space: nowrap;
    }

    div[data-testid="stFileUploader"] ul {
        display: none !important;
    }

    /* Small polish */
    .block-container { padding-top: 1rem; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,0.02); padding: 8px; border-radius: 12px; }
    div[data-testid="stMetricValue"] { font-size: 19px !important; }
    div[data-testid="stMetricLabel"] { font-size: 11px !important; opacity: 0.8; }
    .safeer-subtitle { margin-top: -10px; opacity: 0.85; }

    /* Announcement box */
    .ann-box {
        padding: 10px 12px;
        border-radius: 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 8px;
    }
    .ann-meta {
        font-size: 11px;
        opacity: 0.75;
        margin-top: 4px;
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
st.markdown('<div class="safeer-subtitle">الإدارة / التشغيل / الموارد البشرية / الإشراف / السيارات / الحسابات</div>', unsafe_allow_html=True)
st.divider()

# =========================
# Auth / Roles
# =========================
ROLES = {
    "الإدارة": "admin_password",
    "التشغيل": "ops_password",
    "الموارد البشرية": "hr_password",
    "الإشراف": "sup_password",
    "السيارات / الحركة": "fleet_password",
    "الحسابات": "accounts_password",
}
FALLBACK_PASSWORD = "12345"

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
            ok = (expected and pwd == expected) or ((not expected) and pwd == FALLBACK_PASSWORD)
            if ok:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.rerun()
            else:
                st.error("بيانات الدخول غير صحيحة.")

    if not st.session_state.logged_in:
        st.info("الرجاء تسجيل الدخول من الشريط الجانبي.")
        st.stop()

require_login()
ROLE = st.session_state.role

# =========================
# Sidebar: uploader + filters ONLY (no menu)
# =========================
with st.sidebar:
    st.markdown(f"### المستخدم الحالي: {ROLE}")
    st.divider()

    uploaded_files = st.file_uploader(
        label="تحميل ملفات",
        type=["xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if "excluded_files" not in st.session_state:
        st.session_state.excluded_files = set()

    if uploaded_files:
        with st.expander("📄 الملفات المستخدمة (إلغاء تفعيل ملف)", expanded=False):
            for uf in uploaded_files:
                n = uf.name
                checked = (n not in st.session_state.excluded_files)
                new_checked = st.checkbox(n, value=checked, key=f"use_{n}")
                if new_checked:
                    st.session_state.excluded_files.discard(n)
                else:
                    st.session_state.excluded_files.add(n)

    st.divider()
    search = st.text_input("بحث (المعرف / الاسم)", "")
    min_delivery = st.slider("أقل معدل توصيل", 0.0, 1.0, 0.0, 0.01)
    max_cancel = st.slider("أعلى معدل إلغاء (فلترة)", 0.0, 1.0, 1.0, 0.01)

# =========================
# SQLite
# =========================
def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def db_conn():
    # timeout + WAL helps with Streamlit Cloud + multi-runs
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    con.commit()
    return con

def table_exists(con, name: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def get_table_columns(con, name: str) -> list[str]:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({name})")
    rows = cur.fetchall()
    return [r[1] for r in rows]  # column names

def migrate_announcements_table():
    """
    Fix schema drift:
    - Some DBs have announcements(body NOT NULL)
    - Some have announcements(message)
    We standardize to:
      announcements(id, created_at, created_by_role, body)
    """
    con = db_conn()
    cur = con.cursor()

    if not table_exists(con, "announcements"):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS announcements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                created_by_role TEXT NOT NULL,
                body TEXT NOT NULL
            )
        """)
        con.commit()
        con.close()
        return

    cols = get_table_columns(con, "announcements")
    # Already good
    if "body" in cols and "created_at" in cols and "created_by_role" in cols:
        # Ensure NOT NULL isn't violated by legacy rows (optional safety)
        con.close()
        return

    # Need migration: create new table and copy from old using whichever field exists
    msg_col = None
    if "body" in cols:
        msg_col = "body"
    elif "message" in cols:
        msg_col = "message"
    elif "text" in cols:
        msg_col = "text"

    # Create new table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS announcements_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            created_by_role TEXT NOT NULL,
            body TEXT NOT NULL
        )
    """)

    # Copy data safely if possible
    if msg_col and ("created_at" in cols) and ("created_by_role" in cols):
        cur.execute(f"""
            INSERT INTO announcements_new (created_at, created_by_role, body)
            SELECT
                COALESCE(created_at, ''),
                COALESCE(created_by_role, ''),
                COALESCE({msg_col}, '')
            FROM announcements
        """)
    else:
        # If we can't map, just keep table but make sure new exists
        pass

    # Replace old table
    cur.execute("DROP TABLE IF EXISTS announcements")
    cur.execute("ALTER TABLE announcements_new RENAME TO announcements")

    con.commit()
    con.close()

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

    con.commit()
    con.close()

init_db()
migrate_announcements_table()

def upsert_driver(driver_id: int, driver_name: str = None):
    con = db_conn()
    cur = con.cursor()
    cur.execute("SELECT driver_id, driver_name FROM drivers WHERE driver_id = ?", (int(driver_id),))
    row = cur.fetchone()

    if row is None:
        cur.execute("""
            INSERT INTO drivers (driver_id, driver_name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (int(driver_id), driver_name or "", now_ts(), now_ts()))
    else:
        existing_name = row[1] or ""
        new_name = existing_name if (driver_name is None or str(driver_name).strip() == "") else str(driver_name).strip()
        cur.execute("""
            UPDATE drivers
            SET driver_name=?, updated_at=?
            WHERE driver_id=?
        """, (new_name, now_ts(), int(driver_id)))

    con.commit()
    con.close()

def get_hr_registry() -> pd.DataFrame:
    con = db_conn()
    df = pd.read_sql_query("""
        SELECT
            d.driver_id AS معرف_السائق,
            d.driver_name AS اسم_السائق,
            d.status AS الحالة,
            d.created_at AS تاريخ_الإضافة
        FROM drivers d
        ORDER BY d.driver_id
    """, con)
    con.close()
    return df

def add_announcement(message: str, created_by_role: str):
    migrate_announcements_table()  # ensure schema is correct every time
    msg = (message or "").strip()
    if not msg:
        return

    con = db_conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO announcements (created_at, created_by_role, body) VALUES (?, ?, ?)",
        (now_ts(), str(created_by_role), msg)
    )
    con.commit()
    con.close()

def get_latest_announcements(limit: int = 3) -> pd.DataFrame:
    migrate_announcements_table()
    con = db_conn()
    df = pd.read_sql_query(
        "SELECT created_at, created_by_role, body FROM announcements ORDER BY id DESC LIMIT ?",
        con,
        params=(int(limit),)
    )
    con.close()
    return df

# =========================
# Excel helpers
# =========================
def normalize_col(c: str) -> str:
    return str(c).strip()

@st.cache_data(show_spinner=False)
def read_excel_smart(file_bytes: bytes) -> pd.DataFrame:
    """
    Try header=0 first (normal reports).
    If columns look like data, re-read header=None.
    """
    bio = BytesIO(file_bytes)
    xls = pd.ExcelFile(bio)
    sheet = xls.sheet_names[0]

    bio1 = BytesIO(file_bytes)
    df = pd.read_excel(bio1, sheet_name=sheet, header=0)
    df.columns = [normalize_col(c) for c in df.columns]

    suspicious = False
    if len(df.columns) > 0:
        c0 = str(df.columns[0]).strip()
        if c0.lower() in ["makkah", "riyadh", "jeddah", "medina"] or c0.isdigit():
            suspicious = True
        if any(str(c).startswith("NULL") or str(c).startswith("0.") for c in df.columns[:15]):
            suspicious = True

    if suspicious:
        bio2 = BytesIO(file_bytes)
        df2 = pd.read_excel(bio2, sheet_name=sheet, header=None)
        return df2

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
CANCEL_ALERT_THRESHOLD = 0.002   # 0.20% -> red & priority if >=
ORDERS_TARGET_MONTH = 450

# =========================
# Column mapping (normal performance report)
# =========================
PERF_COLS = {
    "driver_id": ["معرّف السائق", "معرف السائق", "Driver_ID", "driver_id", "id"],
    "first_name": ["اسم السائق", "First Name", "first_name"],
    "last_name": ["اسم السائق.1", "Last Name", "last_name"],
    "delivery_rate": ["معدل اكتمال الطلبات (غير متعلق بالتوصيل)", "معدل التوصيل", "معدل توصيل", "Delivery_Rate", "delivery_rate"],
    "cancel_rate": ["معدل الإلغاء بسبب مشاكل التوصيل", "معدل الغاء", "معدل الإلغاء", "Cancel_Rate", "cancel_rate"],
    "orders_delivered": ["المهام التي تم تسليمها", "طلبات", "الطلبات", "الطلبات المسلمة", "Orders_Delivered", "orders_delivered"],
    "reject_total": ["المهام المرفوضة", "المهام المرفوضة (السائق)", "رفض السائق", "Driver Rejections", "driver_rejections"],
    "work_days": ["اعدد ايام العمل", "عدد ايام العمل", "أيام العمل"],
    "fr": ["FR", "Face Recognition", "Face_Recognition", "التعرف على الوجه"],
    "vda": ["VDA", "vda", "مؤشر VDA"],
}

def _normalize_percent_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    if s.max() > 1.5:
        s = s / 100.0
    return s.clip(0, 1)

def build_performance_report(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Supports:
    - Normal report (Arabic headers)
    - New export (no headers) by positions
    """
    # CASE B: header=None -> integer columns
    if all(isinstance(c, int) for c in df_raw.columns):
        # based on your file:
        # 3: username/name, 6: driver id, 10: rejections, 11: total orders, 12: delivered
        driver_id = safe_to_numeric(df_raw.iloc[:, 6]).astype("Int64")
        driver_name = df_raw.iloc[:, 3].astype(str).str.strip()

        total_orders = safe_to_numeric(df_raw.iloc[:, 11]).fillna(0)
        delivered = safe_to_numeric(df_raw.iloc[:, 12]).fillna(0)
        rejections = safe_to_numeric(df_raw.iloc[:, 10]).fillna(0)

        delivery_rate = (delivered / total_orders.replace(0, pd.NA)).fillna(0).clip(0, 1)
        cancel_rate = ((total_orders - delivered) / total_orders.replace(0, pd.NA)).fillna(0).clip(0, 1)

        out = pd.DataFrame({
            "معرّف السائق": driver_id,
            "اسم السائق": driver_name,
            "معدل توصيل": delivery_rate,
            "معدل الغاء": cancel_rate,
            "طلبات": delivered,
            "المهام المرفوضة": rejections,
            "اعدد ايام العمل": pd.NA,
            "FR": pd.NA,
            "VDA": pd.NA,
        })
        return out

    # CASE A: normal headers
    mapped = {k: pick(df_raw.columns, v) for k, v in PERF_COLS.items()}
    required = ["driver_id", "first_name", "last_name", "delivery_rate", "cancel_rate", "orders_delivered", "reject_total"]
    missing = [k for k in required if not mapped.get(k)]
    if missing:
        st.error("❌ ملف الأداء غير مطابق. الأعمدة المطلوبة غير موجودة: " + ", ".join(missing))
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
    out["معدل توصيل"] = _normalize_percent_series(out["معدل توصيل"])
    out["معدل الغاء"] = _normalize_percent_series(out["معدل الغاء"])

    out["طلبات"] = pd.to_numeric(out["طلبات"], errors="coerce").fillna(0)
    out["المهام المرفوضة"] = pd.to_numeric(out["المهام المرفوضة"], errors="coerce").fillna(0)

    if mapped.get("work_days") and mapped["work_days"] in df_raw.columns:
        out["اعدد ايام العمل"] = safe_to_numeric(df_raw[mapped["work_days"]]).fillna(0)
    else:
        out["اعدد ايام العمل"] = pd.NA

    if mapped.get("fr") and mapped["fr"] in df_raw.columns:
        out["FR"] = safe_to_numeric(df_raw[mapped["fr"]]).fillna(0)
    else:
        out["FR"] = pd.NA

    if mapped.get("vda") and mapped["vda"] in df_raw.columns:
        out["VDA"] = safe_to_numeric(df_raw[mapped["vda"]]).fillna(0)
    else:
        out["VDA"] = pd.NA

    return out

def detect_file_type(df: pd.DataFrame) -> str:
    if all(isinstance(c, int) for c in df.columns):
        return "performance"
    perf_signals = {"معرّف السائق", "اسم السائق", "اسم السائق.1", "معدل الغاء", "معدل توصيل", "طلبات", "المهام المرفوضة"}
    cols = set(map(str, df.columns))
    return "performance" if len(perf_signals.intersection(cols)) >= 3 else "unknown"

# =========================
# Styling (Attention table)
# =========================
def style_attention_table(df):
    sty = df.style.format({
        "معدل توصيل": "{:.2%}",
        "معدل الغاء": "{:.2%}",
        "طلبات": "{:,.0f}",
        "المهام المرفوضة": "{:,.0f}",
    })

    sty = sty.applymap(lambda x: "color:red;font-weight:900;" if float(x) >= CANCEL_ALERT_THRESHOLD else "", subset=["معدل الغاء"])
    sty = sty.applymap(lambda x: "color:red;font-weight:900;" if float(x) < 1.0 else "", subset=["معدل توصيل"])
    sty = sty.applymap(lambda x: "color:red;font-weight:900;" if float(x) < ORDERS_TARGET_MONTH else "", subset=["طلبات"])
    sty = sty.applymap(lambda x: "color:red;font-weight:900;" if float(x) > 0 else "", subset=["المهام المرفوضة"])
    return sty

# =========================
# Build master from uploads
# =========================
def build_master_from_uploads():
    if not uploaded_files:
        return None

    effective_files = [f for f in uploaded_files if f.name not in st.session_state.excluded_files]
    if not effective_files:
        return None

    file_items = []
    for uf in effective_files:
        b = uf.getvalue()
        df = read_excel_smart(b)
        kind = detect_file_type(df)
        file_items.append({"name": uf.name, "df": df, "kind": kind})

    perf_item = next((x for x in file_items if x["kind"] == "performance"), None)
    if perf_item is None:
        perf_item = file_items[0]

    perf = build_performance_report(perf_item["df"])

    # Save driver names to DB
    for _, r in perf.iterrows():
        did = r.get("معرّف السائق")
        name = r.get("اسم السائق")
        if pd.isna(did):
            continue
        upsert_driver(int(did), driver_name=str(name).strip())

    f = perf.copy()

    # Filters
    if search.strip():
        s = search.strip().lower()
        f = f[
            f["اسم السائق"].astype(str).str.lower().str.contains(s, na=False)
            | f["معرّف السائق"].astype(str).str.contains(s, na=False)
        ]

    f = f[(f["معدل توصيل"] >= min_delivery)]
    f = f[(f["معدل الغاء"] <= max_cancel)]

    # Alerts (cancel is BAD if HIGH)
    f["تنبيه الغاء"] = (f["معدل الغاء"] >= CANCEL_ALERT_THRESHOLD).astype(int)
    f["تنبيه توصيل"] = (f["معدل توصيل"] < 1.0).astype(int)
    f["تنبيه طلبات"] = (pd.to_numeric(f["طلبات"], errors="coerce").fillna(0) < ORDERS_TARGET_MONTH).astype(int)
    f["تنبيه رفض"] = (pd.to_numeric(f["المهام المرفوضة"], errors="coerce").fillna(0) > 0).astype(int)

    delivery_gap = (1.0 - f["معدل توصيل"]).clip(lower=0)
    cancel_over = (f["معدل الغاء"] - CANCEL_ALERT_THRESHOLD).clip(lower=0)
    orders_gap = (ORDERS_TARGET_MONTH - pd.to_numeric(f["طلبات"], errors="coerce").fillna(0)).clip(lower=0)
    rejects = pd.to_numeric(f["المهام المرفوضة"], errors="coerce").fillna(0)

    # Priority: cancel dominates (>=0.20% must go red and go up)
    f["أولوية"] = (
        f["تنبيه الغاء"] * 1_000_000
        + cancel_over * 500_000
        + delivery_gap * 50_000
        + f["تنبيه طلبات"] * 10_000
        + orders_gap * 5
        + rejects * 200
    )

    f = f.sort_values(
        ["أولوية", "تنبيه الغاء", "معدل الغاء", "معدل توصيل", "طلبات", "المهام المرفوضة"],
        ascending=[False, False, False, True, True, False]
    ).reset_index(drop=True)

    f["ترتيب المتابعة"] = range(1, len(f) + 1)
    return f

# =========================
# Announcements (all users)
# =========================
def announcements_block():
    try:
        ann = get_latest_announcements(limit=3)
    except Exception:
        ann = pd.DataFrame()

    if len(ann):
        with st.expander("📢 الإعلانات", expanded=True):
            for _, r in ann.iterrows():
                st.markdown(
                    f"""
                    <div class="ann-box">
                        <div>{r.get("body","")}</div>
                        <div class="ann-meta">{r.get("created_at","")} — {r.get("created_by_role","")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.divider()

    with st.sidebar:
        st.divider()
        st.markdown("### 📢 إرسال إعلان")
        with st.form("announcement_form", clear_on_submit=True):
            msg = st.text_area(
                "اكتب الإعلان",
                height=80,
                label_visibility="collapsed",
                placeholder="اكتب إعلاناً قصيراً..."
            )
            submitted = st.form_submit_button("إرسال")
            if submitted:
                add_announcement(msg, ROLE)
                st.success("تم إرسال الإعلان ✅")
                st.rerun()

# =========================
# Pages
# =========================
def page_admin(f: pd.DataFrame | None):
    st.subheader("📊 الإدارة — نظرة (يومي / شهري)")
    if f is None:
        st.info("قم بتحميل ملف/ملفات الأداء لعرض مؤشرات الإدارة.")
        return

    total_drivers = int(f["معرّف السائق"].nunique())
    total_orders = int(pd.to_numeric(f["طلبات"], errors="coerce").fillna(0).sum())
    avg_delivery = float(f["معدل توصيل"].mean()) if len(f) else 0
    avg_cancel = float(f["معدل الغاء"].mean()) if len(f) else 0

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("إجمالي السائقين", f"{total_drivers:,}")
    a2.metric("إجمالي الطلبات المكتملة", f"{total_orders:,}")
    a3.metric("متوسط معدل التوصيل", f"{avg_delivery:.2%}")
    a4.metric("متوسط معدل الإلغاء", f"{avg_cancel:.2%}")

    st.divider()
    st.markdown("### 🚨 الأولوية")
    cols = ["ترتيب المتابعة", "معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة"]
    st.dataframe(style_attention_table(f[cols].head(25)), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📈 توزيع معدل الإلغاء")
    fig = px.histogram(f, x="معدل الغاء", nbins=30, title="توزيع معدل الإلغاء (معدل الغاء)")
    st.plotly_chart(fig, use_container_width=True)

def page_ops(f: pd.DataFrame | None):
    st.subheader("🚚 التشغيل")
    if f is None:
        st.info("ارفع ملف/ملفات للبدء.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("عدد السائقين", f"{len(f):,}")
    k2.metric("متوسط معدل التوصيل", f"{f['معدل توصيل'].mean():.2%}" if len(f) else "—")
    k3.metric("متوسط معدل الإلغاء", f"{f['معدل الغاء'].mean():.2%}" if len(f) else "—")
    k4.metric("عدد الطلبات", f"{int(pd.to_numeric(f['طلبات'], errors='coerce').fillna(0).sum()):,}" if len(f) else "—")

    st.divider()
    st.subheader("🚨 سائقون يحتاجون متابعة (الأولوية أولاً)")
    attention_cols = ["ترتيب المتابعة", "معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة"]
    st.dataframe(style_attention_table(f[attention_cols].head(60)), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔎 بحث سريع عن سائق")
    driver_list = f["اسم السائق"].dropna().unique().tolist()
    selected = st.selectbox("اختر السائق", ["(اختر)"] + driver_list, key="lookup_driver")

    if selected != "(اختر)":
        d = f[f["اسم السائق"] == selected].head(1).iloc[0]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("معدل توصيل %", f"{float(d['معدل توصيل']):.2%}")
        c2.metric("طلبات", f"{int(pd.to_numeric(d['طلبات'], errors='coerce') or 0):,}")
        c3.metric("معدل الغاء %", f"{float(d['معدل الغاء']):.2%}")
        wd = d.get("اعدد ايام العمل")
        c4.metric("اعدد ايام العمل", "—" if pd.isna(wd) else f"{int(float(wd)):,}")
        fr = d.get("FR")
        c5.metric("FR", "—" if pd.isna(fr) else f"{int(float(fr)):,}")
        vda = d.get("VDA")
        c6.metric("VDA", "—" if pd.isna(vda) else f"{int(float(vda)):,}")

    st.divider()
    st.subheader("📋 الجدول النهائي (كامل البيانات)")
    bottom_cols = ["معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة", "اعدد ايام العمل"]
    if "FR" in f.columns:
        bottom_cols.append("FR")
    if "VDA" in f.columns:
        bottom_cols.append("VDA")

    st.dataframe(
        f[bottom_cols].style.format({
            "معدل توصيل": "{:.2%}",
            "معدل الغاء": "{:.2%}",
            "طلبات": "{:,.0f}",
            "المهام المرفوضة": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ تحميل النتائج CSV",
        data=f.to_csv(index=False, encoding="utf-8-sig"),
        file_name="safeer_master_filtered.csv",
        mime="text/csv",
    )

def page_hr():
    st.subheader("👥 الموارد البشرية — سجل السائقين (دائم)")
    registry = get_hr_registry()
    st.dataframe(registry, use_container_width=True, hide_index=True)

def page_supervision(f: pd.DataFrame | None):
    st.subheader("🧭 الإشراف")
    if f is None:
        st.info("ارفع ملف/ملفات للبدء.")
        return
    st.markdown("### 🚨 سائقون يحتاجون متابعة (الأولوية أولاً)")
    cols = ["ترتيب المتابعة", "معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة"]
    st.dataframe(style_attention_table(f[cols].head(80)), use_container_width=True, hide_index=True)

def page_fleet():
    st.subheader("🚗 السيارات / الحركة")
    st.info("جاهز — عند تزويدي بملف السيارات/الحركة (الأعمدة) سأربطه هنا مع الإدارة والتشغيل.")

def page_accounts():
    st.subheader("💰 الحسابات")
    st.info("جاهز — عند تزويدي بملف الحسابات (الأعمدة) سأربطه هنا مع الإدارة والتشغيل.")

# =========================
# Render
# =========================
announcements_block()
f = build_master_from_uploads()

if ROLE == "الإدارة":
    page_admin(f)
elif ROLE == "التشغيل":
    page_ops(f)
elif ROLE == "الموارد البشرية":
    page_hr()
elif ROLE == "الإشراف":
    page_supervision(f)
elif ROLE == "السيارات / الحركة":
    page_fleet()
elif ROLE == "الحسابات":
    page_accounts()
else:
    st.info("الدور غير معروف.")
