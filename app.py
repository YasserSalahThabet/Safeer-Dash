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

    /* Low-key file badge style */
    .file-pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.05);
        margin: 4px 6px 0 0;
        font-size: 12px;
        opacity: 0.9;
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
st.markdown(
    '<div class="safeer-subtitle">الإدارة / التشغيل / الموارد البشرية / الإشراف / السيارات / الحسابات</div>',
    unsafe_allow_html=True
)
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

DEFAULT_PASSWORD_ALL = "12345"


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
            expected = get_secret(ROLES[role], default="").strip()

            # ✅ Fix: if secrets not configured, accept DEFAULT_PASSWORD_ALL
            if (expected and pwd == expected) or (not expected and pwd == DEFAULT_PASSWORD_ALL):
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

    # low-key viewer (and ignore list)
    if "ignored_files" not in st.session_state:
        st.session_state.ignored_files = set()

    if uploaded_files:
        with st.expander("📎 الملفات المرفوعة (إدارة بسيطة)", expanded=False):
            st.caption("حدد الملفات التي تريد تجاهلها (لن تُستخدم في التحليل).")
            for f in uploaded_files:
                colA, colB = st.columns([0.8, 0.2])
                with colA:
                    st.markdown(f"<span class='file-pill'>📄 {f.name}</span>", unsafe_allow_html=True)
                with colB:
                    ignore = st.checkbox(
                        "تجاهل",
                        key=f"ignore_{f.name}",
                        value=(f.name in st.session_state.ignored_files),
                    )
                    if ignore:
                        st.session_state.ignored_files.add(f.name)
                    else:
                        st.session_state.ignored_files.discard(f.name)

    st.divider()
    search = st.text_input("بحث (المعرف / الاسم)", "")
    min_delivery = st.slider("أقل معدل توصيل", 0.0, 1.0, 0.0, 0.01)
    max_cancel = st.slider("أعلى معدل إلغاء (فلترة)", 0.0, 1.0, 1.0, 0.01)

    st.divider()
    st.markdown("### 📣 إعلان")
    with st.expander("إرسال إعلان (للجميع)", expanded=False):
        ann_text = st.text_area("نص الإعلان", height=80, placeholder="اكتب إعلان مختصر…")
        send_ann = st.button("إرسال الإعلان", use_container_width=True)


# =========================
# SQLite helpers
# =========================
def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def table_exists(con, table_name: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def table_columns(con, table_name: str) -> set[str]:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table_name});")
    rows = cur.fetchall()
    return {r[1] for r in rows}  # column name is index 1


# =========================
# DB init + Safe migrations
# =========================
def init_db():
    con = db_conn()
    cur = con.cursor()

    # Drivers registry
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

    # Announcements: support both schemas safely
    if not table_exists(con, "announcements"):
        # create new simple schema
        cur.execute("""
        CREATE TABLE IF NOT EXISTS announcements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            created_by_role TEXT,
            message TEXT
        )
        """)
    else:
        cols = table_columns(con, "announcements")

        # If existing table is old schema (title/body/is_active), create a compatible view-table
        # Strategy: if 'message' missing but 'body' exists -> add 'message' + backfill
        if "message" not in cols and "body" in cols:
            try:
                cur.execute("ALTER TABLE announcements ADD COLUMN message TEXT;")
            except Exception:
                pass
            # backfill message from body
            try:
                cur.execute("UPDATE announcements SET message = body WHERE message IS NULL;")
            except Exception:
                pass

        # If 'created_by_role' missing but 'created_by' exists, add and backfill
        cols = table_columns(con, "announcements")
        if "created_by_role" not in cols and "created_by" in cols:
            try:
                cur.execute("ALTER TABLE announcements ADD COLUMN created_by_role TEXT;")
            except Exception:
                pass
            try:
                cur.execute("UPDATE announcements SET created_by_role = created_by WHERE created_by_role IS NULL;")
            except Exception:
                pass

        # If 'created_at' missing but 'uploaded_at' exists, add and backfill
        cols = table_columns(con, "announcements")
        if "created_at" not in cols and "uploaded_at" in cols:
            try:
                cur.execute("ALTER TABLE announcements ADD COLUMN created_at TEXT;")
            except Exception:
                pass
            try:
                cur.execute("UPDATE announcements SET created_at = uploaded_at WHERE created_at IS NULL;")
            except Exception:
                pass

        # If is_active exists, keep it; we will filter if present

    con.commit()
    con.close()


init_db()


# =========================
# DB operations
# =========================
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
            COALESCE(d.status,'نشط') AS الحالة,
            d.created_at AS تاريخ_الإضافة
        FROM drivers d
        ORDER BY d.driver_id
    """, con)
    con.close()
    return df


def add_announcement(message: str, created_by_role: str):
    message = (message or "").strip()
    if not message:
        return

    con = db_conn()
    cur = con.cursor()
    cols = table_columns(con, "announcements")

    # Choose best insert based on columns available
    if {"created_at", "created_by_role", "message"}.issubset(cols):
        cur.execute(
            "INSERT INTO announcements (created_at, created_by_role, message) VALUES (?, ?, ?)",
            (now_ts(), created_by_role, message)
        )
    elif {"created_at", "created_by_role", "body"}.issubset(cols):
        cur.execute(
            "INSERT INTO announcements (created_at, created_by_role, body) VALUES (?, ?, ?)",
            (now_ts(), created_by_role, message)
        )
    elif {"created_at", "created_by", "body"}.issubset(cols):
        cur.execute(
            "INSERT INTO announcements (created_at, created_by, body) VALUES (?, ?, ?)",
            (now_ts(), created_by_role, message)
        )
    else:
        # last-resort: try generic
        try:
            cur.execute(
                "INSERT INTO announcements (created_at, created_by_role, message) VALUES (?, ?, ?)",
                (now_ts(), created_by_role, message)
            )
        except Exception:
            # if table is too different, avoid crashing
            pass

    con.commit()
    con.close()


def get_latest_announcements(limit: int = 3) -> pd.DataFrame:
    """
    Defensive reader: supports both schemas and avoids crashing app.
    """
    con = db_conn()
    try:
        cols = table_columns(con, "announcements")

        where = ""
        if "is_active" in cols:
            where = "WHERE is_active = 1"

        if {"created_at", "created_by_role", "message"}.issubset(cols):
            q = f"SELECT created_at, created_by_role, message FROM announcements {where} ORDER BY id DESC LIMIT ?"
            df = pd.read_sql_query(q, con, params=(int(limit),))
            return df

        if {"created_at", "created_by_role", "body"}.issubset(cols):
            q = f"SELECT created_at, created_by_role, body as message FROM announcements {where} ORDER BY id DESC LIMIT ?"
            df = pd.read_sql_query(q, con, params=(int(limit),))
            return df

        if {"created_at", "created_by", "body"}.issubset(cols):
            q = f"SELECT created_at, created_by as created_by_role, body as message FROM announcements {where} ORDER BY id DESC LIMIT ?"
            df = pd.read_sql_query(q, con, params=(int(limit),))
            return df

        # fallback: empty
        return pd.DataFrame(columns=["created_at", "created_by_role", "message"])

    except Exception:
        return pd.DataFrame(columns=["created_at", "created_by_role", "message"])
    finally:
        con.close()


# Handle announcements send (from sidebar)
if 'send_ann' in locals() and send_ann:
    add_announcement(ann_text, ROLE)
    st.sidebar.success("تم إرسال الإعلان ✅")
    st.rerun()


# Show announcements on top for ALL users (low-key)
ann = get_latest_announcements(limit=3)
if ann is not None and len(ann):
    with st.container():
        st.markdown("### 📣 الإعلانات")
        for _, r in ann.iterrows():
            created_at = str(r.get("created_at", "") or "")
            by_role = str(r.get("created_by_role", "") or "")
            msg = str(r.get("message", "") or "")
            st.info(f"**{created_at}** — ({by_role})\n\n{msg}")
        st.divider()


# =========================
# Excel helpers
# =========================
def normalize_col(c: str) -> str:
    return str(c).strip()


@st.cache_data(show_spinner=False)
def read_excel_smart(file_bytes: bytes) -> pd.DataFrame:
    """
    Reads first sheet.
    If headers are broken / missing, tries header=None and returns raw columns by index.
    """
    bio = BytesIO(file_bytes)
    xls = pd.ExcelFile(bio)
    sheet = xls.sheet_names[0]

    # First try normal header
    df = pd.read_excel(bio, sheet_name=sheet, header=0)
    df.columns = [normalize_col(c) for c in df.columns]

    # Heuristic: if many column names look like numbers / floats / NULL, treat as no-header format
    bad = 0
    for c in df.columns:
        s = str(c).strip().lower()
        if s in ("null", "nan") or s.startswith("null"):
            bad += 1
        if s.replace(".", "", 1).isdigit():
            bad += 1

    if bad >= max(3, int(len(df.columns) * 0.25)):
        bio2 = BytesIO(file_bytes)
        df2 = pd.read_excel(bio2, sheet_name=sheet, header=None)
        df2.columns = [f"col_{i}" for i in range(df2.shape[1])]
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
CANCEL_ALERT_THRESHOLD = 0.002   # 0.20% (alert if >=)
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
    "orders_delivered": ["المهام التي تم تسليمها", "طلبات", "الطلبات", "الطلبات المسلمة", "Orders_Delivered", "orders_delivered"],
    "reject_total": ["المهام المرفوضة", "المهام المرفوضة (السائق)", "رفض السائق", "Driver Rejections", "driver_rejections"],
    "work_days": ["اعدد ايام العمل", "عدد ايام العمل", "أيام العمل"],
    "fr": ["FR", "Face Recognition", "Face_Recognition", "التعرف على الوجه"],
    "vda": ["VDA", "vda", "مؤشر VDA"],
}


def build_performance_report(df_raw: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    """
    Supports:
      1) Normal Arabic-header template (your usual file)
      2) Headerless export format -> map by column position
    """

    # Case A: Normal header template
    mapped = {k: pick(df_raw.columns, v) for k, v in PERF_COLS.items()}
    required = ["driver_id", "first_name", "last_name", "delivery_rate", "cancel_rate", "orders_delivered", "reject_total"]
    missing = [k for k in required if not mapped.get(k)]

    if not missing:
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

    # Case B: Headerless format
    if all(str(c).startswith("col_") for c in df_raw.columns) and df_raw.shape[1] >= 13:
        out = pd.DataFrame({
            "معرّف السائق": safe_to_numeric(df_raw["col_6"]),
            "اسم السائق": df_raw["col_3"].astype(str).fillna("").str.strip(),
            "معدل توصيل": 1.0,                 # not available in this export
            "معدل الغاء": 0.0,                 # not available in this export
            "طلبات": safe_to_numeric(df_raw["col_12"]),
            "المهام المرفوضة": safe_to_numeric(df_raw["col_10"]),
            "اعدد ايام العمل": safe_to_numeric(df_raw["col_11"]),
            "FR": pd.NA,
            "VDA": pd.NA,
        })

        out["معرّف السائق"] = pd.to_numeric(out["معرّف السائق"], errors="coerce").astype("Int64")
        out["طلبات"] = out["طلبات"].fillna(0)
        out["المهام المرفوضة"] = out["المهام المرفوضة"].fillna(0)
        out["اعدد ايام العمل"] = out["اعدد ايام العمل"].fillna(0)
        out["معدل توصيل"] = pd.to_numeric(out["معدل توصيل"], errors="coerce").fillna(1.0).clip(0, 1)
        out["معدل الغاء"] = pd.to_numeric(out["معدل الغاء"], errors="coerce").fillna(0.0).clip(0, 1)

        st.warning("⚠️ تم تحميل ملف بصيغة بدون عناوين. معدل الإلغاء/التوصيل غير موجود في هذا الملف، وتم تعبئته بقيم افتراضية.")
        return out

    st.error("❌ لم أستطع قراءة هذا الملف كملف أداء.")
    st.stop()


def detect_file_type(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    perf_signals = {"معرّف السائق", "اسم السائق", "اسم السائق.1", "معدل الغاء", "معدل توصيل", "طلبات", "المهام المرفوضة"}
    if len(perf_signals.intersection(cols)) >= 3:
        return "performance"
    if all(str(c).startswith("col_") for c in cols) and df.shape[1] >= 13:
        return "performance"
    return "unknown"


# =========================
# Styling (Attention table)
# =========================
def style_attention_table(df):
    sty = df.style.format({"معدل توصيل": "{:.2%}", "معدل الغاء": "{:.2%}"})
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

    active_files = [f for f in uploaded_files if f.name not in st.session_state.ignored_files]
    if not active_files:
        st.info("كل الملفات الحالية تم تجاهلها. قم بإلغاء تجاهل ملف واحد على الأقل.")
        return None

    file_items = []
    for uf in active_files:
        b = uf.getvalue()
        df = read_excel_smart(b)
        kind = detect_file_type(df)
        file_items.append({"name": uf.name, "df": df, "kind": kind})

    # pick first performance-like file
    perf_item = None
    for item in file_items:
        if item["kind"] == "performance":
            perf_item = item
            break
    if perf_item is None:
        perf_item = file_items[0]

    perf = build_performance_report(perf_item["df"], source_name=perf_item["name"])

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

    # Alerts: cancel is bad if HIGH
    f["تنبيه الغاء"] = (f["معدل الغاء"] >= CANCEL_ALERT_THRESHOLD).astype(int)
    f["تنبيه توصيل"] = (f["معدل توصيل"] < 1.0).astype(int)
    f["تنبيه طلبات"] = (f["طلبات"] < ORDERS_TARGET_MONTH).astype(int)
    f["تنبيه رفض"] = (f["المهام المرفوضة"] > 0).astype(int)

    delivery_gap = (1.0 - f["معدل توصيل"]).clip(lower=0)
    cancel_over = (f["معدل الغاء"] - CANCEL_ALERT_THRESHOLD).clip(lower=0)
    orders_gap = (ORDERS_TARGET_MONTH - f["طلبات"]).clip(lower=0)

    # Priority: cancel dominates
    f["أولوية"] = (
        f["تنبيه الغاء"] * 1_000_000
        + cancel_over * 500_000
        + delivery_gap * 50_000
        + f["تنبيه طلبات"] * 10_000
        + orders_gap * 5
        + f["المهام المرفوضة"] * 200
    )

    f = f.sort_values(
        ["أولوية", "تنبيه الغاء", "معدل الغاء", "معدل توصيل", "طلبات", "المهام المرفوضة"],
        ascending=[False, False, False, True, True, False]
    ).reset_index(drop=True)

    f["ترتيب المتابعة"] = range(1, len(f) + 1)
    return f


# =========================
# Pages (by ROLE only)
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
    st.subheader("🚨 الأولوية")
    attention_cols = ["ترتيب المتابعة", "معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة"]
    st.dataframe(style_attention_table(f[attention_cols].head(60)), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔎 بحث سريع")
    driver_list = f["اسم السائق"].dropna().unique().tolist()
    selected = st.selectbox("اختر السائق", ["(اختر)"] + driver_list, key="lookup_driver")

    if selected != "(اختر)":
        d = f[f["اسم السائق"] == selected].head(1).iloc[0]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("معدل توصيل %", f"{float(d['معدل توصيل']):.2%}")
        c2.metric("طلبات", f"{int(float(pd.to_numeric(d['طلبات'], errors='coerce') or 0)):,}")
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


def page_hr():
    st.subheader("👥 الموارد البشرية — سجل السائقين (دائم)")
    registry = get_hr_registry()
    st.dataframe(registry, use_container_width=True, hide_index=True)


def page_supervision(f: pd.DataFrame | None):
    st.subheader("🧭 الإشراف")
    if f is None:
        st.info("ارفع ملف/ملفات للبدء.")
        return
    st.markdown("### 🚨 سائقون يحتاجون متابعة")
    cols = ["ترتيب المتابعة", "معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة"]
    st.dataframe(style_attention_table(f[cols].head(80)), use_container_width=True, hide_index=True)


def page_fleet():
    st.subheader("🚗 السيارات / الحركة")
    st.info("جاهز — عند تزويدي بملف السيارات/الحركة (الأعمدة) سأربطه هنا مع الإدارة والتشغيل.")


def page_accounts():
    st.subheader("💰 الحسابات")
    st.info("جاهز — عند تزويدي بملف الحسابات (الأعمدة) سأربطه هنا مع الإدارة والتشغيل.")


# =========================
# Render (role decides)
# =========================
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
