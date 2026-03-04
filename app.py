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

LEFT_IMG = ASSETS / "left.jpg"      # left banner
RIGHT_IMG = ASSETS / "right.jpg"    # right banner
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

    /* Hide file list under uploader */
    div[data-testid="stFileUploader"] ul {
        display: none !important;
    }

    /* Small polish */
    .block-container { padding-top: 1rem; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        padding: 8px;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] { font-size: 19px !important; }
    div[data-testid="stMetricLabel"] { font-size: 11px !important; opacity: 0.8; }
    .safeer-subtitle { margin-top: -10px; opacity: 0.85; }
    .muted { opacity: 0.75; font-size: 12px; }
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
    '<div class="safeer-subtitle">الإدارة / التشغيل / الموارد البشرية / الإشراف / السيارات / الحسابات / مسير الرواتب</div>',
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
    "مسير الرواتب": "payroll_password",
}

def get_secret(key: str, default: str = "12345") -> str:
    """
    Fix password issues: if secrets not set, fallback to 12345 (your requirement).
    """
    try:
        return str(st.secrets["auth"][key])
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
            expected = get_secret(ROLES[role], "12345")
            if str(pwd) == str(expected):
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

    # Low-key list of uploaded files + ability to ignore (remove from reading)
    active_names = None
    if uploaded_files:
        with st.expander("📄 الملفات المستخدمة", expanded=False):
            names = [f.name for f in uploaded_files]
            default = names
            active_names = st.multiselect(
                "اختر الملفات التي تريد أن يقرأ منها النظام",
                options=names,
                default=default,
                label_visibility="collapsed"
            )
            st.markdown('<div class="muted">قم بإزالة علامة الملف غير المطلوب.</div>', unsafe_allow_html=True)

    st.divider()
    search = st.text_input("بحث (المعرف / الاسم)", "")
    min_delivery = st.slider("أقل معدل توصيل", 0.0, 1.0, 0.0, 0.01)
    max_cancel = st.slider("أعلى معدل إلغاء (فلترة)", 0.0, 1.0, 1.0, 0.01)

# =========================
# SQLite
# =========================
def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def init_db():
    con = db_conn()
    cur = con.cursor()

    # drivers registry (minimal, stable)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS drivers (
        driver_id INTEGER PRIMARY KEY,
        driver_name TEXT,
        status TEXT DEFAULT 'نشط',
        created_at TEXT,
        updated_at TEXT
    )
    """)

    # announcements: handle old schema problems (message/body)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS announcements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        created_by_role TEXT NOT NULL,
        message TEXT,
        body TEXT
    )
    """)

    con.commit()
    con.close()

init_db()

def ensure_announcements_columns():
    """
    Fix NOT NULL / missing column issues caused by older table schemas.
    We keep both 'message' and 'body' and always write to both.
    """
    con = db_conn()
    cur = con.cursor()
    cols = cur.execute("PRAGMA table_info(announcements)").fetchall()
    col_names = {c[1] for c in cols}

    if "message" not in col_names:
        try:
            cur.execute("ALTER TABLE announcements ADD COLUMN message TEXT")
        except Exception:
            pass

    if "body" not in col_names:
        try:
            cur.execute("ALTER TABLE announcements ADD COLUMN body TEXT")
        except Exception:
            pass

    con.commit()
    con.close()

ensure_announcements_columns()

def add_announcement(msg: str, created_by_role: str):
    msg = (msg or "").strip()
    if not msg:
        return

    ensure_announcements_columns()
    con = db_conn()
    cur = con.cursor()

    # write to BOTH columns to satisfy any NOT NULL constraints on older deployments
    cur.execute(
        "INSERT INTO announcements (created_at, created_by_role, message, body) VALUES (?, ?, ?, ?)",
        (now_ts(), str(created_by_role), msg, msg)
    )

    con.commit()
    con.close()

def get_latest_announcements(limit: int = 3) -> pd.DataFrame:
    ensure_announcements_columns()
    con = db_conn()
    # coalesce in case one of them is used
    df = pd.read_sql_query(
        """
        SELECT
            created_at,
            created_by_role,
            COALESCE(message, body) AS message
        FROM announcements
        ORDER BY id DESC
        LIMIT ?
        """,
        con,
        params=(int(limit),)
    )
    con.close()
    return df

def upsert_driver(driver_id: int, driver_name: str = None):
    con = db_conn()
    cur = con.cursor()
    cur.execute("SELECT driver_id, driver_name FROM drivers WHERE driver_id = ?", (int(driver_id),))
    row = cur.fetchone()

    if row is None:
        cur.execute(
            "INSERT INTO drivers (driver_id, driver_name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (int(driver_id), (driver_name or "").strip(), now_ts(), now_ts())
        )
    else:
        existing_name = (row[1] or "").strip()
        new_name = existing_name
        if driver_name is not None and str(driver_name).strip():
            new_name = str(driver_name).strip()
        cur.execute(
            "UPDATE drivers SET driver_name=?, updated_at=? WHERE driver_id=?",
            (new_name, now_ts(), int(driver_id))
        )

    con.commit()
    con.close()

def get_hr_registry() -> pd.DataFrame:
    con = db_conn()
    df = pd.read_sql_query(
        """
        SELECT
            driver_id AS معرف_السائق,
            driver_name AS اسم_السائق,
            status AS الحالة,
            created_at AS تاريخ_الإضافة
        FROM drivers
        ORDER BY driver_id
        """,
        con
    )
    con.close()
    return df

# =========================
# Announcements UI (low key, for all users)
# =========================
with st.sidebar:
    st.divider()
    with st.expander("📢 إعلان", expanded=False):
        ann_df = get_latest_announcements(limit=3)
        if len(ann_df):
            for _, r in ann_df.iterrows():
                st.caption(f"{r['created_at']} — {r['created_by_role']}")
                st.write(r["message"])
                st.markdown("---")
        else:
            st.caption("لا توجد إعلانات بعد.")

        ann_text = st.text_area("إرسال إعلان", placeholder="اكتب الإعلان هنا...", label_visibility="collapsed", height=80)
        if st.button("إرسال", use_container_width=True):
            add_announcement(ann_text, ROLE)
            st.rerun()

# =========================
# Excel helpers
# =========================
def normalize_col(c) -> str:
    return str(c).strip()

@st.cache_data(show_spinner=False)
def read_excel_smart(file_bytes: bytes) -> pd.DataFrame:
    """
    Reads first sheet.
    If headers look broken (like values), we also keep header=None version available later.
    """
    bio = BytesIO(file_bytes)
    xls = pd.ExcelFile(bio)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(bio, sheet_name=sheet, header=0)
    df.columns = [normalize_col(c) for c in df.columns]
    return df

def read_excel_no_header(file_bytes: bytes) -> pd.DataFrame:
    bio = BytesIO(file_bytes)
    xls = pd.ExcelFile(bio)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(bio, sheet_name=sheet, header=None)
    return df

def safe_to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def pick(df_cols, candidates):
    cols = [normalize_col(c) for c in df_cols]
    for cand in candidates:
        if cand in cols:
            return cand
    return None

def clean_name(first: str, last: str) -> str:
    """
    Fix: ALWAYS ensure a space between first and last.
    Also collapses multiple spaces.
    """
    f = ("" if first is None else str(first)).strip()
    l = ("" if last is None else str(last)).strip()
    full = f"{f} {l}".strip()
    full = " ".join(full.split())
    return full

def looks_like_bad_headers(cols: list[str]) -> bool:
    """
    Heuristic: if many 'columns' are numeric-like or NULL-like, it's not real headers.
    """
    if not cols:
        return True
    bad = 0
    for c in cols[:30]:
        s = str(c).strip()
        if s.upper().startswith("NULL"):
            bad += 1
            continue
        # numeric-like headers
        try:
            float(s)
            bad += 1
        except Exception:
            pass
        # very long integer IDs
        if s.isdigit() and len(s) >= 10:
            bad += 1
    return (bad / min(len(cols), 30)) >= 0.5

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

    "work_days": ["اعدد ايام العمل", "عدد ايام العمل", "أيام العمل", "Work Days", "work_days", "days_worked"],

    "fr": ["FR", "Face Recognition", "Face_Recognition", "التعرف على الوجه", "face_recognition"],
    "vda": ["VDA", "vda", "مؤشر VDA", "مؤشر_إضافي"],
}

def build_performance_report(df_raw: pd.DataFrame) -> pd.DataFrame:
    mapped = {k: pick(df_raw.columns, v) for k, v in PERF_COLS.items()}

    required = ["driver_id", "first_name", "last_name", "delivery_rate", "cancel_rate", "orders_delivered", "reject_total"]
    missing = [k for k in required if not mapped.get(k)]
    if missing:
        st.error("❌ ملف الأداء غير مطابق. الأعمدة المطلوبة غير موجودة: " + ", ".join(missing))
        st.write("الأعمدة الموجودة في الملف:")
        st.write(list(df_raw.columns))
        st.stop()

    driver_name = [
        clean_name(fn, ln)
        for fn, ln in zip(df_raw[mapped["first_name"]], df_raw[mapped["last_name"]])
    ]

    out = pd.DataFrame({
        "معرّف السائق": safe_to_numeric(df_raw[mapped["driver_id"]]),
        "اسم السائق": pd.Series(driver_name, dtype="string"),

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

    # optional direct FR/VDA if present in performance file
    if mapped.get("fr") and mapped["fr"] in df_raw.columns:
        out["FR"] = safe_to_numeric(df_raw[mapped["fr"]]).fillna(0)
    else:
        out["FR"] = pd.NA

    if mapped.get("vda") and mapped["vda"] in df_raw.columns:
        out["VDA"] = safe_to_numeric(df_raw[mapped["vda"]]).fillna(0)
    else:
        out["VDA"] = pd.NA

    return out

def extract_fr_vda_from_headerless_company_file(df_no_header: pd.DataFrame) -> pd.DataFrame | None:
    """
    Supports the 'Company CHARKA...' style file (no headers).
    Based on observed layout:
      - driver_id is column 6
      - FR is column 8 (0/1)
      - VDA is column 9 (0..10-ish)
    If it doesn't match, return None.
    """
    if df_no_header is None or df_no_header.shape[1] < 10:
        return None

    # basic sanity: column 6 should look like big numeric IDs
    did = pd.to_numeric(df_no_header.iloc[:, 6], errors="coerce")
    if did.notna().mean() < 0.6:
        return None

    fr = pd.to_numeric(df_no_header.iloc[:, 8], errors="coerce")
    vda = pd.to_numeric(df_no_header.iloc[:, 9], errors="coerce")

    out = pd.DataFrame({
        "معرّف السائق": did.astype("Int64"),
        "FR": fr,
        "VDA": vda
    }).dropna(subset=["معرّف السائق"])

    # Keep last record per driver_id if duplicates
    out = out.sort_values(["معرّف السائق"]).drop_duplicates("معرّف السائق", keep="last")

    return out

def detect_file_kind(file_bytes: bytes) -> str:
    """
    Detects:
      - performance (must have real headers)
      - frvda_company (headerless company export)
      - unknown
    """
    df = read_excel_smart(file_bytes)
    cols = list(df.columns)

    # performance signals
    perf_signals = {
        "معرّف السائق", "معرف السائق", "اسم السائق", "اسم السائق.1",
        "معدل الإلغاء بسبب مشاكل التوصيل", "معدل الغاء", "معدل توصيل",
        "المهام التي تم تسليمها", "طلبات", "المهام المرفوضة", "المهام المرفوضة (السائق)",
    }
    perf_hits = len(perf_signals.intersection(set(cols)))

    if perf_hits >= 4 and (not looks_like_bad_headers(cols)):
        return "performance"

    # headerless FR/VDA company file detection
    if looks_like_bad_headers(cols):
        dfh = read_excel_no_header(file_bytes)
        extracted = extract_fr_vda_from_headerless_company_file(dfh)
        if extracted is not None and len(extracted):
            return "frvda_company"

    return "unknown"

# =========================
# Styling
# =========================
def style_attention_table(df: pd.DataFrame):
    """
    Fix: integers should not show trailing zeros.
    """
    fmt = {
        "معدل توصيل": "{:.2%}",
        "معدل الغاء": "{:.2%}",
        "طلبات": "{:,.0f}",
        "المهام المرفوضة": "{:,.0f}",
    }
    sty = df.style.format(fmt)

    # cancel is bad if HIGH (>= 0.20%)
    sty = sty.applymap(
        lambda x: "color:red;font-weight:900;" if float(x) >= CANCEL_ALERT_THRESHOLD else "",
        subset=["معدل الغاء"]
    )
    sty = sty.applymap(
        lambda x: "color:red;font-weight:900;" if float(x) < 1.0 else "",
        subset=["معدل توصيل"]
    )
    sty = sty.applymap(
        lambda x: "color:red;font-weight:900;" if float(x) < ORDERS_TARGET_MONTH else "",
        subset=["طلبات"]
    )
    sty = sty.applymap(
        lambda x: "color:red;font-weight:900;" if float(x) > 0 else "",
        subset=["المهام المرفوضة"]
    )
    return sty

# =========================
# Build master from uploads
# =========================
def build_master_from_uploads():
    if not uploaded_files:
        return None, None

    # filter which files are active
    files_in = uploaded_files
    if active_names is not None:
        files_in = [f for f in uploaded_files if f.name in set(active_names)]

    if not files_in:
        return None, None

    file_items = []
    for uf in files_in:
        b = uf.getvalue()
        kind = detect_file_kind(b)
        file_items.append({"name": uf.name, "bytes": b, "kind": kind})

    # choose performance file (required for master)
    perf_item = next((x for x in file_items if x["kind"] == "performance"), None)
    if perf_item is None:
        # if none is detected, try first file as performance and let build_performance_report show error
        perf_item = file_items[0]

    df_perf = read_excel_smart(perf_item["bytes"])
    perf = build_performance_report(df_perf)

    # Save driver names to DB
    for _, r in perf.iterrows():
        did = r.get("معرّف السائق")
        name = r.get("اسم السائق")
        if pd.isna(did):
            continue
        upsert_driver(int(did), driver_name=str(name).strip())

    master = perf.copy()

    # Merge FR/VDA from additional files (including the "Company CHARKA..." format)
    for item in file_items:
        if item["name"] == perf_item["name"]:
            continue

        if item["kind"] == "frvda_company":
            dfh = read_excel_no_header(item["bytes"])
            extra = extract_fr_vda_from_headerless_company_file(dfh)
            if extra is not None and len(extra):
                master = master.merge(extra, on="معرّف السائق", how="left", suffixes=("", "_x"))
            continue

        # otherwise attempt normal header-based FR/VDA merge
        df_other = read_excel_smart(item["bytes"])
        id_col = pick(df_other.columns, PERF_COLS["driver_id"])
        if not id_col:
            continue

        df_id = pd.to_numeric(df_other[id_col], errors="coerce").astype("Int64")

        fr_col = pick(df_other.columns, PERF_COLS["fr"])
        vda_col = pick(df_other.columns, PERF_COLS["vda"])

        if not fr_col and not vda_col:
            continue

        temp = pd.DataFrame({"معرّف السائق": df_id})
        if fr_col:
            temp["FR"] = safe_to_numeric(df_other[fr_col])
        if vda_col:
            temp["VDA"] = safe_to_numeric(df_other[vda_col])

        temp = temp.dropna(subset=["معرّف السائق"]).drop_duplicates("معرّف السائق", keep="last")
        master = master.merge(temp, on="معرّف السائق", how="left", suffixes=("", "_x"))

    # clean up possible duplicate merge columns
    for c in ["FR_x", "VDA_x"]:
        if c in master.columns:
            master.drop(columns=[c], inplace=True)

    if "FR" in master.columns:
        master["FR"] = pd.to_numeric(master["FR"], errors="coerce").fillna(0)
    if "VDA" in master.columns:
        master["VDA"] = pd.to_numeric(master["VDA"], errors="coerce").fillna(0)

    f = master.copy()

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

    return master, f

# =========================
# Pages (by ROLE only)
# =========================
def page_admin(master_all: pd.DataFrame | None, f: pd.DataFrame | None):
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

def page_ops(master_all: pd.DataFrame | None, f: pd.DataFrame | None):
    st.subheader("🚚 التشغيل")
    if f is None:
        st.info("ارفع ملف/ملفات للبدء.")
        return

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("عدد السائقين", f"{int(f['معرّف السائق'].nunique()):,}")
    k2.metric("متوسط معدل التوصيل", f"{f['معدل توصيل'].mean():.2%}" if len(f) else "—")
    k3.metric("متوسط معدل الإلغاء", f"{f['معدل الغاء'].mean():.2%}" if len(f) else "—")
    k4.metric("عدد الطلبات", f"{int(pd.to_numeric(f['طلبات'], errors='coerce').fillna(0).sum()):,}" if len(f) else "—")

    st.divider()

    # Attention table
    st.subheader("🚨 سائقون يحتاجون متابعة (الأولوية أولاً)")
    attention_cols = ["ترتيب المتابعة", "معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة"]
    st.dataframe(style_attention_table(f[attention_cols].head(60)), use_container_width=True, hide_index=True)

    st.divider()

    # Driver lookup (FIXED: full list + no duplicates + show ID in dropdown)
    st.subheader("🔎 البحث عن سائق")
    f_lookup = f.copy()
    f_lookup["معرّف السائق"] = pd.to_numeric(f_lookup["معرّف السائق"], errors="coerce").astype("Int64")

    # Make a stable unique list by (name, id)
    items = (
        f_lookup[["اسم السائق", "معرّف السائق"]]
        .dropna(subset=["معرّف السائق"])
        .drop_duplicates(["اسم السائق", "معرّف السائق"], keep="last")
        .sort_values(["اسم السائق", "معرّف السائق"])
    )

    # label includes BOTH name and id so you don't see one name with different ids without clarity
    items["label"] = items.apply(lambda r: f"{r['اسم السائق']}  —  ({int(r['معرّف السائق'])})", axis=1)
    labels = items["label"].tolist()

    selected_label = st.selectbox("اختر السائق", ["(اختر)"] + labels, key="lookup_driver")
    if selected_label != "(اختر)":
        picked = items[items["label"] == selected_label].head(1).iloc[0]
        did = int(picked["معرّف السائق"])

        row_df = f_lookup[f_lookup["معرّف السائق"].astype("Int64") == did].head(1)
        if len(row_df) == 0:
            st.warning("لم يتم العثور على بيانات هذا السائق داخل الملف الحالي.")
            return
        d = row_df.iloc[0]

        # Metrics (no crashing + correct formatting)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("معدل توصيل %", f"{float(d['معدل توصيل']):.2%}")
        c2.metric("الطلبات", f"{int(pd.to_numeric(d['طلبات'], errors='coerce') or 0):,}")
        c3.metric("معدل الغاء %", f"{float(d['معدل الغاء']):.2%}")

        wd = d.get("اعدد ايام العمل")
        c4.metric("اعدد ايام العمل", "—" if pd.isna(wd) else f"{int(float(wd)):,}")

        fr = d.get("FR")
        c5.metric("FR", "—" if pd.isna(fr) else f"{int(float(fr)):,}")

        vda = d.get("VDA")
        c6.metric("VDA", "—" if pd.isna(vda) else f"{float(vda):.2f}" if pd.notna(vda) else "—")

    st.divider()

    # Bottom table
    st.subheader("📋 الجدول النهائي (كامل البيانات)")
    bottom_cols = ["معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة", "اعدد ايام العمل"]
    if "FR" in f.columns:
        bottom_cols.append("FR")
    if "VDA" in f.columns:
        bottom_cols.append("VDA")

    fmt = {"معدل توصيل": "{:.2%}", "معدل الغاء": "{:.2%}", "طلبات": "{:,.0f}", "المهام المرفوضة": "{:,.0f}"}
    st.dataframe(
        f[bottom_cols].style.format(fmt),
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

def page_supervision(master_all: pd.DataFrame | None, f: pd.DataFrame | None):
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

def page_payroll():
    st.subheader("🧾 مسير الرواتب")
    st.info("جاهز — عند تزويدي بملف الرواتب (الأعمدة) سأربطه هنا مع الموارد البشرية والحسابات.")

# =========================
# Render (role decides)
# =========================
master_all, f = build_master_from_uploads()

if ROLE == "الإدارة":
    page_admin(master_all, f)
elif ROLE == "التشغيل":
    page_ops(master_all, f)
elif ROLE == "الموارد البشرية":
    page_hr()
elif ROLE == "الإشراف":
    page_supervision(master_all, f)
elif ROLE == "السيارات / الحركة":
    page_fleet()
elif ROLE == "الحسابات":
    page_accounts()
elif ROLE == "مسير الرواتب":
    page_payroll()
else:
    st.info("الدور غير معروف.")
