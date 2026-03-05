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

DEFAULT_PASSWORD = "12345"

def get_secret(key: str, default: str = DEFAULT_PASSWORD) -> str:
    """
    Uses st.secrets["auth"][key] if present; otherwise defaults to 12345.
    """
    try:
        auth = st.secrets.get("auth", {})
        val = auth.get(key, None)
        if val is None or str(val).strip() == "":
            return default
        return str(val).strip()
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
            expected = get_secret(ROLES[role])
            if str(pwd).strip() == str(expected).strip():
                st.session_state.logged_in = True
                st.session_state.role = role
                st.rerun()
            else:
                st.error("كلمة المرور غير صحيحة.")

    if not st.session_state.logged_in:
        st.info("الرجاء تسجيل الدخول من الشريط الجانبي.")
        st.stop()

require_login()
ROLE = st.session_state.role

CAN_MANAGE_ANNOUNCEMENTS = ROLE in ("الإدارة", "التشغيل")

# =========================
# Sidebar: uploader + uploaded-file checklist + filters
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

    # Low-key list of uploaded files with enable/disable (so you can "remove" without re-upload)
    enabled_files = []
    if uploaded_files:
        with st.expander("📄 الملفات المرفوعة", expanded=False):
            st.caption("✅ فعّل الملف الذي تريد أن يقرأه النظام. أزل التفعيل لتعطيله.")
            for i, uf in enumerate(uploaded_files):
                key = f"file_enable_{i}_{uf.name}"
                if key not in st.session_state:
                    st.session_state[key] = True
                checked = st.checkbox(uf.name, value=st.session_state[key], key=key)
                if checked:
                    enabled_files.append(uf)
            st.caption(f"الملفات المفعّلة: {len(enabled_files)} / {len(uploaded_files)}")

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

    # We keep both message/body compatibility because older versions may have either one.
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

def ensure_announcements_schema():
    """
    Make sure announcements table has message/body columns.
    Avoid NOT NULL constraint errors by always inserting into any existing NOT NULL column(s).
    """
    con = db_conn()
    cur = con.cursor()
    cur.execute("PRAGMA table_info(announcements)")
    info = cur.fetchall()
    cols = {r[1] for r in info}

    if "message" not in cols:
        cur.execute("ALTER TABLE announcements ADD COLUMN message TEXT")
    if "body" not in cols:
        cur.execute("ALTER TABLE announcements ADD COLUMN body TEXT")

    con.commit()
    con.close()

init_db()
ensure_announcements_schema()

def add_announcement(message: str, created_by_role: str):
    msg = (message or "").strip()
    if not msg:
        return
    ensure_announcements_schema()

    con = db_conn()
    cur = con.cursor()

    # Detect columns again (some deployments may differ)
    cur.execute("PRAGMA table_info(announcements)")
    cols = {r[1] for r in cur.fetchall()}

    fields = ["created_at", "created_by_role"]
    values = [now_ts(), str(created_by_role)]

    if "message" in cols:
        fields.append("message"); values.append(msg)
    if "body" in cols:
        fields.append("body"); values.append(msg)

    placeholders = ", ".join(["?"] * len(fields))
    sql = f"INSERT INTO announcements ({', '.join(fields)}) VALUES ({placeholders})"
    cur.execute(sql, tuple(values))

    con.commit()
    con.close()

def get_latest_announcements(limit: int = 10) -> pd.DataFrame:
    ensure_announcements_schema()
    con = db_conn()
    df = pd.read_sql_query(
        """
        SELECT
            id,
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

def delete_announcement(ann_id: int):
    con = db_conn()
    cur = con.cursor()
    cur.execute("DELETE FROM announcements WHERE id = ?", (int(ann_id),))
    con.commit()
    con.close()

def upsert_driver(driver_id: int, driver_name: str = None):
    con = db_conn()
    cur = con.cursor()
    cur.execute("SELECT driver_id, driver_name FROM drivers WHERE driver_id = ?", (int(driver_id),))
    row = cur.fetchone()

    if row is None:
        cur.execute(
            "INSERT INTO drivers (driver_id, driver_name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (int(driver_id), driver_name or "", now_ts(), now_ts())
        )
    else:
        existing_name = row[1] or ""
        new_name = existing_name if (driver_name is None or str(driver_name).strip() == "") else str(driver_name).strip()
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
            d.driver_id AS معرف_السائق,
            d.driver_name AS اسم_السائق,
            d.status AS الحالة,
            d.created_at AS تاريخ_الإضافة
        FROM drivers d
        ORDER BY d.driver_id
        """,
        con
    )
    con.close()
    return df

# =========================
# Sidebar: Announcements
# ALL users can view
# ONLY الإدارة + التشغيل can create/delete
# =========================
with st.sidebar:
    st.divider()
    with st.expander("📢 الإعلانات", expanded=False):
        ann_df = get_latest_announcements(limit=10)

        # View for everyone
        if len(ann_df):
            for _, r in ann_df.iterrows():
                st.caption(f"{r['created_at']} — {r['created_by_role']}")
                st.write(r["message"] if pd.notna(r["message"]) else "")

                # Delete ONLY for الإدارة + التشغيل
                if CAN_MANAGE_ANNOUNCEMENTS:
                    if st.button("🗑️ حذف", key=f"del_ann_{int(r['id'])}"):
                        delete_announcement(int(r["id"]))
                        st.rerun()

                st.markdown("---")
        else:
            st.caption("لا توجد إعلانات بعد.")

        # Create ONLY for الإدارة + التشغيل
        if CAN_MANAGE_ANNOUNCEMENTS:
            ann_text = st.text_area(
                "إرسال إعلان",
                placeholder="اكتب الإعلان هنا...",
                label_visibility="collapsed",
                height=80
            )
            if st.button("إرسال", use_container_width=True):
                add_announcement(ann_text, ROLE)
                st.rerun()
        else:
            st.caption("يمكن للإدارة والتشغيل فقط إضافة/حذف الإعلانات.")

# =========================
# Excel helpers
# =========================
def normalize_col(c: str) -> str:
    return str(c).strip()

def looks_like_bad_header(cols) -> bool:
    """
    Detect cases where the first row was treated as header but it's actually data:
    columns become mostly numeric strings / NULL / 0.1 / etc.
    """
    if not cols:
        return True
    cols = [str(c) for c in cols]
    bad = 0
    for c in cols[:30]:
        cc = c.strip().lower()
        if cc in ("nan", "null", "none") or cc.startswith("null"):
            bad += 1
            continue
        # numeric-like headers
        try:
            float(cc)
            bad += 1
            continue
        except Exception:
            pass
        if cc.replace(".", "", 1).isdigit():
            bad += 1
    return bad >= max(8, int(len(cols[:30]) * 0.6))

@st.cache_data(show_spinner=False)
def read_excel_bytes_smart(file_bytes: bytes) -> pd.DataFrame:
    """
    Try header=0; if it looks wrong, try header=1 then header=2.
    """
    bio0 = BytesIO(file_bytes)
    xls = pd.ExcelFile(bio0)
    sheet = xls.sheet_names[0]

    for hdr in (0, 1, 2):
        bio = BytesIO(file_bytes)
        df = pd.read_excel(bio, sheet_name=sheet, header=hdr)
        df.columns = [normalize_col(c) for c in df.columns]
        if not looks_like_bad_header(list(df.columns)):
            return df

    # fallback
    bio = BytesIO(file_bytes)
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

def fmt_int_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    # keep NA as NA, otherwise int
    return x.round(0).astype("Int64")

# =========================
# RULES / TARGETS
# =========================
CANCEL_ALERT_THRESHOLD = 0.002   # 0.20% (alert if >=)
ORDERS_TARGET_MONTH = 450

# =========================
# Column mapping (Performance + FR/VDA)
# =========================
PERF_COLS = {
    "driver_id": ["معرّف السائق", "معرف السائق", "Driver_ID", "driver_id", "id", "ID"],
    "first_name": ["اسم السائق", "First Name", "first_name", "الاسم الأول", "اسم الموظف"],
    "last_name": ["اسم السائق.1", "Last Name", "last_name", "اسم العائلة", "اسم الموظف.1"],

    "delivery_rate": ["معدل اكتمال الطلبات (غير متعلق بالتوصيل)", "معدل التوصيل", "معدل توصيل", "Delivery_Rate", "delivery_rate", "Delivery rate"],
    "cancel_rate": ["معدل الإلغاء بسبب مشاكل التوصيل", "معدل غاء", "معدل الغاء", "معدل الإلغاء", "Cancel_Rate", "cancel_rate", "Cancel rate"],
    "orders_delivered": ["المهام التي تم تسليمها", "طلبات", "الطلبات", "عدد الطلبات", "Orders_Delivered", "orders_delivered"],
    "reject_total": ["المهام المرفوضة", "المهام المرفوضة (السائق)", "رفض السائق", "Driver Rejections", "driver_rejections"],

    "work_days": ["اعدد ايام العمل", "عدد ايام العمل", "أيام العمل", "Work Days"],
    "fr": ["FR", "Face Recognition", "Face_Recognition", "التعرف على الوجه", "Face recognition"],
    "vda": ["VDA", "vda", "مؤشر VDA", "VDA %", "VDA score"],
}

def normalize_driver_name(first: str, last: str) -> str:
    full = f"{str(first).strip()} {str(last).strip()}"
    full = " ".join(full.split())  # collapse multiple spaces into one
    return full.strip()

def build_performance_report(df_raw: pd.DataFrame) -> pd.DataFrame:
    mapped = {k: pick(df_raw.columns, v) for k, v in PERF_COLS.items()}

    required = ["driver_id", "first_name", "last_name", "delivery_rate", "cancel_rate", "orders_delivered", "reject_total"]
    missing = [k for k in required if not mapped.get(k)]
    if missing:
        st.error("❌ ملف الأداء غير مطابق. الأعمدة المطلوبة غير موجودة: " + ", ".join(missing))
        st.write("الأعمدة الموجودة في الملف:")
        st.write(list(df_raw.columns))
        st.stop()

    # ✅ FIX: ALWAYS ensure a single space between first/last
    driver_name = [
        normalize_driver_name(a, b)
        for a, b in zip(df_raw[mapped["first_name"]].astype(str), df_raw[mapped["last_name"]].astype(str))
    ]

    out = pd.DataFrame({
        "معرّف السائق": fmt_int_series(df_raw[mapped["driver_id"]]),
        "اسم السائق": pd.Series(driver_name, dtype="string"),

        "معدل توصيل": safe_to_numeric(df_raw[mapped["delivery_rate"]]),
        "معدل الغاء": safe_to_numeric(df_raw[mapped["cancel_rate"]]),
        "طلبات": safe_to_numeric(df_raw[mapped["orders_delivered"]]),
        "المهام المرفوضة": safe_to_numeric(df_raw[mapped["reject_total"]]),
    })

    out["معدل توصيل"] = out["معدل توصيل"].fillna(0).clip(0, 1)
    out["معدل الغاء"] = out["معدل الغاء"].fillna(0).clip(0, 1)

    # ✅ FIX: integers, no extra zeros
    out["طلبات"] = fmt_int_series(out["طلبات"]).fillna(0)
    out["المهام المرفوضة"] = fmt_int_series(out["المهام المرفوضة"]).fillna(0)

    if mapped.get("work_days") and mapped["work_days"] in df_raw.columns:
        out["اعدد ايام العمل"] = fmt_int_series(df_raw[mapped["work_days"]]).fillna(0)
    else:
        out["اعدد ايام العمل"] = pd.NA

    # If performance file includes FR/VDA
    if mapped.get("fr") and mapped["fr"] in df_raw.columns:
        out["FR"] = fmt_int_series(df_raw[mapped["fr"]]).fillna(0)
    else:
        out["FR"] = pd.NA

    if mapped.get("vda") and mapped["vda"] in df_raw.columns:
        out["VDA"] = fmt_int_series(df_raw[mapped["vda"]]).fillna(0)
    else:
        out["VDA"] = pd.NA

    return out

def detect_file_kind(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    perf_signals = {"معرّف السائق", "معرف السائق", "اسم السائق", "اسم السائق.1", "معدل الغاء", "معدل توصيل", "طلبات", "عدد الطلبات"}
    fr_signals = {"FR", "Face Recognition", "Face_Recognition", "التعرف على الوجه"}
    vda_signals = {"VDA", "مؤشر VDA"}

    hits_perf = len(perf_signals.intersection(cols))
    hits_fr = len(fr_signals.intersection(cols))
    hits_vda = len(vda_signals.intersection(cols))

    if hits_perf >= 3:
        return "performance"
    if (hits_fr + hits_vda) >= 1:
        return "frvda"
    return "unknown"

def extract_frvda(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    This supports your "Company CHARKA..." format being uploaded ONLY for FR/VDA.
    We only need: driver_id + FR + VDA
    """
    id_col = pick(df.columns, PERF_COLS["driver_id"])
    fr_col = pick(df.columns, PERF_COLS["fr"])
    vda_col = pick(df.columns, PERF_COLS["vda"])

    if not id_col:
        return None
    if not fr_col and not vda_col:
        return None

    out = pd.DataFrame({"معرّف السائق": fmt_int_series(df[id_col])})
    if fr_col:
        out["FR"] = fmt_int_series(df[fr_col]).fillna(0)
    if vda_col:
        out["VDA"] = fmt_int_series(df[vda_col]).fillna(0)

    out = out.dropna(subset=["معرّف السائق"]).drop_duplicates("معرّف السائق")
    return out

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
# Build master from uploads (stable lookup fix)
# =========================
def build_master_from_uploads():
    files_to_use = enabled_files if uploaded_files else []
    if not files_to_use:
        return None, None

    items = []
    for uf in files_to_use:
        b = uf.getvalue()
        df = read_excel_bytes_smart(b)
        kind = detect_file_kind(df)
        items.append({"name": uf.name, "df": df, "kind": kind})

    perf_items = [x for x in items if x["kind"] == "performance"]
    if not perf_items:
        # no performance file -> nothing to build
        return None, None

    # Use first performance file (you can disable others using the checklist)
    perf_item = perf_items[0]
    perf = build_performance_report(perf_item["df"])

    # Save driver names to DB
    for _, r in perf.iterrows():
        did = r.get("معرّف السائق")
        name = r.get("اسم السائق")
        if pd.isna(did):
            continue
        upsert_driver(int(did), driver_name=str(name).strip())

    # master_all = PERF base
    master_all = perf.copy()

    # Merge FR/VDA from any FRVDA file(s) (your Company CHARKA file)
    for it in items:
        if it["name"] == perf_item["name"]:
            continue
        if it["kind"] != "frvda":
            continue
        ext = extract_frvda(it["df"])
        if ext is None or ext.empty:
            continue
        master_all = master_all.merge(ext, on="معرّف السائق", how="left", suffixes=("", "_x"))

    # Fill FR/VDA columns if missing
    if "FR" in master_all.columns:
        master_all["FR"] = fmt_int_series(master_all["FR"]).fillna(0)
    else:
        master_all["FR"] = pd.NA
    if "VDA" in master_all.columns:
        master_all["VDA"] = fmt_int_series(master_all["VDA"]).fillna(0)
    else:
        master_all["VDA"] = pd.NA

    # ===== FILTERED (used for tables) =====
    f = master_all.copy()

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
    f["تنبيه طلبات"] = (pd.to_numeric(f["طلبات"], errors="coerce").fillna(0) < ORDERS_TARGET_MONTH).astype(int)
    f["تنبيه رفض"] = (pd.to_numeric(f["المهام المرفوضة"], errors="coerce").fillna(0) > 0).astype(int)

    delivery_gap = (1.0 - f["معدل توصيل"]).clip(lower=0)
    cancel_over = (f["معدل الغاء"] - CANCEL_ALERT_THRESHOLD).clip(lower=0)
    orders_gap = (ORDERS_TARGET_MONTH - pd.to_numeric(f["طلبات"], errors="coerce").fillna(0)).clip(lower=0)

    # Priority: cancel dominates
    f["أولوية"] = (
        f["تنبيه الغاء"] * 1_000_000
        + cancel_over * 500_000
        + delivery_gap * 50_000
        + f["تنبيه طلبات"] * 10_000
        + orders_gap * 5
        + pd.to_numeric(f["المهام المرفوضة"], errors="coerce").fillna(0) * 200
    )

    f = f.sort_values(
        ["أولوية", "تنبيه الغاء", "معدل الغاء", "معدل توصيل", "طلبات", "المهام المرفوضة"],
        ascending=[False, False, False, True, True, False]
    ).reset_index(drop=True)

    f["ترتيب المتابعة"] = range(1, len(f) + 1)

    return master_all.reset_index(drop=True), f

# =========================
# Pages
# =========================
def page_admin(master_all: pd.DataFrame | None, f: pd.DataFrame | None):
    st.subheader("📊 الإدارة — نظرة (يومي / شهري)")
    if master_all is None or f is None:
        st.info("قم بتحميل ملف/ملفات الأداء لعرض مؤشرات الإدارة.")
        return

    total_drivers = int(master_all["معرّف السائق"].nunique())
    total_orders = int(pd.to_numeric(master_all["طلبات"], errors="coerce").fillna(0).sum())
    avg_delivery = float(master_all["معدل توصيل"].mean()) if len(master_all) else 0
    avg_cancel = float(master_all["معدل الغاء"].mean()) if len(master_all) else 0

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("إجمالي السائقين", f"{total_drivers:,}")
    a2.metric("إجمالي الطلبات المكتملة", f"{total_orders:,}")
    a3.metric("متوسط معدل التوصيل", f"{avg_delivery:.2%}")
    a4.metric("متوسط معدل الإلغاء", f"{avg_cancel:.2%}")

    st.divider()
    st.markdown("### 🚨 الأولوية")
    cols = ["ترتيب المتابعة", "معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة"]
    st.dataframe(style_attention_table(f[cols].head(25)), use_container_width=True, hide_index=True)

def render_driver_lookup(master_all: pd.DataFrame):
    """
    ✅ Stable Driver Lookup:
    - Always built from master_all (unfiltered), so it never "disappears"
    - Dropdown label includes BOTH name + ID to avoid duplicates
    """
    st.subheader("🔎 البحث عن سائق")

    base = master_all.copy()
    base = base.dropna(subset=["معرّف السائق", "اسم السائق"]).copy()

    base["معرّف السائق"] = fmt_int_series(base["معرّف السائق"])
    base["اسم السائق"] = base["اسم السائق"].astype(str).apply(lambda x: " ".join(str(x).split()))

    # Build stable label
    base["label"] = base["اسم السائق"].astype(str) + " — " + base["معرّف السائق"].astype(str)

    # Unique by ID
    base = base.drop_duplicates(subset=["معرّف السائق"]).sort_values(["اسم السائق", "معرّف السائق"]).reset_index(drop=True)

    labels = base["label"].tolist()
    selected = st.selectbox("اختر السائق", ["(اختر)"] + labels, key="lookup_driver_label")

    if selected == "(اختر)":
        return

    sel_row = base[base["label"] == selected].head(1)
    if sel_row.empty:
        st.info("لا توجد بيانات لهذا السائق.")
        return

    driver_id = int(sel_row["معرّف السائق"].iloc[0])
    d = master_all[master_all["معرّف السائق"].astype("Int64") == driver_id].head(1).iloc[0]

    # Metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("معدل توصيل %", f"{float(d['معدل توصيل']):.2%}")
    c2.metric("الطلبات", f"{int(pd.to_numeric(d['طلبات'], errors='coerce') or 0):,}")
    c3.metric("معدل الغاء %", f"{float(d['معدل الغاء']):.2%}")

    wd = d.get("اعدد ايام العمل", pd.NA)
    c4.metric("اعدد ايام العمل", "—" if pd.isna(wd) else f"{int(pd.to_numeric(wd, errors='coerce') or 0):,}")

    fr = d.get("FR", pd.NA)
    c5.metric("FR", "—" if pd.isna(fr) else f"{int(pd.to_numeric(fr, errors='coerce') or 0):,}")

    vda = d.get("VDA", pd.NA)
    c6.metric("VDA", "—" if pd.isna(vda) else f"{int(pd.to_numeric(vda, errors='coerce') or 0):,}")

def page_ops(master_all: pd.DataFrame | None, f: pd.DataFrame | None):
    st.subheader("🚚 التشغيل")
    if master_all is None or f is None:
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

    # ✅ FIX: Driver lookup always uses master_all (not filtered)
    render_driver_lookup(master_all)

    st.divider()
    st.subheader("📋 الجدول النهائي (كامل البيانات)")
    show_cols = ["معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة", "اعدد ايام العمل", "FR", "VDA"]
    show_cols = [c for c in show_cols if c in f.columns]
    df_show = f[show_cols].copy()

    # Ensure integers show clean
    if "طلبات" in df_show.columns:
        df_show["طلبات"] = fmt_int_series(df_show["طلبات"]).fillna(0)
    if "المهام المرفوضة" in df_show.columns:
        df_show["المهام المرفوضة"] = fmt_int_series(df_show["المهام المرفوضة"]).fillna(0)
    if "اعدد ايام العمل" in df_show.columns:
        df_show["اعدد ايام العمل"] = fmt_int_series(df_show["اعدد ايام العمل"])

    st.dataframe(
        df_show.style.format({"معدل توصيل": "{:.2%}", "معدل الغاء": "{:.2%}"}),
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
    if master_all is None or f is None:
        st.info("ارفع ملف/ملفات للبدء.")
        return
    cols = ["ترتيب المتابعة", "معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة"]
    st.dataframe(style_attention_table(f[cols].head(80)), use_container_width=True, hide_index=True)

    st.divider()
    render_driver_lookup(master_all)

def page_fleet():
    st.subheader("🚗 السيارات / الحركة")
    st.info("جاهز — عند تزويدي بملف السيارات/الحركة (الأعمدة) سأربطه هنا مع الإدارة والتشغيل.")

def page_accounts():
    st.subheader("💰 الحسابات")
    st.info("جاهز — عند تزويدي بملف الحسابات (الأعمدة) سأربطه هنا مع الإدارة والتشغيل.")

# =========================
# Payroll (مسير الرواتب)
# =========================
PAYROLL_COLS = {
    "driver_id": ["ID", "معرّف السائق", "معرف السائق", "Driver_ID", "driver_id"],
    "driver_name": ["اسم الموظف", "اسم السائق", "الاسم", "Name"],
    "orders": ["عدد الطلبات", "الطلبات", "طلبات", "Orders"],
    "base_salary": ["الراتب الاساسي", "الراتب الأساسي", "Base Salary", "الأساسي"],
}

def find_payroll_file(files_to_use) -> tuple[str, pd.DataFrame] | tuple[None, None]:
    """
    Pick the first file that looks like payroll by columns.
    """
    for uf in files_to_use:
        df = read_excel_bytes_smart(uf.getvalue())
        cols = set(df.columns)
        if (pick(df.columns, PAYROLL_COLS["orders"]) and pick(df.columns, PAYROLL_COLS["driver_id"])):
            # payroll usually has base salary too, but not required
            return uf.name, df
    return None, None

def page_payroll():
    st.subheader("🧾 مسير الرواتب")

    files_to_use = enabled_files if uploaded_files else []
    if not files_to_use:
        st.info("قم بتحميل ملف مسير الرواتب.")
        return

    fname, df = find_payroll_file(files_to_use)
    if df is None:
        st.error("❌ لم يتم العثور على ملف مسير الرواتب ضمن الملفات المفعّلة.")
        st.write("الأعمدة الموجودة في الملفات المفعّلة:")
        for uf in files_to_use:
            dfx = read_excel_bytes_smart(uf.getvalue())
            st.write(uf.name, list(dfx.columns))
        return

    id_col = pick(df.columns, PAYROLL_COLS["driver_id"])
    name_col = pick(df.columns, PAYROLL_COLS["driver_name"])
    orders_col = pick(df.columns, PAYROLL_COLS["orders"])
    base_col = pick(df.columns, PAYROLL_COLS["base_salary"])

    if not id_col or not orders_col:
        st.error("❌ ملف مسير الرواتب غير مطابق (ينقصه ID و/أو عدد الطلبات).")
        st.write(list(df.columns))
        return

    out = pd.DataFrame()
    out["ID"] = fmt_int_series(df[id_col])
    out["اسم الموظف"] = df[name_col].astype(str).apply(lambda x: " ".join(str(x).split())) if name_col else ""

    out["عدد الطلبات"] = fmt_int_series(df[orders_col]).fillna(0)

    if base_col:
        base_num = pd.to_numeric(df[base_col], errors="coerce")
    else:
        base_num = pd.Series([pd.NA] * len(df))

    # ✅ Rule (your final clarification): if orders < 450, ignore base salary and use orders * 7
    calc_alt = pd.to_numeric(out["عدد الطلبات"], errors="coerce").fillna(0) * 7
    use_alt = pd.to_numeric(out["عدد الطلبات"], errors="coerce").fillna(0) < ORDERS_TARGET_MONTH

    out["المستحق"] = base_num
    out.loc[use_alt, "المستحق"] = calc_alt.loc[use_alt]
    out["المستحق"] = pd.to_numeric(out["المستحق"], errors="coerce").fillna(0).round(2)

    out["طريقة الاحتساب"] = "الراتب الاساسي"
    out.loc[use_alt, "طريقة الاحتساب"] = "عدد الطلبات × 7 (أقل من 450)"

    st.caption(f"الملف المستخدم: {fname}")
    st.dataframe(out, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ تحميل مسير الرواتب (CSV)",
        data=out.to_csv(index=False, encoding="utf-8-sig"),
        file_name="payroll_calculated.csv",
        mime="text/csv",
    )

# =========================
# Render
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