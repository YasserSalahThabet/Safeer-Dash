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

    div[data-testid="stFileUploader"] ul { display: none !important; }

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
    "مسير الرواتب": "payroll_password",   # ✅ ADDED
}
DEFAULT_PASSWORD = "12345"

def get_secret(key: str, default: str = DEFAULT_PASSWORD) -> str:
    try:
        auth = st.secrets.get("auth", {})
        val = auth.get(key, None)
        if val is None or str(val).strip() == "":
            return default
        return str(val)
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
            expected = get_secret(ROLES[role])  # defaults to 12345 if missing
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
    con = db_conn()
    cur = con.cursor()
    cur.execute("PRAGMA table_info(announcements)")
    cols = [r[1] for r in cur.fetchall()]
    if "message" not in cols:
        cur.execute("ALTER TABLE announcements ADD COLUMN message TEXT")
    if "body" not in cols:
        cur.execute("ALTER TABLE announcements ADD COLUMN body TEXT")
    con.commit()
    con.close()

ensure_announcements_columns()

def add_announcement(message: str, created_by_role: str):
    if str(created_by_role) not in ("الإدارة", "التشغيل"):
        return
    msg = (message or "").strip()
    if not msg:
        return
    ensure_announcements_columns()
    con = db_conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO announcements (created_at, created_by_role, message, body) VALUES (?, ?, ?, ?)",
        (now_ts(), str(created_by_role), msg, msg)
    )
    con.commit()
    con.close()

def get_latest_announcements(limit: int = 10) -> pd.DataFrame:
    ensure_announcements_columns()
    con = db_conn()
    df = pd.read_sql_query(
        """
        SELECT id, created_at, created_by_role, COALESCE(message, body) AS message
        FROM announcements
        ORDER BY id DESC
        LIMIT ?
        """,
        con,
        params=(int(limit),)
    )
    con.close()
    return df

def delete_announcement(ann_id: int, role: str):
    if str(role) not in ("الإدارة", "التشغيل"):
        return
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
        SELECT d.driver_id AS معرف_السائق,
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
# =========================
with st.sidebar:
    st.divider()
    with st.expander("📢 الإعلانات", expanded=False):
        ann_df = get_latest_announcements(limit=10)

        if len(ann_df):
            for _, r in ann_df.iterrows():
                st.caption(f"{r['created_at']} — {r['created_by_role']}")
                st.write(r["message"])

                if CAN_MANAGE_ANNOUNCEMENTS:
                    if st.button("🗑️ حذف", key=f"del_ann_{int(r['id'])}"):
                        delete_announcement(int(r["id"]), ROLE)
                        st.rerun()
                st.markdown("---")
        else:
            st.caption("لا توجد إعلانات بعد.")

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
CANCEL_ALERT_THRESHOLD = 0.002   # 0.20%
ORDERS_TARGET_MONTH = 450

# =========================
# Column mapping (Performance)
# =========================
PERF_COLS = {
    "driver_id": ["معرّف السائق", "معرف السائق", "Driver_ID", "driver_id", "id"],
    "first_name": ["اسم السائق", "First Name", "first_name"],
    "last_name": ["اسم السائق.1", "Last Name", "last_name"],
    "delivery_rate": ["معدل اكتمال الطلبات (غير متعلق بالتوصيل)", "معدل التوصيل", "معدل توصيل", "Delivery_Rate", "delivery_rate"],
    "cancel_rate": ["معدل الإلغاء بسبب مشاكل التوصيل", "معدل غاء", "معدل الغاء", "معدل الإلغاء", "Cancel_Rate", "cancel_rate"],
    "orders_delivered": ["المهام التي تم تسليمها", "طلبات", "الطلبات", "الطلبات المسلمة", "Orders_Delivered", "orders_delivered"],
    "reject_total": ["المهام المرفوضة", "المهام المرفوضة (السائق)", "رفض السائق", "Driver Rejections", "driver_rejections"],
    "work_days": ["اعدد ايام العمل", "عدد ايام العمل", "أيام العمل"],
    "fr": ["FR", "Face Recognition", "Face_Recognition", "التعرف على الوجه"],
    "vda": ["VDA", "vda", "مؤشر VDA"],
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

def detect_file_type(cols: set[str]) -> str:
    perf_signals = {"معرّف السائق", "اسم السائق", "اسم السائق.1", "معدل الغاء", "معدل توصيل", "طلبات", "المهام المرفوضة"}
    return "performance" if len(perf_signals.intersection(cols)) >= 3 else "unknown"

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
    files_to_use = enabled_files if uploaded_files else []
    if not files_to_use:
        st.session_state["master_all"] = None
        return None

    file_items = []
    for uf in files_to_use:
        b = uf.getvalue()
        df = read_first_sheet_excel_bytes(b)
        kind = detect_file_type(set(df.columns))
        file_items.append({"name": uf.name, "df": df, "kind": kind})

    perf_item = None
    for item in file_items:
        if item["kind"] == "performance":
            perf_item = item
            break
    if perf_item is None:
        perf_item = file_items[0]

    perf = build_performance_report(perf_item["df"])
    st.session_state["master_all"] = perf.copy()

    for _, r in perf.iterrows():
        did = r.get("معرّف السائق")
        name = r.get("اسم السائق")
        if pd.isna(did):
            continue
        upsert_driver(int(did), driver_name=str(name).strip())

    f = perf.copy()

    if search.strip():
        s = search.strip().lower()
        f = f[
            f["اسم السائق"].str.lower().str.contains(s, na=False)
            | f["معرّف السائق"].astype(str).str.contains(s, na=False)
        ]
    f = f[(f["معدل توصيل"] >= min_delivery)]
    f = f[(f["معدل الغاء"] <= max_cancel)]

    f["تنبيه الغاء"] = (f["معدل الغاء"] >= CANCEL_ALERT_THRESHOLD).astype(int)
    f["تنبيه توصيل"] = (f["معدل توصيل"] < 1.0).astype(int)
    f["تنبيه طلبات"] = (f["طلبات"] < ORDERS_TARGET_MONTH).astype(int)
    f["تنبيه رفض"] = (f["المهام المرفوضة"] > 0).astype(int)

    delivery_gap = (1.0 - f["معدل توصيل"]).clip(lower=0)
    cancel_over = (f["معدل الغاء"] - CANCEL_ALERT_THRESHOLD).clip(lower=0)
    orders_gap = (ORDERS_TARGET_MONTH - f["طلبات"]).clip(lower=0)

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
# Pages
# =========================
def page_admin(f: pd.DataFrame | None):
    st.subheader("📊 الإدارة — نظرة  (يومي / شهري)")
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
    st.subheader("🔎 بحث عن سائق")

    master_all = st.session_state.get("master_all", None)
    if master_all is None or len(master_all) == 0:
        st.info("لا توجد بيانات كافية لعرض بحث عن سائق.")
        return

    master_all = master_all.copy()
    master_all["معرّف السائق"] = pd.to_numeric(master_all["معرّف السائق"], errors="coerce").astype("Int64")
    master_all["اسم السائق"] = master_all["اسم السائق"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    master_all = master_all.dropna(subset=["معرّف السائق"]).drop_duplicates(subset=["معرّف السائق"], keep="first")
    master_all["اختيار"] = master_all["اسم السائق"] + " (" + master_all["معرّف السائق"].astype(str) + ")"

    selected = st.selectbox("اختر السائق", ["(اختر)"] + master_all["اختيار"].tolist())

    if selected != "(اختر)":
        did = int(selected.split("(")[-1].replace(")", "").strip())
        row = master_all[master_all["معرّف السائق"] == did].head(1).iloc[0]

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("معدل توصيل %", f"{float(row['معدل توصيل']):.2%}")
        c2.metric("طلبات", f"{int(float(row['طلبات'])):,}")
        c3.metric("معدل الغاء %", f"{float(row['معدل الغاء']):.2%}")
        wd = row.get("اعدد ايام العمل")
        c4.metric("اعدد ايام العمل", "—" if pd.isna(wd) else f"{int(float(wd)):,}")
        fr = row.get("FR")
        c5.metric("FR", "—" if pd.isna(fr) else f"{int(float(fr)):,}")
        vda = row.get("VDA")
        c6.metric("VDA", "—" if pd.isna(vda) else f"{int(float(vda)):,}")

    st.divider()
    st.subheader("📋 الجدول النهائي (كامل البيانات)")
    st.dataframe(
        f[["معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة"]]
        .style.format({"معدل توصيل": "{:.2%}", "معدل الغاء": "{:.2%}"}),
        use_container_width=True,
        hide_index=True
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
# Render
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
elif ROLE == "مسير الرواتب":
    page_payroll()
else:
    st.info("الدور غير معروف.")