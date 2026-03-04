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

# IMPORTANT: set these to your real filenames in /assets
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
# CSS (Streamlit 1.54 uploader button + small polish)
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

    /* Low-key announcements box */
    .ann-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 10px;
        font-size: 13px;
        opacity: 0.95;
    }
    .ann-meta { opacity: 0.7; font-size: 12px; }
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

DEFAULT_PASSWORD = "12345"  # as requested

def get_secret(key: str, default: str = DEFAULT_PASSWORD) -> str:
    """
    If secrets are not configured, fallback to DEFAULT_PASSWORD.
    """
    try:
        v = st.secrets.get("auth", {}).get(key, None)
        if v is None or str(v).strip() == "":
            return default
        return str(v)
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
            expected = get_secret(ROLES[role], DEFAULT_PASSWORD)
            if pwd == expected:
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
# SQLite
# =========================
def db_conn():
    # bigger timeout helps with 'database is locked' issues on Streamlit Cloud
    return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def init_db():
    con = db_conn()
    cur = con.cursor()

    # Drivers registry (basic)
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

    # Announcements (central web notifications)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS announcements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        created_by_role TEXT NOT NULL,
        message TEXT NOT NULL
    )
    """)

    con.commit()
    con.close()

init_db()

def upsert_driver(driver_id: int, driver_name: str = None):
    con = db_conn()
    cur = con.cursor()
    cur.execute("SELECT driver_id, driver_name FROM drivers WHERE driver_id = ?", (int(driver_id),))
    row = cur.fetchone()

    if row is None:
        cur.execute("""
            INSERT INTO drivers (driver_id, driver_name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (int(driver_id), (driver_name or "").strip(), now_ts(), now_ts()))
    else:
        existing_name = (row[1] or "").strip()
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

# ---- Announcements helpers ----
def add_announcement(message: str, created_by_role: str):
    msg = (message or "").strip()
    if not msg:
        return

    con = db_conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO announcements (created_at, created_by_role, message) VALUES (?, ?, ?)",
        (now_ts(), str(created_by_role), msg)
    )
    con.commit()
    con.close()

def get_latest_announcements(limit: int = 3) -> pd.DataFrame:
    con = db_conn()
    df = pd.read_sql_query(
        "SELECT created_at, created_by_role, message FROM announcements ORDER BY id DESC LIMIT ?",
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
def read_excel_bytes_prefer_report(file_bytes: bytes) -> pd.DataFrame:
    """
    Reads Excel from bytes.
    Prefer sheet named like 'report' (any case) if exists, else first sheet.
    """
    bio = BytesIO(file_bytes)
    xls = pd.ExcelFile(bio)

    # prefer sheet name "report"
    preferred = None
    for s in xls.sheet_names:
        if str(s).strip().lower() == "report":
            preferred = s
            break
    sheet = preferred if preferred else xls.sheet_names[0]

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
# Rules / Targets
# =========================
CANCEL_ALERT_THRESHOLD = 0.002   # 0.20% (alert if >=)
ORDERS_TARGET_MONTH = 450

# =========================
# Column mapping (Performance)
# =========================
PERF_COLS = {
    "driver_id": ["معرّف السائق", "معرف السائق", "Driver_ID", "driver_id", "id", "ID", "Driver Id", "DriverID"],
    "first_name": ["اسم السائق", "First Name", "first_name", "الاسم الأول", "الاسم"],
    "last_name": ["اسم السائق.1", "Last Name", "last_name", "اسم العائلة", "الاسم الأخير", "اللقب"],
    "delivery_rate": ["معدل اكتمال الطلبات (غير متعلق بالتوصيل)", "معدل التوصيل", "معدل توصيل", "Delivery_Rate", "delivery_rate", "Delivery Rate"],
    "cancel_rate": ["معدل الإلغاء بسبب مشاكل التوصيل", "معدل الغاء", "معدل الإلغاء", "Cancel_Rate", "cancel_rate", "Cancel Rate", "Cancellation Rate"],
    "orders_delivered": ["المهام التي تم تسليمها", "طلبات", "الطلبات", "الطلبات المسلمة", "Orders_Delivered", "orders_delivered", "Completed Orders", "Orders Completed"],
    "reject_total": ["المهام المرفوضة", "المهام المرفوضة (السائق)", "رفض السائق", "Driver Rejections", "driver_rejections", "Rejected Tasks"],
    "work_days": ["اعدد ايام العمل", "عدد ايام العمل", "أيام العمل", "Work Days", "Days Worked"],
    "fr": ["FR", "Face Recognition", "Face_Recognition", "التعرف على الوجه"],
    "vda": ["VDA", "vda", "مؤشر VDA"],
}

def build_performance_report(df_raw: pd.DataFrame) -> pd.DataFrame:
    mapped = {k: pick(df_raw.columns, v) for k, v in PERF_COLS.items()}

    required = ["driver_id", "delivery_rate", "cancel_rate", "orders_delivered", "reject_total"]
    missing = [k for k in required if not mapped.get(k)]
    if missing:
        st.error("❌ ملف الأداء غير مطابق. الأعمدة المطلوبة غير موجودة: " + ", ".join(missing))
        st.write("الأعمدة الموجودة في الملف:")
        st.write(list(df_raw.columns))
        st.stop()

    # driver name: if no first/last, try a single name column
    if mapped.get("first_name") and mapped.get("last_name"):
        driver_name = (
            df_raw[mapped["first_name"]].astype(str).str.strip().fillna("")
            + " "
            + df_raw[mapped["last_name"]].astype(str).str.strip().fillna("")
        ).str.replace(r"\s+", " ", regex=True).str.strip()
    elif mapped.get("first_name"):
        driver_name = df_raw[mapped["first_name"]].astype(str).str.strip()
    else:
        # last resort
        driver_name = pd.Series(["—"] * len(df_raw))

    out = pd.DataFrame({
        "معرّف السائق": safe_to_numeric(df_raw[mapped["driver_id"]]),
        "اسم السائق": driver_name,

        "معدل توصيل": safe_to_numeric(df_raw[mapped["delivery_rate"]]),
        "معدل الغاء": safe_to_numeric(df_raw[mapped["cancel_rate"]]),
        "طلبات": safe_to_numeric(df_raw[mapped["orders_delivered"]]),
        "المهام المرفوضة": safe_to_numeric(df_raw[mapped["reject_total"]]),
    })

    out["معرّف السائق"] = pd.to_numeric(out["معرّف السائق"], errors="coerce").astype("Int64")

    # percent columns normalized [0..1]
    out["معدل توصيل"] = out["معدل توصيل"].fillna(0).clip(0, 1)
    out["معدل الغاء"] = out["معدل الغاء"].fillna(0).clip(0, 1)

    # integer-like columns -> Int64 (fix .000000)
    for col in ["طلبات", "المهام المرفوضة"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(0).astype("Int64")

    # optional extra cols
    if mapped.get("work_days") and mapped["work_days"] in df_raw.columns:
        out["اعدد ايام العمل"] = pd.to_numeric(df_raw[mapped["work_days"]], errors="coerce").round(0).astype("Int64")
    else:
        out["اعدد ايام العمل"] = pd.NA

    if mapped.get("fr") and mapped["fr"] in df_raw.columns:
        out["FR"] = pd.to_numeric(df_raw[mapped["fr"]], errors="coerce").round(0).astype("Int64")
    else:
        out["FR"] = pd.NA

    if mapped.get("vda") and mapped["vda"] in df_raw.columns:
        out["VDA"] = pd.to_numeric(df_raw[mapped["vda"]], errors="coerce").round(0).astype("Int64")
    else:
        out["VDA"] = pd.NA

    return out

def detect_file_type(cols: set[str]) -> str:
    perf_signals = {"معرّف السائق", "معرف السائق", "معدل الغاء", "معدل توصيل", "طلبات", "المهام المرفوضة", "معدل الإلغاء بسبب مشاكل التوصيل"}
    hits = len(perf_signals.intersection(cols))
    return "performance" if hits >= 3 else "unknown"

# =========================
# Styling (Attention table)
# =========================
def style_attention_table(df):
    # ensure formatting: percents + integers (no extra zeros)
    fmt = {
        "معدل توصيل": "{:.2%}",
        "معدل الغاء": "{:.2%}",
        "طلبات": "{:.0f}",
        "المهام المرفوضة": "{:.0f}",
    }
    sty = df.style.format(fmt)

    # cancel is BAD if HIGH (>= 0.20%)
    sty = sty.applymap(lambda x: "color:red;font-weight:900;" if float(x) >= CANCEL_ALERT_THRESHOLD else "", subset=["معدل الغاء"])
    # delivery is BAD if < 100%
    sty = sty.applymap(lambda x: "color:red;font-weight:900;" if float(x) < 1.0 else "", subset=["معدل توصيل"])
    # orders target
    sty = sty.applymap(lambda x: "color:red;font-weight:900;" if float(x) < ORDERS_TARGET_MONTH else "", subset=["طلبات"])
    # rejections
    sty = sty.applymap(lambda x: "color:red;font-weight:900;" if float(x) > 0 else "", subset=["المهام المرفوضة"])
    return sty

# =========================
# Sidebar: uploader + filters + files list + announcements
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

    # Low-key: show uploaded file names and allow excluding some
    selected_names = []
    if uploaded_files:
        with st.expander("📄 الملفات المرفوعة (اختياري)", expanded=False):
            names = [f.name for f in uploaded_files]
            selected_names = st.multiselect(
                "اختر الملفات التي تريد استخدامها",
                options=names,
                default=names
            )
    else:
        selected_names = []

    st.divider()
    search = st.text_input("بحث (المعرف / الاسم)", "")
    min_delivery = st.slider("أقل معدل توصيل", 0.0, 1.0, 0.0, 0.01)
    max_cancel = st.slider("أعلى معدل إلغاء (فلترة)", 0.0, 1.0, 1.0, 0.01)

    st.divider()
    with st.expander("📣 إعلان (للجميع)", expanded=False):
        ann_text = st.text_area("اكتب إعلان قصير", placeholder="مثال: الرجاء متابعة السائقين ذوي الأولوية اليوم.", height=90)
        if st.button("إرسال الإعلان", use_container_width=True):
            try:
                add_announcement(ann_text, ROLE)
                st.success("تم إرسال الإعلان.")
            except Exception as e:
                st.error("تعذر إرسال الإعلان. تحقق من قاعدة البيانات.")
                st.exception(e)

# =========================
# Build master from uploads
# =========================
def build_master_from_uploads():
    if not uploaded_files:
        return None

    # filter by user selection (exclude unwanted files)
    use_files = [f for f in uploaded_files if (not selected_names) or (f.name in selected_names)]
    if not use_files:
        return None

    file_items = []
    for uf in use_files:
        b = uf.getvalue()
        df = read_excel_bytes_prefer_report(b)
        kind = detect_file_type(set(df.columns))
        file_items.append({"name": uf.name, "df": df, "kind": kind})

    # pick first performance file
    perf_item = next((it for it in file_items if it["kind"] == "performance"), None)
    if perf_item is None:
        # if still none, try to build anyway from first (will error with columns list)
        perf_item = file_items[0]

    perf = build_performance_report(perf_item["df"])

    # Save driver names to DB (for HR registry)
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
    f["تنبيه طلبات"] = (pd.to_numeric(f["طلبات"], errors="coerce").fillna(0) < ORDERS_TARGET_MONTH).astype(int)
    f["تنبيه رفض"] = (pd.to_numeric(f["المهام المرفوضة"], errors="coerce").fillna(0) > 0).astype(int)

    delivery_gap = (1.0 - f["معدل توصيل"]).clip(lower=0)
    cancel_over = (f["معدل الغاء"] - CANCEL_ALERT_THRESHOLD).clip(lower=0)
    orders_gap = (ORDERS_TARGET_MONTH - pd.to_numeric(f["طلبات"], errors="coerce").fillna(0)).clip(lower=0)
    rejects = pd.to_numeric(f["المهام المرفوضة"], errors="coerce").fillna(0)

    # Priority: cancel dominates, then delivery, then orders, then rejects
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
# Pages (by ROLE only)
# =========================
def render_announcements_strip():
    try:
        ann = get_latest_announcements(limit=3)
        if ann is not None and len(ann) > 0:
            st.markdown("### 📣 آخر الإعلانات")
            for _, r in ann.iterrows():
                created_at = r.get("created_at", "")
                created_by_role = r.get("created_by_role", "")
                msg = r.get("message", "")
                st.markdown(
                    f"""
                    <div class="ann-box">
                        <div>{msg}</div>
                        <div class="ann-meta">{created_at} — {created_by_role}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.divider()
    except Exception as e:
        # do not crash the app for announcements
        st.warning("تعذر تحميل الإعلانات حالياً.")
        # optionally show details in dev:
        # st.exception(e)

def page_admin(f: pd.DataFrame | None):
    st.subheader("📊 الإدارة — نظرة (يومي / شهري)")
    render_announcements_strip()

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
    render_announcements_strip()

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
    st.subheader("🔎 بحث سريع ")
    driver_list = f["اسم السائق"].dropna().unique().tolist()
    selected = st.selectbox("اختر السائق", ["(اختر)"] + driver_list, key="lookup_driver_ops")

    if selected != "(اختر)":
        d = f[f["اسم السائق"] == selected].head(1).iloc[0]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("معدل توصيل %", f"{float(d['معدل توصيل']):.2%}")
        c2.metric("طلبات", f"{int(pd.to_numeric(d['طلبات'], errors='coerce') or 0):,}")
        c3.metric("معدل الغاء %", f"{float(d['معدل الغاء']):.2%}")

        wd = d.get("اعدد ايام العمل")
        c4.metric("اعدد ايام العمل", "—" if pd.isna(wd) else f"{int(wd):,}")

        fr = d.get("FR")
        c5.metric("FR", "—" if pd.isna(fr) else f"{int(fr):,}")

        vda = d.get("VDA")
        c6.metric("VDA", "—" if pd.isna(vda) else f"{int(vda):,}")

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
            "طلبات": "{:.0f}",
            "المهام المرفوضة": "{:.0f}",
            "اعدد ايام العمل": "{:.0f}",
            "FR": "{:.0f}",
            "VDA": "{:.0f}",
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
    render_announcements_strip()
    registry = get_hr_registry()
    st.dataframe(registry, use_container_width=True, hide_index=True)

def page_supervision(f: pd.DataFrame | None):
    st.subheader("🧭 الإشراف")
    render_announcements_strip()
    if f is None:
        st.info("ارفع ملف/ملفات للبدء.")
        return
    st.markdown("### 🚨 سائقون يحتاجون متابعة")
    cols = ["ترتيب المتابعة", "معرّف السائق", "اسم السائق", "معدل توصيل", "معدل الغاء", "طلبات", "المهام المرفوضة"]
    st.dataframe(style_attention_table(f[cols].head(80)), use_container_width=True, hide_index=True)

def page_fleet():
    st.subheader("🚗 السيارات / الحركة")
    render_announcements_strip()
    st.info("جاهز — عند تزويدي بملف السيارات/الحركة (الأعمدة) سأربطه هنا مع الإدارة والتشغيل.")

def page_accounts():
    st.subheader("💰 الحسابات")
    render_announcements_strip()
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
