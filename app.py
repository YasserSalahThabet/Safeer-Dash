import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Driver Performance", layout="wide")
st.title("🚚 Driver Performance Dashboard")

AR_HEADER_MARKERS = ["معرف السائق", "اسم السائق", "معدل التوصيل", "معدل الإلغاء"]

COLS = {
    "driver_id": ["معرف السائق"],
    "user_name": ["اسم اليوزر"],
    "driver_name": ["اسم السائق"],
    "completed": ["طلب مكتمل"],
    "cancelled": ["طلب ملغي"],
    "driver_reject": ["رفض السائق"],
    "auto_reject": ["رفض تلقائي"],
    "cancel_rate": ["معدل الإلغاء"],
    "delivery_rate": ["معدل التوصيل"],
}

def normalize(x):
    return str(x).strip()

def find_header_row(df_preview, max_rows=30):
    for r in range(min(max_rows, len(df_preview))):
        row_vals = [normalize(v) for v in df_preview.iloc[r].tolist()]
        hits = sum(1 for m in AR_HEADER_MARKERS if m in row_vals)
        if hits >= 2:
            return r
    return 1

def pick_col(df_cols, candidates):
    cols = [normalize(c) for c in df_cols]
    for cand in candidates:
        if cand in cols:
            return cand
    return None

@st.cache_data
def load_excel(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = xls.sheet_names[0]

    preview = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, nrows=40)
    header_row = find_header_row(preview)

    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)
    df.columns = [normalize(c) for c in df.columns]
    return df

def to_report(df_raw):
    mapped = {k: pick_col(df_raw.columns, v) for k, v in COLS.items()}

    out = pd.DataFrame({
        "Driver_ID": df_raw[mapped["driver_id"]],
        "Driver_Name": df_raw[mapped["driver_name"]],
        "Delivery_Rate": pd.to_numeric(df_raw[mapped["delivery_rate"]], errors="coerce"),
        "Cancel_Rate": pd.to_numeric(df_raw[mapped["cancel_rate"]], errors="coerce"),
        "Orders_Completed": pd.to_numeric(df_raw[mapped["completed"]], errors="coerce"),
        "Orders_Cancelled": pd.to_numeric(df_raw[mapped["cancelled"]], errors="coerce"),
        "Driver_Rejections": pd.to_numeric(df_raw[mapped["driver_reject"]], errors="coerce"),
    })

    out = out.fillna(0)

    # Performance Score 0–1
    score_0_100 = (
        (out["Delivery_Rate"] * 100) * 0.6
        + ((1 - out["Cancel_Rate"]) * 100) * 0.3
        + (100 - (out["Driver_Rejections"] * 10)) * 0.1
    )

    out["Performance_Score"] = (score_0_100 / 100).clip(0, 1)
    out = out.sort_values("Performance_Score", ascending=True)

    return out


# 🔴 Styling Rules
def style_attention(df):

    def red_if_reject(x):
        return "color: red; font-weight: 700;" if x > 0 else ""

    def red_if_not_100(x):
        return "color: red; font-weight: 700;" if round(float(x), 6) != 1.0 else ""

    def red_if_cancel_high(x):
        # 0.20% = 0.002
        return "color: red; font-weight: 700;" if float(x) >= 0.002 else ""

    sty = df.style.format({
        "Performance_Score": "{:.2%}",
        "Delivery_Rate": "{:.2%}",
        "Cancel_Rate": "{:.2%}",
    })

    sty = sty.applymap(red_if_reject, subset=["Driver_Rejections"])
    sty = sty.applymap(red_if_not_100, subset=["Performance_Score"])
    sty = sty.applymap(red_if_cancel_high, subset=["Cancel_Rate"])

    return sty


# Sidebar
with st.sidebar:
    uploaded = st.file_uploader("Upload raw Excel template (.xlsx)", type=["xlsx"])

if not uploaded:
    st.warning("Upload Excel file.")
    st.stop()

raw = load_excel(uploaded)
report = to_report(raw)

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Avg Delivery Rate", f"{report['Delivery_Rate'].mean():.2%}")
c2.metric("Avg Cancel Rate", f"{report['Cancel_Rate'].mean():.2%}")
c3.metric("Avg Score", f"{report['Performance_Score'].mean():.2%}")

st.divider()

st.subheader("🚨 Drivers (Low Performance First)")
st.dataframe(style_attention(report), use_container_width=True, hide_index=True)

st.download_button(
    "Download CSV",
    data=report.to_csv(index=False, encoding="utf-8-sig"),
    file_name="drivers_filtered.csv",
)