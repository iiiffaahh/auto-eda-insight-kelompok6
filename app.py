"""
app.py — Auto EDA Insight Dashboard (REVAMPED)
Data Science Programming — Kelompok 6
"""

from pathlib import Path
import base64
import mimetypes
import sys
import os
import datetime
import time
import io
import json
import re

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="Auto EDA Insight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backend.data_loader import load_file
from backend.descriptive_stats import numeric_stats, categorical_stats
try:
    from backend.data_cleaning import (
        snapshot, drop_duplicates, drop_missing_rows,
        fill_missing_mean, fill_missing_median, fill_missing_mode,
        drop_column, convert_dtype,
    )
except ModuleNotFoundError:
    from backend.preprocessing import (
        snapshot, drop_duplicates, drop_missing_rows,
        fill_missing_mean, fill_missing_median, fill_missing_mode,
        drop_column, convert_dtype,
    )
from backend.visualization import (
    plot_histogram, plot_boxplot, plot_density, plot_qq, plot_violin,
    plot_bar, plot_pie, plot_pareto,
    plot_scatter, plot_correlation_heatmap, plot_regression,
    plot_boxplot_by_cat, plot_grouped_bar, plot_time_series,
)
from backend.insight_generator import generate_insights

# ══════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════
DEFAULT_STATE = {
    "authenticated": False,
    "username": "",
    "user_role": "viewer",
    "df": None,
    "df_original": None,
    "history": [],
    "cleaning_log": [],
    "before_snap": None,
    "after_snap": None,
    "before_df": None,
    "after_df": None,
    "last_cleaning_operation": "",
    "cleaning_notice": "",
    "active_page": "🏠 Dashboard",
    "nav_radio": "🏠 Dashboard",
    "active_file": None,
    "last_upload_signature": None,
    "ui_theme": "Dark Mode",
    "_scroll_to_main": False,
    "register_mode": False,
    "users_db": {"admin": {"password": "eda2025", "role": "admin", "name": "Admin"}, "clara": {"password": "kelompok6", "role": "member", "name": "Clara"}, "naisya": {"password": "kelompok6", "role": "member", "name": "Naisya"}, "iffah": {"password": "kelompok6", "role": "member", "name": "Iffah"}, "fifi": {"password": "kelompok6", "role": "member", "name": "Fifi"}, "nurul": {"password": "kelompok6", "role": "member", "name": "Iffah"}, "aisya": {"password": "kelompok6", "role": "member", "name": "Naisya"}},
}
for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════
#  NAVIGATION STRUCTURE — categorised sidebar
# ══════════════════════════════════════════════════════
NAV_CATEGORIES = {
    "🏠 HOME": ["🏠 Dashboard"],
    "📁 DATA MANAGEMENT": ["📤 Upload Data", "👁️ Data Preview", "📌 Dataset Info"],
    "🧹 CLEANING": ["🧹 Data Cleaning"],
    "📊 STATISTICS": ["📈 Statistik — Numerik", "📊 Statistik — Kategorik"],
    "📉 VISUALIZATION": ["📉 Visualisasi Numerik", "🎨 Visualisasi Kategorik", "🔗 Bivariate & Multivariat", "📦 Kategorik vs Numerik", "⏱️ Time Series"],
    "💡 INSIGHTS & REPORT": ["💡 Insights", "📄 Download Report"],
    "🗂️ HISTORY": ["🗂️ Riwayat Upload"],
}

ALL_PAGES = [p for pages in NAV_CATEGORIES.values() for p in pages]

# ══════════════════════════════════════════════════════
#  COLOUR TOKENS — Excel Finance Dashboard inspired
#  Deep navy/slate + gold + teal accents
# ══════════════════════════════════════════════════════
DARK_CSS = """
    --bg:       #1a0845;
    --panel:    #1e0f3d;
    --panel-2:  #2a1258;
    --stroke:   rgba(139, 92, 246, .35);
    --text:     #f0eeff;
    --muted:    #8b78c4;
    --accent:   #7c3aed;
    --cyan:     #06b6d4;
    --gold:     #f59e0b;
    --green:    #10b981;
    --red:      #f43f5e;
    --amber:    #f97316;
    --violet:   #8b5cf6;
    --pink:     #ec4899;
    --chip-bg:  rgba(124,58,237,.12);
    --input-bg: #0f0b24;
    --input-border: rgba(139,92,246,.45);
    --shadow:   0 18px 44px rgba(0,0,0,.5);
"""
LIGHT_CSS = """
    --bg:       #eef8f2;
    --panel:    rgba(255,255,255,.72);
    --panel-2:  rgba(239,252,246,.86);
    --stroke:   rgba(20,121,86,.18);
    --text:     #123328;
    --muted:    #4f7464;
    --accent:   #12946b;
    --cyan:     #0e9f9a;
    --gold:     #b7791f;
    --green:    #138a51;
    --red:      #dc2626;
    --amber:    #d97706;
    --chip-bg:  rgba(18,148,107,.08);
    --input-bg: rgba(255,255,255,.78);
    --input-border: rgba(18,148,107,.28);
    --shadow:   0 18px 44px rgba(63,128,109,.18);
"""


# ══════════════════════════════════════════════════════
#  GLOBAL CSS — loaded from frontend/static/css/style.css
# ══════════════════════════════════════════════════════
import os as _os
_css_path = _os.path.join(_os.path.dirname(__file__), "frontend", "static", "css", "style.css")
with open(_css_path, "r", encoding="utf-8") as _f:
    _static_css = _f.read()

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');
:root {{ {DARK_CSS} }}
{_static_css}
</style>"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════
#  THEME INJECTION
# ══════════════════════════════════════════════════════
def inject_theme_css():
    is_light = "Light" in st.session_state.get("ui_theme", "🌙 Dark Mode")
    vars_css = LIGHT_CSS if is_light else DARK_CSS
    if is_light:
        app_bg = "background: radial-gradient(circle at 8% 10%, rgba(22,163,74,.22), transparent 30%), radial-gradient(circle at 92% 12%, rgba(124,58,237,.16), transparent 34%), radial-gradient(circle at 74% 88%, rgba(6,182,212,.16), transparent 36%), linear-gradient(135deg, #dff5ea 0%, #f8fff9 48%, #eef4ff 100%) !important;"
        app_bg_image = "radial-gradient(circle at 8% 10%, rgba(22,163,74,.22), transparent 30%), radial-gradient(circle at 92% 12%, rgba(124,58,237,.16), transparent 34%), radial-gradient(circle at 74% 88%, rgba(6,182,212,.16), transparent 36%), linear-gradient(135deg, #dff5ea 0%, #f8fff9 48%, #eef4ff 100%)"
        sb_bg = "linear-gradient(180deg, rgba(217,243,231,.98) 0%, rgba(231,249,240,.97) 46%, rgba(226,236,255,.96) 100%)"
        sb_var = "--sb-bg: rgba(217,243,231,.98); --app-bg-image: " + app_bg_image + ";"
    else:
        app_bg = "background: linear-gradient(135deg, #2e1065 0%, #1e0a4a 40%, #1a0533 100%) !important;"
        app_bg_image = "radial-gradient(ellipse at 0% 0%, #3b0764 0%, transparent 50%), radial-gradient(ellipse at 100% 0%, #1e1b4b 0%, transparent 50%), radial-gradient(ellipse at 50% 100%, #4c1d95 0%, transparent 55%), radial-gradient(ellipse at 100% 100%, #1a0533 0%, transparent 50%), linear-gradient(135deg, #2e1065 0%, #1e0a4a 35%, #0f0627 65%, #1a0533 100%)"
        sb_bg = "#1e0a4a"
        sb_var = "--sb-bg: #150732; --app-bg-image: " + app_bg_image + ";"
    st.markdown(f"""<style>
    :root {{ {vars_css} {sb_var} }}
    .stApp {{ {app_bg} }}
    section[data-testid="stSidebar"] {{ background: {sb_bg} !important; }}
    a.anchor-link, .anchor-link {{ display:none !important; visibility:hidden !important; }}
    .notice-success {{
        padding:13px 16px;
        border-radius:14px;
        border:1px solid {"rgba(22,163,74,.20)" if is_light else "rgba(16,185,129,.28)"};
        background:{"linear-gradient(135deg, rgba(22,163,74,.13), rgba(6,182,212,.08))" if is_light else "linear-gradient(135deg, rgba(16,185,129,.16), rgba(34,211,238,.08))"};
        color:{"#14532d" if is_light else "#d1fae5"};
        font-weight:850;
        margin:10px 0;
    }}
    .notice-info {{
        padding:13px 16px;
        border-radius:14px;
        border:1px solid {"rgba(14,159,154,.22)" if is_light else "rgba(139,92,246,.30)"};
        background:{"linear-gradient(135deg, rgba(6,182,212,.10), rgba(255,255,255,.58))" if is_light else "linear-gradient(135deg, rgba(124,58,237,.16), rgba(6,182,212,.08))"};
        color:var(--text);
        font-weight:760;
        margin:10px 0;
    }}
    .clean-title {{
        font-size:22px;
        font-weight:950;
        letter-spacing:-.2px;
        color:var(--text);
        margin:22px 0 10px;
    }}
    .soft-panel {{
        background:{"rgba(255,255,255,.70)" if is_light else "rgba(255,255,255,.035)"};
        border:1px solid var(--stroke);
        border-radius:22px;
        box-shadow:var(--shadow);
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius:24px !important;
        border:1px solid {"rgba(20,121,86,.16)" if is_light else "rgba(139,92,246,.28)"} !important;
        background:{"linear-gradient(145deg, rgba(255,255,255,.78), rgba(232,249,240,.72))" if is_light else "linear-gradient(145deg, rgba(31,15,68,.90), rgba(22,11,54,.86))"} !important;
        box-shadow:{"0 18px 48px rgba(31,111,83,.13)" if is_light else "0 18px 48px rgba(0,0,0,.28)"} !important;
    }}
    .bento-card, .metric-card, .eda-card, .pf-card, .smart-card, .viz-card, .panel-card {{
        border-radius:22px !important;
        border:1px solid {"rgba(20,121,86,.16)" if is_light else "rgba(139,92,246,.28)"} !important;
        background:{"linear-gradient(145deg, rgba(255,255,255,.78), rgba(232,249,240,.72))" if is_light else "linear-gradient(145deg, rgba(38,18,82,.92), rgba(22,11,54,.88))"} !important;
        box-shadow:{"0 18px 48px rgba(31,111,83,.14)" if is_light else "0 18px 48px rgba(0,0,0,.28)"} !important;
    }}
    .metric-card {{
        min-height:118px;
        padding:20px 22px !important;
        display:flex; flex-direction:column; justify-content:space-between; gap:8px;
        position:relative; overflow:hidden;
    }}
    .metric-card:before {{
        content:""; position:absolute; width:120px; height:120px; right:-44px; top:-48px;
        border-radius:50%; background:{"rgba(18,148,107,.10)" if is_light else "rgba(124,58,237,.16)"};
    }}
    .metric-card .metric-icon {{
        width:40px; height:40px; border-radius:14px; display:flex; align-items:center; justify-content:center;
        background:{"rgba(18,148,107,.10)" if is_light else "rgba(124,58,237,.16)"}; color:var(--accent); font-weight:950;
    }}
    .metric-card .metric-label {{
        font-size:12px !important; font-weight:950 !important; letter-spacing:1.1px; color:var(--muted) !important; text-transform:uppercase;
    }}
    .metric-card .metric-value {{
        font-size:32px !important; font-weight:950 !important; color:var(--text) !important; line-height:1.05;
    }}
    .feature-card {{
        border-radius:24px; padding:22px; min-height:170px;
        border:1px solid {"rgba(20,121,86,.16)" if is_light else "rgba(139,92,246,.26)"};
        background:{"linear-gradient(145deg, rgba(255,255,255,.82), rgba(224,246,236,.72))" if is_light else "linear-gradient(145deg, rgba(39,18,85,.96), rgba(19,8,48,.94))"};
        box-shadow:{"0 20px 55px rgba(31,111,83,.14)" if is_light else "0 20px 55px rgba(0,0,0,.32)"};
    }}
    .feature-card-title {{font-size:22px;font-weight:950;color:var(--text);margin-bottom:6px;}}
    .feature-card-sub {{font-size:13px;font-weight:750;color:var(--muted);line-height:1.55;}}
    .status-card {{
        padding:18px 22px; border-radius:20px; font-size:16px; font-weight:900; line-height:1.65;
        background:{"linear-gradient(135deg, rgba(20,184,166,.12), rgba(22,163,74,.10))" if is_light else "linear-gradient(135deg, rgba(6,182,212,.16), rgba(16,185,129,.12))"};
        color:{"#14532d" if is_light else "#d1fae5"};
        border:1px solid {"rgba(18,148,107,.22)" if is_light else "rgba(16,185,129,.28)"};
        margin:16px 0;
    }}
    div[data-testid="stMetric"] {{
        border-radius:22px !important;
        padding:18px 20px !important;
        border:1px solid {"rgba(20,121,86,.16)" if is_light else "rgba(139,92,246,.28)"} !important;
        background:{"linear-gradient(145deg, rgba(255,255,255,.80), rgba(232,249,240,.72))" if is_light else "linear-gradient(145deg, rgba(38,18,82,.92), rgba(22,11,54,.88))"} !important;
        box-shadow:{"0 18px 48px rgba(31,111,83,.14)" if is_light else "0 18px 48px rgba(0,0,0,.28)"} !important;
    }}
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
        font-size:12px !important; font-weight:950 !important; color:var(--muted) !important; letter-spacing:.5px !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-size:34px !important; font-weight:950 !important; color:var(--text) !important;
    }}
    </style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════
def fmt_int(value):
    try: return f"{int(value):,}".replace(",", ".")
    except: return str(value)

def clean_ui_label(label: str) -> str:
    """Remove leading decorative emoji/icons from labels while preserving internal page keys."""
    label = str(label)
    # Remove common leading emoji/codepoint decorations plus extra spaces
    return re.sub(r'^[^A-Za-z0-9#]+\s*', '', label).strip()

def clean_section_label(label: str) -> str:
    """Remove the category icon and format sidebar section title."""
    label = str(label)
    label = re.sub(r'^[^A-Za-z0-9]+\s*', '', label).strip()
    return label.title()

def strip_decorative_emoji(value: str) -> str:
    """Remove decorative emoji symbols from UI messages while keeping normal text."""
    value = str(value)
    value = re.sub(r'[\U00010000-\U0010ffff]', '', value)
    value = re.sub(r'[✅❌⚠️✔✖✕🔍🔎📌📊📈📉📄📤📥🧹🧠✨♻️🔁🗑️📁📦💡🎨🔗⏳🚀🏠👁️]', '', value)
    value = value.replace('  ', ' ').strip()
    return value

def file_size_label(size_bytes):
    if not size_bytes: return "-"
    units = ["B","KB","MB","GB"]; size = float(size_bytes); idx = 0
    while size >= 1024 and idx < len(units)-1: size /= 1024; idx += 1
    return f"{size:.2f} {units[idx]}"

def dataset_summary(df):
    if df is None: return {"rows":0,"cols":0,"numeric":0,"category":0,"missing":0,"duplicate":0,"date":0}
    return {
        "rows": df.shape[0], "cols": df.shape[1],
        "numeric": len(df.select_dtypes(include="number").columns),
        "category": len(df.select_dtypes(include=["object","category","bool"]).columns),
        "missing": int(df.isna().sum().sum()), "duplicate": int(df.duplicated().sum()),
        "date": len([c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]),
    }

def data_quality_score(df):
    if df is None or df.empty: return 0, "No Data"
    total = max(df.shape[0]*df.shape[1], 1)
    mp = df.isna().sum().sum()/total*100; dp = df.duplicated().sum()/max(df.shape[0],1)*100
    score = int(round(max(0, min(100, 100-mp-dp))))
    label = "Excellent" if score>=90 else "Good" if score>=75 else "Need Cleaning" if score>=55 else "Poor"
    return score, label

def metric_card(icon, label, value):
    label_clean = strip_decorative_emoji(label)
    # A compact bento-style tile. Icon is ignored when empty so no blank square appears.
    icon_html = f'<div class="metric-icon">{strip_decorative_emoji(icon)}</div>' if str(icon).strip() else ''
    return f"""<div class="metric-card">{icon_html}<div class="metric-label">{label_clean}</div><div class="metric-value">{fmt_int(value)}</div></div>"""

def go_to(page):
    st.session_state.active_page = page; st.session_state.nav_radio = page; st.session_state._scroll_to_main = True

def logout():
    for k in ["authenticated","username","user_role","df","df_original","active_page","nav_radio","before_snap","after_snap","before_df","after_df","cleaning_log","active_file","last_upload_signature","cleaning_notice"]:
        st.session_state[k] = DEFAULT_STATE.get(k, None) if k in DEFAULT_STATE else None
    st.session_state["authenticated"] = False
    st.session_state["active_page"] = "🏠 Dashboard"
    st.session_state["nav_radio"] = "🏠 Dashboard"

def scroll_to_main():
    if not st.session_state.get("_scroll_to_main", False): return
    components.html("""<script>setTimeout(()=>{const d=window.parent.document;const t=d.getElementById('main-anchor');if(t)t.scrollIntoView({behavior:'smooth'});else window.parent.scrollTo({top:0,behavior:'smooth'});},180);</script>""", height=0)
    st.session_state._scroll_to_main = False

def themed_dataframe_style(df_style):
    is_light = "Light" in st.session_state.get("ui_theme","🌙 Dark Mode")
    if is_light: bg,head,text,border = "#ffffff","#e8f5ec","#0a2218","#b7e4c7"
    else: bg,head,text,border = "#0d1526","#111e35","#e4eeff","#1e3a6b"
    return (df_style.style
        .set_table_styles([
            {"selector":"thead th","props":[("background-color",head),("color",text),("font-weight","900"),("border",f"1px solid {border}")]},
            {"selector":"tbody td","props":[("background-color",bg),("color",text),("border",f"1px solid {border}")]},
        ])
        .set_properties(**{"background-color":bg,"color":text,"border-color":border})
    )


def get_display_name(username=None):
    """Return user-facing name for sidebar/account display."""
    username = username or st.session_state.get("username", "")
    db = st.session_state.get("users_db", {})
    info = db.get(username, {}) if username else {}
    name = str(info.get("name", "")).strip()
    if name:
        return name
    return username.title() if username else "Guest"


def render_paginated_table(data, key="table", title=None, page_size_default=10, height=430):
    """Full-width dataframe viewer with search + pagination that never crashes.

    Notes:
    - `title` is intentionally ignored to avoid duplicate section headings.
    - height is dynamic so small tables don't show lots of empty grid rows.
    """
    if data is None:
        st.info("Tidak ada data untuk ditampilkan.")
        return
    try:
        data = pd.DataFrame(data).copy()
    except Exception:
        st.write(data)
        return
    if data.empty:
        st.info("Dataset kosong.")
        return

    top_left, top_right = st.columns([3, 1])
    with top_left:
        search = st.text_input("Search data", placeholder="Cari teks/angka di semua kolom...", key=f"{key}_search")
    with top_right:
        page_options = [10, 25, 50, 100, 200]
        idx = page_options.index(page_size_default) if page_size_default in page_options else 0
        page_size = st.selectbox("Rows/page", page_options, index=idx, key=f"{key}_page_size")

    view = data
    if search:
        s_text = str(search).lower()
        try:
            mask = view.astype(str).apply(lambda row: row.str.lower().str.contains(s_text, na=False).any(), axis=1)
            view = view[mask]
        except Exception:
            pass

    total_rows = len(view)
    total_pages = max(1, int(np.ceil(total_rows / page_size)))
    page_key = f"{key}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    st.session_state[page_key] = max(1, min(int(st.session_state[page_key]), total_pages))
    page = st.session_state[page_key]
    start = (page - 1) * page_size
    end = min(start + page_size, total_rows)
    current_view = view.iloc[start:end]

    # Dynamic table height: fit the number of shown rows, capped by the requested max height.
    shown_rows = max(1, len(current_view))
    dynamic_height = min(height, max(118, 38 * shown_rows + 48))

    try:
        st.dataframe(themed_dataframe_style(current_view), use_container_width=True, height=dynamic_height)
    except Exception:
        st.dataframe(current_view, use_container_width=True, height=dynamic_height)

    p1, p2, p3, p4 = st.columns([1, 1, 3, 1])
    with p1:
        if st.button("‹ Prev", key=f"{key}_prev", use_container_width=True, disabled=page <= 1):
            st.session_state[page_key] = max(1, page - 1)
            st.rerun()
    with p2:
        if st.button("Next ›", key=f"{key}_next", use_container_width=True, disabled=page >= total_pages):
            st.session_state[page_key] = min(total_pages, page + 1)
            st.rerun()
    with p3:
        shown_start = start + 1 if total_rows else 0
        st.caption(f"Halaman {page} dari {total_pages} · Menampilkan {shown_start}-{end} dari {fmt_int(total_rows)} baris hasil filter · Total asli {fmt_int(len(data))} baris")
    with p4:
        new_page = st.number_input("Page", min_value=1, max_value=total_pages, value=page, key=f"{key}_jump", label_visibility="collapsed")
        if int(new_page) != page:
            st.session_state[page_key] = int(new_page)
            st.rerun()



def _safe_to_datetime(series):
    """Convert many common date formats safely without crashing on newer pandas."""
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.to_datetime(series.astype(str), errors="coerce")


def _is_year_like(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return False
    ratio = ((vals >= 1900) & (vals <= 2100) & (vals % 1 == 0)).mean()
    unique_count = vals.nunique()
    return ratio >= 0.6 and unique_count >= 2


def detect_time_candidates(df):
    """Detect real date/time columns or year-like columns.

    Important: this function DOES NOT create a fake index timeline. If no valid
    time column is detected, Time Series page will show an info message instead
    of forcing an analysis.
    """
    candidates = []
    if df is None or df.empty:
        return candidates

    hints = (
        "date", "tanggal", "time", "datetime", "timestamp",
        "year", "tahun", "month", "bulan", "periode", "period"
    )

    for col in df.columns:
        ser = df[col]
        col_lower = str(col).lower().strip()

        if pd.api.types.is_datetime64_any_dtype(ser):
            candidates.append((col, "datetime"))
            continue

        # Numeric year column, e.g. 2020, 2021, 2022
        if ("year" in col_lower or "tahun" in col_lower) and _is_year_like(ser):
            candidates.append((col, "year"))
            continue

        # Date-like name: try parse, but only accept if most rows parse correctly
        if any(h in col_lower for h in hints):
            parsed = _safe_to_datetime(ser)
            parse_ratio = parsed.notna().mean() if len(parsed) else 0
            if parse_ratio >= 0.45:
                candidates.append((col, "parse"))
                continue
            if _is_year_like(ser):
                candidates.append((col, "year"))
                continue

        # Generic object/string column: detect real date strings
        if ser.dtype == "object" or str(ser.dtype).startswith("string"):
            parsed = _safe_to_datetime(ser)
            parse_ratio = parsed.notna().mean() if len(parsed) else 0
            if parse_ratio >= 0.65:
                candidates.append((col, "parse"))

    # de-duplicate while preserving order
    seen = set()
    out = []
    for col, kind in candidates:
        if col not in seen:
            seen.add(col)
            out.append((col, kind))
    return out


def _convert_to_datetime(series, kind="parse"):
    if kind == "year":
        vals = pd.to_numeric(series, errors="coerce")
        out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        mask = vals.notna()
        out.loc[mask] = pd.to_datetime(vals.loc[mask].astype(int).astype(str) + "-01-01", errors="coerce")
        return out
    return _safe_to_datetime(series)


def render_universal_time_series(df):
    """Time-series page that is safe for any dataset.

    If a dataset has a real time column, the page analyzes it. If not, it shows a
    clear info message and does not crash. It no longer forces an artificial row-index
    timeline because that can mislead users during presentation.
    """
    st.markdown("## Time Series Analytics")
    st.caption("Auto-detection kolom waktu. Jika dataset tidak memiliki kolom tanggal/waktu, halaman ini akan memberi info tanpa error.")

    if df is None or df.empty:
        st.warning("Dataset belum tersedia atau kosong.")
        return

    work = df.copy()
    time_candidates = detect_time_candidates(work)

    if not time_candidates:
        st.info("Dataset ini tidak memiliki kolom time series yang valid. Fitur Time Series dilewati, tetapi fitur EDA lain tetap dapat digunakan.")
        st.markdown("""
        <div class="panel-card" style="margin-top:14px;">
            <b>Contoh kolom waktu yang dapat dianalisis:</b><br>
            <span style="opacity:.85;">date, tanggal, timestamp, datetime, year/tahun, month/bulan, order_date, created_at</span>
        </div>
        """, unsafe_allow_html=True)
        return

    time_labels = [str(c[0]) for c in time_candidates]
    time_kind_map = {str(col): kind for col, kind in time_candidates}

    numeric_cols = [str(c) for c in work.select_dtypes(include="number").columns.tolist()]

    c1, c2, c3, c4 = st.columns([1.35, 1.35, 1, 1])
    date_sel = c1.selectbox("Kolom waktu", time_labels, key="ts_date_real_only")

    # Value options must not duplicate the chosen time column.
    value_options = [c for c in numeric_cols if c != date_sel]
    value_options.append("Jumlah Baris / Frekuensi")

    default_value_index = 0
    if value_options and value_options[0] == "Jumlah Baris / Frekuensi":
        default_value_index = 0

    val_sel = c2.selectbox("Nilai dianalisis", value_options, index=default_value_index, key="ts_value_real_only")
    period_label = c3.selectbox("Agregasi", ["Harian", "Mingguan", "Bulanan", "Tahunan"], index=2, key="ts_period_real_only")
    agg_label = c4.selectbox("Metode", ["Sum", "Mean", "Count"], index=0, key="ts_agg_real_only")
    window = st.slider("Moving Average", 2, 30, 7, key="ts_window_real_only")

    kind = time_kind_map.get(date_sel, "parse")
    ts = pd.DataFrame(index=work.index)
    ts["_time"] = _convert_to_datetime(work[date_sel], kind)

    if val_sel == "Jumlah Baris / Frekuensi" or val_sel not in work.columns:
        ts["_value"] = 1
        value_note = "jumlah baris/frekuensi"
        agg_func = "sum"
    else:
        ts["_value"] = pd.to_numeric(work[val_sel], errors="coerce")
        value_note = f"kolom nilai `{val_sel}`"
        agg_func = {"Sum": "sum", "Mean": "mean", "Count": "count"}.get(agg_label, "sum")

    ts = ts.dropna(subset=["_time", "_value"])

    if ts.empty:
        st.warning("Kolom waktu terdeteksi, tetapi nilainya tidak berhasil dikonversi menjadi tanggal. Coba pilih kolom waktu lain jika tersedia.")
        return

    freq_map = {"Harian": "D", "Mingguan": "W", "Bulanan": "M", "Tahunan": "Y"}
    freq = freq_map.get(period_label, "M")

    try:
        ts["periode"] = ts["_time"].dt.to_period(freq).dt.to_timestamp()
    except Exception:
        st.warning("Agregasi waktu gagal dibuat. Coba pilih agregasi lain atau kolom waktu lain.")
        return

    grouped = ts.groupby("periode", as_index=False).agg(value=("_value", agg_func)).sort_values("periode")
    grouped["moving_avg"] = grouped["value"].rolling(window=window, min_periods=1).mean()

    if grouped.empty:
        st.warning("Data time series kosong setelah agregasi.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(metric_card("", "Periode", len(grouped)), unsafe_allow_html=True)
    k2.markdown(metric_card("Σ", "Total", f"{grouped['value'].sum():,.2f}"), unsafe_allow_html=True)
    k3.markdown(metric_card("μ", "Rata-rata", f"{grouped['value'].mean():,.2f}"), unsafe_allow_html=True)
    trend_delta = grouped["value"].iloc[-1] - grouped["value"].iloc[0] if len(grouped) > 1 else 0
    k4.markdown(metric_card("", "Trend Δ", f"{trend_delta:,.2f}"), unsafe_allow_html=True)

    theme_mode = "light" if "Light" in st.session_state.get("ui_theme", "🌙 Dark Mode") else "dark"
    st.plotly_chart(
        plot_time_series(
            grouped.rename(columns={"periode": "Periode", "value": "Nilai", "moving_avg": "Moving Average"}),
            "Periode", "Nilai", window=window, theme=theme_mode, ma_col="Moving Average",
            title=f"Trend Time Series — {value_note}"
        ),
        use_container_width=True
    )

    if len(grouped) >= 2:
        direction = "naik" if trend_delta > 0 else "turun" if trend_delta < 0 else "stabil"
        st.markdown(f'<div class="status-card">Time series berhasil dianalisis menggunakan kolom waktu <b>{date_sel}</b> dan {value_note}. Tren akhir cenderung <b>{direction}</b> sebesar {trend_delta:,.2f} dari periode awal ke akhir.</div>', unsafe_allow_html=True)
    else:
        st.info("Time series berhasil dibuat, tetapi hanya ada 1 periode setelah agregasi. Coba ubah agregasi menjadi Harian/Mingguan untuk detail lebih banyak.")

    render_paginated_table(
        grouped.rename(columns={"periode": "Periode", "value": "Nilai", "moving_avg": "Moving Average"}),
        key="time_series_result",
        page_size_default=10,
        height=320
    )


# ══════════════════════════════════════════════════════
#  TEAM MEMBERS & PHOTO UTILITIES
# ══════════════════════════════════════════════════════
TEAM_MEMBERS = [
    {"nim":"52250037","name":"Nurul Iffah","aliases":["nurul","nurul_iffah","52250037"]},
    {"nim":"52250040","name":"Naisya Hafitz Mufidah","aliases":["naisya","naisyah","aisya","52250040"]},
    {"nim":"52250039","name":"Clara Maisie Wanghili","aliases":["clara","52250039"]},
    {"nim":"52250009","name":"Dhea Putri Khasanah","aliases":["dhea","52250009"]},
    {"nim":"52250038","name":"Fifi Muthia Pitaloka","aliases":["fifi","52250038"]},
]

def _image_data_uri(path):
    try:
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    except: return None

def _find_member_photo(member):
    dirs = [BASE_DIR/"frontend"/"static"/"assets"/"images"]
    exts = [".png",".jpg",".jpeg"]
    aliases = list(dict.fromkeys([*member.get("aliases",[]), member["nim"]]))
    for folder in dirs:
        if not folder.exists(): continue
        for alias in aliases:
            for ext in exts:
                p = folder/f"{alias}{ext}"
                if p.exists(): return p
        for f in folder.iterdir():
            if f.suffix.lower() in exts:
                stem = f.stem.lower()
                if any(a.lower() in stem for a in aliases): return f
    return None

def _get_avatar_path(member):
    img_dir = BASE_DIR/"frontend"/"static"/"assets"/"images"
    for name in [f"avatar_{member['nim']}.png", f"avatar_{member['aliases'][0]}.png"] if member.get("aliases") else [f"avatar_{member['nim']}.png"]:
        p = img_dir/name
        if p.exists(): return str(p)
    p = _find_member_photo(member)
    return str(p) if p else None

def _member_avatar_html(member, size=80, border_color="var(--cyan)"):
    path = _get_avatar_path(member)
    initials = "".join(p[0] for p in member["name"].split()[:2]).upper()
    if path:
        uri = _image_data_uri(Path(path))
        if uri:
            return f'<img src="{uri}" alt="{member["name"]}" style="width:{size}px;height:{size}px;border-radius:50%;object-fit:cover;object-position:center top;border:3px solid {border_color};box-shadow:0 0 16px rgba(34,211,238,.35);display:block;margin:0 auto;">'
    return f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--cyan));color:#fff;font-size:{size//4}px;font-weight:900;display:flex;align-items:center;justify-content:center;margin:0 auto;border:3px solid {border_color};">{initials}</div>'

def render_team_grid(card_class="", size=80):
    cards = []
    for m in TEAM_MEMBERS:
        avatar = _member_avatar_html(m, size)
        cards.append(f"""
        <div class="team-card-auth {card_class}">
            {avatar}
            <div class="team-name-auth">{m['name']}</div>
            <div class="team-nim-auth">{m['nim']}</div>
        </div>""")
    st.markdown(f'<div class="team-grid-auth">{"".join(cards)}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  AUTH PAGES — Sign In + Sign Up
# ══════════════════════════════════════════════════════
def auth_page():
    inject_theme_css()
    mode = st.session_state.get("register_mode", False)

    st.markdown("""
    <style>
    [data-testid="stHeader"],[data-testid="stToolbar"],
    [data-testid="stDecoration"],footer { display:none!important; }
    html, body, .stApp, [data-testid="stAppViewContainer"], .main {
        height:100vh!important;
        max-height:100vh!important;
        overflow:hidden!important;
    }
    .block-container {
        padding:.55rem 1rem!important;
        max-width:940px!important;
        margin:0 auto!important;
        min-height:100vh!important;
        display:flex!important;
        flex-direction:column!important;
        justify-content:center!important;
    }
    div[data-testid="InputInstructions"],
    div[data-testid="stTextInput"] small,
    [data-testid="stTextInput"] [data-testid="InputInstructions"] {
        display:none!important;
        visibility:hidden!important;
        height:0!important;
        margin:0!important;
        padding:0!important;
    }

    /* ── Uniform 4-side input border ── */
    [data-testid="stTextInput"]>div,
    [data-testid="stTextInput"]>div>div,
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] {
        background:transparent!important; border:none!important;
        box-shadow:none!important; outline:none!important;
    }
    [data-testid="stTextInput"] input {
        background:rgba(255,255,255,.07)!important;
        border-top:   1.5px solid rgba(255,255,255,.2)!important;
        border-right: 1.5px solid rgba(255,255,255,.2)!important;
        border-bottom:1.5px solid rgba(255,255,255,.2)!important;
        border-left:  1.5px solid rgba(255,255,255,.2)!important;
        border-radius:12px!important;
        padding:12px 16px!important; font-size:14px!important;
        color:#f0eeff!important; width:100%!important;
        outline:none!important; box-shadow:none!important;
    }
    [data-testid="stTextInput"] input:focus {
        border-top:   1.5px solid #7c3aed!important;
        border-right: 1.5px solid #7c3aed!important;
        border-bottom:1.5px solid #7c3aed!important;
        border-left:  1.5px solid #7c3aed!important;
        box-shadow:0 0 0 3px rgba(124,58,237,.18)!important;
        outline:none!important;
    }
    [data-testid="stTextInput"] input::placeholder { color:rgba(255,255,255,.3)!important; }
    [data-testid="stWidgetLabel"] { display:none!important; }

    div[data-testid="stFormSubmitButton"]>button {
        background:linear-gradient(135deg,#7c3aed,#4c1d95)!important;
        border:none!important; border-radius:12px!important;
        color:#fff!important; font-size:15px!important;
        font-weight:900!important; min-height:48px!important;
        box-shadow:0 8px 24px rgba(124,58,237,.4)!important;
        letter-spacing:1px!important; text-transform:uppercase!important;
        transition: all .25s ease!important;
    }
    div[data-testid="stFormSubmitButton"]>button:hover {
        box-shadow:0 12px 32px rgba(124,58,237,.6)!important;
        transform:translateY(-1px)!important;
    }
    .f-label { font-size:11px; font-weight:800; color:rgba(196,181,253,.7);
               letter-spacing:.8px; text-transform:uppercase;
               margin:10px 0 5px; display:block; }

    /* ── Native st.container(border=True) styling — used for BOTH panels ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius:24px!important;
        transition: all .45s cubic-bezier(.4,0,.2,1)!important;
        animation: fadeSlide .45s ease;
    }
    @keyframes fadeSlide {
        from { opacity:0; transform: translateX(12px); }
        to   { opacity:1; transform: translateX(0); }
    }
    /* Purple "switch" panel container */
    .switch-box div[data-testid="stVerticalBlockBorderWrapper"] {
        background:linear-gradient(160deg,#7c3aed 0%,#4c1d95 60%,#2e1065 100%)!important;
        border:none!important;
        box-shadow:0 20px 60px rgba(124,58,237,.3)!important;
    }
    .switch-box div[data-testid="stVerticalBlock"] { gap:0.4rem!important; }
    /* Form panel container */
    .form-box div[data-testid="stVerticalBlockBorderWrapper"] {
        background:#15092f!important;
        border:1px solid rgba(139,92,246,.25)!important;
        box-shadow:0 20px 60px rgba(0,0,0,.4)!important;
    }

    /* Switch button styling */
    .switch-box div[data-testid="stButton"]>button {
        border-radius:999px!important; font-weight:900!important;
        font-size:12.5px!important; min-height:42px!important;
        border:1.5px solid rgba(255,255,255,.6)!important;
        background:rgba(255,255,255,.1)!important; color:#fff!important;
        letter-spacing:1.5px!important; text-transform:uppercase!important;
        transition: all .2s ease!important;
    }
    .switch-box div[data-testid="stButton"]>button:hover {
        background:rgba(255,255,255,.25)!important;
        transform:translateY(-1px)!important;
    }

    /* Team strip inside switch panel */
    .team-strip-row {
        display:grid; grid-template-columns:repeat(5,1fr);
        gap:6px; margin-top:8px;
    }
    .tm-s { text-align:center; }
    .tm-s-name { font-size:9px; font-weight:900; color:#fff; margin-top:5px; line-height:1.2; }
    .tm-s-nim  { font-size:7.5px; font-family:'JetBrains Mono',monospace; color:rgba(255,255,255,.7); margin-top:1px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Mode-dependent text & order ──
    if not mode:
        purple_title, purple_text = "Hello, Friend!", "Belum punya akun? Daftar sekarang untuk mengakses semua fitur Auto EDA Insight Dashboard."
        purple_btn = "Register"
        form_title = "Login"
        form_first = True
    else:
        purple_title, purple_text = "Welcome Back!", "Sudah punya akun? Masuk untuk melanjutkan eksplorasi data kamu di Auto EDA Insight."
        purple_btn = "Login"
        form_title = "Create Account"
        form_first = False

    # ── Logo header ──
    st.markdown(
        '<div style="text-align:center;padding:0 0 10px;">'
        '<div style="font-size:22px;filter:drop-shadow(0 0 16px rgba(139,92,246,.9));margin-bottom:6px;">◈</div>'
        '<div style="font-size:24px;font-weight:950;color:#fff;letter-spacing:-1px;'
        'text-shadow:0 0 30px rgba(139,92,246,.7);">Auto EDA Insight</div>'
        '<div style="font-family:JetBrains Mono,monospace;font-size:9.5px;color:#f59e0b;'
        'letter-spacing:3px;text-transform:uppercase;margin-top:4px;">◆Data Science Programming · Kelompok 6 · ITSB</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── TEAM photos HTML (used inside switch panel) ──
    team_cards = []
    for m in TEAM_MEMBERS:
        av = _member_avatar_html(m, size=38, border_color="rgba(255,255,255,.6)")
        team_cards.append(
            '<div class="tm-s">' + av +
            '<div class="tm-s-name">' + m["name"].split()[0] + '</div>' +
            '<div class="tm-s-nim">'  + m["nim"] + '</div></div>'
        )
    team_html = '<div class="team-strip-row">' + "".join(team_cards) + '</div>'

    # ── TWO-COLUMN LAYOUT — order swaps based on mode ──
    col_a, col_b = st.columns([1, 1], gap="medium")
    form_col   = col_a if form_first else col_b
    purple_col = col_b if form_first else col_a

    # ── PURPLE SWITCH PANEL — native container, everything INSIDE ──
    with purple_col:
        st.markdown('<div class="switch-box">', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                f'<div style="text-align:center;color:#fff;padding:8px 4px;">'
                f'<div style="font-size:21px;font-weight:950;margin-bottom:6px;letter-spacing:-.5px;">{purple_title}</div>'
                f'<div style="font-size:11.5px;opacity:.85;line-height:1.45;margin-bottom:10px;">{purple_text}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            _, bcenter, _ = st.columns([1,1.6,1])
            with bcenter:
                if st.button(purple_btn, key="toggle_mode", use_container_width=True):
                    st.session_state.register_mode = not mode
                    st.rerun()
            st.markdown(team_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── FORM PANEL — native container, everything INSIDE ──
    with form_col:
        st.markdown('<div class="form-box">', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                f'<div style="font-size:20px;font-weight:950;margin-bottom:8px;color:#fff;text-align:center;">{form_title}</div>',
                unsafe_allow_html=True
            )

            if not mode:
                with st.form("login_form", clear_on_submit=False):
                    st.markdown('<span class="f-label">Username</span>', unsafe_allow_html=True)
                    username = st.text_input("_u", placeholder="Username kamu",
                                             label_visibility="collapsed", key="login_user")
                    st.markdown('<span class="f-label">Password</span>', unsafe_allow_html=True)
                    password = st.text_input("_p", placeholder="Password kamu",
                                             type="password", label_visibility="collapsed", key="login_pass")
                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                    submitted = st.form_submit_button("Sign In", use_container_width=True)
                if submitted:
                    db = st.session_state.users_db
                    username = username.strip().lower()
                    if username in db and db[username]["password"] == password:
                        st.session_state.authenticated = True
                        st.session_state.username    = username
                        st.session_state.user_role   = db[username].get("role","member")
                        st.session_state.active_page = "🏠 Dashboard"
                        st.session_state.nav_radio   = "🏠 Dashboard"
                        st.rerun()
                    else:
                        st.error("Username atau password salah.")
            else:
                with st.form("register_form", clear_on_submit=True):
                    st.markdown('<span class="f-label">Username</span>', unsafe_allow_html=True)
                    new_user  = st.text_input("_ru", placeholder="Pilih username unik",  label_visibility="collapsed")
                    st.markdown('<span class="f-label">Nama Lengkap</span>', unsafe_allow_html=True)
                    new_name  = st.text_input("_rn", placeholder="Nama lengkap kamu",    label_visibility="collapsed")
                    st.markdown('<span class="f-label">Password</span>', unsafe_allow_html=True)
                    new_pass  = st.text_input("_rp", placeholder="Min. 6 karakter",      type="password", label_visibility="collapsed")
                    st.markdown('<span class="f-label">Ulangi Password</span>', unsafe_allow_html=True)
                    new_pass2 = st.text_input("_rp2", placeholder="Konfirmasi password", type="password", label_visibility="collapsed")
                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                    reg_submit = st.form_submit_button("Sign Up", use_container_width=True)
                if reg_submit:
                    new_user = new_user.strip().lower()
                    new_name = new_name.strip() or new_user.title()
                    if not new_user or not new_pass:   st.error("Username & password wajib diisi.")
                    elif len(new_pass) < 6:            st.error("Password minimal 6 karakter.")
                    elif new_pass != new_pass2:        st.error("Konfirmasi password tidak cocok.")
                    elif new_user in st.session_state.users_db: st.error("Username sudah dipakai.")
                    else:
                        st.session_state.users_db[new_user] = {"password":new_pass,"role":"member","name":new_name}
                        st.success("Akun berhasil dibuat! Silakan Masuk.")
                        st.session_state.register_mode = False; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Keyboard helper: Enter di username pindah ke password, Enter di password langsung Sign In.
    components.html("""
    <script>
    (function(){
      const doc = window.parent.document;
      function visible(el){
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      }
      function setupLoginEnter(){
        const inputs = Array.from(doc.querySelectorAll('input')).filter(visible);
        const user = inputs.find(x => (x.getAttribute('placeholder') || '') === 'Username kamu');
        const pass = inputs.find(x => (x.getAttribute('placeholder') || '') === 'Password kamu');
        if(user && pass && !user.dataset.enterFocusFixed){
          user.dataset.enterFocusFixed = '1';
          user.addEventListener('keydown', function(e){
            if(e.key === 'Enter'){
              e.preventDefault();
              e.stopPropagation();
              pass.focus();
            }
          }, true);
        }
        if(pass && !pass.dataset.enterSubmitFixed){
          pass.dataset.enterSubmitFixed = '1';
          pass.addEventListener('keydown', function(e){
            if(e.key === 'Enter'){
              const buttons = Array.from(doc.querySelectorAll('button'));
              const btn = buttons.find(b => (b.innerText || '').trim().toLowerCase() === 'sign in');
              if(btn){
                e.preventDefault();
                e.stopPropagation();
                btn.click();
              }
            }
          }, true);
        }
      }
      setupLoginEnter();
      setTimeout(setupLoginEnter, 300);
      setTimeout(setupLoginEnter, 900);
    })();
    </script>
    """, height=0)


# ══════════════════════════════════════════════════════
#  SIDEBAR — Categorised navigation
# ══════════════════════════════════════════════════════
def render_sidebar():
    inject_theme_css()
    is_light = "Light" in st.session_state.get("ui_theme","🌙 Dark Mode")

    with st.sidebar:
        st.markdown(f"""
        <style>
        section[data-testid="stSidebar"] {{
            background: {"linear-gradient(180deg, rgba(217,243,231,.98) 0%, rgba(226,246,237,.98) 45%, rgba(224,235,255,.97) 100%)" if is_light else "#150732"} !important;
            border-right: 1px solid {"rgba(18,148,107,.20)" if is_light else "rgba(139,92,246,.25)"} !important;
            transition: width .25s ease, min-width .25s ease, max-width .25s ease, transform .25s ease !important;
        }}
        [data-testid="stAppViewContainer"] .main .block-container {{
            max-width: 100% !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
        }}
        section[data-testid="stSidebar"][aria-expanded="false"],
        section[data-testid="stSidebar"][data-collapsed="true"] {{
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            overflow: hidden !important;
        }}
        section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child,
        section[data-testid="stSidebar"][data-collapsed="true"] > div:first-child {{
            display: none !important;
        }}
        section[data-testid="stSidebar"] > div:first-child {{ padding: 16px 14px !important; }}
        [data-testid="stSidebarCollapseButton"] {{ visibility:visible !important; }}

        .sb-brand {{
            display:flex; align-items:center; gap:10px;
            padding: 4px 4px 16px;
            border-bottom: 1px solid {"rgba(22,163,74,.12)" if is_light else "rgba(139,92,246,.18)"};
            margin-bottom: 12px;
        }}
        .sb-logo-box {{
            width:38px; height:38px; border-radius:11px; flex-shrink:0;
            background: {"linear-gradient(135deg,#16a34a,#14532d)" if is_light else "linear-gradient(135deg,#7c3aed,#4c1d95)"};
            display:flex; align-items:center; justify-content:center;
            font-size:18px; color:#fff; box-shadow:{"0 8px 20px rgba(18,148,107,.26)" if is_light else "0 4px 14px rgba(124,58,237,.4)"};
        }}
        .sb-brand-title {{ font-size:14px; font-weight:950; color:{"#0a2218" if is_light else "#fff"}; line-height:1.2; }}
        .sb-brand-sub {{ font-size:9px; font-family:'JetBrains Mono',monospace; color:{"#16a34a" if is_light else "#7c3aed"};
                         letter-spacing:2px; text-transform:uppercase; font-weight:800; }}
        .sb-user-name {{ margin-top:5px; display:inline-flex; gap:5px; align-items:center;
                         padding:4px 9px; border-radius:999px; font-size:10.5px; font-weight:900;
                         color:{"#14532d" if is_light else "#e9d5ff"};
                         background:{"rgba(22,163,74,.10)" if is_light else "rgba(124,58,237,.18)"}; }}

        /* Top-level nav buttons (no group) */
        section[data-testid="stSidebar"] [data-testid="stButton"] > button {{
            width:100% !important; text-align:left !important; justify-content:flex-start !important;
            border-radius:10px !important; border:none !important;
            background:transparent !important; box-shadow:none !important;
            color:{"#2d5a3d" if is_light else "rgba(224,217,255,.85)"} !important;
            font-size:13px !important; font-weight:700 !important;
            padding:9px 12px !important; min-height:38px !important;
            margin:1px 0 !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {{
            background:{"rgba(22,163,74,.07)" if is_light else "rgba(124,58,237,.15)"} !important;
            color:{"#0a2218" if is_light else "#fff"} !important; transform:none !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {{
            background:{"linear-gradient(135deg,#16a34a,#14532d)" if is_light else "linear-gradient(135deg,#7c3aed,#4c1d95)"} !important;
            color:#fff !important; font-weight:900 !important;
            box-shadow:{"0 3px 10px rgba(22,163,74,.35)" if is_light else "0 3px 10px rgba(124,58,237,.35)"} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]:hover {{
            background:{"linear-gradient(135deg,#22c55e,#15803d)" if is_light else "linear-gradient(135deg,#8b4ff5,#5b21b6)"} !important; color:#fff !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="secondary"] {{
            border:1px solid {"rgba(22,163,74,.25)" if is_light else "rgba(139,92,246,.25)"} !important;
        }}

        /* Group header — collapsible look with chevron */
        .sb-group-header {{
            display:flex; align-items:center; gap:10px;
            padding:10px 12px; border-radius:10px; cursor:pointer;
            font-size:13px; font-weight:900; color:{"#0a2218" if is_light else "#fff"};
            margin-top:4px;
        }}
        .sb-group-header.open {{
            background:{"#e8f5ec" if is_light else "rgba(124,58,237,.12)"};
        }}
        .sb-group-icon {{
            width:30px; height:30px; border-radius:9px; flex-shrink:0;
            display:flex; align-items:center; justify-content:center; font-size:15px;
            background:{"#d1fae5" if is_light else "rgba(124,58,237,.18)"};
        }}
        .sb-group-chevron {{ margin-left:auto; font-size:11px; color:{"#6aad84" if is_light else "rgba(196,181,253,.5)"}; }}

        /* Sub-item indentation guide */
        .sb-subitem-wrap {{
            margin-left:18px; border-left:1.5px solid {"#b7e4c7" if is_light else "rgba(139,92,246,.2)"};
            padding-left:10px; margin-bottom:4px;
        }}
        section[data-testid="stSidebar"] .sb-subitem-wrap [data-testid="stButton"] > button {{
            font-size:12.5px !important; font-weight:600 !important; padding:7px 10px !important; min-height:32px !important;
        }}

        /* Expander used as collapsible group — strip default styling */
        section[data-testid="stSidebar"] [data-testid="stExpander"] {{
            border:none !important; background:transparent !important; box-shadow:none !important;
            margin-bottom:2px !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
            padding:10px 12px !important; border-radius:10px !important;
            font-size:13px !important; font-weight:900 !important;
            color:{"#0a2218" if is_light else "#fff"} !important;
            background:transparent !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
            background:{"#e8f5ec" if is_light else "rgba(124,58,237,.1)"} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] summary {{
            background:{"#e8f5ec" if is_light else "rgba(124,58,237,.12)"} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stExpander"] > div > div {{
            border:none !important; padding:2px 0 4px 14px !important;
        }}

        .sb-divider {{ height:1px; background:{"rgba(22,163,74,.12)" if is_light else "rgba(139,92,246,.18)"}; margin:10px 4px; }}
        section[data-testid="stSidebar"] [data-testid="stToggle"] {{ margin:0 0 2px 0 !important; }}
        section[data-testid="stSidebar"] [data-testid="stToggle"] label {{ min-height:32px !important; }}
        section[data-testid="stSidebar"] [data-testid="stToggle"] div[role="switch"] {{ transform:scale(.88); transform-origin:left center; }}

        .sb-dataset-card {{
            margin-top:8px; padding:12px 14px; border-radius:14px;
            background:{"#e8f5ec" if is_light else "rgba(124,58,237,.1)"};
            border:1px solid {"rgba(22,163,74,.15)" if is_light else "rgba(139,92,246,.22)"};
        }}
        .sb-dataset-label {{ font-size:9px; font-weight:900; letter-spacing:1.5px; text-transform:uppercase;
                              color:{"#16a34a" if is_light else "#7c3aed"}; margin-bottom:4px; }}
        .sb-dataset-name {{ font-size:12px; font-weight:900; color:{"#0a2218" if is_light else "#fff"};
                            word-break:break-word; line-height:1.3; }}
        .sb-dataset-meta {{ font-size:11px; color:{"#4a6b56" if is_light else "rgba(196,181,253,.6)"}; margin-top:4px; }}
        </style>
        """, unsafe_allow_html=True)

        if st.session_state.active_page not in ALL_PAGES:
            st.session_state.active_page = "🏠 Dashboard"
        active = st.session_state.active_page

        # ── TOP THEME SWITCH ──
        dark_now = not is_light
        top_cols = st.columns([1.2, 1.0], gap="small")
        with top_cols[0]:
            new_dark = st.toggle("Dark Mode", value=dark_now, key="theme_toggle_switch", label_visibility="collapsed")
        wanted_theme = "Dark Mode" if new_dark else "Light Mode"
        if wanted_theme != st.session_state.get("ui_theme"):
            st.session_state.ui_theme = wanted_theme
            st.rerun()
        st.markdown(
            '<div style="margin-top:-26px;margin-left:54px;font-size:10px;font-weight:950;'
            'letter-spacing:1.3px;text-transform:uppercase;color:var(--muted);line-height:1;">'
            + ("Dark Mode" if dark_now else "Light Mode") + '</div>'
            '<div style="height:14px"></div>',
            unsafe_allow_html=True
        )

        # ── BRAND + ACCOUNT ──
        display_name = get_display_name()
        st.markdown(
            '<div class="sb-brand">'
            '<div class="sb-logo-box">◈</div>'
            '<div><div class="sb-brand-title">Auto EDA Insight</div>'
            '<div class="sb-brand-sub">KELOMPOK 6</div>'
            '<div class="sb-user-name">' + display_name + '</div></div>'
            '</div>',
            unsafe_allow_html=True
        )

        # ── HOME — standalone top-level item (like "Profile" in reference) ──
        for page in NAV_CATEGORIES.get("🏠 HOME", []):
            is_active = (page == active)
            if st.button(clean_ui_label(page), key=f"nav_{page}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.active_page = page
                st.session_state._scroll_to_main = True
                st.rerun()

        # ── GROUPED NAVIGATION via collapsible expanders ──
        group_icons = {}
        for cat_name, pages in NAV_CATEGORIES.items():
            if cat_name == "🏠 HOME":
                continue
            clean_label = clean_section_label(cat_name)
            icon = ""
            group_has_active = active in pages
            with st.expander(clean_label, expanded=group_has_active):
                for page in pages:
                    is_active = (page == active)
                    if st.button(clean_ui_label(page), key=f"nav_{page}", use_container_width=True,
                                 type="primary" if is_active else "secondary"):
                        st.session_state.active_page = page
                        st.session_state._scroll_to_main = True
                        st.rerun()

        # ── DATASET STATUS ──
        df = st.session_state.df
        if df is not None:
            s = dataset_summary(df)
            score, qlabel = data_quality_score(df)
            fname = st.session_state.active_file.get("name","-") if st.session_state.active_file else "-"
            badge_clr = "#10b981" if score>=75 else "#f43f5e" if score<55 else "#f97316"
            st.markdown(
                '<div class="sb-dataset-card">'
                '<div class="sb-dataset-label">Active Dataset</div>'
                '<div class="sb-dataset-name">' + fname + '</div>'
                '<div class="sb-dataset-meta">' + fmt_int(s["rows"]) + ' baris · ' + str(s["cols"]) + ' kolom</div>'
                '<div style="margin-top:6px;display:inline-block;padding:3px 10px;border-radius:999px;'
                'font-size:10px;font-weight:900;background:' + badge_clr + '22;color:' + badge_clr + ';">'
                + qlabel + ' · ' + str(score) + '/100</div>'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # ── LOGOUT ──
        if st.button("Logout", key="sb_logout", use_container_width=True, on_click=logout):
            st.rerun()

# ══════════════════════════════════════════════════════
#  DATA CLEANING — Multi-choice table UI
# ══════════════════════════════════════════════════════
CLEANING_OPS = [
    {
        "id": "drop_dup",
        "name": "Hapus Baris Duplikat",
        "icon": "🔁",
        "category": "Deduplication",
        "description": "Menghapus baris yang identik secara keseluruhan. Cocok bila ada data entry ganda.",
        "impact": "High",
        "when": "Ada duplikat terdeteksi",
        "fn_key": "drop_duplicates",
        "needs_col": False,
        "needs_dtype": False,
    },
    {
        "id": "drop_missing",
        "name": "Hapus Baris Missing Values",
        "icon": "🗑️",
        "category": "Missing Value",
        "description": "Menghapus semua baris yang memiliki minimal satu nilai kosong (NaN).",
        "impact": "High",
        "when": "Missing value tidak bisa diisi / sudah banyak",
        "fn_key": "drop_missing_rows",
        "needs_col": False,
        "needs_dtype": False,
    },
    {
        "id": "fill_mean",
        "name": "Isi Missing → Mean",
        "icon": "μ",
        "category": "Missing Value",
        "description": "Mengisi nilai kosong numerik dengan rata-rata kolom tersebut.",
        "impact": "Medium",
        "when": "Distribusi simetris / tidak banyak outlier",
        "fn_key": "fill_missing_mean",
        "needs_col": False,
        "needs_dtype": False,
    },
    {
        "id": "fill_median",
        "name": "Isi Missing → Median",
        "icon": "M",
        "category": "Missing Value",
        "description": "Mengisi nilai kosong numerik dengan median kolom tersebut. Lebih robust dari mean.",
        "impact": "Medium",
        "when": "Ada outlier atau distribusi skewed",
        "fn_key": "fill_missing_median",
        "needs_col": False,
        "needs_dtype": False,
    },
    {
        "id": "fill_mode",
        "name": "Isi Missing → Mode",
        "icon": "♻",
        "category": "Missing Value",
        "description": "Mengisi nilai kosong semua kolom (numerik & kategorik) dengan nilai paling sering muncul.",
        "impact": "Low",
        "when": "Kolom kategorik atau distribusi jelas miring ke satu nilai",
        "fn_key": "fill_missing_mode",
        "needs_col": False,
        "needs_dtype": False,
    },
    {
        "id": "drop_col",
        "name": "Hapus Kolom",
        "icon": "❌",
        "category": "Column Management",
        "description": "Menghapus kolom yang tidak relevan dari dataset.",
        "impact": "High",
        "when": "Kolom tidak diperlukan / terlalu banyak missing",
        "fn_key": "drop_column",
        "needs_col": True,
        "needs_dtype": False,
    },
    {
        "id": "convert_dtype",
        "name": "Ubah Tipe Data Kolom",
        "icon": "🔄",
        "category": "Type Conversion",
        "description": "Mengkonversi tipe data sebuah kolom ke tipe yang lebih sesuai.",
        "impact": "Medium",
        "when": "Tipe data terdeteksi salah saat import",
        "fn_key": "convert_dtype",
        "needs_col": True,
        "needs_dtype": True,
    },
]

FN_MAP = {
    "drop_duplicates": drop_duplicates,
    "drop_missing_rows": drop_missing_rows,
    "fill_missing_mean": fill_missing_mean,
    "fill_missing_median": fill_missing_median,
    "fill_missing_mode": fill_missing_mode,
}

def render_cleaning_page(df):
    s = dataset_summary(df)
    st.markdown("## Data Cleaning")
    st.markdown(f"""
    <div class="eda-card eda-card-sm" style="margin-bottom:16px;">
        <div style="display:flex; gap:24px; flex-wrap:wrap; font-size:15px; font-weight:700;">
            <span>Dataset: <b>{s['rows']} baris × {s['cols']} kolom</b></span>
            <span style="color:var(--amber);">Missing: <b>{fmt_int(s['missing'])}</b></span>
            <span style="color:var(--red);">Duplikat: <b>{fmt_int(s['duplicate'])}</b></span>
        </div>
    </div>""", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Pilih & Jalankan Operasi", "Log & Before/After", "Reset"])

    with tab1:
        st.markdown("### Pilih Operasi Cleaning")
        st.markdown('<div class="callout" style="margin-bottom:16px;">Centang satu atau beberapa operasi di bawah, atur parameter jika diperlukan, lalu klik <b>Jalankan Operasi Terpilih</b>.</div>', unsafe_allow_html=True)

        # Build category groups
        cats = {}
        for op in CLEANING_OPS:
            cats.setdefault(op["category"], []).append(op)

        selected_ops = []
        col_extras = {}  # op_id → (col, dtype)

        for cat_name, ops in cats.items():
            st.markdown(f'<div class="section-chip">{cat_name}</div>', unsafe_allow_html=True)
            # Table header
            st.markdown("""
            <table class="clean-table">
              <thead><tr>
                <th style="width:40px;"></th>
                <th>Operasi</th><th>Kategori</th><th>Dampak</th><th>Kapan digunakan?</th>
              </tr></thead>
            </table>""", unsafe_allow_html=True)

            for op in ops:
                impact_cls = "impact-high" if op["impact"]=="High" else "impact-medium" if op["impact"]=="Medium" else "impact-low"
                col_cb, col_info = st.columns([1, 5])
                with col_cb:
                    checked = st.checkbox("", key=f"op_{op['id']}", label_visibility="collapsed")
                with col_info:
                    st.markdown(f"""
                    <table class="clean-table" style="margin-top:0;">
                      <tbody><tr>
                        <td style="width:40px;"></td>
                        <td><span class="op-badge">{op['name']}</span><br>
                            <span style="font-size:12px; color:var(--muted);">{op['description']}</span></td>
                        <td><span class="badge">{op['category']}</span></td>
                        <td class="{impact_cls}">{op['impact']}</td>
                        <td style="font-size:13px; color:var(--muted);">{op['when']}</td>
                      </tr></tbody>
                    </table>""", unsafe_allow_html=True)

                if checked:
                    selected_ops.append(op)
                    # Extra params if needed
                    if op["needs_col"] or op["needs_dtype"]:
                        with st.expander(f"Parameter untuk: {op['name']}", expanded=True):
                            extra_col = None; extra_dtype = None
                            if op["needs_col"]:
                                extra_col = st.selectbox(f"Pilih kolom ({op['name']})", df.columns, key=f"col_{op['id']}")
                            if op["needs_dtype"]:
                                extra_dtype = st.selectbox("Target tipe data", ["float64","int64","str","datetime64[ns]"], key=f"dtype_{op['id']}")
                            col_extras[op["id"]] = (extra_col, extra_dtype)
            st.markdown("<br>", unsafe_allow_html=True)

        if selected_ops:
            st.markdown(f'<div class="callout"><b>{len(selected_ops)} operasi dipilih:</b> {", ".join([o["name"] for o in selected_ops])}</div>', unsafe_allow_html=True)

        run_clicked = st.button("Jalankan Operasi Terpilih", type="primary", use_container_width=True, key="run_cleaning")
        status_box = st.empty()
        if st.session_state.get("cleaning_notice") and not run_clicked:
            status_box.markdown('<div class="status-card">' + st.session_state.cleaning_notice + '</div>', unsafe_allow_html=True)

        if run_clicked:
            if not selected_ops:
                st.warning("Pilih minimal 1 operasi terlebih dahulu.")
            else:
                current_df = st.session_state.df.copy()
                all_msgs = []
                st.session_state.cleaning_notice = ""
                progress = st.progress(0, text="Menyiapkan proses cleaning...")
                with st.spinner("Cleaning sedang diproses..."):
                    for i, op in enumerate(selected_ops, start=1):
                        status_box.markdown(f'<div class="notice-info">Menjalankan operasi {i}/{len(selected_ops)}: {op["name"]}</div>', unsafe_allow_html=True)
                        before_df_snap = current_df.copy()
                        extras = col_extras.get(op["id"], (None, None))
                        extra_col, extra_dtype = extras
                        try:
                            time.sleep(0.08)
                            if op["fn_key"] in FN_MAP:
                                new_df, bef, aft, msg = FN_MAP[op["fn_key"]](current_df)
                            elif op["fn_key"] == "drop_column" and extra_col:
                                new_df, bef, aft, msg = drop_column(current_df, extra_col)
                            elif op["fn_key"] == "convert_dtype" and extra_col and extra_dtype:
                                new_df, bef, aft, msg = convert_dtype(current_df, extra_col, extra_dtype)
                            else:
                                st.warning(f"Operasi '{op['name']}' membutuhkan parameter tambahan."); continue
                            current_df = new_df
                            st.session_state.before_snap = bef
                            st.session_state.after_snap = aft
                            st.session_state.before_df = before_df_snap
                            st.session_state.after_df = current_df.copy()
                            st.session_state.last_cleaning_operation = op["name"]
                            ts = datetime.datetime.now().strftime("%H:%M:%S")
                            st.session_state.cleaning_log.append(f"[{ts}] {msg}")
                            all_msgs.append(msg)
                            progress.progress(i / len(selected_ops), text=f"Selesai {i}/{len(selected_ops)} operasi")
                        except Exception as e:
                            st.error(f"Error pada '{op['name']}': {e}")
                st.session_state.df = current_df
                if all_msgs:
                    st.session_state.cleaning_notice = "Proses cleaning selesai: " + " | ".join(all_msgs)
                    status_box.markdown('<div class="status-card">' + st.session_state.cleaning_notice + '</div>', unsafe_allow_html=True)
                st.rerun()

    with tab2:
        if st.session_state.before_snap and st.session_state.after_snap:
            b, a = st.session_state.before_snap, st.session_state.after_snap
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Baris Before", fmt_int(b["shape"][0]))
            c2.metric("Baris After", fmt_int(a["shape"][0]), delta=fmt_int(a["shape"][0]-b["shape"][0]))
            c3.metric("Missing Before", fmt_int(b["missing_total"]))
            c4.metric("Missing After", fmt_int(a["missing_total"]), delta=fmt_int(a["missing_total"]-b["missing_total"]))
            # Before/After dataframes
            if st.session_state.before_df is not None and st.session_state.after_df is not None:
                st.markdown('<div class="clean-title">Before Cleaning</div>', unsafe_allow_html=True)
                render_paginated_table(st.session_state.before_df, key="clean_before", page_size_default=10, height=360)
                st.markdown('<div class="clean-title">After Cleaning</div>', unsafe_allow_html=True)
                render_paginated_table(st.session_state.after_df, key="clean_after", page_size_default=10, height=360)
        else:
            st.info("Jalankan operasi cleaning untuk melihat perbandingan before/after.")

        if st.session_state.cleaning_log:
            st.markdown("#### Log Operasi")
            for log_entry in reversed(st.session_state.cleaning_log):
                st.markdown(f'<div class="log-chip">• {log_entry}</div>', unsafe_allow_html=True)

    with tab3:
        if st.session_state.df_original is not None:
            if st.button("Reset ke Data Original", use_container_width=True):
                st.session_state.df = st.session_state.df_original.copy()
                st.session_state.cleaning_log = []
                st.session_state.before_snap = None; st.session_state.after_snap = None
                st.session_state.before_df = None; st.session_state.after_df = None
                st.session_state.last_cleaning_operation = ""
                st.session_state.cleaning_notice = ""
                st.success("Data berhasil direset ke kondisi awal."); st.rerun()
        else:
            st.info("Tidak ada data original untuk direset.")


# ══════════════════════════════════════════════════════
#  INSIGHTS BUILDER
# ══════════════════════════════════════════════════════
def build_initial_intelligent_insights(df):
    if df is None: return ["Upload dataset untuk melihat initial intelligent insights."]
    insights = []
    rows, cols = df.shape
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    missing_total = int(df.isna().sum().sum())
    dup_total = int(df.duplicated().sum())
    total_cells = max(rows*cols, 1)
    insights.append(f"Dataset memiliki {fmt_int(rows)} baris dan {fmt_int(cols)} kolom ({len(num_cols)} numerik, {len(cat_cols)} kategorik).")
    if missing_total:
        top_missing = df.isna().sum().sort_values(ascending=False).head(3)
        top_missing = top_missing[top_missing>0]
        detail = ", ".join([f"{i} ({fmt_int(v)})" for i,v in top_missing.items()])
        insights.append(f"Missing value: {fmt_int(missing_total)} sel ({missing_total/total_cells*100:.1f}%). Kolom terbanyak: {detail}.")
    else:
        insights.append("Dataset bersih dari missing value — siap untuk analisis lanjutan.")
    if dup_total:
        insights.append(f"Terdapat {fmt_int(dup_total)} baris duplikat. Disarankan Hapus Baris Duplikat.")
    else:
        insights.append("Tidak ada baris duplikat terdeteksi.")
    if num_cols:
        outlier_notes = []
        for col in num_cols[:8]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s)<4: continue
            q1,q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3-q1
            if iqr==0: continue
            n = ((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum()
            if n: outlier_notes.append(f"{col} ({fmt_int(n)})")
        if outlier_notes:
            insights.append("Potensi outlier: " + ", ".join(outlier_notes[:4]) + ".")
        if len(num_cols)>=2:
            corr = df[num_cols].corr(numeric_only=True).abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            stacked = upper.stack().sort_values(ascending=False)
            if not stacked.empty:
                (a,b),val = stacked.index[0], stacked.iloc[0]
                insights.append(f"Korelasi numerik terkuat: {a} ↔ {b} ({val:.2f}).")
    if cat_cols:
        top_cat = cat_cols[0]
        mode = df[top_cat].mode(dropna=True)
        if not mode.empty:
            insights.append(f"Nilai paling sering di kolom {top_cat}: {mode.iloc[0]}.")
    return insights[:7]


# ══════════════════════════════════════════════════════
#  REPORT GENERATOR
# ══════════════════════════════════════════════════════
def generate_html_report(df, insights, cleaning_log, meta):
    s = dataset_summary(df)
    score, qlabel = data_quality_score(df)
    insights_html = "".join([f"<li style='margin:6px 0;'>{i}</li>" for i in insights])
    log_html = "".join([f"<li style='margin:4px 0;font-family:monospace;'>{l}</li>" for l in cleaning_log]) if cleaning_log else "<li>Tidak ada operasi cleaning.</li>"

    # Stats table
    stats_html = ""
    try:
        num_stats = numeric_stats(df)
        if not num_stats.empty:
            stats_html = num_stats.to_html(classes="report-table", border=0, float_format=lambda x: f"{x:.2f}")
    except: pass

    now = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>Auto EDA Insight — Report</title>
<style>
* {{ box-sizing: border-box; margin:0; padding:0; }}
body {{ font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif; background:#060b17; color:#e4eeff; padding:40px; }}
.container {{ max-width:1100px; margin:0 auto; }}
h1 {{ font-size:36px; font-weight:900; color:#4f93ff; letter-spacing:-1px; }}
h2 {{ font-size:22px; font-weight:800; color:#22d3ee; margin:28px 0 12px; border-bottom:2px solid rgba(79,147,255,.25); padding-bottom:8px; }}
h3 {{ font-size:17px; font-weight:700; color:#a3c4ff; margin:16px 0 8px; }}
.header {{ background:linear-gradient(135deg,#0d1526,#111e35); border:1px solid rgba(79,147,255,.35); border-radius:20px; padding:30px 36px; margin-bottom:30px; }}
.subtitle {{ font-family:monospace; font-size:12px; color:#f0c050; letter-spacing:3px; margin-top:6px; }}
.generated {{ font-size:13px; color:#7a8fb4; margin-top:12px; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:24px; }}
.kpi-box {{ background:#0d1526; border:1px solid rgba(79,147,255,.3); border-radius:14px; padding:18px; text-align:center; }}
.kpi-num  {{ font-size:34px; font-weight:900; color:#4f93ff; }}
.kpi-lbl  {{ font-size:12px; color:#7a8fb4; margin-top:4px; font-weight:700; text-transform:uppercase; }}
.section  {{ background:#0d1526; border:1px solid rgba(79,147,255,.25); border-radius:16px; padding:22px 26px; margin-bottom:20px; }}
ul {{ padding-left:22px; }}
li {{ margin:6px 0; font-size:14px; }}
.badge {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:900; border:1px solid rgba(79,147,255,.35); background:rgba(79,147,255,.1); color:#4f93ff; }}
.badge-green {{ border-color:rgba(34,197,94,.4); background:rgba(34,197,94,.1); color:#22c55e; }}
.badge-red {{ border-color:rgba(248,113,113,.4); background:rgba(248,113,113,.1); color:#f87171; }}
.badge-gold {{ border-color:rgba(240,192,80,.4); background:rgba(240,192,80,.1); color:#f0c050; }}
.quality-bar-bg {{ height:10px; border-radius:999px; background:rgba(255,255,255,.08); overflow:hidden; margin:10px 0; }}
.quality-bar {{ height:10px; border-radius:999px; background:linear-gradient(90deg,#4f93ff,#22d3ee); }}
.report-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.report-table th {{ background:#111e35; color:#a3c4ff; padding:8px 12px; text-align:left; border-bottom:2px solid rgba(79,147,255,.25); }}
.report-table td {{ padding:7px 12px; border-bottom:1px solid rgba(79,147,255,.12); color:#e4eeff; }}
.footer {{ text-align:center; font-size:12px; color:#7a8fb4; margin-top:40px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>◈ Auto EDA Insight</h1>
    <div class="subtitle">◆ DATA SCIENCE PROGRAMMING — KELOMPOK 6 · ITSB ◆</div>
    <div class="generated">Generated: {now}</div>
    <div style="margin-top:10px;">
      <span class="badge badge-{'green' if score>=75 else 'red' if score<55 else 'gold'}">Quality Score: {qlabel} · {score}/100</span>
      <span class="badge" style="margin-left:6px;">Dataset: {meta.get('name','-')}</span>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi-box"><div class="kpi-num">{fmt_int(s['rows'])}</div><div class="kpi-lbl">Total Rows</div></div>
    <div class="kpi-box"><div class="kpi-num">{fmt_int(s['cols'])}</div><div class="kpi-lbl">Columns</div></div>
    <div class="kpi-box" style="border-color:rgba(251,191,36,.3);"><div class="kpi-num" style="color:#fbbf24;">{fmt_int(s['missing'])}</div><div class="kpi-lbl">Missing</div></div>
    <div class="kpi-box" style="border-color:rgba(248,113,113,.3);"><div class="kpi-num" style="color:#f87171;">{fmt_int(s['duplicate'])}</div><div class="kpi-lbl">Duplicates</div></div>
  </div>

  <div class="section">
    <h2>Dataset Overview</h2>
    <div class="quality-bar-bg"><div class="quality-bar" style="width:{score}%;"></div></div>
    <p style="font-size:13px;margin-top:4px;color:#7a8fb4;">Data Quality Score: <b style="color:#22d3ee;">{score}/100 — {qlabel}</b></p>
    <ul style="margin-top:12px;">
      <li>Format: <b>{meta.get('format','-')}</b></li>
      <li>Ukuran File: <b>{file_size_label(meta.get('size_bytes'))}</b></li>
      <li>Kolom Numerik: <b>{s['numeric']}</b> | Kategorik: <b>{s['category']}</b></li>
      <li>Kolom Datetime: <b>{s['date']}</b></li>
    </ul>
  </div>

  <div class="section">
    <h2>Initial Intelligent Insights</h2>
    <ul>{insights_html}</ul>
  </div>

  <div class="section">
    <h2>Cleaning Log</h2>
    <ul>{log_html}</ul>
  </div>

  {'<div class="section"><h2>Descriptive Statistics (Numerik)</h2>' + stats_html + '</div>' if stats_html else ''}

  <div class="footer">
    <p>Auto EDA Insight Dashboard · Data Science Programming · Kelompok 6 · ITSB · {now}</p>
  </div>
</div>
</body>
</html>"""


def render_report_page(df):
    st.markdown("## Download Report")
    if df is None:
        st.warning("Belum ada data. Upload file terlebih dahulu.")
        return

    s = dataset_summary(df)
    score, qlabel = data_quality_score(df)
    insights = build_initial_intelligent_insights(df)
    cleaning_log = st.session_state.cleaning_log
    meta = st.session_state.active_file or {}
    is_light = "Light" in st.session_state.get("ui_theme", "Dark Mode")

    report_css = f"""
    <style>
    .report-summary {{
        border-radius:24px;
        padding:20px 22px;
        border:1px solid {("rgba(20,121,86,.18)" if is_light else "rgba(139,92,246,.28)")};
        background:{("linear-gradient(135deg, rgba(255,255,255,.76), rgba(223,245,234,.78), rgba(235,241,255,.72))" if is_light else "linear-gradient(135deg, rgba(42,18,88,.92), rgba(30,15,61,.92))")};
        box-shadow:var(--shadow);
        margin-bottom:22px;
    }}
    .report-summary-grid {{
        display:grid;
        grid-template-columns:repeat(4,1fr);
        gap:16px;
    }}
    .report-stat {{
        border-radius:18px;
        padding:16px 18px;
        background:{("rgba(255,255,255,.55)" if is_light else "rgba(255,255,255,.045)")};
        border:1px solid {("rgba(20,121,86,.14)" if is_light else "rgba(139,92,246,.18)")};
    }}
    .report-stat .lbl {{
        font-size:11px;
        font-weight:900;
        text-transform:uppercase;
        letter-spacing:1.2px;
        color:var(--muted);
    }}
    .report-stat .val {{
        margin-top:6px;
        font-size:28px;
        font-weight:950;
        color:var(--text);
        line-height:1;
    }}
    .download-card {{
        border-radius:24px;
        padding:22px 22px 20px;
        min-height:230px;
        display:flex;
        flex-direction:column;
        justify-content:space-between;
        border:1px solid rgba(255,255,255,.22);
        box-shadow:0 18px 42px rgba(0,0,0,.16);
        position:relative;
        overflow:hidden;
    }}
    .download-card::after {{
        content:"";
        position:absolute;
        width:130px;
        height:130px;
        border-radius:999px;
        right:-45px;
        top:-45px;
        background:rgba(255,255,255,.25);
    }}
    .download-card .tiny {{
        font-size:11px;
        font-weight:950;
        text-transform:uppercase;
        letter-spacing:1.4px;
        opacity:.72;
        margin-bottom:8px;
    }}
    .download-card h3 {{
        margin:0 0 14px 0;
        font-size:24px !important;
        font-weight:950 !important;
        color:#12221a !important;
        line-height:1.12;
    }}
    .download-card p {{
        margin:0;
        font-size:14px;
        font-weight:760;
        color:rgba(18,34,26,.78);
        line-height:1.62;
    }}
    .download-card.dark h3, .download-card.dark p, .download-card.dark .tiny {{
        color:white !important;
    }}
    .download-card.dark p {{ opacity:.86; }}
    div[data-testid="stDownloadButton"] button {{
        margin-top:8px !important;
        border-radius:15px !important;
        min-height:44px !important;
        font-weight:900 !important;
        border:1px solid var(--stroke) !important;
        background:var(--panel) !important;
        color:var(--text) !important;
        box-shadow:none !important;
    }}
    @media(max-width:1000px) {{
        .report-summary-grid {{ grid-template-columns:repeat(2,1fr); }}
    }}
    </style>
    """
    st.markdown(report_css, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="report-summary">
        <div class="report-summary-grid">
            <div class="report-stat"><div class="lbl">Total Rows</div><div class="val">{fmt_int(s['rows'])}</div></div>
            <div class="report-stat"><div class="lbl">Columns</div><div class="val">{fmt_int(s['cols'])}</div></div>
            <div class="report-stat"><div class="lbl">Quality Score</div><div class="val">{score}/100</div></div>
            <div class="report-stat"><div class="lbl">Cleaning Ops</div><div class="val">{len(cleaning_log)}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    html_report = generate_html_report(df, insights, cleaning_log, meta)
    csv_data = df.to_csv(index=False)
    summary_data = {
        "generated_at": datetime.datetime.now().isoformat(),
        "dataset": meta,
        "summary": {k: int(v) if isinstance(v, (np.integer,)) else v for k, v in s.items()},
        "quality_score": score,
        "quality_label": qlabel,
        "insights": [strip_decorative_emoji(i) for i in insights],
        "cleaning_log": cleaning_log,
    }
    try:
        num_stats = numeric_stats(df)
        if not num_stats.empty:
            summary_data["numeric_stats"] = num_stats.to_dict()
    except Exception:
        pass
    json_str = json.dumps(summary_data, ensure_ascii=False, indent=2, default=str)

    excel_bytes = None
    excel_error = None
    try:
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            df.head(5000).to_excel(writer, sheet_name="Dataset", index=False)
            try:
                ns = numeric_stats(df)
                if not ns.empty:
                    ns.to_excel(writer, sheet_name="Numeric Stats", index=False)
            except Exception:
                pass
            try:
                cs = categorical_stats(df)
                if not cs.empty:
                    cs.to_excel(writer, sheet_name="Categorical Stats", index=False)
            except Exception:
                pass
            pd.DataFrame({"Insight": [strip_decorative_emoji(i) for i in insights]}).to_excel(writer, sheet_name="Insights", index=False)
            if cleaning_log:
                pd.DataFrame({"Cleaning Log": cleaning_log}).to_excel(writer, sheet_name="Cleaning Log", index=False)
        excel_buf.seek(0)
        excel_bytes = excel_buf.getvalue()
    except Exception as e:
        excel_error = str(e)

    st.markdown("### Pilih Format Download")
    dark_class = "dark" if not is_light else ""
    card_colors = {
        "html": "linear-gradient(135deg,#dff7ff,#d8fbe9)",
        "csv": "linear-gradient(135deg,#e9ddff,#dff7ff)",
        "excel": "linear-gradient(135deg,#dcfce7,#fef3c7)",
        "json": "linear-gradient(135deg,#ffe4ef,#e9ddff)",
    }
    if not is_light:
        card_colors = {
            "html": "linear-gradient(135deg,#2563eb,#0891b2)",
            "csv": "linear-gradient(135deg,#7c3aed,#2563eb)",
            "excel": "linear-gradient(135deg,#16a34a,#0f766e)",
            "json": "linear-gradient(135deg,#db2777,#7c3aed)",
        }

    cards = st.columns(4, gap="medium")
    with cards[0]:
        st.markdown(f'<div class="download-card {dark_class}" style="background:{card_colors["html"]};"><div><div class="tiny">Web Report</div><h3>HTML Report</h3><p>Laporan lengkap dalam format web dan bisa dibuka langsung di browser.</p></div></div>', unsafe_allow_html=True)
        st.download_button("Download HTML", html_report.encode("utf-8"), file_name=f"eda_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html", mime="text/html", use_container_width=True)
    with cards[1]:
        st.markdown(f'<div class="download-card {dark_class}" style="background:{card_colors["csv"]};"><div><div class="tiny">Clean Dataset</div><h3>Dataset CSV</h3><p>Export dataset aktif setelah proses cleaning dalam format CSV.</p></div></div>', unsafe_allow_html=True)
        st.download_button("Download CSV", csv_data.encode("utf-8"), file_name=f"dataset_cleaned_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
    with cards[2]:
        st.markdown(f'<div class="download-card {dark_class}" style="background:{card_colors["excel"]};"><div><div class="tiny">Workbook</div><h3>Excel Report</h3><p>Berisi dataset, statistik numerik/kategorik, insight, dan log cleaning.</p></div></div>', unsafe_allow_html=True)
        if excel_bytes:
            st.download_button("Download Excel", excel_bytes, file_name=f"eda_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        else:
            st.warning(f"Excel export belum tersedia: {excel_error}")
    with cards[3]:
        st.markdown(f'<div class="download-card {dark_class}" style="background:{card_colors["json"]};"><div><div class="tiny">Summary Data</div><h3>Summary JSON</h3><p>Ringkasan dataset dan insight otomatis untuk dokumentasi lanjutan.</p></div></div>', unsafe_allow_html=True)
        st.download_button("Download JSON", json_str.encode("utf-8"), file_name=f"eda_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", use_container_width=True)


# ══════════════════════════════════════════════════════
#  HOME DASHBOARD
# ══════════════════════════════════════════════════════
def plot_dtype_donut(df):
    counts = df.dtypes.astype(str).value_counts()
    is_light = "Light" in st.session_state.get("ui_theme","🌙 Dark Mode")
    bg = "#f8f5ff" if is_light else "#1a0a3e"
    text_color = "#1e0a4a" if is_light else "#f0eeff"
    fig, ax = plt.subplots(figsize=(4, 3), facecolor=bg)
    ax.set_facecolor(bg)
    colors = ["#7c3aed","#06b6d4","#f59e0b","#10b981","#f43f5e","#a78bfa"]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index.astype(str), autopct="%1.0f%%",
        startangle=90, pctdistance=0.72,
        wedgeprops={"width":0.48,"edgecolor":bg,"linewidth":2},
        colors=colors[:len(counts)])
    for t in texts: t.set_color(text_color); t.set_fontsize(8); t.set_fontweight("bold")
    for t in autotexts: t.set_color("white"); t.set_fontsize(7); t.set_fontweight("bold")
    plt.tight_layout(pad=0.5)
    return fig


def _summary_paragraph(df, s):
    """Short natural-language summary of the dataset."""
    lines = []
    lines.append(f"Dataset memiliki **{s['rows']:,} baris** dan **{s['cols']} kolom** "
                 f"({s['numeric']} numerik, {s['category']} kategorik).")
    if s["missing"]:
        pct = s["missing"] / max(s["rows"]*s["cols"],1) * 100
        lines.append(f"Terdapat **{s['missing']:,} nilai kosong** ({pct:.1f}%) — cleaning disarankan.")
    else:
        lines.append("Tidak ada nilai kosong — dataset bersih.")
    if s["duplicate"]:
        lines.append(f"Ditemukan **{s['duplicate']} baris duplikat**.")
    return " ".join(lines)


def render_home_dashboard():
    df = st.session_state.df
    meta = st.session_state.active_file or {}
    s = dataset_summary(df)
    score, qlabel = data_quality_score(df)
    insights = build_initial_intelligent_insights(df)
    is_light = "Light" in st.session_state.get("ui_theme", "Dark Mode")
    theme_mode = "light" if is_light else "dark"

    num_cols = df.select_dtypes(include="number").columns.tolist() if df is not None else []
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist() if df is not None else []

    if "dash_num_col" not in st.session_state or st.session_state.dash_num_col not in num_cols:
        st.session_state.dash_num_col = num_cols[0] if num_cols else None
    if "dash_cat_col" not in st.session_state or st.session_state.dash_cat_col not in cat_cols:
        st.session_state.dash_cat_col = cat_cols[0] if cat_cols else None

    sel_num = st.session_state.dash_num_col
    sel_cat = st.session_state.dash_cat_col

    if is_light:
        bg_main = "linear-gradient(135deg,#dff5ea 0%,#f8fff9 48%,#eef4ff 100%)"
        text_main = "#10281f"
        text_mute = "#526f63"
        card_bg = "rgba(255,255,255,.78)"
        card_bdr = "rgba(20,121,86,.16)"
        panel_bg = "linear-gradient(145deg, rgba(255,255,255,.82), rgba(232,249,240,.74))"
        accent = "#16a34a"
        accent2 = "#06b6d4"
        nav_active = "linear-gradient(135deg,#16a34a,#0f766e)"
        kpi_colors = ["#16a34a", "#06b6d4", "#f59e0b", "#10b981", "#ec4899"]
        donut_bg = "rgba(255,255,255,.0)"
    else:
        bg_main = "linear-gradient(135deg,#2e1065 0%,#1e0a4a 45%,#150732 100%)"
        text_main = "#f3efff"
        text_mute = "#a9a0d0"
        card_bg = "rgba(31,15,68,.90)"
        card_bdr = "rgba(139,92,246,.28)"
        panel_bg = "linear-gradient(145deg, rgba(39,18,85,.94), rgba(19,8,48,.92))"
        accent = "#7c3aed"
        accent2 = "#22d3ee"
        nav_active = "linear-gradient(135deg,#7c3aed,#4c1d95)"
        kpi_colors = ["#7c3aed", "#22d3ee", "#f59e0b", "#10b981", "#ec4899"]
        donut_bg = "rgba(255,255,255,.0)"

    st.markdown(f"""
    <style>
    .stApp {{ background:{bg_main} !important; }}
    .block-container {{ padding-top:.6rem !important; padding-left:.8rem !important; padding-right:.8rem !important; max-width:100% !important; }}
    .pf-brand {{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;flex-wrap:wrap;gap:12px;}}
    .pf-brand-left {{display:flex;align-items:center;gap:12px;}}
    .pf-logo {{width:44px;height:44px;border-radius:16px;background:{nav_active};display:flex;align-items:center;justify-content:center;color:#fff;font-weight:950;box-shadow:0 14px 32px rgba(0,0,0,{'.16' if is_light else '.36'});}}
    .pf-title-main {{font-size:22px;font-weight:950;color:{text_main};letter-spacing:-.4px;}}
    .pf-sub-main {{font-family:JetBrains Mono,monospace;font-size:12px;font-weight:850;letter-spacing:2.5px;color:{text_mute};text-transform:uppercase;}}
    .pf-stat-strip {{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:10px 0 18px;}}
    .pf-stat-mini {{
        border-radius:20px;
        padding:16px 20px;
        min-height:86px;
        background:linear-gradient(135deg, var(--stat-main) 0%, var(--stat-soft) 100%);
        border:1px solid var(--stat-border);
        box-shadow:0 12px 30px rgba(0,0,0,{'.07' if is_light else '.24'});
        display:flex;
        flex-direction:column;
        justify-content:center;
        position:relative;
        overflow:hidden;
    }}
    .pf-stat-mini:after {{
        content:"";
        position:absolute;
        width:92px;
        height:92px;
        right:-32px;
        top:-38px;
        border-radius:999px;
        background:rgba(255,255,255,{'.38' if is_light else '.07'});
        pointer-events:none;
    }}
    .pf-stat-label {{font-size:12px;font-weight:950;color:{text_mute};position:relative;z-index:1;}}
    .pf-stat-value {{font-size:23px;font-weight:950;color:{text_main};line-height:1.1;position:relative;z-index:1;margin-top:6px;}}
    .pf-nav-row [data-testid="stButton"]>button {{border-radius:999px!important;min-height:40px!important;font-size:12px!important;font-weight:950!important;text-transform:uppercase!important;letter-spacing:.6px!important;background:{card_bg}!important;color:{text_main}!important;border:1px solid {card_bdr}!important;box-shadow:none!important;}}
    .pf-nav-row [data-testid="stButton"]>button[kind="primary"] {{background:{nav_active}!important;color:#fff!important;border:none!important;box-shadow:0 10px 24px rgba(0,0,0,{'.12' if is_light else '.32'})!important;}}
    .pf-panel-title {{font-size:22px;font-weight:950;color:{text_main};margin-bottom:3px;letter-spacing:-.2px;}}
    .pf-panel-sub {{font-size:13px;font-weight:760;color:{text_mute};margin-bottom:12px;}}
    .pf-pill {{display:inline-flex;align-items:center;border-radius:999px;padding:6px 12px;font-size:12px;font-weight:900;background:{'rgba(22,163,74,.11)' if is_light else 'rgba(124,58,237,.20)'};color:{text_main};border:1px solid {card_bdr};}}
    .pf-kpi-grid {{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin:16px 0;}}
    .pf-kpi-card {{border-radius:22px;padding:18px 20px;min-height:116px;background:{panel_bg};border:1px solid {card_bdr};box-shadow:0 16px 42px rgba(0,0,0,{'.08' if is_light else '.25'});position:relative;overflow:hidden;}}
    .pf-kpi-card:after {{content:"";position:absolute;right:-38px;top:-46px;width:120px;height:120px;border-radius:999px;background:rgba(255,255,255,{'.42' if is_light else '.06'});}}
    .pf-kpi-val {{font-size:30px;font-weight:950;line-height:1;color:{text_main};margin-top:18px;}}
    .pf-kpi-lbl {{font-size:11px;font-weight:950;letter-spacing:1.15px;color:{text_mute};text-transform:uppercase;}}
    .pf-row {{display:flex;align-items:center;justify-content:space-between;padding:11px 0;border-bottom:1px solid {card_bdr};gap:12px;}}
    .pf-row:last-child {{border-bottom:none;}}
    .pf-row span {{font-size:13px;font-weight:850;color:{text_mute};}}
    .pf-row b {{font-size:14px;font-weight:950;color:{text_main};}}
    .pf-col-icon {{width:42px;height:42px;border-radius:15px;background:linear-gradient(135deg, rgba(124,58,237,.18), rgba(34,211,238,.14));flex-shrink:0;}}
    .pf-summary {{border-radius:16px;padding:14px 16px;border:1px solid {card_bdr};background:{'rgba(255,255,255,.52)' if is_light else 'rgba(255,255,255,.045)'};font-weight:780;line-height:1.7;color:{text_main};}}
    .pf-ins-row {{display:flex;gap:10px;align-items:flex-start;padding:8px 0;color:{text_main};font-weight:760;}}
    .pf-ins-dot {{width:8px;height:8px;border-radius:99px;flex-shrink:0;margin-top:8px;background:{accent};}}
    @media(max-width:1100px){{.pf-stat-strip,.pf-kpi-grid{{grid-template-columns:repeat(2,minmax(0,1fr));}}}}
    </style>
    """, unsafe_allow_html=True)

    team_av = "".join([_member_avatar_html(m, size=34, border_color=(accent if is_light else "#7c3aed")) for m in TEAM_MEMBERS])
    st.markdown(
        '<div class="pf-brand"><div class="pf-brand-left"><div class="pf-logo">◇</div><div>'
        '<div class="pf-title-main">Auto EDA Insight</div><div class="pf-sub-main">Data Science Programming · Kelompok 6 · ITSB</div>'
        '</div></div><div style="display:flex;gap:7px;align-items:center;">' + team_av + '</div></div>',
        unsafe_allow_html=True,
    )

    miss_pct_total = (s["missing"] / max(s["rows"] * s["cols"], 1) * 100) if df is not None else 0
    stat_items = [
        ("Total Rows", fmt_int(s["rows"]), kpi_colors[0]),
        ("Columns", str(s["cols"]), kpi_colors[1]),
        ("Missing Rate", f"{miss_pct_total:.1f}%", kpi_colors[2]),
        ("Quality Score", f"{score}/100", kpi_colors[3]),
    ]
    stat_html = "".join([
        f'<div class="pf-stat-mini" style="--stat-main:{clr}22;--stat-soft:{clr}0B;--stat-border:{clr}66;">'
        f'<div class="pf-stat-label">{lbl}</div><div class="pf-stat-value">{val}</div></div>'
        for lbl, val, clr in stat_items
    ])
    st.markdown('<div class="pf-stat-strip">' + stat_html + '</div>', unsafe_allow_html=True)

    nav_pages = [
        ("🏠 Dashboard", "Dashboard"), ("🧹 Data Cleaning", "Cleaning"),
        ("📈 Statistik — Numerik", "Num Stats"), ("📊 Statistik — Kategorik", "Cat Stats"),
        ("📉 Visualisasi Numerik", "Num Viz"), ("🔗 Bivariate & Multivariat", "Bivariate"),
        ("💡 Insights", "Insights"), ("📄 Download Report", "Report"),
    ]
    st.markdown('<div class="pf-nav-row">', unsafe_allow_html=True)
    nav_cols = st.columns(len(nav_pages), gap="small")
    for col, (dest, label) in zip(nav_cols, nav_pages):
        if col.button(label, key=f"topnav_{dest}", use_container_width=True, type="primary" if st.session_state.active_page == dest else "secondary"):
            st.session_state.active_page = dest
            st.session_state._scroll_to_main = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Main overview keeps the previous dashboard layout, but panels are native containers so text and box stay unified.
    r1a, r1b = st.columns([3.2, 1.05], gap="medium")
    with r1a:
        with st.container(border=True):
            miss_pct = (s["missing"] / max(s["rows"] * s["cols"], 1) * 100) if df is not None else 0
            st.markdown(
                '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:8px;">'
                '<div><div class="pf-panel-title">Visual Overview Dataset</div><div class="pf-panel-sub">Histogram dan tipe data dalam satu panel agar dashboard terlihat menyatu.</div></div>'
                '<div style="display:flex;gap:8px;flex-wrap:wrap;">'
                f'<span class="pf-pill">Valid {100-miss_pct:.0f}%</span><span class="pf-pill">Missing {miss_pct:.0f}%</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            c_hist, c_dtype = st.columns([1.65, 1.0], gap="medium")
            with c_hist:
                if df is not None and sel_num:
                    try:
                        fig_hist = plot_histogram(df, sel_num, theme=theme_mode)
                        fig_hist.update_layout(height=310, margin=dict(l=30, r=10, t=38, b=34), title=f"Distribusi — {sel_num}")
                        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": True, "responsive": True})
                    except Exception as e:
                        st.caption(f"Chart error: {e}")
                else:
                    st.markdown('<div class="notice-info">Upload data dan pilih kolom numerik untuk melihat histogram.</div>', unsafe_allow_html=True)
            with c_dtype:
                if df is not None:
                    try:
                        dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
                        dtype_counts.columns = ["Tipe Data", "Jumlah"]
                        colors = (["#16a34a", "#06b6d4", "#f59e0b", "#7c3aed", "#ec4899"] if is_light else ["#7c3aed", "#22d3ee", "#f59e0b", "#10b981", "#ec4899"])
                        fig_dtype = px.pie(dtype_counts, names="Tipe Data", values="Jumlah", hole=.52, title="Tipe Data", color_discrete_sequence=colors)
                        fig_dtype.update_traces(textposition="inside", textinfo="percent+label")
                        fig_dtype.update_layout(height=310, margin=dict(l=5, r=5, t=38, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_main, size=12), legend=dict(orientation="h", y=-.06, x=0))
                        st.plotly_chart(fig_dtype, use_container_width=True, config={"displayModeBar": True, "responsive": True})
                    except Exception as e:
                        st.caption(f"Chart error: {e}")
                else:
                    st.markdown('<div class="notice-info">Upload data untuk melihat komposisi tipe data.</div>', unsafe_allow_html=True)
    with r1b:
        st.markdown(
            f'<div style="border-radius:28px;padding:24px;background:{nav_active};color:#fff;box-shadow:0 20px 46px rgba(0,0,0,{.12 if is_light else .35});min-height:365px;display:flex;flex-direction:column;justify-content:space-between;">'
            '<div><div style="font-size:12px;font-weight:950;letter-spacing:1.6px;text-transform:uppercase;opacity:.82;">Quality Score</div>'
            f'<div style="font-size:46px;font-weight:950;line-height:1.05;margin-top:12px;">{score}<span style="font-size:18px;">/100</span></div>'
            f'<div style="font-size:13px;font-weight:850;margin-top:4px;opacity:.88;">{qlabel}</div></div>'
            f'<div><div style="height:7px;border-radius:999px;background:rgba(255,255,255,.25);overflow:hidden;margin:16px 0;"><div style="height:100%;width:{score}%;background:white;border-radius:999px;"></div></div>'
            f'<div style="font-size:12px;font-weight:850;line-height:1.9;opacity:.9;">{fmt_int(s["rows"])} baris · {s["cols"]} kolom<br>{fmt_int(s["missing"])} missing · {fmt_int(s["duplicate"])} duplikat<br>{meta.get("name", "Belum ada dataset")[:28]}</div></div></div>',
            unsafe_allow_html=True,
        )

    kpi_data = [("Rows", fmt_int(s["rows"])), ("Columns", fmt_int(s["cols"])), ("Numerik", str(s["numeric"])), ("Kategorik", str(s["category"])), ("Missing", fmt_int(s["missing"]))]
    kpi_html = "".join([f'<div class="pf-kpi-card" style="border-left:4px solid {kpi_colors[i%len(kpi_colors)]};"><div class="pf-kpi-val" style="color:{kpi_colors[i%len(kpi_colors)]};">{val}</div><div class="pf-kpi-lbl">{lbl}</div></div>' for i, (lbl, val) in enumerate(kpi_data)])
    st.markdown('<div class="pf-kpi-grid">' + kpi_html + '</div>', unsafe_allow_html=True)

    r2a, r2b, r2c = st.columns([1.25, 1.65, 1.45], gap="medium")
    with r2a:
        with st.container(border=True):
            st.markdown(f'<div class="pf-panel-title">{sel_num or "Kolom numerik"}</div><div class="pf-panel-sub">Statistik ringkas</div>', unsafe_allow_html=True)
            if df is not None and sel_num:
                vals = pd.to_numeric(df[sel_num], errors="coerce").dropna()
                if len(vals):
                    rows = [("Mean", f"{vals.mean():.2f}", kpi_colors[0]), ("Median", f"{vals.median():.2f}", kpi_colors[1]), ("Std", f"{vals.std():.2f}", kpi_colors[2]), ("Min", f"{vals.min():.2f}", kpi_colors[3]), ("Max", f"{vals.max():.2f}", kpi_colors[4])]
                    st.markdown("".join([f'<div class="pf-row"><span>{k}</span><b style="color:{c};">{v}</b></div>' for k, v, c in rows]), unsafe_allow_html=True)
                else:
                    st.markdown('<div class="notice-info">Kolom numerik tidak memiliki nilai valid.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="notice-info">Pilih kolom numerik.</div>', unsafe_allow_html=True)
    with r2b:
        with st.container(border=True):
            st.markdown(f'<div class="pf-panel-title">Top Nilai — {sel_cat or "Kolom kategori"}</div><div class="pf-panel-sub">6 kategori terbanyak</div>', unsafe_allow_html=True)
            if df is not None and sel_cat:
                vc = df[sel_cat].astype(str).value_counts().head(6).reset_index()
                vc.columns = [sel_cat, "count"]
                fig_top = px.bar(vc, x="count", y=sel_cat, orientation="h", color=sel_cat, color_discrete_sequence=["#7c3aed", "#8b5cf6", "#a78bfa", "#06b6d4", "#67e8f9", "#c4b5fd"], text="count")
                fig_top.update_layout(height=300, margin=dict(l=8, r=8, t=8, b=26), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_mute, size=11), showlegend=False, xaxis_title=None, yaxis_title=None)
                fig_top.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": True, "responsive": True})
            else:
                st.markdown('<div class="notice-info">Pilih kolom kategorik.</div>', unsafe_allow_html=True)
    with r2c:
        with st.container(border=True):
            st.markdown('<div class="pf-panel-title">Ringkasan Kolom</div><div class="pf-panel-sub">Statistik per kolom numerik</div>', unsafe_allow_html=True)
            if df is not None and num_cols:
                rows_html = ""
                for i, colname in enumerate(num_cols[:6]):
                    vals = pd.to_numeric(df[colname], errors="coerce").dropna()
                    meanv = f"{vals.mean():,.1f}" if len(vals) else "—"
                    clr = kpi_colors[i % len(kpi_colors)]
                    rows_html += f'<div class="pf-row"><div style="display:flex;align-items:center;gap:12px;"><div class="pf-col-icon" style="background:{clr}22;"></div><b>{colname[:18]}</b></div><b>{meanv}</b></div>'
                st.markdown(rows_html, unsafe_allow_html=True)
            else:
                st.markdown('<div class="notice-info">Upload data untuk melihat ringkasan kolom.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    r3a, r3b = st.columns([1.0, 1.65], gap="medium")
    with r3a:
        with st.container(border=True):
            st.markdown('<div class="pf-panel-title">Kontrol Visualisasi</div><div class="pf-panel-sub">Pilih kolom, chart dashboard akan ikut berubah</div>', unsafe_allow_html=True)
            if num_cols:
                new_num = st.selectbox("Kolom Numerik (chart utama)", num_cols, index=num_cols.index(sel_num) if sel_num in num_cols else 0, key="dash_num_sel")
                if new_num != st.session_state.dash_num_col:
                    st.session_state.dash_num_col = new_num
                    st.rerun()
            if cat_cols:
                new_cat = st.selectbox("Kolom Kategorik", cat_cols, index=cat_cols.index(sel_cat) if sel_cat in cat_cols else 0, key="dash_cat_sel")
                if new_cat != st.session_state.dash_cat_col:
                    st.session_state.dash_cat_col = new_cat
                    st.rerun()
    with r3b:
        with st.container(border=True):
            summary_txt = _summary_paragraph(df, s) if df is not None else "Upload dataset untuk melihat ringkasan."
            st.markdown('<div class="pf-panel-title">Ringkasan & Insights</div><div class="pf-panel-sub">Rangkuman otomatis dataset kamu</div>', unsafe_allow_html=True)
            st.markdown('<div class="pf-summary">' + strip_decorative_emoji(summary_txt).replace("**", "") + '</div>', unsafe_allow_html=True)
            dot_colors = ["#7c3aed", "#06b6d4", "#10b981", "#f97316", "#ec4899", "#f59e0b"]
            rows = "".join([f'<div class="pf-ins-row"><div class="pf-ins-dot" style="background:{dot_colors[i%len(dot_colors)]};"></div><div>{strip_decorative_emoji(ins).replace("**", "")}</div></div>' for i, ins in enumerate(insights[:4])])
            st.markdown(rows, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  FILE PROCESSING
# ══════════════════════════════════════════════════════
def process_uploaded_file(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    sig = f"{uploaded_file.name}:{len(file_bytes)}"
    if st.session_state.last_upload_signature == sig and st.session_state.df is not None:
        return st.session_state.df, None, None
    raw_dir = BASE_DIR/"data"/"raw"; raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir/uploaded_file.name).write_bytes(file_bytes)
    # Use a fresh BytesIO so seek/read always works regardless of Streamlit state
    file_like = io.BytesIO(file_bytes)
    file_like.name = uploaded_file.name
    df_new, text_content, err = load_file(file_like)
    if err: return None, text_content, err
    ext = uploaded_file.name.split(".")[-1].upper()
    st.session_state.active_file = {"name":uploaded_file.name,"format":ext,"size_bytes":len(file_bytes),"saved_path":str(Path("data")/"raw"/uploaded_file.name),"uploaded_at":datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}
    st.session_state.last_upload_signature = sig
    if df_new is not None:
        st.session_state.df = df_new; st.session_state.df_original = df_new.copy()
        st.session_state.cleaning_log = []
        st.session_state.cleaning_notice = ""
        st.session_state.before_snap = st.session_state.after_snap = st.session_state.before_df = st.session_state.after_df = None
        st.session_state.history.append({"name":uploaded_file.name,"rows":df_new.shape[0],"cols":df_new.shape[1],"time":datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),"df":df_new.copy(),"meta":st.session_state.active_file.copy()})
    return df_new, text_content, None


# ══════════════════════════════════════════════════════
#  MAIN DASHBOARD ROUTER
# ══════════════════════════════════════════════════════
def show_loading(label="Memuat..."):
    st.markdown(f'<div class="eda-card"><div style="font-weight:800;">{label}</div><div class="prog-track"><div class="prog-fill" style="width:100%;"></div></div></div>', unsafe_allow_html=True)

def main_dashboard():
    render_sidebar()
    menu = st.session_state.active_page
    with st.container():
        st.markdown('<div id="main-anchor"></div>', unsafe_allow_html=True)
        scroll_to_main()
        df = st.session_state.df

        if menu == "🏠 Dashboard":
            render_home_dashboard()

        elif menu == "📤 Upload Data":
            st.markdown("## Upload Data")
            st.caption("Upload dataset CSV, Excel, JSON, atau TXT.")
            uploaded = st.file_uploader("Drag & drop file di sini", type=["csv","xlsx","xls","json","txt"], label_visibility="collapsed")
            if uploaded:
                progress_box = st.empty()
                progress = st.progress(0, text=f"Uploading {uploaded.name} ...")
                try:
                    for pct, label in [(20, "Menerima file..."), (45, "Membaca struktur dataset..."), (70, "Mendeteksi tipe data...")]:
                        progress.progress(pct, text=f"{label}")
                        time.sleep(0.08)
                    with progress_box.container():
                        show_loading(f"Memproses dataset: {uploaded.name}")
                    df_new, text_c, err = process_uploaded_file(uploaded)
                    progress.progress(100, text="Dataset berhasil diproses.")
                    time.sleep(0.08)
                finally:
                    progress.empty()
                    progress_box.empty()

                if err: st.error(f"{err}")
                elif df_new is not None:
                    st.markdown(f'<div class="notice-success"><b>{uploaded.name}</b> — {df_new.shape[0]} baris × {df_new.shape[1]} kolom berhasil diupload.</div>', unsafe_allow_html=True)
                    render_paginated_table(df_new, key="upload_preview", page_size_default=10, height=380)
                    st.button("Lihat Dashboard", use_container_width=True, on_click=go_to, args=("🏠 Dashboard",))
                elif text_c:
                    st.markdown('<div class="notice-success">File teks berhasil dimuat.</div>', unsafe_allow_html=True)
                    st.text_area("Isi File", text_c, height=300)
            else:
                st.markdown('<div class="notice-info">Pilih file untuk memulai analisis.</div>', unsafe_allow_html=True)

        elif menu == "👁️ Data Preview":
            if df is None: st.warning("Upload file terlebih dahulu.")
            else:
                st.markdown("## Data Preview")
                render_paginated_table(df, key="data_preview", page_size_default=25, height=520)

        elif menu == "📌 Dataset Info":
            if df is None: st.warning("Upload file terlebih dahulu.")
            else:
                st.markdown("## Dataset Information")
                s = dataset_summary(df)
                cols = st.columns(6, gap="medium")
                for cw, item in zip(cols, [("","Rows",s["rows"]),("","Columns",s["cols"]),("","Numeric",s["numeric"]),("","Category",s["category"]),("","Missing",s["missing"]),("","Duplicate",s["duplicate"])]):
                    with cw: st.markdown(metric_card(*item), unsafe_allow_html=True)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                dtype_df = pd.DataFrame({"Column":df.columns,"Data Type":df.dtypes.astype(str).values,"Non-Null":df.notnull().sum().values,"Null":df.isnull().sum().values,"Null %":(df.isnull().sum()/len(df)*100).round(2).astype(str).values+"%","Sample":[str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0]>0 else "—" for c in df.columns]})
                render_paginated_table(dtype_df, key="dataset_info", page_size_default=25, height=480)

        elif menu == "🧹 Data Cleaning":
            if df is None: st.warning("Upload file terlebih dahulu.")
            else: render_cleaning_page(df)

        elif menu == "📈 Statistik — Numerik":
            if df is None: st.warning("Upload file terlebih dahulu.")
            else:
                st.markdown("## Statistik Deskriptif — Numerik")
                stats = numeric_stats(df)
                if stats.empty: st.info("Tidak ada kolom numerik.")
                else: render_paginated_table(stats, key="numeric_stats", page_size_default=25, height=480)

        elif menu == "📊 Statistik — Kategorik":
            if df is None: st.warning("Upload file terlebih dahulu.")
            else:
                st.markdown("## Statistik Deskriptif — Kategorik")
                stats = categorical_stats(df)
                if stats.empty: st.info("Tidak ada kolom kategorik.")
                else: render_paginated_table(stats, key="categorical_stats", page_size_default=25, height=480)

        elif menu == "📉 Visualisasi Numerik":
            if df is None: st.warning("Upload file terlebih dahulu.")
            else:
                num_cols = df.select_dtypes(include="number").columns.tolist()
                if not num_cols: st.info("Tidak ada kolom numerik.")
                else:
                    theme_mode = "light" if "Light" in st.session_state.get("ui_theme", "🌙 Dark Mode") else "dark"
                    st.markdown("## Visualisasi Numerik")
                    st.markdown('<div class="viz-card"><b>Pilih kolom numerik untuk eksplorasi visual interaktif yang lebih rapi dan mudah dibaca.</b></div>', unsafe_allow_html=True)
                    col_sel = st.selectbox("Pilih Kolom Numerik", num_cols)
                    series = df[col_sel].dropna()
                    m1,m2,m3,m4 = st.columns(4, gap="medium")
                    m1.markdown(metric_card("Σ","Non-null",len(series)), unsafe_allow_html=True)
                    m2.markdown(metric_card("μ","Mean",f"{series.mean():.2f}" if len(series) else 0), unsafe_allow_html=True)
                    m3.markdown(metric_card("M","Median",f"{series.median():.2f}" if len(series) else 0), unsafe_allow_html=True)
                    m4.markdown(metric_card("","Missing",int(df[col_sel].isna().sum())), unsafe_allow_html=True)
                    chart_tabs = st.tabs(["Histogram","Boxplot","Violin","Density","QQ Plot"])
                    with chart_tabs[0]: st.plotly_chart(plot_histogram(df, col_sel, theme=theme_mode), use_container_width=True)
                    with chart_tabs[1]: st.plotly_chart(plot_boxplot(df, col_sel, theme=theme_mode), use_container_width=True)
                    with chart_tabs[2]: st.plotly_chart(plot_violin(df, col_sel, theme=theme_mode), use_container_width=True)
                    with chart_tabs[3]: st.plotly_chart(plot_density(df, col_sel, theme=theme_mode), use_container_width=True)
                    with chart_tabs[4]: st.plotly_chart(plot_qq(df, col_sel, theme=theme_mode), use_container_width=True)

        elif menu == "🎨 Visualisasi Kategorik":
            if df is None: st.warning("Upload file terlebih dahulu.")
            else:
                cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
                if not cat_cols: st.info("Tidak ada kolom kategorik.")
                else:
                    theme_mode = "light" if "Light" in st.session_state.get("ui_theme", "🌙 Dark Mode") else "dark"
                    st.markdown("## Visualisasi Kategorik")
                    col_sel = st.selectbox("Pilih Kolom Kategorik", cat_cols)
                    m1,m2,m3 = st.columns(3)
                    m1.markdown(metric_card("","Unique",df[col_sel].nunique()), unsafe_allow_html=True)
                    m2.markdown(metric_card("","Top Value",str(df[col_sel].mode().iloc[0])[:18] if not df[col_sel].mode().empty else "-"), unsafe_allow_html=True)
                    m3.markdown(metric_card("","Missing",int(df[col_sel].isna().sum())), unsafe_allow_html=True)
                    chart_tabs = st.tabs(["Bar Chart","Pareto Chart","Pie Chart"])
                    with chart_tabs[0]: st.plotly_chart(plot_bar(df, col_sel, theme=theme_mode), use_container_width=True)
                    with chart_tabs[1]: st.plotly_chart(plot_pareto(df, col_sel, theme=theme_mode), use_container_width=True)
                    with chart_tabs[2]: st.plotly_chart(plot_pie(df, col_sel, theme=theme_mode), use_container_width=True)

        elif menu == "🔗 Bivariate & Multivariat":
            if df is None: st.warning("Upload file terlebih dahulu.")
            else:
                num_cols = df.select_dtypes(include="number").columns.tolist()
                if len(num_cols) < 2: st.info("Minimal 2 kolom numerik dibutuhkan.")
                else:
                    theme_mode = "light" if "Light" in st.session_state.get("ui_theme", "🌙 Dark Mode") else "dark"
                    st.markdown("## Bivariate & Multivariat")
                    chart_tabs = st.tabs(["Heatmap Korelasi","Scatter Plot","Regresi Linear"])
                    with chart_tabs[0]: st.plotly_chart(plot_correlation_heatmap(df, theme=theme_mode), use_container_width=True)
                    with chart_tabs[1]:
                        c1,c2 = st.columns(2)
                        x_col = c1.selectbox("Kolom X", num_cols)
                        y_col = c2.selectbox("Kolom Y", num_cols, index=min(1,len(num_cols)-1))
                        st.plotly_chart(plot_scatter(df, x_col, y_col, theme=theme_mode), use_container_width=True)
                    with chart_tabs[2]:
                        c1,c2 = st.columns(2)
                        x_col = c1.selectbox("X (Independen)", num_cols, key="reg_x")
                        y_col = c2.selectbox("Y (Dependen)", num_cols, index=min(1,len(num_cols)-1), key="reg_y")
                        st.plotly_chart(plot_regression(df, x_col, y_col, theme=theme_mode), use_container_width=True)

        elif menu == "📦 Kategorik vs Numerik":
            if df is None: st.warning("Upload file terlebih dahulu.")
            else:
                cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
                num_cols = df.select_dtypes(include="number").columns.tolist()
                if not cat_cols or not num_cols: st.info("Dibutuhkan minimal 1 kolom kategorik dan 1 numerik.")
                else:
                    theme_mode = "light" if "Light" in st.session_state.get("ui_theme", "🌙 Dark Mode") else "dark"
                    st.markdown("## Kategorik vs Numerik")
                    c1,c2 = st.columns(2)
                    cat_sel = c1.selectbox("Kolom Kategorik", cat_cols)
                    num_sel = c2.selectbox("Kolom Numerik", num_cols)
                    ca,cb = st.columns(2)
                    with ca: st.plotly_chart(plot_boxplot_by_cat(df, cat_sel, num_sel, theme=theme_mode), use_container_width=True)
                    with cb: st.plotly_chart(plot_grouped_bar(df, cat_sel, num_sel, theme=theme_mode), use_container_width=True)

        elif menu == "⏱️ Time Series":
            if df is None: st.warning("Upload file terlebih dahulu.")
            else:
                render_universal_time_series(df)

        elif menu == "💡 Insights":
            if df is None: st.warning("Upload file terlebih dahulu.")
            else:
                st.markdown("## Initial Intelligent Insight Generation")
                st.caption("Insight otomatis dibuat dari struktur dataset, kualitas data, missing value, duplikasi, outlier, dan korelasi.")

                initial_insights = build_initial_intelligent_insights(df)
                with st.spinner("Menyusun insight otomatis..."):
                    insights = generate_insights(df)

                s = dataset_summary(df)
                score, qlabel = data_quality_score(df)
                is_light = "Light" in st.session_state.get("ui_theme", "🌙 Dark Mode")
                panel_bg = "#ffffff" if is_light else "#1a1640"
                soft_bg = "#f0faf4" if is_light else "#24104e"
                border = "rgba(22,163,74,.18)" if is_light else "rgba(139,92,246,.28)"
                text_color = "#0a2218" if is_light else "#f4f1ff"
                muted = "#4a6b56" if is_light else "#aaa1d6"

                st.markdown(f"""
                <style>
                .insight-hero {{
                    background: linear-gradient(135deg, {"#dcfce7,#ffffff" if is_light else "#28145f,#180934"});
                    border:1px solid {border};
                    border-radius:22px;
                    padding:20px 22px;
                    box-shadow:0 12px 34px rgba(0,0,0,{".08" if is_light else ".35"});
                    margin-bottom:16px;
                }}
                .insight-grid {{
                    display:grid;
                    grid-template-columns: repeat(4, minmax(0,1fr));
                    gap:12px;
                    margin-top:14px;
                }}
                .insight-mini {{
                    background:{soft_bg};
                    border:1px solid {border};
                    border-radius:16px;
                    padding:14px 15px;
                }}
                .insight-mini .lbl {{
                    color:{muted};
                    font-size:11px;
                    font-weight:900;
                    text-transform:uppercase;
                    letter-spacing:1px;
                }}
                .insight-mini .val {{
                    color:{text_color};
                    font-size:24px;
                    font-weight:950;
                    margin-top:4px;
                }}
                .insight-list {{
                    display:grid;
                    grid-template-columns: repeat(2, minmax(0,1fr));
                    gap:12px;
                    margin-top:10px;
                }}
                .smart-card {{
                    background:{panel_bg};
                    border:1px solid {border};
                    border-radius:18px;
                    padding:15px 16px;
                    min-height:92px;
                    box-shadow:0 8px 22px rgba(0,0,0,{".05" if is_light else ".22"});
                }}
                .smart-card .tag {{
                    display:inline-flex;
                    padding:4px 9px;
                    border-radius:999px;
                    font-size:10px;
                    font-weight:900;
                    letter-spacing:.8px;
                    text-transform:uppercase;
                    margin-bottom:8px;
                }}
                .smart-card .txt {{
                    color:{text_color};
                    font-size:14px;
                    font-weight:750;
                    line-height:1.55;
                }}
                @media(max-width: 1000px) {{
                    .insight-grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
                    .insight-list {{ grid-template-columns: 1fr; }}
                }}
                </style>
                <div class="insight-hero">
                    <div style="display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;">
                        <div>
                            <div style="font-size:22px;font-weight:950;color:{text_color};">Smart Dataset Interpretation</div>
                            <div style="font-size:13px;font-weight:700;color:{muted};margin-top:4px;">
                                Ringkasan cepat supaya hasil EDA lebih mudah dipahami saat presentasi.
                            </div>
                        </div>
                        <div style="font-size:13px;font-weight:900;color:{text_color};background:{soft_bg};border:1px solid {border};border-radius:999px;padding:8px 13px;">
                            Quality: {score}/100 · {qlabel}
                        </div>
                    </div>
                    <div class="insight-grid">
                        <div class="insight-mini"><div class="lbl">Rows</div><div class="val">{fmt_int(s["rows"])}</div></div>
                        <div class="insight-mini"><div class="lbl">Columns</div><div class="val">{fmt_int(s["cols"])}</div></div>
                        <div class="insight-mini"><div class="lbl">Missing</div><div class="val">{fmt_int(s["missing"])}</div></div>
                        <div class="insight-mini"><div class="lbl">Duplicate</div><div class="val">{fmt_int(s["duplicate"])}</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                color_pool = [
                    ("Overview", "#7c3aed"), ("Quality", "#06b6d4"),
                    ("Cleaning", "#10b981"), ("Pattern", "#f59e0b"),
                    ("Correlation", "#ec4899"), ("Recommendation", "#f97316")
                ]
                cards_html = '<div class="insight-list">'
                all_insights = list(initial_insights[:4]) + list(insights[:6])
                for i, ins in enumerate(all_insights, 1):
                    tag, color = color_pool[(i-1) % len(color_pool)]
                    clean_ins = strip_decorative_emoji(str(ins).replace("**", "").replace("`", ""))
                    cards_html += (
                        f'<div class="smart-card">'
                        f'<div class="tag" style="background:{color}22;color:{color};">{i:02d} · {tag}</div>'
                        f'<div class="txt">{clean_ins}</div>'
                        f'</div>'
                    )
                cards_html += '</div>'
                st.markdown(cards_html, unsafe_allow_html=True)

                st.markdown("### Rekomendasi Lanjutan")
                recs = []
                if s["missing"] > 0:
                    recs.append("Lakukan data cleaning pada missing value sebelum analisis lanjutan.")
                if s["duplicate"] > 0:
                    recs.append("Hapus atau validasi baris duplikat agar hasil statistik tidak bias.")
                if s["numeric"] >= 2:
                    recs.append("Gunakan Bivariate & Multivariat untuk membaca korelasi antar variabel numerik.")
                if s["category"] > 0:
                    recs.append("Gunakan Visualisasi Kategorik untuk melihat kategori yang paling dominan.")
                if not recs:
                    recs.append("Dataset sudah cukup bersih dan siap untuk visualisasi serta pelaporan.")

                for rec in recs:
                    st.markdown(f'<div class="smart-card" style="min-height:unset;margin-bottom:8px;"><div class="txt">{rec}</div></div>', unsafe_allow_html=True)

        elif menu == "📄 Download Report":
            render_report_page(df)

        elif menu == "🗂️ Riwayat Upload":
            st.markdown("## Riwayat Upload")
            history = st.session_state.history
            if not history: st.info("Belum ada file yang pernah diupload di sesi ini.")
            else:
                for i, h in enumerate(reversed(history), 1):
                    with st.expander(f"{h['name']} · {h['time']} · {h['rows']} baris × {h['cols']} kolom"):
                        render_paginated_table(h["df"], key=f"history_{i}", page_size_default=10, height=360)
                        if st.button(f"Muat ulang", key=f"reload_{i}"):
                            st.session_state.df = h["df"].copy()
                            st.session_state.df_original = h["df"].copy()
                            st.session_state.active_file = h.get("meta",{"name":h["name"]})
                            st.session_state.cleaning_log = []
                            st.success(f"Dataset '{h['name']}' dimuat ulang."); st.rerun()


# ══════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════
inject_theme_css()
if not st.session_state.authenticated:
    auth_page()
else:
    main_dashboard()
