"""Week 3 dashboard: forecast vs current stock -> shortage / excess risk -> AI agent explanation -> human decision.

Run from the project folder:   streamlit run week3_dashboard.py
Needs:                          pip install streamlit requests python-dotenv altair matplotlib
Reads Week 1/2 files, so run the Week 2 notebook first. Keep inventory_risk.py next to this file.

AI features:
  - Claude (via the hackathon's AWS LLM Gateway, OpenAI-compatible /v1 endpoint) rewords the
    static rule text into a natural recommendation; falls back to the rule text if the gateway
    isn't configured.
  - Per-product multi-turn chat with quick scenario prompt chips.
  - Configure once via a .env file next to this script (see .env.example), or set
    LLM_GATEWAY_URL / LLM_GATEWAY_API_KEY / LLM_MODEL as real environment variables.
  - Deterministic guardrail check ensures AI explanations stay anchored to mathematical truth.
"""
import math
import os
from datetime import datetime
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads .env next to this file into os.environ, if present — must run before
                # _gateway_configured()/_call_llm_gateway() read LLM_GATEWAY_* below

from inventory_risk import (
    DEFAULT_EXCESS_RATIO, STATUS, assess, horizon_forecast, recommend_action,
    validate_stock_file, detect_outlier_weeks, rolling_wape, drift_flag,
    estimate_cost_impact, guardrail_check,
)

# ---------------------------------------------------------------- page configuration
st.set_page_config(
    page_title="Demand & Inventory Risk Engine",
    layout="wide",
    page_icon="📦",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- modern enterprise UI theme
st.markdown("""
<style>
/* Modern typography and neutral executive background */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}

/* Header & Brand Identity */
.app-header {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}
.app-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.app-subtitle {
    font-size: 0.88rem;
    color: #64748b;
    margin: 0;
}
.header-badge-row {
    display: flex;
    gap: 8px;
    margin-top: 10px;
    flex-wrap: wrap;
}
.badge-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
    border: 1px solid #e2e8f0;
    background: #f1f5f9;
    color: #334155;
}
.badge-chip.online {
    background: #ecfdf5;
    border-color: #a7f3d0;
    color: #065f46;
}
.badge-chip.warning {
    background: #fffbeb;
    border-color: #fde68a;
    color: #92400e;
}

/* Executive Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #0f172a !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid #334155 !important;
    border-radius: 10px;
    margin-bottom: 8px;
}
[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
}

/* Polished Metric Cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 14px 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
}

/* Custom Status KPI Cards */
.status-kpi-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 14px 16px;
    border: 1px solid #e2e8f0;
    border-left-width: 5px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    transition: all 0.2s ease;
}
.status-kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
}
.status-kpi-card .count-val {
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 4px;
}
.status-kpi-card .status-label {
    font-size: 0.85rem;
    font-weight: 700;
    color: #334155;
    display: flex;
    align-items: center;
    gap: 6px;
}
.status-kpi-card .status-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 4px;
}

/* AI Copilot Card */
.copilot-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8faff 100%);
    border: 1px solid #c7d2fe;
    border-radius: 14px;
    padding: 20px 22px;
    margin: 12px 0 16px 0;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.06);
    position: relative;
    overflow: hidden;
}
.copilot-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #4f46e5, #06b6d4, #10b981);
}
.copilot-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}
.copilot-title {
    font-size: 0.96rem;
    font-weight: 700;
    color: #1e1b4b;
    display: flex;
    align-items: center;
    gap: 8px;
}
.copilot-body {
    font-size: 0.95rem;
    line-height: 1.55;
    color: #1e293b;
    font-weight: 500;
    background: #ffffff;
    border: 1px solid #e0e7ff;
    border-radius: 10px;
    padding: 14px 16px;
}
.guardrail-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 6px;
}
.guardrail-chip.pass {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
}
.guardrail-chip.warn {
    background: #fff1f2;
    color: #be123c;
    border: 1px solid #fecdd3;
}

/* Modern Tab Styling */
div[data-baseweb="tab-list"] {
    gap: 6px;
    background: #f1f5f9;
    padding: 4px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    margin-bottom: 16px;
}
div[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    color: #475569 !important;
    background: transparent !important;
    border: none !important;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background: #ffffff !important;
    color: #1e293b !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
}

/* Buttons */
.stButton>button {
    background: #2563eb;
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0.5rem 1.1rem;
    transition: all 0.15s ease;
}
.stButton>button:hover {
    background: #1d4ed8;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
}
[data-testid="stForm"] button[kind="formSubmit"] {
    background: #059669 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
[data-testid="stForm"] button[kind="formSubmit"]:hover {
    background: #047857 !important;
    box-shadow: 0 2px 8px rgba(5, 150, 105, 0.25) !important;
}

/* Quick prompt chips */
.prompt-chip-btn {
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
    color: #334155 !important;
    font-size: 0.8rem !important;
    padding: 4px 10px !important;
    border-radius: 9999px !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
SKIP = {".venv", "venv", "node_modules", ".git", "__pycache__", "site-packages"}


# ---------------------------------------------------------------- LLM gateway helpers
# Reads the same three env vars used throughout this project's gateway setup:
#   LLM_GATEWAY_URL, LLM_GATEWAY_API_KEY, LLM_MODEL
# Talks to the gateway's documented OpenAI-compatible endpoint at <LLM_GATEWAY_URL>/v1/chat/completions.
def _gateway_configured() -> bool:
    return bool(os.environ.get("LLM_GATEWAY_URL", "").strip() and os.environ.get("LLM_GATEWAY_API_KEY", "").strip())


def _call_llm_gateway(messages: list, max_retries: int = 3, backoff: int = 3) -> str:
    import time

    url = os.environ.get("LLM_GATEWAY_URL", "").rstrip("/") + "/v1/chat/completions"
    key = os.environ.get("LLM_GATEWAY_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")
    payload = {"model": model, "messages": messages, "temperature": 0.3}

    # The gateway's docs confirm X-API-Key auth on its native endpoint; the /v1 OpenAI-compatible
    # endpoint should take a standard Bearer token, but that's not independently confirmed here —
    # so try Bearer first and fall back to X-API-Key on an auth rejection, rather than guessing once.
    header_variants = [
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        {"X-API-Key": key, "Content-Type": "application/json"},
    ]

    last_exc = None
    for headers in header_variants:
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code in (401, 403) and headers is header_variants[0]:
                    break  # try the other auth header instead of retrying the same one
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                last_exc = exc
                if attempt == max_retries:
                    break
                time.sleep(backoff * attempt)  # ALB can rate-limit rapid successive calls
    raise last_exc


def _build_product_context(pid, a, info, stock, weeks, start_label) -> str:
    bias_note = ""
    bp = info.get("bias_pct")
    if bp is not None and not pd.isna(bp):
        direction = "over-forecasting" if bp > 0 else "under-forecasting"
        bias_note = f"\n- Forecast bias: {bp:+.1f}% ({direction})"

    return (
        f"Product: {pid}\n"
        f"Planning window: {weeks} week(s) from {start_label}\n"
        f"Forecast model: {info['final_model']} (human review: {info['human_decision']})\n"
        f"Model WAPE: {info['wape']:.1f}%  |  MAE: {info['mae']:.1f} units/week{bias_note}\n"
        f"\nInventory figures:\n"
        f"- Current stock:    {stock:,.0f} units\n"
        f"- Forecast demand:  {a['demand']:,.0f} units (over the horizon)\n"
        f"- Safety stock:     {a['need'] - a['demand']:,.0f} units\n"
        f"- Stock needed:     {a['need']:,.0f} units\n"
        f"- Stock gap:        {a['gap']:+,.0f} units (positive = surplus, negative = shortfall)\n"
        f"- Weeks of cover:   {a['cover_weeks']:.1f}\n"
        f"- Risk status:      {a['status']}\n"
        f"- Shortfall units:  {a['shortfall_units']:,}\n"
        f"- Top-up to cover high-demand case: {a['topup_units']:,} units\n"
        f"- Excess units above threshold: {a['excess_units']:,}\n"
    )


def ai_recommendation(api_key: str, product_context: str, rule_text: str) -> str:
    if not _gateway_configured():
        return rule_text

    system_prompt = (
        "You are an inventory planning assistant embedded in a supply-chain dashboard. "
        "You receive structured data about a single product's forecast demand, current stock, "
        "and forecast model quality. Your job is to give the planner a clear, specific, "
        "actionable recommendation in 3–5 sentences. "
        "Mention the key numbers. Flag any model reliability concerns (high WAPE, strong bias). "
        "Be direct — no preamble, no sign-off. Plain prose only, no bullet points or markdown."
    )
    user_prompt = (
        f"=== PRODUCT DATA ===\n{product_context}\n\n"
        f"=== RULE-BASED BASELINE (for reference) ===\n{rule_text}\n\n"
        "Now write your recommendation:"
    )

    try:
        return _call_llm_gateway([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as exc:
        return f"{rule_text}\n\n_(AI unavailable: {exc})_"


def ai_chat_response(api_key: str, product_context: str, history: list, user_msg: str) -> str:
    if not _gateway_configured():
        return "LLM gateway not configured. Set LLM_GATEWAY_URL / LLM_GATEWAY_API_KEY / LLM_MODEL on the server."

    system = (
        "You are an inventory planning assistant. Answer questions about the product below. "
        "Keep answers concise and grounded in the numbers provided. "
        "If asked about scenarios (e.g. 'what if I order X more?'), work through the maths. "
        "Plain prose only."
        f"\n\n=== CURRENT PRODUCT DATA ===\n{product_context}"
    )

    try:
        messages = [{"role": "system", "content": system}]
        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            if turn.get("assistant"):
                messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": user_msg})

        return _call_llm_gateway(messages)
    except Exception as exc:
        return f"_(AI error: {exc})_"


def find_file(name, required=True):
    candidates = list(BASE.rglob(name)) + list(BASE.parent.glob(name))
    for p in candidates:
        if not SKIP & set(p.parts):
            return p
    if required:
        st.error(f"Couldn't find **{name}** near {BASE}. Please ensure Week 2 files are available.")
        st.stop()
    return None


# ---------------------------------------------------------------- load data
weekly = pd.read_csv(find_file("weekly_sales.csv"), parse_dates=["date"])
final_fc = pd.read_csv(find_file("final_forecast.csv"), parse_dates=["date"])
metrics = pd.read_csv(find_file("model_metrics.csv"))
model_forecasts = pd.read_csv(find_file("model_forecasts.csv"), parse_dates=["date"])
MODEL_COL_MAP = {
    "Moving Average": "moving_average",
    "Holt-Winters": "holt_winters",
    "XGBoost": "xgboost",
    "Croston": "croston",
    "Ensemble": "ensemble",
}
summary_path = find_file("model_selection_summary.csv")
summary = pd.read_csv(summary_path)
OUT_DIR = summary_path.parent

stock_path = find_file("current_stock.csv", required=False)
using_sample = stock_path is None
stock_file = pd.read_csv(stock_path or find_file("current_stock_SAMPLE.csv"))
if "safety_stock" not in stock_file.columns:
    stock_file["safety_stock"] = 0

STOCK_ISSUES = validate_stock_file(stock_file, known_products=summary["product_id"].unique())
DECISIONS_PATH = OUT_DIR / "planner_decisions.csv"
CHART_DIR = OUT_DIR / "charts"

_avg = weekly.groupby("date")["quantity_sold"].mean()
PARTIAL_WEEKS = list(_avg[_avg < 0.5 * _avg.median()].index)

# ---------------------------------------------------------------- session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "ai_rec_cache" not in st.session_state:
    st.session_state.ai_rec_cache = {}
if "staged_prompt" not in st.session_state:
    st.session_state.staged_prompt = None
if "prefilled_note" not in st.session_state:
    st.session_state.prefilled_note = ""

# ---------------------------------------------------------------- sidebar navigation
with st.sidebar:
    st.markdown("### ⚙️ Planning Engine")
    weeks = st.slider("Planning horizon (weeks)", 1, 4, 1)

    fc_dates = sorted(final_fc["date"].unique())
    date_labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in fc_dates]
    partial_labels = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in PARTIAL_WEEKS}
    trailing_partial = sum(1 for lab in reversed(date_labels) if lab in partial_labels)
    default_idx = max(0, len(date_labels) - weeks - trailing_partial)

    start_label = st.selectbox("Planning start week", date_labels, index=default_idx)
    start = pd.Timestamp(start_label)

    excess_ratio = st.slider("Excess threshold (x expected demand)", 1.1, 3.0, float(DEFAULT_EXCESS_RATIO), 0.1)

    st.markdown("---")
    with st.expander("📦 Current Stock & What-If Editor"):
        st.caption("Live edit inventory to simulate stock adjustments:")
        stock_df = st.data_editor(
            stock_file[["product_id", "current_stock", "safety_stock"]],
            hide_index=True,
            width="stretch",
        )

    if STOCK_ISSUES:
        with st.expander(f"⚠️ Stock Issues ({len(STOCK_ISSUES)})", expanded=not using_sample):
            for issue in STOCK_ISSUES:
                st.write(f"- {issue}")

    with st.expander("💲 Unit Economics (Cost / ROI)"):
        st.caption("Enter unit metrics to quantify financial risk:")
        unit_margin = st.number_input("Margin lost per unit short ($)", min_value=0.0, value=0.0, step=0.1)
        unit_cost = st.number_input("Unit cost ($)", min_value=0.0, value=0.0, step=0.1)
        holding_cost_pct = (
            st.number_input("Weekly holding cost (% of unit cost)", min_value=0.0, value=0.0, step=0.1) / 100.0
        )

    st.markdown("---")
    st.markdown("### 🤖 Copilot Intelligence")
    if _gateway_configured():
        st.caption(f"✓ Connected to LLM gateway ({os.environ.get('LLM_MODEL', '')})")
    else:
        st.caption("⚠️ LLM_GATEWAY_URL / LLM_GATEWAY_API_KEY not set on this server — using rule-based fallback.")

ai_enabled = _gateway_configured()


# ---------------------------------------------------------------- computation helpers
def model_info(pid):
    s = summary[summary["product_id"] == pid].iloc[0]
    m = metrics[(metrics["product_id"] == pid) & (metrics["model"] == s["final_model"]) & (metrics["split"] == s["eval_window"])]
    r = m.iloc[0] if len(m) else None
    return {
        "final_model": s["final_model"],
        "human_decision": s["human_decision"],
        "window": s["eval_window"],
        "selection_window": s["selection_window"] if "selection_window" in summary.columns else None,
        "wape": float(r["wape"]) if r is not None else float(s["final_model_wape"]),
        "mae": float(r["mae"]) if r is not None else np.nan,
        "bias": float(r["bias"]) if r is not None else np.nan,
        "bias_pct": float(r["bias_pct"]) if r is not None else np.nan,
    }


rows, details = [], {}
shortfall_total, excess_total = 0, 0
for pid in sorted(summary["product_id"]):
    info = model_info(pid)
    demand, used = horizon_forecast(final_fc, pid, start, weeks)
    srow = stock_df[stock_df["product_id"] == pid]
    if srow.empty or used.empty:
        rows.append({
            "Product": pid,
            "Status": STATUS["nostock"],
            "Forecast Demand": round(demand) if len(used) else np.nan,
            "Current Stock": np.nan,
            "Gap": np.nan,
            "Cover Weeks": np.nan,
            "Model": info["final_model"],
            "WAPE %": round(info["wape"], 1),
            "Bias %": round(info["bias_pct"], 1),
        })
        continue
    stock, safety = float(srow.iloc[0]["current_stock"]), float(srow.iloc[0]["safety_stock"])
    a = assess(stock, demand, info["wape"], weeks, safety, excess_ratio)
    details[pid] = (info, a, stock, len(used))
    shortfall_total += a["shortfall_units"]
    excess_total += a["excess_units"]
    rows.append({
        "Product": pid,
        "Status": a["status"],
        "Forecast Demand": round(demand),
        "Current Stock": round(stock),
        "Gap": round(a["gap"]),
        "Cover Weeks": round(a["cover_weeks"], 1),
        "Model": info["final_model"],
        "WAPE %": round(info["wape"], 1),
        "Bias %": round(info["bias_pct"], 1),
    })

overview = pd.DataFrame(rows)
counts = overview["Status"].value_counts()

# ---------------------------------------------------------------- top header banner
st.markdown(f"""
<div class="app-header">
    <div class="app-title">
        <span>📦 Demand Forecast & Inventory Risk Engine</span>
    </div>
    <div class="app-subtitle">
        Predictive supply chain risk assessment • Model benchmarking • Human-in-the-loop AI copilot
    </div>
    <div class="header-badge-row">
        <span class="badge-chip">⏱️ Window: <strong>{weeks} week(s) from {start_label}</strong></span>
        <span class="badge-chip {'online' if ai_enabled else ''}">
            {'✨ Claude Copilot: Active' if ai_enabled else '⚠️ AI Offline: Rule Mode'}
        </span>
        <span class="badge-chip {'warning' if using_sample else 'online'}">
            {'⚠️ Feed: Sample Stock' if using_sample else '✓ Feed: Production Stock'}
        </span>
        <span class="badge-chip">📊 Tracked SKUs: <strong>{len(summary)}</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- executive KPI cards
STATUS_THEMES = {
    "shortage": {"color": "#dc2626", "bg": "#fef2f2", "border": "#fecaca", "icon": "🔴"},
    "watch": {"color": "#d97706", "bg": "#fffbeb", "border": "#fde68a", "icon": "🟠"},
    "balanced": {"color": "#059669", "bg": "#f0fdf4", "border": "#bbf7d0", "icon": "🟢"},
    "excess": {"color": "#4f46e5", "bg": "#eef2ff", "border": "#c7d2fe", "icon": "🔵"},
}

kpi_cols = st.columns(4)
for col, key in zip(kpi_cols, ["shortage", "watch", "balanced", "excess"]):
    th = STATUS_THEMES[key]
    val = int(counts.get(STATUS[key], 0))
    subtext = {
        "shortage": f"{shortfall_total:,} total units shortfall" if shortfall_total > 0 else "No active shortfalls",
        "watch": "Within WAPE model error band",
        "balanced": "Optimal stock coverage",
        "excess": f"{excess_total:,} units above threshold" if excess_total > 0 else "No excess holding risk",
    }[key]
    col.markdown(
        f"""
        <div class="status-kpi-card" style="background:{th['bg']}; border-color:{th['border']}; border-left-color:{th['color']};">
            <div class="count-val" style="color:{th['color']};">{val}</div>
            <div class="status-label">{th['icon']} {STATUS[key]}</div>
            <div class="status-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- main tab navigation
tab_portfolio, tab_product, tab_copilot, tab_models, tab_audit = st.tabs([
    "📊 Portfolio Risk Matrix",
    "🔍 Product Deep-Dive",
    "✨ AI Agent Copilot",
    "📈 Model Diagnostics & Drift",
    "📋 Planner Decision & Audit",
])


# ================================================================ TAB 1: PORTFOLIO RISK
with tab_portfolio:
    st.subheader("All Products Risk Matrix")
    st.caption("Global overview of stock health across selected horizon. Click column headers to sort.")

    # Status filter pills
    status_filter = st.radio(
        "Filter by Risk Status",
        options=["All", STATUS["shortage"], STATUS["watch"], STATUS["balanced"], STATUS["excess"]],
        horizontal=True,
    )

    filtered_df = overview if status_filter == "All" else overview[overview["Status"] == status_filter]

    st.dataframe(
        filtered_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Product": st.column_config.TextColumn("Product SKU", width="small"),
            "Status": st.column_config.TextColumn("Inventory Status"),
            "Forecast Demand": st.column_config.NumberColumn("Forecast Demand", format="%d units"),
            "Current Stock": st.column_config.NumberColumn("Current Stock", format="%d units"),
            "Gap": st.column_config.NumberColumn("Stock Gap", format="%+d units"),
            "Cover Weeks": st.column_config.NumberColumn("Weeks of Cover", format="%.1f w"),
            "Model": st.column_config.TextColumn("Selected Model"),
            "WAPE %": st.column_config.NumberColumn("WAPE", format="%.1f%%"),
            "Bias %": st.column_config.NumberColumn("Bias", format="%+.1f%%"),
        },
    )

    with st.expander("🛡️ Governance & Model Limitations (Section 7 Compliance Matrix)"):
        st.markdown(
            """
            | Risk Domain | Risk Mitigation & Architectural Control |
            |---|---|
            | **Sample Stock Data** | Automated schema validation (`validate_stock_file`) runs upon loading; warnings stay visible in sidebar. |
            | **Outliers & Data Quality** | Outlier detection via robust trailing-median z-scores flags irregular spikes/dips. |
            | **Historical Model Drift** | Real-time rolling 6-week WAPE monitor detects accuracy slippage vs selection-time baseline. |
            | **Automation Bias** | Human planners must explicitly verify model WAPE & bias prior to submitting sign-offs. |
            | **LLM Reasoning Integrity** | Deterministic business calculus in `inventory_risk.py`; LLM outputs are checked by guardrail checks. |
            | **Cost / ROI Uncertainty** | Planners can dynamically inject custom unit margins & holding rates for live financial impact simulations. |
            """
        )


# ================================================================ TAB 2: PRODUCT DEEP-DIVE
with tab_product:
    sel_col1, sel_col2 = st.columns([1, 2])
    with sel_col1:
        pid = st.selectbox("Select Product to Inspect", sorted(summary["product_id"]), key="deepdive_pid")

    if pid not in details:
        st.info("No stock data or forecast available for this SKU in the selected window.")
    else:
        info, a, stock, n_used = details[pid]
        with sel_col2:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; height: 100%; gap: 12px; padding-top: 24px;">
                    <span style="font-weight: 700; color: #334155;">Status: {a['status']}</span>
                    <span style="color: #64748b;">•</span>
                    <span style="color: #64748b;">Selected Model: <strong>{info['final_model']}</strong></span>
                    <span style="color: #64748b;">•</span>
                    <span style="color: #64748b;">Review: <strong>{info['human_decision']}</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if n_used < weeks:
            st.warning(f"Note: Only {n_used} forecast week(s) are available from {start_label}; horizon was shortened.")

        # Key Metrics Row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Current Stock", f"{stock:,.0f} units")
        mcol2.metric(f"Forecast Demand ({weeks}w)", f"{a['demand']:,.0f} units")
        mcol3.metric(
            "Stock Gap",
            f"{a['gap']:+,.0f} units",
            delta=f"{a['cover_weeks']:.1f} weeks cover",
            delta_color="normal" if a['gap'] >= 0 else "inverse",
        )
        mcol4.metric(
            "Safety Stock Buffer",
            f"{a['need'] - a['demand']:,.0f} units",
            help="Additional buffer required above baseline demand",
        )

        # Financial Impact Simulator
        cost_est = estimate_cost_impact(
            a,
            unit_margin=unit_margin,
            holding_cost_pct_per_week=holding_cost_pct,
            unit_cost=unit_cost,
            weeks=weeks,
        )
        if cost_est["inputs_provided"]:
            fcol1, fcol2 = st.columns(2)
            if a["status_key"] in ("shortage", "watch"):
                fcol1.metric("Estimated Lost Margin (Stockout)", f"${cost_est['estimated_lost_margin']:,.0f}")
            if a["status_key"] == "excess":
                fcol2.metric("Estimated Holding Cost", f"${cost_est['estimated_excess_holding_cost']:,.0f}")

        # Visual Stock Level Breakdown Gauge
        st.markdown("<h4 style='font-size: 1.05rem; margin-top: 15px;'>Stock Position vs. Risk Boundaries</h4>", unsafe_allow_html=True)
        gauge_data = pd.DataFrame([
            {"Tier": "1. Current Stock", "Units": stock, "Category": "Stock On Hand", "Color": "#2563eb"},
            {"Tier": "2. Expected Demand", "Units": a["demand"], "Category": "Demand", "Color": "#059669"},
            {"Tier": "3. Required (Demand + Safety)", "Units": a["need"], "Category": "Buffer Need", "Color": "#d97706"},
            {"Tier": "4. High-Demand (+WAPE)", "Units": a["high"], "Category": "High Case", "Color": "#dc2626"},
        ])
        gauge_chart = (
            alt.Chart(gauge_data)
            .mark_bar(cornerRadiusEnd=4, height=22)
            .encode(
                x=alt.X("Units:Q", title="Units"),
                y=alt.Y("Tier:N", title="", sort=None),
                color=alt.Color("Color:N", scale=None),
                tooltip=[alt.Tooltip("Tier:N"), alt.Tooltip("Units:Q", format=",.0f")],
            )
            .properties(height=160)
        )
        st.altair_chart(gauge_chart, width="stretch")

        # Interactive Trend Chart
        st.markdown("<h4 style='font-size: 1.05rem; margin-top: 20px;'>52-Week Actuals & Forecast Horizon</h4>", unsafe_allow_html=True)
        hist = weekly[weekly["product_id"] == pid].sort_values("date").tail(52).copy()
        ff = final_fc[final_fc["product_id"] == pid].sort_values("date").copy()
        horizon = ff[ff["date"] >= start].head(weeks).copy()
        outliers = detect_outlier_weeks(weekly, pid)

        # Build clean layered Altair visualization
        hist_clean = hist.rename(columns={"quantity_sold": "quantity"})[["date", "quantity"]].copy()
        hist_clean["Series"] = "Historical Sales"

        fc_clean = ff.rename(columns={"forecast_quantity": "quantity"})[["date", "quantity"]].copy()
        fc_clean["Series"] = f"Forecast ({info['final_model']})"

        chart_data = pd.concat([hist_clean, fc_clean], ignore_index=True)

        chart_base = alt.Chart(chart_data).encode(
            x=alt.X("date:T", title="Week Date", axis=alt.Axis(format="%b %Y", grid=True, gridColor="#f1f5f9")),
            y=alt.Y("quantity:Q", title="Units / Week", axis=alt.Axis(grid=True, gridColor="#f1f5f9")),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["Historical Sales", f"Forecast ({info['final_model']})"],
                    range=["#2563eb", "#db2777"],
                ),
                legend=alt.Legend(title="", orient="top-left"),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Week", format="%Y-%m-%d"),
                alt.Tooltip("quantity:Q", title="Quantity", format=",.0f"),
                alt.Tooltip("Series:N"),
            ],
        )

        lines = chart_base.mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=25))

        # Planning window highlight rect
        layers = [lines]
        if len(horizon):
            h_df = pd.DataFrame([{
                "start": horizon["date"].min() - pd.Timedelta(days=3.5),
                "end": horizon["date"].max() + pd.Timedelta(days=3.5),
            }])
            horizon_rect = alt.Chart(h_df).mark_rect(opacity=0.15, color="#6366f1").encode(
                x="start:T", x2="end:T"
            )
            layers.insert(0, horizon_rect)

        # Incomplete weeks highlight
        if PARTIAL_WEEKS:
            p_in_view = [d for d in PARTIAL_WEEKS if hist["date"].min() <= d <= hist["date"].max()]
            if p_in_view:
                p_df = pd.DataFrame([{"x1": d - pd.Timedelta(days=3.5), "x2": d + pd.Timedelta(days=3.5)} for d in p_in_view])
                partial_rect = alt.Chart(p_df).mark_rect(opacity=0.18, color="#f97316").encode(x="x1:T", x2="x2:T")
                layers.insert(0, partial_rect)

        # Outlier points
        if len(outliers):
            out_in_view = outliers[outliers["date"].between(hist["date"].min(), hist["date"].max())]
            if len(out_in_view):
                out_points = (
                    alt.Chart(out_in_view)
                    .mark_point(color="#dc2626", size=80, shape="cross", strokeWidth=2)
                    .encode(
                        x="date:T",
                        y="quantity_sold:Q",
                        tooltip=[
                            alt.Tooltip("date:T", title="Outlier Date", format="%Y-%m-%d"),
                            alt.Tooltip("quantity_sold:Q", title="Units", format=",.0f"),
                            alt.Tooltip("z_score:Q", title="Robust Z-Score", format="%.2f"),
                        ],
                    )
                )
                layers.append(out_points)

        final_chart = alt.layer(*layers).properties(height=340).interactive()
        st.altair_chart(final_chart, width="stretch")

        if len(outliers):
            with st.expander(f"⚠️ {len(outliers)} Unusual Sales Week(s) Detected in History"):
                st.caption("Robust z-score vs trailing 12-week median. Examine for promotions or stockouts.")
                out_display = outliers.copy()
                num_cols = out_display.select_dtypes(include=[np.number]).columns
                out_display[num_cols] = out_display[num_cols].round(2)
                st.dataframe(out_display, hide_index=True, width="stretch")


# ================================================================ TAB 3: AI AGENT COPILOT
with tab_copilot:
    st.subheader(f"AI Supply Chain Copilot — {pid}")
    st.caption("Multi-turn reasoning assistant grounded in deterministic inventory calculus.")

    product_context = _build_product_context(pid, a, info, stock, weeks, start_label)
    rule_text = recommend_action(a, weeks, info)
    cache_key = (pid, weeks, start_label)

    if ai_enabled:
        if cache_key not in st.session_state.ai_rec_cache:
            with st.spinner("🤖 Claude is synthesizing supply-chain recommendation..."):
                rec_text = ai_recommendation(None, product_context, rule_text)
            st.session_state.ai_rec_cache[cache_key] = rec_text
        else:
            rec_text = st.session_state.ai_rec_cache[cache_key]

        gr_issues = guardrail_check(rec_text, a)

        # Copilot Hero Card
        chip_html = (
            '<span class="guardrail-chip pass">✓ Verified by Deterministic Calculus</span>'
            if not gr_issues
            else f'<span class="guardrail-chip warn">⚠️ Guardrail Flag ({len(gr_issues)} discrepancy)</span>'
        )

        st.markdown(
            f"""
            <div class="copilot-card">
                <div class="copilot-header">
                    <div class="copilot-title">✨ Claude Inventory Advisor (via AWS LLM Gateway)</div>
                    <div>{chip_html}</div>
                </div>
                <div class="copilot-body">{rec_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if gr_issues:
            with st.expander("⚠️ Guardrail Discrepancy Notice"):
                for issue in gr_issues:
                    st.write(f"- {issue}")

        col_act1, col_act2 = st.columns([1, 2])
        with col_act1:
            if st.button("🔄 Regenerate Analysis", width="stretch"):
                if cache_key in st.session_state.ai_rec_cache:
                    del st.session_state.ai_rec_cache[cache_key]
                st.rerun()
        with col_act2:
            if st.button("📋 Adopt Recommendation for Decision Note", width="stretch"):
                st.session_state.prefilled_note = rec_text
                st.success("Copied to Planner Decision form!")

        with st.expander("🔍 View Deterministic Rule-Based Baseline"):
            st.write(rule_text)
    else:
        st.markdown(
            f"""
            <div class="copilot-card">
                <div class="copilot-header">
                    <div class="copilot-title">⚙️ Rule-Based Recommendation Engine</div>
                    <span class="guardrail-chip pass">Deterministic Rules Active</span>
                </div>
                <div class="copilot-body">{rule_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("ℹ️ Set LLM_GATEWAY_URL / LLM_GATEWAY_API_KEY / LLM_MODEL on the server to activate autonomous multi-turn reasoning.")

    # Multi-turn Copilot Chat
    st.markdown("---")
    st.markdown("#### 💬 Ask the Inventory Copilot")

    if not ai_enabled:
        st.info("LLM gateway not configured on the server — ask a planner to set LLM_GATEWAY_URL / LLM_GATEWAY_API_KEY.")
    else:
        if pid not in st.session_state.chat_history:
            st.session_state.chat_history[pid] = []

        chat_hist = st.session_state.chat_history[pid]

        # Suggested Prompt Chips
        st.caption("Quick Scenarios:")
        qc1, qc2, qc3 = st.columns(3)
        if qc1.button("⚡ What if demand spikes by 25%?", key=f"q1_{pid}", width="stretch"):
            st.session_state.staged_prompt = "What would happen to our risk status if demand spikes by 25%?"
        if qc2.button("🛡️ Explain safety stock buffer", key=f"q2_{pid}", width="stretch"):
            st.session_state.staged_prompt = "Can you break down how the safety stock buffer protects against forecast error?"
        if qc3.button("📦 What if order is delayed 1 week?", key=f"q3_{pid}", width="stretch"):
            st.session_state.staged_prompt = "What is the stockout exposure if incoming orders are delayed by 1 week?"

        # Render conversation history
        for turn in chat_hist:
            with st.chat_message("user"):
                st.write(turn["user"])
            with st.chat_message("assistant", avatar="🤖"):
                st.write(turn["assistant"])

        # Determine prompt input (typed or via quick chip)
        chat_val = st.chat_input(f"Ask about {pid} (e.g. 'What if I order 300 more units?')")
        active_prompt = chat_val or st.session_state.staged_prompt

        if active_prompt:
            st.session_state.staged_prompt = None
            with st.chat_message("user"):
                st.write(active_prompt)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analyzing inventory scenario..."):
                    reply = ai_chat_response(
                        None,
                        product_context,
                        chat_hist,
                        active_prompt,
                    )
                st.write(reply)
            chat_hist.append({"user": active_prompt, "assistant": reply})
            st.session_state.chat_history[pid] = chat_hist
            st.rerun()

        if chat_hist:
            if st.button("Clear Chat History", key=f"clear_{pid}"):
                st.session_state.chat_history[pid] = []
                st.rerun()


# ================================================================ TAB 4: MODEL DIAGNOSTICS & DRIFT
with tab_models:
    st.subheader(f"Model Diagnostics & Forecast Drift — {pid}")

    diag_col1, diag_col2 = st.columns([1, 1])
    with diag_col1:
        st.markdown(f"**Active Model:** `{info['final_model']}`")
        dm1, dm2, dm3 = st.columns(3)
        dm1.metric("Model WAPE", f"{info['wape']:.1f}%")
        dm2.metric("MAE", f"{info['mae']:,.0f} units")
        dm3.metric("Bias", f"{info['bias']:+,.0f}/wk")
        if info["selection_window"]:
            st.caption(
                f"Trained on '{info['selection_window']}' window; accuracy measured on independent "
                f"held-out '{info['window']}' evaluation window."
            )

    with diag_col2:
        st.markdown("**Candidate Algorithm Comparison**")
        cmp = metrics[(metrics["product_id"] == pid) & (metrics["split"] == info["window"])].copy()
        cmp_num_cols = cmp.select_dtypes(include=[np.number]).columns
        cmp[cmp_num_cols] = cmp[cmp_num_cols].round(2)
        st.dataframe(
            cmp[["model", "mae", "wape", "bias", "bias_pct"]],
            hide_index=True,
            width="stretch",
        )

    # Rolling WAPE Drift Monitor
    st.markdown("---")
    st.markdown("#### Rolling Forecast Drift Monitor (6-Week WAPE)")
    model_col = MODEL_COL_MAP.get(info["final_model"])

    if model_col and model_col in model_forecasts.columns:
        rw = rolling_wape(model_forecasts, pid, model_col, window=6)
        has_drift = drift_flag(rw["rolling_wape"], info["wape"], tolerance_pt=3.0)

        dcol_chart, dcol_card = st.columns([3, 1])
        with dcol_chart:
            drift_base = alt.Chart(rw).encode(
                x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y", grid=True, gridColor="#f1f5f9")),
                y=alt.Y("rolling_wape:Q", title="WAPE %", axis=alt.Axis(grid=True, gridColor="#f1f5f9")),
                tooltip=[
                    alt.Tooltip("date:T", title="Week", format="%Y-%m-%d"),
                    alt.Tooltip("rolling_wape:Q", title="Rolling WAPE", format=".1f"),
                ],
            )
            drift_line = drift_base.mark_line(color="#0d9488", strokeWidth=2.4, point=True)
            drift_rule = alt.Chart(pd.DataFrame([{"wape": info["wape"]}])).mark_rule(
                color="#e11d48", strokeDash=[4, 4], strokeWidth=2
            ).encode(y="wape:Q")

            st.altair_chart((drift_line + drift_rule).properties(height=230), width="stretch")

        with dcol_card:
            if has_drift:
                st.error("⚠️ **Model Drift Detected**\nRecent accuracy has degraded beyond acceptable tolerance.")
            else:
                st.success("✓ **Model Accuracy Stable**\nRolling error remains well within tolerance limits.")

        st.caption("Drift guardrail monitors trailing accuracy to prevent silent decay in live production.")
    else:
        st.caption("Rolling WAPE is not available for this model configuration.")

    chart_png = CHART_DIR / f"{pid}_model_comparison.png"
    if chart_png.exists():
        with st.expander("View Week 2 Model Evaluation Diagnostic"):
            st.image(str(chart_png), width="stretch")


# ================================================================ TAB 5: PLANNER DECISION & AUDIT
with tab_audit:
    st.subheader(f"Human-in-the-Loop Decision Sign-Off — {pid}")
    st.caption("Log formal planner approval or override. All submissions are recorded in the audit trail.")

    with st.form(f"decision_form_{pid}"):
        choice = st.radio(
            "Planning Decision",
            ["Approve agent recommendation", "Override recommendation"],
            horizontal=True,
        )
        reviewed = st.checkbox(
            f"I have reviewed model WAPE ({info['wape']:.1f}%) and bias ({info['bias_pct']:+.1f}%) before deciding.",
            help="Automation bias control: ensures statistical limits were reviewed.",
        )
        override_action = st.text_input(
            "If overriding: what is the alternative replenishment action?",
            placeholder="e.g. Order 450 units instead of 300 due to planned promotion",
        )
        note = st.text_area(
            "Decision Rationale / Notes",
            value=st.session_state.prefilled_note,
            placeholder="Provide context or justify your override...",
        )
        submitted = st.form_submit_button("💾 Save & Log Decision")

    if submitted:
        if not reviewed:
            st.error("Please confirm you have reviewed the model WAPE and bias before signing off.")
        elif choice == "Override recommendation" and not (override_action.strip() and note.strip()):
            st.error("Overrides require both the specific action and a supporting rationale.")
        else:
            used_rec = st.session_state.ai_rec_cache.get(cache_key, rule_text)
            record = pd.DataFrame([{
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "product_id": pid,
                "planning_start": start_label,
                "horizon_weeks": weeks,
                "status": a["status_key"],
                "forecast_demand": round(a["demand"]),
                "current_stock": round(stock),
                "agent_recommendation": used_rec,
                "decision": choice,
                "planner_action": override_action.strip() if choice == "Override recommendation" else "As recommended",
                "note": note.strip(),
                "stock_source": "SAMPLE" if using_sample else "Real",
                "ai_used": ai_enabled,
                "reviewed_error_and_bias": reviewed,
            }])
            record.to_csv(DECISIONS_PATH, mode="a", header=not DECISIONS_PATH.exists(), index=False)
            st.session_state.prefilled_note = ""
            st.success(f"✓ Decision successfully logged to {DECISIONS_PATH.name}")

    if DECISIONS_PATH.exists():
        log = pd.read_csv(DECISIONS_PATH)
        log_prod = log[log["product_id"] == pid]
        st.markdown("---")
        st.markdown("#### 📜 Historic Audit Trail")
        if len(log_prod):
            st.dataframe(
                log_prod[["timestamp", "planning_start", "horizon_weeks", "status", "decision", "planner_action", "note", "ai_used"]].tail(10),
                hide_index=True,
                width="stretch",
            )
            csv_data = log_prod.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Audit Log (CSV)",
                data=csv_data,
                file_name=f"audit_log_{pid}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.caption("No historical decisions recorded for this SKU yet.")
