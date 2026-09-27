"""Week 3 inventory-risk logic (no Streamlit in here, so it can be tested on its own).

Idea: compare the forecast demand for the planning horizon with current stock.
  need      = forecast demand + safety stock
  high      = forecast demand * (1 + WAPE) + safety stock     (a "demand runs high" case)
  excess at = forecast demand * excess_ratio + safety stock

  stock < need                  -> Shortage risk
  need <= stock < high          -> Watch      (covered, but inside the model's normal error)
  stock > excess at             -> Excess risk
  otherwise                     -> Balanced
"""
import math

import numpy as np
import pandas as pd

STATUS = {
    "shortage": "🔴 Shortage risk",
    "watch": "🟠 Watch",
    "balanced": "🟢 Balanced",
    "excess": "🔵 Excess risk",
    "nostock": "⚪ No stock data",
}
DEFAULT_EXCESS_RATIO = 1.5
BIAS_NOTE_PCT = 2.0   # mention model bias in the explanation once it is at least this large


def horizon_forecast(final_forecast, product_id, start, weeks):
    """Sum of the selected model's forecast over `weeks` weeks starting at `start`.

    Returns (total_demand, rows_used). rows_used can be shorter than `weeks`
    if the forecast file runs out of weeks.
    """
    f = final_forecast[final_forecast["product_id"] == product_id].sort_values("date")
    f = f[f["date"] >= pd.Timestamp(start)].head(weeks)
    return float(f["forecast_quantity"].sum()), f


def assess(stock, demand, wape_pct, weeks=1, safety_stock=0.0, excess_ratio=DEFAULT_EXCESS_RATIO):
    """Classify stock against forecast demand. All quantities are units over the whole horizon."""
    need = demand + safety_stock
    high = demand * (1 + wape_pct / 100.0) + safety_stock
    excess_from = demand * excess_ratio + safety_stock

    if stock < need:
        key = "shortage"
    elif stock < high:
        key = "watch"
    elif stock > excess_from:
        key = "excess"
    else:
        key = "balanced"

    gap = stock - need
    return {
        "status_key": key,
        "status": STATUS[key],
        "demand": demand,
        "need": need,
        "high": high,
        "excess_from": excess_from,
        "gap": gap,
        "shortfall_units": int(math.ceil(-gap)) if gap < 0 else 0,
        "topup_units": int(math.ceil(high - stock)) if key in ("shortage", "watch") else 0,
        "excess_units": int(math.floor(gap)) if key == "excess" else 0,
        "cover_weeks": (stock / (demand / weeks)) if demand > 0 else np.inf,
    }


def recommend_action(a, weeks, model):
    """Rule-based recommendation text.

    `a` is the dict from assess(); `model` has final_model, wape, bias_pct, human_decision.
    The rules decide the action; an LLM (if you connect one) should only reword this text.
    """
    d, s = f"{a['demand']:,.0f}", f"{weeks} week{'s' if weeks != 1 else ''}"
    key = a["status_key"]

    if key == "shortage":
        text = (f"Replenish about {a['shortfall_units']:,} units to cover the expected demand of {d} over the next {s}. "
                f"Ordering up to {a['topup_units']:,} units would also cover a high-demand case (forecast +{model['wape']:.0f}%).")
    elif key == "watch":
        text = (f"Stock covers the expected demand of {d} over the next {s}, but only by {int(a['gap']):,} units, which is "
                f"within the model's normal error (±{model['wape']:.0f}%). Monitor closely or top up by up to {a['topup_units']:,} units.")
    elif key == "excess":
        text = (f"Stock is about {a['cover_weeks']:.1f} weeks of cover against {s} of expected demand ({d} units), "
                f"roughly {a['excess_units']:,} units above what is needed. Consider reducing or delaying upcoming purchases. "
                f"If this product is perishable, holding this much stock may raise the risk of waste.")
    else:
        text = (f"No action needed. Stock covers the expected demand of {d} over the next {s} "
                f"with {int(a['gap']):,} units to spare, without being excessive.")

    bias = model.get("bias_pct")
    if bias is not None and not pd.isna(bias):
        if bias <= -BIAS_NOTE_PCT:
            text += (f" Caution: {model['final_model']} has under-forecast by {abs(bias):.1f}% on past weeks, "
                     f"so real demand may run higher than the forecast.")
        elif bias >= BIAS_NOTE_PCT:
            text += (f" Note: {model['final_model']} has over-forecast by {bias:.1f}% on past weeks, "
                     f"so real demand may run lower than the forecast.")
    if model.get("human_decision") == "pending_review":
        text += " The forecast model has not been reviewed by a planner yet."
    return text


# =============================================================================
# Controls added to respond to Section 7 (Risks, Controls and Limitations) of
# the proposal. Each function below is deterministic and independently
# testable — none of it depends on Streamlit or an LLM.
# =============================================================================

# ---------------------------------------------------------------- (2) Sample / real stock data quality
REQUIRED_STOCK_COLS = ["product_id", "current_stock", "safety_stock"]


def validate_stock_file(stock_df: pd.DataFrame, known_products=None) -> list:
    """Schema / sanity checks for a stock feed (sample or real).

    Returns a list of human-readable issue strings; an empty list means the
    file passed all checks. This is the "validate units/timestamps" control
    called out under 'Sample stock data' in the proposal's risk table.
    """
    issues = []
    missing_cols = [c for c in REQUIRED_STOCK_COLS if c not in stock_df.columns]
    if missing_cols:
        issues.append(f"Missing required column(s): {missing_cols}. Expected {REQUIRED_STOCK_COLS}.")
        return issues  # can't check further without the columns

    if stock_df["product_id"].duplicated().any():
        dupes = stock_df.loc[stock_df["product_id"].duplicated(), "product_id"].unique().tolist()
        issues.append(f"Duplicate product_id rows: {dupes}. Only the first row per product is used downstream.")

    for col in ["current_stock", "safety_stock"]:
        non_numeric = pd.to_numeric(stock_df[col], errors="coerce").isna() & stock_df[col].notna()
        if non_numeric.any():
            issues.append(f"Non-numeric values in '{col}' for: {stock_df.loc[non_numeric, 'product_id'].tolist()}")
        neg = pd.to_numeric(stock_df[col], errors="coerce") < 0
        if neg.any():
            issues.append(f"Negative values in '{col}' for: {stock_df.loc[neg, 'product_id'].tolist()}")

    if known_products is not None:
        missing_products = sorted(set(known_products) - set(stock_df["product_id"]))
        if missing_products:
            issues.append(f"No stock row for forecasted product(s): {missing_products}. They will show 'No stock data'.")

    if "as_of" in stock_df.columns:
        as_of = pd.to_datetime(stock_df["as_of"], errors="coerce")
        stale = as_of < (pd.Timestamp.now() - pd.Timedelta(days=7))
        if stale.any():
            issues.append(f"Stock snapshot older than 7 days for: {stock_df.loc[stale, 'product_id'].tolist()}")
    return issues


# ---------------------------------------------------------------- (5) Outliers and data quality
def detect_outlier_weeks(weekly: pd.DataFrame, product_id: str, z_thresh: float = 3.0,
                          target_col: str = "quantity_sold") -> pd.DataFrame:
    """Flag weeks whose demand is an unusual spike/drop for that product (robust z-score on
    a trailing 12-week window), independent of the fixed PARTIAL_WEEKS list used for the
    known incomplete-data week. Returns the flagged rows only."""
    s = weekly[weekly["product_id"] == product_id].sort_values("date").copy()
    med = s[target_col].rolling(12, min_periods=6).median()
    mad = (s[target_col] - med).abs().rolling(12, min_periods=6).median()
    robust_z = (s[target_col] - med) / (1.4826 * mad.replace(0, np.nan))
    s["robust_z"] = robust_z
    return s[s["robust_z"].abs() >= z_thresh][["date", target_col, "robust_z"]]


# ---------------------------------------------------------------- (3) Historical drift monitoring
def rolling_wape(model_forecasts: pd.DataFrame, product_id: str, model_col: str, window: int = 6) -> pd.DataFrame:
    """Rolling WAPE over the last `window` scored weeks for one product/model, so a planner
    (or a scheduled job) can see accuracy degrade before it shows up in a quarterly review."""
    s = model_forecasts[model_forecasts["product_id"] == product_id].sort_values("date").copy()
    err = (s[model_col] - s["actual"]).abs()
    roll_err = err.rolling(window, min_periods=max(2, window // 2)).sum()
    roll_actual = s["actual"].rolling(window, min_periods=max(2, window // 2)).sum()
    s["rolling_wape"] = 100 * roll_err / roll_actual
    return s[["date", "actual", model_col, "rolling_wape"]]


def drift_flag(rolling_wape_series: pd.Series, baseline_wape: float, tolerance_pt: float = 3.0) -> bool:
    """True if the most recent rolling WAPE has drifted more than `tolerance_pt` points above the
    WAPE the model was selected on. A simple, explainable trigger for 'retrain/reselect on a
    defined cadence' rather than an unexplained accuracy drop."""
    recent = rolling_wape_series.dropna()
    if recent.empty or pd.isna(baseline_wape):
        return False
    return bool(recent.iloc[-1] - baseline_wape > tolerance_pt)


# ---------------------------------------------------------------- (8) Cost / ROI uncertainty
def estimate_cost_impact(a: dict, unit_margin: float = 0.0, holding_cost_pct_per_week: float = 0.0,
                          unit_cost: float = 0.0, weeks: int = 1) -> dict:
    """Optional, clearly-labelled $ estimate. Only meaningful once real unit economics are
    supplied — everything defaults to 0 so the dashboard never invents a number. This is a
    rough order-of-magnitude figure for a pilot business case, not an accounting result."""
    shortage_cost = a.get("shortfall_units", 0) * unit_margin
    excess_units = a.get("excess_units", 0)
    excess_holding_cost = excess_units * unit_cost * holding_cost_pct_per_week * weeks
    return {
        "estimated_lost_margin": round(shortage_cost, 2),
        "estimated_excess_holding_cost": round(excess_holding_cost, 2),
        "inputs_provided": bool(unit_margin or unit_cost or holding_cost_pct_per_week),
    }


# ---------------------------------------------------------------- (7) LLM explanation risk (guardrail)
def guardrail_check(ai_text: str, a: dict, tolerance: float = 0.15) -> list:
    """Cheap, deterministic sanity check on an LLM-written recommendation: does it contain
    numbers roughly consistent with the computed facts, and does it avoid stating the
    opposite risk status? This does NOT verify meaning/nuance — it only catches gross
    numeric drift or a flipped status, so a planner knows when to double-check the text
    instead of trusting fluent prose by default."""
    problems = []
    if not ai_text:
        return problems
    digits = [float(n.replace(",", "")) for n in _re_findall_numbers(ai_text)]

    def _has_close_match(value):
        if value in (None, 0) or pd.isna(value):
            return True  # nothing meaningful to check against
        return any(abs(d - abs(value)) <= tolerance * max(abs(value), 1) for d in digits)

    if not _has_close_match(a.get("demand")):
        problems.append("Forecast demand figure isn't clearly reflected in the AI text.")

    status_key = a.get("status_key")
    opposite = {"shortage": "excess", "excess": "shortage"}.get(status_key)
    if opposite and opposite in ai_text.lower():
        problems.append(f"AI text mentions '{opposite}' while the computed status is '{status_key}'.")
    return problems


def _re_findall_numbers(text: str) -> list:
    import re
    return re.findall(r"-?\d[\d,]*\.?\d*", text)
