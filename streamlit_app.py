"""
CRAT Analyzer — Children Reading Acuity Test

Streamlit application for CRAT data entry, reading-speed calculation,
curve fitting, clinical metric calculation, and visualization.

This version does NOT require matplotlib or seaborn.
It uses Altair for plotting and has a fallback fitting method if scipy is unavailable.

Run locally:
    streamlit run streamlit_app.py
"""

# ============================================================
# Imports
# ============================================================

from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# scipy is optional. If unavailable, the app still runs using a grid-search fallback.
try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except Exception:
    curve_fit = None
    SCIPY_AVAILABLE = False


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="CRAT Analyzer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Constants
# ============================================================

N_CHARACTERS_PER_CARD = 18

# 17 print sizes from 1.3 to -0.3 logMAR in steps of 0.1
PRINT_SIZES = np.round(np.arange(1.3, -0.31, -0.1), 1)


# ============================================================
# Core Calculations
# ============================================================

def calculate_reading_speed(time_seconds: float, errors: int) -> float:
    """
    Calculate Chinese Reading Speed.

    CRS = 60 * (18 - errors) / time_seconds
    """
    if pd.isna(time_seconds) or time_seconds <= 0:
        return np.nan

    errors = int(np.clip(errors, 0, N_CHARACTERS_PER_CARD))
    correct_characters = N_CHARACTERS_PER_CARD - errors

    return 60.0 * correct_characters / float(time_seconds)


def calculate_cra(number_attempted: int, cumulative_errors: int) -> float:
    """
    Calculate Chinese Reading Acuity.

    CRA = 1.4 - (Number of sentences read * 0.1)
          + (Total cumulative errors * 0.0056)
    """
    return 1.4 - (number_attempted * 0.1) + (cumulative_errors * 0.0056)


def exponential_plateau_model(x, plateau, bottom, rate, x_shift):
    """
    Exponential rise-to-plateau model.

    y = bottom + (plateau - bottom) * (1 - exp(-rate * max(x - x_shift, 0)))

    Interpretation:
    - plateau = CMRS
    - x is print size in logMAR
    """
    x = np.asarray(x, dtype=float)
    effective_x = np.maximum(x - x_shift, 0)
    return bottom + (plateau - bottom) * (1 - np.exp(-rate * effective_x))


def fit_with_scipy(x, y):
    """
    Fit model using scipy.optimize.curve_fit.
    """
    plateau_init = max(float(np.nanmax(y)), 1.0)
    bottom_init = max(float(np.nanmin(y)), 0.0)
    rate_init = 3.0
    x_shift_init = float(np.nanmin(x))

    p0 = [plateau_init, bottom_init, rate_init, x_shift_init]

    lower_bounds = [
        0.0,      # plateau
        0.0,      # bottom
        0.01,     # rate
        -0.6,     # x_shift
    ]

    upper_bounds = [
        max(plateau_init * 3.0, 500.0),  # plateau
        max(plateau_init * 2.0, 400.0),  # bottom
        50.0,                            # rate
        1.5,                             # x_shift
    ]

    popt, _ = curve_fit(
        exponential_plateau_model,
        x,
        y,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000,
    )

    return popt


def fit_with_grid_search(x, y):
    """
    Fallback non-linear fitting method that does not require scipy.

    This is less sophisticated than scipy curve_fit but allows the app
    to continue running if scipy is unavailable on Streamlit Cloud.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    y_max = max(float(np.nanmax(y)), 1.0)
    y_min = max(float(np.nanmin(y)), 0.0)

    plateau_candidates = np.linspace(y_max, max(y_max * 1.8, y_max + 50), 25)
    bottom_candidates = np.linspace(0, y_min, 15)
    rate_candidates = np.linspace(0.5, 15.0, 30)
    x_shift_candidates = np.linspace(-0.5, min(float(np.nanmin(x)), 0.5), 25)

    best_params = None
    best_sse = np.inf

    for plateau in plateau_candidates:
        for bottom in bottom_candidates:
            if plateau <= bottom:
                continue

            for rate in rate_candidates:
                for x_shift in x_shift_candidates:
                    y_pred = exponential_plateau_model(
                        x,
                        plateau,
                        bottom,
                        rate,
                        x_shift,
                    )
                    sse = np.sum((y - y_pred) ** 2)

                    if sse < best_sse:
                        best_sse = sse
                        best_params = [plateau, bottom, rate, x_shift]

    return np.asarray(best_params, dtype=float)


def fit_reading_curve(df_valid: pd.DataFrame):
    """
    Fit the CRAT reading-speed curve.

    Returns a dictionary with:
    - success
    - method
    - params
    - x_fit
    - y_fit
    - cmrs
    - message
    """
    result = {
        "success": False,
        "method": None,
        "params": None,
        "x_fit": None,
        "y_fit": None,
        "cmrs": np.nan,
        "message": "",
    }

    if df_valid.empty:
        result["message"] = "No valid data available for curve fitting."
        return result

    x = df_valid["Print Size logMAR"].to_numpy(dtype=float)
    y = df_valid["CRS chars/min"].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        result["message"] = (
            "Insufficient valid data for non-linear fitting. "
            "At least 3 valid tested cards are recommended."
        )
        return result

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    try:
        if SCIPY_AVAILABLE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = fit_with_scipy(x, y)

            method = "Non-linear exponential plateau model using scipy"

        else:
            params = fit_with_grid_search(x, y)
            method = "Non-linear exponential plateau model using fallback grid search"

        x_fit = np.linspace(-0.4, 1.4, 500)
        y_fit = exponential_plateau_model(x_fit, *params)

        plateau = float(params[0])

        result.update(
            {
                "success": True,
                "method": method,
                "params": params,
                "x_fit": x_fit,
                "y_fit": y_fit,
                "cmrs": plateau,
                "message": "Curve fit completed successfully.",
            }
        )

        return result

    except Exception as exc:
        result["message"] = f"Curve fitting failed: {exc}"
        return result


def calculate_ccps(fit_result: dict, threshold_fraction: float):
    """
    Calculate CCPS from fitted model.

    CCPS = x where fitted speed reaches threshold_fraction * CMRS.
    """
    cmrs = fit_result.get("cmrs", np.nan)
    params = fit_result.get("params", None)

    if not np.isfinite(cmrs) or cmrs <= 0 or params is None:
        return np.nan, np.nan

    threshold_speed = threshold_fraction * cmrs

    plateau, bottom, rate, x_shift = params
    denominator = plateau - bottom

    if denominator <= 0 or rate <= 0:
        return np.nan, threshold_speed

    proportion = (threshold_speed - bottom) / denominator

    if proportion <= 0:
        ccps = x_shift
    elif proportion >= 1:
        ccps = np.nan
    else:
        ccps = x_shift - np.log(1.0 - proportion) / rate

    return float(ccps), float(threshold_speed)


# ============================================================
# Session State
# ============================================================

def make_default_dataframe():
    return pd.DataFrame(
        {
            "Tested": [True] * len(PRINT_SIZES),
            "Print Size logMAR": PRINT_SIZES,
            "Time seconds": [np.nan] * len(PRINT_SIZES),
            "Errors": [0] * len(PRINT_SIZES),
        }
    )


if "crat_data" not in st.session_state:
    st.session_state["crat_data"] = make_default_dataframe()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("CRAT Controls")

threshold_percent = st.sidebar.slider(
    "CCPS threshold percentage",
    min_value=80,
    max_value=95,
    value=90,
    step=1,
    help="CCPS is the logMAR print size where speed reaches this percentage of CMRS.",
)

threshold_fraction = threshold_percent / 100.0

st.sidebar.markdown("---")

if st.sidebar.button("Reset table"):
    st.session_state["crat_data"] = make_default_dataframe()
    st.rerun()

st.sidebar.markdown("### Dependency status")

if SCIPY_AVAILABLE:
    st.sidebar.success("scipy available")
else:
    st.sidebar.warning("scipy unavailable. Using fallback fitting.")

st.sidebar.info(
    "This version does not require matplotlib or seaborn."
)


# ============================================================
# Main Interface
# ============================================================

st.title("Children Reading Acuity Test CRAT Analyzer")

st.caption(
    "Clinical tool for calculating CRS, CRA, CMRS, and CCPS from CRAT data."
)

with st.expander("Mathematical definitions", expanded=False):
    st.markdown(
        r"""
        **Chinese Reading Speed CRS**

        $$
        CRS = \frac{60 \times (18 - Errors)}{Time}
        $$

        **Chinese Reading Acuity CRA**

        $$
        CRA = 1.4 - (Number\ of\ sentences\ read \times 0.1)
        + (Total\ cumulative\ errors \times 0.0056)
        $$

        **Curve model**

        $$
        y = bottom + (plateau - bottom)
        \left(1 - e^{-rate \cdot max(x - x_{shift}, 0)}\right)
        $$

        The fitted plateau is interpreted as **CMRS**.
        """
    )


# ============================================================
# Patient Information
# ============================================================

st.header("1. Patient Information")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    patient_id = st.text_input(
        "Patient ID / Name",
        placeholder="Enter patient ID or name",
    )

with col2:
    examiner = st.text_input(
        "Examiner",
        placeholder="Optional",
    )

with col3:
    test_date = st.date_input(
        "Test date",
        value=datetime.today(),
    )


# ============================================================
# Data Entry
# ============================================================

st.header("2. CRAT Data Entry")

st.write(
    "Enter reading time and number of errors for each card. "
    "Uncheck **Tested** for skipped or untested cards."
)

edited_df = st.data_editor(
    st.session_state["crat_data"],
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    column_config={
        "Tested": st.column_config.CheckboxColumn(
            "Tested",
            help="Uncheck if the card was skipped or not tested.",
            default=True,
        ),
        "Print Size logMAR": st.column_config.NumberColumn(
            "Print Size logMAR",
            format="%.1f",
            disabled=True,
        ),
        "Time seconds": st.column_config.NumberColumn(
            "Time seconds",
            min_value=0.0,
            step=0.1,
            format="%.2f",
            help="Reading time in seconds. Must be greater than 0.",
        ),
        "Errors": st.column_config.NumberColumn(
            "Errors",
            min_value=0,
            max_value=18,
            step=1,
            format="%d",
            help="Number of reading errors from 0 to 18.",
        ),
    },
    key="crat_editor",
)

st.session_state["crat_data"] = edited_df.copy()


# ============================================================
# Processing
# ============================================================

df = edited_df.copy()

df["Time seconds"] = pd.to_numeric(df["Time seconds"], errors="coerce")
df["Errors"] = pd.to_numeric(df["Errors"], errors="coerce").fillna(0).astype(int)
df["Errors"] = df["Errors"].clip(0, N_CHARACTERS_PER_CARD)

df["CRS chars/min"] = df.apply(
    lambda row: calculate_reading_speed(row["Time seconds"], row["Errors"])
    if bool(row["Tested"])
    else np.nan,
    axis=1,
)

df_tested = df[df["Tested"] == True].copy()

df_valid = df_tested[
    df_tested["Time seconds"].notna()
    & (df_tested["Time seconds"] > 0)
    & df_tested["CRS chars/min"].notna()
    & np.isfinite(df_tested["CRS chars/min"])
].copy()

number_attempted = int(df_tested.shape[0])
cumulative_errors = int(df_tested["Errors"].sum()) if number_attempted > 0 else 0

cra = calculate_cra(number_attempted, cumulative_errors)

fit_result = fit_reading_curve(df_valid)

cmrs = fit_result.get("cmrs", np.nan)
ccps, threshold_speed = calculate_ccps(fit_result, threshold_fraction)


# ============================================================
# Clinical Results
# ============================================================

st.header("3. Clinical Results")

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Cards attempted", number_attempted)

with m2:
    st.metric("Cumulative errors", cumulative_errors)

with m3:
    st.metric("CRA", f"{cra:.3f} logMAR")

with m4:
    st.metric(
        "CMRS",
        f"{cmrs:.1f} chars/min" if np.isfinite(cmrs) else "NA",
    )

with m5:
    st.metric(
        f"CCPS at {threshold_percent}%",
        f"{ccps:.3f} logMAR" if np.isfinite(ccps) else "NA",
    )

if fit_result["success"]:
    st.success(fit_result["message"])
else:
    st.warning(fit_result["message"])

if number_attempted > 0 and len(df_valid) < number_attempted:
    st.warning(
        "Some tested cards have missing or invalid times and were excluded from fitting."
    )


# ============================================================
# Processed Data
# ============================================================

st.header("4. Processed Data")

display_df = df.copy()
display_df["CRS chars/min"] = display_df["CRS chars/min"].round(2)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# Visualization with Altair
# ============================================================

st.header("5. CRAT Reading-Speed Curve")

chart_layers = []

# Observed data
observed_df = df_valid[
    ["Print Size logMAR", "CRS chars/min"]
].copy()

if not observed_df.empty:
    observed_points = (
        alt.Chart(observed_df)
        .mark_circle(size=100, color="#1f77b4", opacity=0.9)
        .encode(
            x=alt.X(
                "Print Size logMAR:Q",
                scale=alt.Scale(domain=[1.4, -0.4]),
                title="Print Size logMAR",
            ),
            y=alt.Y(
                "CRS chars/min:Q",
                title="Reading Speed characters/min",
            ),
            tooltip=[
                alt.Tooltip("Print Size logMAR:Q", format=".1f"),
                alt.Tooltip("CRS chars/min:Q", format=".2f"),
            ],
        )
    )
    chart_layers.append(observed_points)

# Fitted curve
if fit_result["x_fit"] is not None and fit_result["y_fit"] is not None:
    fit_df = pd.DataFrame(
        {
            "Print Size logMAR": fit_result["x_fit"],
            "CRS chars/min": fit_result["y_fit"],
        }
    )

    fitted_curve = (
        alt.Chart(fit_df)
        .mark_line(color="#d62728", strokeWidth=3)
        .encode(
            x=alt.X(
                "Print Size logMAR:Q",
                scale=alt.Scale(domain=[1.4, -0.4]),
                title="Print Size logMAR",
            ),
            y=alt.Y(
                "CRS chars/min:Q",
                title="Reading Speed characters/min",
            ),
            tooltip=[
                alt.Tooltip("Print Size logMAR:Q", format=".3f"),
                alt.Tooltip("CRS chars/min:Q", format=".2f"),
            ],
        )
    )
    chart_layers.append(fitted_curve)

# CMRS horizontal line
if np.isfinite(cmrs):
    cmrs_df = pd.DataFrame(
        {
            "y": [cmrs],
            "label": [f"CMRS = {cmrs:.1f} chars/min"],
        }
    )

    cmrs_line = (
        alt.Chart(cmrs_df)
        .mark_rule(color="#2ca02c", strokeDash=[6, 4], strokeWidth=2)
        .encode(
            y="y:Q",
            tooltip=["label:N"],
        )
    )
    chart_layers.append(cmrs_line)

# Threshold horizontal line
if np.isfinite(threshold_speed):
    threshold_df = pd.DataFrame(
        {
            "y": [threshold_speed],
            "label": [f"{threshold_percent}% CMRS = {threshold_speed:.1f} chars/min"],
        }
    )

    threshold_line = (
        alt.Chart(threshold_df)
        .mark_rule(color="#ff7f0e", strokeDash=[6, 4], strokeWidth=2)
        .encode(
            y="y:Q",
            tooltip=["label:N"],
        )
    )
    chart_layers.append(threshold_line)

# CCPS vertical line
if np.isfinite(ccps):
    ccps_df = pd.DataFrame(
        {
            "x": [ccps],
            "label": [f"CCPS = {ccps:.3f} logMAR"],
        }
    )

    ccps_line = (
        alt.Chart(ccps_df)
        .mark_rule(color="#9467bd", strokeDash=[6, 4], strokeWidth=2)
        .encode(
            x="x:Q",
            tooltip=["label:N"],
        )
    )
    chart_layers.append(ccps_line)

# CRA vertical line
if np.isfinite(cra):
    cra_df = pd.DataFrame(
        {
            "x": [cra],
            "label": [f"CRA = {cra:.3f} logMAR"],
        }
    )

    cra_line = (
        alt.Chart(cra_df)
        .mark_rule(color="#8c564b", strokeDash=[2, 4], strokeWidth=2)
        .encode(
            x="x:Q",
            tooltip=["label:N"],
        )
    )
    chart_layers.append(cra_line)

if chart_layers:
    final_chart = (
        alt.layer(*chart_layers)
        .properties(
            width="container",
            height=520,
            title="CRAT Reading Speed vs. Print Size",
        )
        .resolve_scale(
            y="shared",
            x="shared",
        )
        .configure_axis(
            grid=True,
            labelFontSize=12,
            titleFontSize=14,
        )
        .configure_title(
            fontSize=18,
            anchor="start",
        )
    )

    st.altair_chart(final_chart, use_container_width=True)

    st.info(
        f"""
        **Clinical summary**

        Patient: {patient_id if patient_id else "Not specified"}  
        CRA: {cra:.3f} logMAR  
        CMRS: {cmrs:.1f} chars/min if available  
        CCPS at {threshold_percent}%: {ccps:.3f} logMAR if available  
        Fit method: {fit_result.get("method", "NA")}
        """
    )

else:
    st.info("Enter valid CRAT data to generate the chart.")


# ============================================================
# Export
# ============================================================

st.header("6. Export Results")

export_df = df.copy()

export_df.insert(0, "Patient ID / Name", patient_id)
export_df.insert(1, "Examiner", examiner)
export_df.insert(2, "Test Date", str(test_date))

export_df["CRA logMAR"] = cra
export_df["CMRS chars/min"] = cmrs
export_df[f"CCPS {threshold_percent}% logMAR"] = ccps
export_df[f"Threshold Speed {threshold_percent}% chars/min"] = threshold_speed
export_df["Fit Method"] = fit_result.get("method")
export_df["Fit Message"] = fit_result.get("message")

csv = export_df.to_csv(index=False).encode("utf-8-sig")

safe_patient_id = patient_id.replace(" ", "_") if patient_id else "patient"

st.download_button(
    label="Download results as CSV",
    data=csv,
    file_name=f"CRAT_results_{safe_patient_id}_{test_date}.csv",
    mime="text/csv",
)

st.caption(
    "This tool is intended to assist clinical analysis. "
    "Results should be interpreted by a qualified eye-care professional."
)
