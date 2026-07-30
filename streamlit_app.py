"""
CRAT Analyzer — Children Reading Acuity Test

Streamlit app for:
1. CRAT clinical data entry
2. Reading speed calculation
3. CRA calculation
4. Exponential decay-to-asymptote fitting
5. CMRS and CCPS calculation
6. Altair visualization

This version avoids matplotlib/seaborn to prevent deployment issues on Streamlit Cloud.

Run:
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

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except Exception:
    curve_fit = None
    SCIPY_AVAILABLE = False


# ============================================================
# Page configuration
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

# CRAT has 17 cards from 1.3 to -0.3 logMAR in 0.1 steps.
PRINT_SIZES = np.round(np.arange(1.3, -0.31, -0.1), 1)

# Clinical default for CCPS.
DEFAULT_CCPS_THRESHOLD_PERCENT = 90


# ============================================================
# Core CRAT calculations
# ============================================================

def calculate_reading_speed(time_seconds: float, errors: int) -> float:
    """
    Calculate Chinese Reading Speed, CRS.

    CRS = 60 * (18 - errors) / time_seconds

    Parameters
    ----------
    time_seconds : float
        Reading time in seconds.
    errors : int
        Number of reading errors.

    Returns
    -------
    float
        Reading speed in characters per minute.
    """
    if pd.isna(time_seconds) or time_seconds <= 0:
        return np.nan

    errors = int(np.clip(errors, 0, N_CHARACTERS_PER_CARD))
    correct_characters = N_CHARACTERS_PER_CARD - errors

    return 60.0 * correct_characters / float(time_seconds)


def calculate_cra(number_attempted: int, cumulative_errors: int) -> float:
    """
    Calculate Chinese Reading Acuity, CRA.

    CRA = 1.4 - (Number of sentences read * 0.1)
          + (Total cumulative errors * 0.0056)

    Here, "sentences read" is interpreted as the number of CRAT cards attempted.
    """
    return 1.4 - (number_attempted * 0.1) + (cumulative_errors * 0.0056)


# ============================================================
# Exponential decay-to-asymptote model
# ============================================================

def exponential_decay_to_asymptote(x, cmrs, amplitude, rate):
    """
    Exponential decay-to-asymptote model for reading speed.

    Model:
        y = CMRS - amplitude * exp(-rate * (x - x_ref))

    In this app, x_ref is fixed globally during fitting to the smallest
    print size in the fitted range. To keep curve_fit simple, x_ref is
    handled by transforming x before calling this function.

    Equivalent transformed model:
        y = CMRS - amplitude * exp(-rate * x_transformed)

    where:
        x_transformed = x - x_ref

    Interpretation:
    - CMRS is the asymptotic maximum reading speed.
    - amplitude determines how far below CMRS the curve starts.
    - rate controls how quickly reading speed approaches CMRS.
    """
    return cmrs - amplitude * np.exp(-rate * x)


def predict_exponential_decay(x_original, params, x_ref):
    """
    Predict reading speed from the fitted exponential decay-to-asymptote model.

    Parameters
    ----------
    x_original : array-like
        Original print size in logMAR.
    params : array-like
        Fitted parameters [cmrs, amplitude, rate].
    x_ref : float
        Reference x-value used for transformation.

    Returns
    -------
    array-like
        Predicted reading speed.
    """
    cmrs, amplitude, rate = params
    x_transformed = np.asarray(x_original, dtype=float) - x_ref
    return exponential_decay_to_asymptote(x_transformed, cmrs, amplitude, rate)


def fit_exponential_decay_scipy(x, y):
    """
    Fit the exponential decay-to-asymptote model using scipy curve_fit.

    The model is:
        y = CMRS - amplitude * exp(-rate * (x - x_ref))

    x_ref is chosen as the smallest x-value among valid tested cards.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_ref = float(np.nanmin(x))
    x_transformed = x - x_ref

    y_max = float(np.nanmax(y))
    y_min = float(np.nanmin(y))
    y_range = max(y_max - y_min, 1.0)

    # Initial guesses
    cmrs_init = max(y_max * 1.05, y_max + 1.0)
    amplitude_init = max(cmrs_init - y_min, 1.0)
    rate_init = 3.0

    p0 = [cmrs_init, amplitude_init, rate_init]

    # Bounds
    lower_bounds = [
        y_max,       # CMRS should be at least observed maximum speed
        0.0,         # amplitude
        0.001,       # rate
    ]

    upper_bounds = [
        max(y_max * 3.0, 500.0),   # CMRS
        max(y_range * 10.0, 500.0), # amplitude
        50.0,                      # rate
    ]

    popt, pcov = curve_fit(
        exponential_decay_to_asymptote,
        x_transformed,
        y,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000,
    )

    return popt, pcov, x_ref


def fit_exponential_decay_grid_search(x, y):
    """
    Fallback fitting method if scipy is unavailable.

    This performs a simple grid search over CMRS, amplitude, and rate.
    It is less precise than scipy curve_fit but keeps the app functional.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_ref = float(np.nanmin(x))
    x_transformed = x - x_ref

    y_max = float(np.nanmax(y))
    y_min = float(np.nanmin(y))
    y_range = max(y_max - y_min, 1.0)

    cmrs_candidates = np.linspace(y_max, max(y_max * 1.8, y_max + 80.0), 50)
    amplitude_candidates = np.linspace(0.0, max(y_range * 5.0, 300.0), 50)
    rate_candidates = np.linspace(0.1, 20.0, 60)

    best_params = None
    best_sse = np.inf

    for cmrs in cmrs_candidates:
        for amplitude in amplitude_candidates:
            for rate in rate_candidates:
                y_pred = exponential_decay_to_asymptote(
                    x_transformed,
                    cmrs,
                    amplitude,
                    rate,
                )

                sse = np.sum((y - y_pred) ** 2)

                if sse < best_sse:
                    best_sse = sse
                    best_params = np.array([cmrs, amplitude, rate], dtype=float)

    return best_params, None, x_ref


def fit_reading_curve(df_valid: pd.DataFrame):
    """
    Fit reading speed against print size using the exponential decay-to-asymptote model.

    Parameters
    ----------
    df_valid : pd.DataFrame
        DataFrame with valid rows containing:
        - Print Size logMAR
        - CRS chars/min

    Returns
    -------
    dict
        Fit results.
    """
    result = {
        "success": False,
        "method": None,
        "params": None,
        "x_ref": None,
        "x_fit": None,
        "y_fit": None,
        "cmrs": np.nan,
        "message": "",
    }

    if df_valid.empty:
        result["message"] = "No valid CRAT data available for curve fitting."
        return result

    x = df_valid["Print Size logMAR"].to_numpy(dtype=float)
    y = df_valid["CRS chars/min"].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        result["message"] = (
            "Insufficient valid data for exponential fitting. "
            "At least 3 valid tested cards are recommended."
        )
        return result

    # Sort by print size ascending for stable fitting.
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    try:
        if SCIPY_AVAILABLE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params, pcov, x_ref = fit_exponential_decay_scipy(x, y)

            method = "Exponential decay-to-asymptote model using scipy"

        else:
            params, pcov, x_ref = fit_exponential_decay_grid_search(x, y)
            method = "Exponential decay-to-asymptote model using fallback grid search"

        x_fit = np.linspace(-0.4, 1.4, 600)
        y_fit = predict_exponential_decay(x_fit, params, x_ref)

        cmrs = float(params[0])

        result.update(
            {
                "success": True,
                "method": method,
                "params": params,
                "x_ref": x_ref,
                "x_fit": x_fit,
                "y_fit": y_fit,
                "cmrs": cmrs,
                "message": "Exponential decay-to-asymptote fitting completed successfully.",
            }
        )

        return result

    except Exception as exc:
        result["message"] = f"Exponential fitting failed: {exc}"
        return result


def calculate_ccps_from_exponential_decay(fit_result: dict, threshold_fraction: float = 0.90):
    """
    Calculate Chinese Critical Print Size, CCPS.

    CCPS is defined as the print size where the fitted reading speed reaches:

        threshold_fraction * CMRS

    For model:
        y = CMRS - amplitude * exp(-rate * (x - x_ref))

    Solve for x:

        threshold * CMRS = CMRS - amplitude * exp(-rate * (x - x_ref))

        amplitude * exp(-rate * (x - x_ref)) = CMRS - threshold * CMRS

        exp(-rate * (x - x_ref)) = CMRS * (1 - threshold) / amplitude

        x = x_ref - ln[CMRS * (1 - threshold) / amplitude] / rate

    Returns
    -------
    tuple
        ccps, threshold_speed
    """
    cmrs = fit_result.get("cmrs", np.nan)
    params = fit_result.get("params", None)
    x_ref = fit_result.get("x_ref", None)

    if params is None or x_ref is None:
        return np.nan, np.nan

    if not np.isfinite(cmrs) or cmrs <= 0:
        return np.nan, np.nan

    cmrs, amplitude, rate = params

    threshold_speed = threshold_fraction * cmrs

    if amplitude <= 0 or rate <= 0:
        return np.nan, float(threshold_speed)

    ratio = (cmrs - threshold_speed) / amplitude

    if ratio <= 0:
        return np.nan, float(threshold_speed)

    ccps = x_ref - np.log(ratio) / rate

    return float(ccps), float(threshold_speed)


# ============================================================
# Data initialization
# ============================================================

def make_default_dataframe():
    """
    Create default CRAT data-entry table.
    """
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
# Sidebar controls
# ============================================================

st.sidebar.title("CRAT Controls")

st.sidebar.markdown("### CCPS threshold")

threshold_percent = st.sidebar.slider(
    "Percent of CMRS used for CCPS",
    min_value=80,
    max_value=95,
    value=DEFAULT_CCPS_THRESHOLD_PERCENT,
    step=1,
    help=(
        "CRAT critical print size is commonly calculated at 90% of "
        "maximum reading speed."
    ),
)

threshold_fraction = threshold_percent / 100.0

st.sidebar.markdown("---")

if st.sidebar.button("Reset CRAT table"):
    st.session_state["crat_data"] = make_default_dataframe()
    st.rerun()

st.sidebar.markdown("### Fitting method")

st.sidebar.info(
    """
    The app fits an exponential decay-to-asymptote model:

    y = CMRS - A × exp[-k × (x - x_ref)]

    CCPS is calculated at 90% of CMRS by default.
    """
)

if SCIPY_AVAILABLE:
    st.sidebar.success("scipy available")
else:
    st.sidebar.warning("scipy unavailable. Using fallback grid-search fitting.")


# ============================================================
# Main title
# ============================================================

st.title("Children Reading Acuity Test CRAT Analyzer")

st.caption(
    "Clinical tool for calculating CRS, CRA, CMRS, and CCPS using an "
    "exponential decay-to-asymptote model."
)

with st.expander("Model and clinical definitions", expanded=False):
    st.markdown(
        r"""
        ### Chinese Reading Speed CRS

        $$
        CRS = \frac{60 \times (18 - Errors)}{Time}
        $$

        ### Chinese Reading Acuity CRA

        $$
        CRA = 1.4 - (Number\ of\ sentences\ read \times 0.1)
        + (Total\ cumulative\ errors \times 0.0056)
        $$

        ### Exponential decay-to-asymptote fitting model

        $$
        y = CMRS - A e^{-k(x - x_{ref})}
        $$

        where:

        - $$y$$ is reading speed in characters/min.
        - $$x$$ is print size in logMAR.
        - $$CMRS$$ is the asymptotic Chinese Maximum Reading Speed.
        - $$A$$ is the amplitude.
        - $$k$$ is the exponential rate constant.
        - $$x_{ref}$$ is the smallest fitted print size.

        ### CCPS

        Chinese Critical Print Size is calculated as the print size where:

        $$
        y = 0.90 \times CMRS
        $$

        by default.
        """
    )


# ============================================================
# Patient information
# ============================================================

st.header("1. Patient Information")

col_patient_1, col_patient_2, col_patient_3 = st.columns([2, 2, 1])

with col_patient_1:
    patient_id = st.text_input(
        "Patient ID / Name",
        placeholder="Enter patient ID or name",
    )

with col_patient_2:
    examiner = st.text_input(
        "Examiner",
        placeholder="Optional",
    )

with col_patient_3:
    test_date = st.date_input(
        "Test date",
        value=datetime.today(),
    )


# ============================================================
# Data entry
# ============================================================

st.header("2. CRAT Data Entry")

st.write(
    "Enter reading time and errors for each card. "
    "Uncheck **Tested** for cards that were skipped or not tested."
)

edited_df = st.data_editor(
    st.session_state["crat_data"],
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    column_config={
        "Tested": st.column_config.CheckboxColumn(
            "Tested",
            help="Uncheck if this card was skipped or not tested.",
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
# Data processing
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

ccps, threshold_speed = calculate_ccps_from_exponential_decay(
    fit_result,
    threshold_fraction=threshold_fraction,
)


# ============================================================
# Clinical results
# ============================================================

st.header("3. Clinical Results")

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

with metric_1:
    st.metric("Cards attempted", number_attempted)

with metric_2:
    st.metric("Cumulative errors", cumulative_errors)

with metric_3:
    st.metric("CRA", f"{cra:.3f} logMAR")

with metric_4:
    st.metric(
        "CMRS",
        f"{cmrs:.1f} chars/min" if np.isfinite(cmrs) else "NA",
    )

with metric_5:
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
        "Some tested cards have missing or invalid reading times and were excluded "
        "from CRS calculation and curve fitting."
    )


# ============================================================
# Processed data table
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
# Visualization using Altair
# ============================================================

st.header("5. CRAT Reading-Speed Curve")

chart_layers = []

# Observed CRS data
observed_df = df_valid[
    ["Print Size logMAR", "CRS chars/min"]
].copy()

if not observed_df.empty:
    observed_points = (
        alt.Chart(observed_df)
        .mark_circle(size=110, color="#1f77b4", opacity=0.9)
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
            "label": [
                f"{threshold_percent}% CMRS = {threshold_speed:.1f} chars/min"
            ],
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
            height=540,
            title="CRAT Reading Speed vs. Print Size",
        )
        .resolve_scale(
            x="shared",
            y="shared",
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

    st.markdown("### Clinical Summary")

    st.info(
        f"""
        **Patient:** {patient_id if patient_id else "Not specified"}  
        **CRA:** {cra:.3f} logMAR  
        **CMRS:** {cmrs:.1f} chars/min  
        **CCPS at {threshold_percent}% CMRS:** {ccps:.3f} logMAR  
        **Threshold speed:** {threshold_speed:.1f} chars/min  
        **Fit method:** {fit_result.get("method", "NA")}
        """
        if np.isfinite(cmrs) and np.isfinite(ccps)
        else
        f"""
        **Patient:** {patient_id if patient_id else "Not specified"}  
        **CRA:** {cra:.3f} logMAR  
        **CMRS:** NA  
        **CCPS at {threshold_percent}% CMRS:** NA  
        **Fit method:** {fit_result.get("method", "NA")}
        """
    )

else:
    st.info("Enter valid CRAT data to generate the fitted curve.")


# ============================================================
# Export results
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

if fit_result.get("params") is not None:
    params = fit_result["params"]
    export_df["Fit CMRS parameter"] = params[0]
    export_df["Fit amplitude parameter"] = params[1]
    export_df["Fit rate parameter"] = params[2]
    export_df["Fit x_ref"] = fit_result.get("x_ref")

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
