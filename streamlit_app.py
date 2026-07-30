"""
CRAT Analyzer — Children Reading Acuity Test

Streamlit application for clinical data entry, reading-speed calculation,
non-linear curve fitting, and visualization of CRAT results.

Run with:
    streamlit run crat_analyzer.py
"""

# ============================================================
# Imports
# ============================================================
import streamlit as st
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Included per requirement.
# Note: Streamlit does not natively render ipywidgets in the same way as Jupyter.
# This app uses Streamlit-native widgets for production-ready deployment.
try:
    import ipywidgets as widgets  # noqa: F401
except ImportError:
    widgets = None


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="CRAT Analyzer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", context="talk")


# ============================================================
# Constants
# ============================================================

N_CHARACTERS_PER_CARD = 18

# CRAT print sizes: 1.3 to -0.3 logMAR in steps of 0.1
PRINT_SIZES = np.round(np.arange(1.3, -0.31, -0.1), 1)

DEFAULT_TIME_SECONDS = np.nan
DEFAULT_ERRORS = 0


# ============================================================
# Mathematical Functions
# ============================================================

def calculate_reading_speed(time_seconds: float, errors: int) -> float:
    """
    Calculate Chinese Reading Speed, CRS.

    CRS = 60 * (18 - Reading Errors) / Reading Time

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

    errors = max(0, min(int(errors), N_CHARACTERS_PER_CARD))
    correct_chars = N_CHARACTERS_PER_CARD - errors

    return 60.0 * correct_chars / time_seconds


def calculate_cra(number_attempted: int, cumulative_errors: int) -> float:
    """
    Calculate Chinese Reading Acuity, CRA.

    CRA = 1.4 - (Number of sentences read * 0.1)
          + (Total cumulative errors * 0.0056)

    Parameters
    ----------
    number_attempted : int
        Number of tested/attempted CRAT cards.
    cumulative_errors : int
        Total cumulative reading errors across attempted cards.

    Returns
    -------
    float
        Chinese Reading Acuity in logMAR.
    """
    return 1.4 - (number_attempted * 0.1) + (cumulative_errors * 0.0056)


def exponential_plateau_model(x, plateau, bottom, rate, x_shift):
    """
    Exponential rise-to-plateau model.

    This model assumes reading speed increases as print size becomes larger
    and approaches an asymptotic plateau.

    y = bottom + (plateau - bottom) * (1 - exp(-rate * max(x - x_shift, 0)))

    Parameters
    ----------
    x : array-like
        Print size in logMAR.
    plateau : float
        Asymptotic maximum reading speed, i.e., CMRS.
    bottom : float
        Lower asymptote / floor reading speed.
    rate : float
        Growth rate.
    x_shift : float
        Horizontal shift / approximate starting point of growth.

    Returns
    -------
    array-like
        Predicted reading speed.
    """
    x = np.asarray(x)
    effective_x = np.maximum(x - x_shift, 0)
    return bottom + (plateau - bottom) * (1 - np.exp(-rate * effective_x))


def fit_reading_curve(df_valid: pd.DataFrame):
    """
    Fit exponential plateau curve to valid CRAT data.

    Parameters
    ----------
    df_valid : pd.DataFrame
        DataFrame containing valid tested rows with columns:
        'Print Size logMAR' and 'CRS chars/min'.

    Returns
    -------
    dict
        Dictionary containing fitting results and metadata.
    """
    x = df_valid["Print Size logMAR"].to_numpy(dtype=float)
    y = df_valid["CRS chars/min"].to_numpy(dtype=float)

    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]

    result = {
        "success": False,
        "method": None,
        "params": None,
        "message": "",
        "x_fit": None,
        "y_fit": None,
        "cmrs": np.nan,
    }

    if len(x) < 3:
        result["message"] = (
            "Insufficient valid data for non-linear curve fitting. "
            "At least 3 valid tested cards are recommended."
        )
        return result

    # Sort by ascending print size for fitting/interpolation.
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    try:
        plateau_init = max(np.nanmax(y_sorted), 1.0)
        bottom_init = max(np.nanmin(y_sorted), 0.0)
        rate_init = 3.0
        x_shift_init = min(x_sorted)

        p0 = [plateau_init, bottom_init, rate_init, x_shift_init]

        lower_bounds = [
            0.0,                 # plateau
            0.0,                 # bottom
            0.01,                # rate
            -0.6,                # x_shift
        ]

        upper_bounds = [
            max(plateau_init * 3, 500.0),  # plateau
            max(plateau_init * 2, 400.0),  # bottom
            50.0,                         # rate
            1.5,                          # x_shift
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                exponential_plateau_model,
                x_sorted,
                y_sorted,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000,
            )

        plateau, bottom, rate, x_shift = popt

        x_fit = np.linspace(-0.4, 1.4, 500)
        y_fit = exponential_plateau_model(x_fit, *popt)

        result.update(
            {
                "success": True,
                "method": "Non-linear exponential plateau model",
                "params": popt,
                "covariance": pcov,
                "message": "Curve fit completed successfully.",
                "x_fit": x_fit,
                "y_fit": y_fit,
                "cmrs": plateau,
            }
        )

        return result

    except Exception as exc:
        # Graceful fallback to interpolation if curve fitting fails.
        try:
            unique_x, unique_idx = np.unique(x_sorted, return_index=True)
            unique_y = y_sorted[unique_idx]

            if len(unique_x) >= 2:
                interpolation = interp1d(
                    unique_x,
                    unique_y,
                    kind="linear",
                    fill_value="extrapolate",
                    bounds_error=False,
                )

                x_fit = np.linspace(-0.4, 1.4, 500)
                y_fit = interpolation(x_fit)
                cmrs = np.nanmax(unique_y)

                result.update(
                    {
                        "success": False,
                        "method": "Linear interpolation fallback",
                        "params": None,
                        "message": (
                            "Non-linear fitting failed. "
                            f"Using linear interpolation fallback. Details: {exc}"
                        ),
                        "x_fit": x_fit,
                        "y_fit": y_fit,
                        "cmrs": cmrs,
                    }
                )
                return result

        except Exception as interp_exc:
            result["message"] = (
                "Both non-linear fitting and interpolation failed. "
                f"Fit error: {exc}. Interpolation error: {interp_exc}"
            )
            return result

    return result


def calculate_ccps_from_fit(fit_result: dict, threshold_fraction: float):
    """
    Calculate Chinese Critical Print Size, CCPS.

    CCPS is defined as the print size where fitted reading speed reaches
    a threshold fraction of CMRS.

    Parameters
    ----------
    fit_result : dict
        Output from fit_reading_curve.
    threshold_fraction : float
        Fraction of CMRS, e.g., 0.90 for 90%.

    Returns
    -------
    tuple
        ccps, threshold_speed
    """
    cmrs = fit_result.get("cmrs", np.nan)

    if not np.isfinite(cmrs) or cmrs <= 0:
        return np.nan, np.nan

    threshold_speed = threshold_fraction * cmrs

    # If non-linear model succeeded, solve analytically.
    if fit_result.get("params") is not None:
        plateau, bottom, rate, x_shift = fit_result["params"]

        denominator = plateau - bottom

        if denominator <= 0:
            return np.nan, threshold_speed

        proportion = (threshold_speed - bottom) / denominator

        if proportion <= 0:
            ccps = x_shift
        elif proportion >= 1:
            ccps = np.nan
        else:
            ccps = x_shift - np.log(1 - proportion) / rate

        return ccps, threshold_speed

    # Fallback for interpolation: find x closest to threshold on fitted curve.
    x_fit = fit_result.get("x_fit")
    y_fit = fit_result.get("y_fit")

    if x_fit is None or y_fit is None:
        return np.nan, threshold_speed

    finite_mask = np.isfinite(x_fit) & np.isfinite(y_fit)
    x_fit = x_fit[finite_mask]
    y_fit = y_fit[finite_mask]

    if len(x_fit) == 0:
        return np.nan, threshold_speed

    idx = np.argmin(np.abs(y_fit - threshold_speed))
    ccps = x_fit[idx]

    return ccps, threshold_speed


# ============================================================
# Data Initialization
# ============================================================

def get_default_crat_dataframe():
    """
    Create default CRAT data-entry DataFrame.
    """
    return pd.DataFrame(
        {
            "Tested": [True] * len(PRINT_SIZES),
            "Print Size logMAR": PRINT_SIZES,
            "Time seconds": [DEFAULT_TIME_SECONDS] * len(PRINT_SIZES),
            "Errors": [DEFAULT_ERRORS] * len(PRINT_SIZES),
        }
    )


if "crat_data" not in st.session_state:
    st.session_state["crat_data"] = get_default_crat_dataframe()


# ============================================================
# Sidebar Controls
# ============================================================

st.sidebar.title("CRAT Controls")

threshold_percent = st.sidebar.slider(
    "CCPS threshold percentage of CMRS",
    min_value=80,
    max_value=95,
    value=90,
    step=1,
    help="CCPS is the print size where reading speed reaches this percentage of CMRS.",
)

threshold_fraction = threshold_percent / 100.0

st.sidebar.markdown("---")

if st.sidebar.button("Reset CRAT table", type="secondary"):
    st.session_state["crat_data"] = get_default_crat_dataframe()
    st.rerun()

st.sidebar.markdown(
    """
    **Clinical notes**

    - CRS is calculated per card.
    - CRA uses the number of tested cards and cumulative errors.
    - CMRS is estimated as the fitted plateau.
    - CCPS is calculated from the fitted curve.
    """
)


# ============================================================
# Main UI
# ============================================================

st.title("Children Reading Acuity Test CRAT Analyzer")
st.caption(
    "Interactive clinical tool for calculating CRA, CRS, CMRS, and CCPS "
    "from Children Reading Acuity Test data."
)

with st.expander("Mathematical definitions", expanded=False):
    st.markdown(
        r"""
        **Chinese Reading Speed CRS**

        $$
        CRS = \frac{60 \times (18 - \text{Reading Errors})}
        {\text{Reading Time in seconds}}
        $$

        **Chinese Reading Acuity CRA**

        $$
        CRA = 1.4 - (\text{Number of sentences read} \times 0.1)
        + (\text{Total cumulative errors} \times 0.0056)
        $$

        **Curve model**

        An exponential rise-to-plateau model is fitted:

        $$
        y = bottom + (plateau - bottom)
        \left(1 - e^{-rate \cdot \max(x - x_{shift}, 0)}\right)
        $$

        where the fitted plateau is interpreted as **CMRS**.
        """
    )


# ============================================================
# Patient Information
# ============================================================

st.header("1. Patient Information")

col_patient_1, col_patient_2, col_patient_3 = st.columns([2, 2, 1])

with col_patient_1:
    patient_id = st.text_input(
        "Patient ID / Name",
        value="",
        placeholder="Enter patient ID or name",
    )

with col_patient_2:
    examiner = st.text_input(
        "Examiner",
        value="",
        placeholder="Optional",
    )

with col_patient_3:
    test_date = st.date_input(
        "Test date",
        value=datetime.today(),
    )


# ============================================================
# Data Entry
# ============================================================

st.header("2. CRAT Data Entry")

st.markdown(
    """
    Enter reading time and errors for each card.  
    Uncheck **Tested** for skipped or untested cards.
    """
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
            help="CRAT print size in logMAR.",
            format="%.1f",
            disabled=True,
        ),
        "Time seconds": st.column_config.NumberColumn(
            "Time seconds",
            help="Reading time in seconds. Must be > 0 for tested cards.",
            min_value=0.0,
            step=0.1,
            format="%.2f",
        ),
        "Errors": st.column_config.NumberColumn(
            "Errors",
            help="Number of reading errors, from 0 to 18.",
            min_value=0,
            max_value=18,
            step=1,
            format="%d",
        ),
    },
    key="crat_editor",
)

st.session_state["crat_data"] = edited_df.copy()


# ============================================================
# Data Processing
# ============================================================

df = edited_df.copy()

df["Time seconds"] = pd.to_numeric(df["Time seconds"], errors="coerce")
df["Errors"] = pd.to_numeric(df["Errors"], errors="coerce").fillna(0).astype(int)
df["Errors"] = df["Errors"].clip(lower=0, upper=N_CHARACTERS_PER_CARD)

df["CRS chars/min"] = df.apply(
    lambda row: calculate_reading_speed(row["Time seconds"], row["Errors"])
    if row["Tested"]
    else np.nan,
    axis=1,
)

df_tested = df[df["Tested"]].copy()
df_valid = df_tested[
    df_tested["CRS chars/min"].notna()
    & np.isfinite(df_tested["CRS chars/min"])
    & (df_tested["Time seconds"] > 0)
].copy()

number_attempted = int(df_tested.shape[0])
cumulative_errors = int(df_tested["Errors"].sum()) if number_attempted > 0 else 0
cra = calculate_cra(number_attempted, cumulative_errors)

fit_result = fit_reading_curve(df_valid)
ccps, threshold_speed = calculate_ccps_from_fit(fit_result, threshold_fraction)
cmrs = fit_result.get("cmrs", np.nan)


# ============================================================
# Results Summary
# ============================================================

st.header("3. Clinical Results")

metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = st.columns(5)

with metric_col_1:
    st.metric("Cards attempted", f"{number_attempted}")

with metric_col_2:
    st.metric("Cumulative errors", f"{cumulative_errors}")

with metric_col_3:
    st.metric("CRA logMAR", f"{cra:.3f}" if np.isfinite(cra) else "NA")

with metric_col_4:
    st.metric(
        "CMRS chars/min",
        f"{cmrs:.1f}" if np.isfinite(cmrs) else "NA",
    )

with metric_col_5:
    st.metric(
        f"CCPS at {threshold_percent}%",
        f"{ccps:.3f} logMAR" if np.isfinite(ccps) else "NA",
    )

if fit_result["message"]:
    if fit_result["success"]:
        st.success(fit_result["message"])
    else:
        st.warning(fit_result["message"])

if len(df_tested) > 0 and len(df_valid) < len(df_tested):
    st.warning(
        "Some tested cards have missing or invalid reading times and were excluded "
        "from CRS calculation and curve fitting."
    )


# ============================================================
# Processed Data Table
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
# Visualization
# ============================================================

st.header("5. CRAT Reading-Speed Curve")

fig, ax = plt.subplots(figsize=(12, 7))

# Plot valid observed data.
if not df_valid.empty:
    sns.scatterplot(
        data=df_valid,
        x="Print Size logMAR",
        y="CRS chars/min",
        s=110,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
        label="Observed CRS",
        zorder=5,
    )

# Plot fitted curve.
x_fit = fit_result.get("x_fit")
y_fit = fit_result.get("y_fit")

if x_fit is not None and y_fit is not None:
    ax.plot(
        x_fit,
        y_fit,
        color="#d62728",
        linewidth=3,
        label=fit_result.get("method", "Fitted curve"),
        zorder=4,
    )

# Horizontal CMRS line.
if np.isfinite(cmrs):
    ax.axhline(
        cmrs,
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        label=f"CMRS = {cmrs:.1f} chars/min",
    )

# Horizontal threshold speed line.
if np.isfinite(threshold_speed):
    ax.axhline(
        threshold_speed,
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
        label=f"{threshold_percent}% CMRS = {threshold_speed:.1f}",
    )

# Vertical CCPS line.
if np.isfinite(ccps):
    ax.axvline(
        ccps,
        color="#9467bd",
        linestyle="--",
        linewidth=2,
        label=f"CCPS = {ccps:.3f} logMAR",
    )

# Vertical CRA line.
if np.isfinite(cra):
    ax.axvline(
        cra,
        color="#8c564b",
        linestyle=":",
        linewidth=2.5,
        label=f"CRA = {cra:.3f} logMAR",
    )

# Standard vision-science inverted x-axis.
ax.set_xlim(1.4, -0.4)
ax.set_xticks(np.round(np.arange(1.4, -0.41, -0.1), 1))

# Y-axis limits.
all_y_values = []

if not df_valid.empty:
    all_y_values.extend(df_valid["CRS chars/min"].dropna().tolist())

if y_fit is not None:
    all_y_values.extend(pd.Series(y_fit).dropna().tolist())

if np.isfinite(cmrs):
    all_y_values.append(cmrs)

if np.isfinite(threshold_speed):
    all_y_values.append(threshold_speed)

if all_y_values:
    y_max = max(all_y_values)
    ax.set_ylim(bottom=0, top=max(y_max * 1.15, 10))
else:
    ax.set_ylim(bottom=0, top=100)

ax.set_xlabel("Print Size logMAR")
ax.set_ylabel("Reading Speed characters/min")
ax.set_title("CRAT Reading Speed vs. Print Size")

# Clinical summary text box.
summary_text = (
    f"Patient: {patient_id if patient_id else 'Not specified'}\n"
    f"Date: {test_date}\n"
    f"CRA: {cra:.3f} logMAR\n"
    f"CMRS: {cmrs:.1f} chars/min" if np.isfinite(cmrs) else
    f"Patient: {patient_id if patient_id else 'Not specified'}\n"
    f"Date: {test_date}\n"
    f"CRA: {cra:.3f} logMAR\n"
    f"CMRS: NA"
)

summary_text += (
    f"\nCCPS {threshold_percent}%: {ccps:.3f} logMAR"
    if np.isfinite(ccps)
    else f"\nCCPS {threshold_percent}%: NA"
)

ax.text(
    0.02,
    0.98,
    summary_text,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round,pad=0.45",
        facecolor="white",
        edgecolor="gray",
        alpha=0.9,
    ),
)

ax.legend(loc="lower left", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

st.pyplot(fig)


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

st.download_button(
    label="Download results as CSV",
    data=csv,
    file_name=f"CRAT_results_{patient_id if patient_id else 'patient'}_{test_date}.csv",
    mime="text/csv",
)

st.caption(
    "This tool is intended to assist clinical analysis. "
    "Results should be interpreted by a qualified eye-care professional."
)
