# app.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="notebook")


# ============================================================
# Constants
# ============================================================

N_CHARACTERS_PER_CARD = 18
LOGMAR_LEVELS = np.round(np.arange(1.3, -0.31, -0.1), 1)
DEFAULT_THRESHOLD_PERCENT = 90


# ============================================================
# Model functions
# ============================================================

def exponential_plateau_model(logmar, cmrs, slope, transition):
    y = cmrs * (1 - np.exp(-slope * (logmar - transition)))
    return np.maximum(y, 0)


def logistic_plateau_model(logmar, cmrs, slope, midpoint):
    return cmrs / (1 + np.exp(-slope * (logmar - midpoint)))


# ============================================================
# Calculation functions
# ============================================================

def calculate_reading_speed(time_seconds, errors):
    if pd.isna(time_seconds) or pd.isna(errors):
        return np.nan

    if time_seconds <= 0:
        return np.nan

    errors = max(errors, 0)
    errors = min(errors, N_CHARACTERS_PER_CARD)

    correct_characters = N_CHARACTERS_PER_CARD - errors

    return 60 * correct_characters / time_seconds


def calculate_cra(number_sentences_read, total_errors):
    return 1.4 - (number_sentences_read * 0.1) + (total_errors * 0.0056)


def estimate_ccps_from_curve(x_grid, y_grid, cmrs, threshold_fraction):
    if x_grid is None or y_grid is None:
        return np.nan

    if not np.isfinite(cmrs) or cmrs <= 0:
        return np.nan

    threshold_speed = threshold_fraction * cmrs

    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    valid = np.isfinite(x_grid) & np.isfinite(y_grid)
    x_grid = x_grid[valid]
    y_grid = y_grid[valid]

    if len(x_grid) < 2:
        return np.nan

    if threshold_speed < np.nanmin(y_grid) or threshold_speed > np.nanmax(y_grid):
        return np.nan

    try:
        df_tmp = pd.DataFrame({
            "x": x_grid,
            "y": y_grid
        }).dropna()

        df_tmp = df_tmp.sort_values("y")
        df_tmp = df_tmp.drop_duplicates(subset="y")

        if len(df_tmp) < 2:
            return np.nan

        interpolator = interp1d(
            df_tmp["y"],
            df_tmp["x"],
            bounds_error=False,
            fill_value=np.nan
        )

        return float(interpolator(threshold_speed))

    except Exception:
        return np.nan


# ============================================================
# Curve fitting
# ============================================================

def fit_reading_curve(df_tested, threshold_percent=90):

    x = df_tested["logMAR"].values.astype(float)
    y = df_tested["ReadingSpeed"].values.astype(float)

    valid = np.isfinite(x) & np.isfinite(y) & (y >= 0)
    x = x[valid]
    y = y[valid]

    threshold_fraction = threshold_percent / 100

    results = {
        "success": False,
        "method": None,
        "message": "",
        "params": None,
        "CMRS": np.nan,
        "CCPS": np.nan,
        "threshold_speed": np.nan,
        "x_grid": None,
        "y_grid": None
    }

    if len(x) < 3:
        cmrs_empirical = np.nanmax(y) if len(y) > 0 else np.nan

        results.update({
            "success": False,
            "method": "Insufficient data",
            "message": "Insufficient valid data points for nonlinear curve fitting.",
            "CMRS": cmrs_empirical,
            "threshold_speed": threshold_fraction * cmrs_empirical
            if np.isfinite(cmrs_empirical) else np.nan
        })

        return results

    x_grid = np.linspace(-0.4, 1.4, 500)

    initial_cmrs = max(np.nanmax(y), 1)
    initial_slope = 4.0
    initial_transition = np.nanmedian(x)

    lower_bounds = [0, 0.01, -1.0]
    upper_bounds = [1000, 50.0, 2.0]

    try:
        popt, pcov = curve_fit(
            exponential_plateau_model,
            x,
            y,
            p0=[initial_cmrs, initial_slope, initial_transition],
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000
        )

        cmrs, slope, transition = popt
        y_grid = exponential_plateau_model(x_grid, *popt)

        ccps = estimate_ccps_from_curve(
            x_grid=x_grid,
            y_grid=y_grid,
            cmrs=cmrs,
            threshold_fraction=threshold_fraction
        )

        results.update({
            "success": True,
            "method": "Exponential plateau model",
            "message": "Nonlinear exponential plateau fit completed successfully.",
            "params": popt,
            "CMRS": cmrs,
            "CCPS": ccps,
            "threshold_speed": threshold_fraction * cmrs,
            "x_grid": x_grid,
            "y_grid": y_grid
        })

        return results

    except Exception as e_exp:
        exp_error = str(e_exp)

    try:
        popt, pcov = curve_fit(
            logistic_plateau_model,
            x,
            y,
            p0=[initial_cmrs, 8.0, np.nanmedian(x)],
            bounds=([0, 0.01, -1.0], [1000, 50.0, 2.0]),
            maxfev=20000
        )

        cmrs, slope, midpoint = popt
        y_grid = logistic_plateau_model(x_grid, *popt)

        ccps = estimate_ccps_from_curve(
            x_grid=x_grid,
            y_grid=y_grid,
            cmrs=cmrs,
            threshold_fraction=threshold_fraction
        )

        results.update({
            "success": True,
            "method": "Fallback logistic plateau model",
            "message": "Exponential model failed; logistic model was used instead.",
            "params": popt,
            "CMRS": cmrs,
            "CCPS": ccps,
            "threshold_speed": threshold_fraction * cmrs,
            "x_grid": x_grid,
            "y_grid": y_grid
        })

        return results

    except Exception as e_log:
        logistic_error = str(e_log)

    cmrs_empirical = np.nanmax(y)
    threshold_speed = threshold_fraction * cmrs_empirical

    results.update({
        "success": False,
        "method": "Empirical fallback",
        "message": (
            "Nonlinear model fitting failed. "
            f"Exponential error: {exp_error}. Logistic error: {logistic_error}."
        ),
        "CMRS": cmrs_empirical,
        "CCPS": np.nan,
        "threshold_speed": threshold_speed
    })

    return results


# ============================================================
# Main CRAT analysis function
# ============================================================

def analyze_crat_data(patient_id, raw_rows, threshold_percent=90):

    df = pd.DataFrame(raw_rows)

    df["TimeSeconds"] = pd.to_numeric(df["TimeSeconds"], errors="coerce")
    df["Errors"] = pd.to_numeric(df["Errors"], errors="coerce")
    df["Tested"] = df["Tested"].astype(bool)

    df_tested = df[
        (df["Tested"]) &
        (df["TimeSeconds"].notna()) &
        (df["TimeSeconds"] > 0)
    ].copy()

    if df_tested.empty:
        raise ValueError(
            "No valid tested cards were entered. "
            "Please enter at least one card with Tested checked and Time > 0."
        )

    df_tested["Errors"] = df_tested["Errors"].fillna(0)
    df_tested["Errors"] = df_tested["Errors"].clip(
        lower=0,
        upper=N_CHARACTERS_PER_CARD
    )

    df_tested["ReadingSpeed"] = df_tested.apply(
        lambda row: calculate_reading_speed(
            row["TimeSeconds"],
            row["Errors"]
        ),
        axis=1
    )

    number_sentences_read = len(df_tested)
    total_errors = df_tested["Errors"].sum()
    cra = calculate_cra(number_sentences_read, total_errors)

    fit_results = fit_reading_curve(
        df_tested,
        threshold_percent=threshold_percent
    )

    summary = {
        "PatientID": patient_id,
        "NumberSentencesRead": number_sentences_read,
        "TotalErrors": total_errors,
        "CRA": cra,
        "CMRS": fit_results["CMRS"],
        "CCPS": fit_results["CCPS"],
        "ThresholdPercent": threshold_percent,
        "ThresholdSpeed": fit_results["threshold_speed"],
        "FitMethod": fit_results["method"],
        "FitSuccess": fit_results["success"],
        "FitMessage": fit_results["message"]
    }

    return df, df_tested, summary, fit_results


# ============================================================
# Plotting function
# ============================================================

def plot_crat_results(df_tested, summary, fit_results):

    patient_id = summary["PatientID"]
    cra = summary["CRA"]
    cmrs = summary["CMRS"]
    ccps = summary["CCPS"]
    threshold_percent = summary["ThresholdPercent"]
    threshold_speed = summary["ThresholdSpeed"]

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.scatterplot(
        data=df_tested,
        x="logMAR",
        y="ReadingSpeed",
        s=90,
        color="#1f77b4",
        edgecolor="black",
        ax=ax,
        label="Measured reading speed"
    )

    df_plot = df_tested.sort_values("logMAR", ascending=True)

    ax.plot(
        df_plot["logMAR"],
        df_plot["ReadingSpeed"],
        color="#1f77b4",
        alpha=0.35,
        linewidth=1.5
    )

    if fit_results["x_grid"] is not None and fit_results["y_grid"] is not None:
        ax.plot(
            fit_results["x_grid"],
            fit_results["y_grid"],
            color="#d62728",
            linewidth=2.8,
            label=f"Fitted curve: {fit_results['method']}"
        )

    if np.isfinite(cmrs):
        ax.axhline(
            cmrs,
            color="#2ca02c",
            linestyle="--",
            linewidth=2,
            label=f"CMRS = {cmrs:.1f} char/min"
        )

    if np.isfinite(threshold_speed):
        ax.axhline(
            threshold_speed,
            color="#ff7f0e",
            linestyle="--",
            linewidth=2,
            label=f"{threshold_percent:.0f}% CMRS = {threshold_speed:.1f} char/min"
        )

    if np.isfinite(ccps):
        ax.axvline(
            ccps,
            color="#9467bd",
            linestyle="--",
            linewidth=2,
            label=f"CCPS = {ccps:.2f} logMAR"
        )

    if np.isfinite(cra):
        ax.axvline(
            cra,
            color="#8c564b",
            linestyle=":",
            linewidth=2.5,
            label=f"CRA = {cra:.2f} logMAR"
        )

    ax.set_title(
        "Children Reading Acuity Test (CRAT) Analysis",
        fontsize=17,
        weight="bold"
    )

    ax.set_xlabel("Print Size / Reading Acuity (logMAR)", fontsize=13)
    ax.set_ylabel("Chinese Reading Speed (characters/min)", fontsize=13)

    ax.set_xlim(-0.4, 1.4)

    y_max_candidates = [
        df_tested["ReadingSpeed"].max(),
        cmrs if np.isfinite(cmrs) else np.nan
    ]

    y_max = np.nanmax(y_max_candidates)

    if np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(0, y_max * 1.25)
    else:
        ax.set_ylim(0, 300)

    ccps_text = f"{ccps:.2f}" if np.isfinite(ccps) else "Not estimable"
    cmrs_text = f"{cmrs:.1f}" if np.isfinite(cmrs) else "Not estimable"

    summary_text = (
        f"Patient ID: {patient_id}\n"
        f"CRA: {cra:.2f} logMAR\n"
        f"CMRS: {cmrs_text} char/min\n"
        f"CCPS: {ccps_text} logMAR\n"
        f"Threshold: {threshold_percent:.0f}% of CMRS\n"
        f"Sentences read: {summary['NumberSentencesRead']}\n"
        f"Total errors: {summary['TotalErrors']:.0f}"
    )

    ax.text(
        0.03,
        0.97,
        summary_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.9,
            edgecolor="gray"
        )
    )

    ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


# ============================================================
# Streamlit web interface
# ============================================================

st.set_page_config(
    page_title="CRAT Analysis Tool",
    layout="wide"
)

st.title("Children Reading Acuity Test (CRAT) Analysis Tool")

st.markdown("""
Enter reading time and number of errors for each tested card.

- **Time**: reading time in seconds  
- **Errors**: skipped, mispronounced, or substituted characters  
- Immediate self-corrections should not be counted as errors  
- Reading speed is automatically corrected for errors  
""")

patient_id = st.text_input("Patient ID", value="")

threshold_percent = st.slider(
    "CCPS threshold (% CMRS)",
    min_value=80,
    max_value=95,
    value=DEFAULT_THRESHOLD_PERCENT,
    step=1
)

st.subheader("Card Data Entry")

default_df = pd.DataFrame({
    "logMAR": LOGMAR_LEVELS,
    "Tested": True,
    "TimeSeconds": 0.0,
    "Errors": 0
})

edited_df = st.data_editor(
    default_df,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "logMAR": st.column_config.NumberColumn(
            "logMAR",
            format="%.1f",
            disabled=True
        ),
        "Tested": st.column_config.CheckboxColumn("Tested"),
        "TimeSeconds": st.column_config.NumberColumn(
            "Time seconds",
            min_value=0.0,
            step=0.1,
            format="%.2f"
        ),
        "Errors": st.column_config.NumberColumn(
            "Errors",
            min_value=0,
            max_value=18,
            step=1
        ),
    }
)

col1, col2 = st.columns([1, 4])

with col1:
    analyze_button = st.button("Analyze CRAT", type="primary")

with col2:
    st.caption("Rows with TimeSeconds ≤ 0 are ignored.")

if analyze_button:

    try:
        if patient_id.strip() == "":
            patient_id = "Unnamed patient"

        raw_rows = edited_df.to_dict(orient="records")

        df_all, df_tested, summary, fit_results = analyze_crat_data(
            patient_id=patient_id,
            raw_rows=raw_rows,
            threshold_percent=threshold_percent
        )

        st.success("Analysis completed.")

        st.subheader("CRAT Results Summary")

        summary_display = pd.DataFrame([{
            "Patient ID": summary["PatientID"],
            "Sentences read": summary["NumberSentencesRead"],
            "Total errors": summary["TotalErrors"],
            "CRA logMAR": summary["CRA"],
            "CMRS char/min": summary["CMRS"],
            "CCPS logMAR": summary["CCPS"],
            "Threshold %": summary["ThresholdPercent"],
            "Threshold speed char/min": summary["ThresholdSpeed"],
            "Fit method": summary["FitMethod"],
            "Fit success": summary["FitSuccess"]
        }])

        st.dataframe(
            summary_display,
            use_container_width=True
        )

        st.subheader("Tested Card-Level Data")

        st.dataframe(
            df_tested[
                ["logMAR", "TimeSeconds", "Errors", "ReadingSpeed"]
            ],
            use_container_width=True
        )

        if not fit_results["success"]:
            st.warning(fit_results["message"])
        else:
            st.info(fit_results["message"])

        st.subheader("CRAT Plot")

        fig = plot_crat_results(df_tested, summary, fit_results)
        st.pyplot(fig)

    except Exception as e:
        st.error(str(e))
