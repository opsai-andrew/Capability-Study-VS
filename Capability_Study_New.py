import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st

st.set_page_config(page_title="Capability Study", layout="wide")
st.title("Capability Study Analyzer")


# ---------- helpers ----------
def moving_range_sigma(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3: return np.nan
    mr = np.abs(np.diff(x))
    d2 = 1.128
    return np.mean(mr) / d2

def iqr_filter(x, k=1.5):
    s = pd.Series(x, dtype=float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return s[(s >= lo) & (s <= hi)].values

def nonparametric_ppk(x, lsl, usl):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 5: return np.nan, np.nan, np.nan
    p_above = (np.sum(x > usl) + 0.5) / (n + 1.0)
    p_below = (np.sum(x < lsl) + 0.5) / (n + 1.0)
    z_upper = stats.norm.isf(p_above)
    z_lower = stats.norm.isf(p_below)
    return np.nanmin([z_upper, z_lower]) / 3.0, p_below, p_above

def capability_indices(x, lsl, usl, target=np.nan, use_mr=True):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 5: raise ValueError("Not enough data points (need >= 5).")

    mu = np.mean(x)
    s_overall = np.std(x, ddof=1)
    Pp  = (usl - lsl) / (6.0 * s_overall) if s_overall > 0 else np.nan
    Ppu = (usl - mu) / (3.0 * s_overall) if s_overall > 0 else np.nan
    Ppl = (mu - lsl) / (3.0 * s_overall) if s_overall > 0 else np.nan
    Ppk = np.nanmin([Ppu, Ppl])

    sigma_within = moving_range_sigma(x) if use_mr else np.nan
    if np.isfinite(sigma_within) and sigma_within > 0:
        Cp  = (usl - lsl) / (6.0 * sigma_within)
        Cpu = (usl - mu) / (3.0 * sigma_within)
        Cpl = (mu - lsl) / (3.0 * sigma_within)
        Cpk = np.nanmin([Cpu, Cpl])
    else:
        Cp = Cpk = np.nan

    if s_overall > 0:
        z_upper = (usl - mu) / s_overall
        z_lower = (mu - lsl) / s_overall
        ppm_upper = (1.0 - stats.norm.cdf(z_upper)) * 1e6
        ppm_lower = (1.0 - stats.norm.cdf(z_lower)) * 1e6
    else:
        ppm_upper = ppm_lower = np.nan

    ppk_np, p_below, p_above = nonparametric_ppk(x, lsl, usl)

    try:
        k2_stat, k2_p = stats.normaltest(x)
    except Exception:
        k2_stat, k2_p = (np.nan, np.nan)

    try:
        ad_res = stats.anderson(x, dist='norm')
        ad_stat = ad_res.statistic
        ad_crit = list(zip(ad_res.significance_level, ad_res.critical_values))
    except Exception:
        ad_stat, ad_crit = (np.nan, [])

    return dict(
        n=n, mean=mu, stdev_overall=s_overall, min=np.min(x), max=np.max(x),
        Pp=Pp, Ppk=Ppk, Cp=Cp, Cpk=Cpk, sigma_within_mR=sigma_within,
        ppm_below=ppm_lower, ppm_above=ppm_upper,
        ppm_total=(ppm_lower + ppm_upper) if np.isfinite(ppm_lower) and np.isfinite(ppm_upper) else np.nan,
        Ppk_nonparametric=ppk_np, p_below_emp=p_below, p_above_emp=p_above,
        normaltest_K2_stat=k2_stat, normaltest_p=k2_p,
        anderson_stat=ad_stat, anderson_crit=ad_crit
    )

def plot_hist_with_specs(x, lsl, usl, bins=30, title="Histogram"):
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    mu, s = np.mean(x), np.std(x, ddof=1)
    counts, bin_edges, _ = ax.hist(x, bins=bins, alpha=0.6, edgecolor='k')
    ax.axvline(lsl, linestyle='--', linewidth=2, label='LSL')
    ax.axvline(mu,  linestyle='-',  linewidth=2, label='Mean')
    ax.axvline(usl, linestyle='--', linewidth=2, label='USL')
    if s > 0:
        xs = np.linspace(bin_edges[0], bin_edges[-1], 400)
        pdf = stats.norm.pdf(xs, loc=mu, scale=s)
        area = np.sum(counts) * (bin_edges[1] - bin_edges[0])
        ax.plot(xs, pdf * area, linewidth=2, label='Normal Overlay')
    ax.set_title(title); ax.set_xlabel("Value"); ax.set_ylabel("Frequency"); ax.legend()
    return fig

def plot_qq(x, title="Q-Q Plot (Normal)"):
    fig = plt.figure(figsize=(8,5))
    stats.probplot(np.asarray(x, dtype=float), dist="norm", plot=plt)
    plt.title(title)
    return fig

def plot_time_series(x, lsl, usl, title="Run Chart with Specs"):
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    ax.plot(range(1, len(x)+1), x, marker='o', linewidth=1)
    ax.axhline(lsl, linestyle='--', linewidth=2, label='LSL')
    ax.axhline(usl, linestyle='--', linewidth=2, label='USL')
    ax.set_title(title); ax.set_xlabel('Sample Index'); ax.set_ylabel('Value'); ax.legend()
    return fig

# ---------- UI ----------
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a dataset to begin.")
    st.stop()

# Read file
if uploaded.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

st.success(f"Loaded '{uploaded.name}' • rows={len(df)}, cols={len(df.columns)}")

col = st.selectbox("Select column", list(df.columns))
c1, c2, c3 = st.columns(3)
lsl = c1.number_input("LSL", value=0.0)
usl = c2.number_input("USL", value=1.0)
target = c3.number_input("Target (optional)", value=0.0, step=0.1, format="%.6f")

c4, c5, c6 = st.columns([1,1,1])
use_outliers = c4.checkbox("Apply IQR outlier filter", value=False)
k_val = c5.number_input("IQR k", value=1.5, step=0.1, format="%.2f")
use_mr = c6.checkbox("Use moving range for within sigma (Cp/Cpk)", value=True)

bins = st.slider("Histogram bins", min_value=10, max_value=80, step=2, value=30)
show_run = st.checkbox("Show time series chart", value=True)

# Compute
x_raw = pd.to_numeric(df[col], errors="coerce").values.astype(float)
x = iqr_filter(x_raw, k=k_val) if use_outliers else x_raw[~np.isnan(x_raw)]
if len(x) < 5:
    st.error("Not enough data after filtering (need >= 5).")
    st.stop()

res = capability_indices(x, lsl, usl, target=target, use_mr=use_mr)

st.subheader("Summary")
st.write({
    "n": res["n"],
    "mean": float(res["mean"]),
    "stdev_overall": float(res["stdev_overall"]),
    "min": float(res["min"]),
    "max": float(res["max"]),
    "Pp": float(res["Pp"]), "Ppk": float(res["Ppk"]),
    "Cp": float(res["Cp"]), "Cpk": float(res["Cpk"]),
    "sigma_within_mR": float(res["sigma_within_mR"]),
    "ppm_below": float(res["ppm_below"]),
    "ppm_above": float(res["ppm_above"]),
    "ppm_total": float(res["ppm_total"]),
    "Ppk_nonparametric": float(res["Ppk_nonparametric"]),
    "normaltest_K2_stat": float(res["normaltest_K2_stat"]),
    "normaltest_p": float(res["normaltest_p"]),
    "anderson_stat": float(res["anderson_stat"]),
})

st.subheader("Charts")
st.pyplot(plot_hist_with_specs(x, lsl, usl, bins=bins, title=f"Histogram: {col}"))
st.pyplot(plot_qq(x, title=f"Q-Q Plot: {col}"))
if show_run:
    st.pyplot(plot_time_series(x, lsl, usl, title=f"Run Chart: {col}"))
