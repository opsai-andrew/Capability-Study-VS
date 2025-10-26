# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------- Stats helpers ----------
def moving_range_sigma(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3: return np.nan
    mr = np.abs(np.diff(x))
    d2 = 1.128
    return np.mean(mr) / d2

def nonparametric_ppk(x, lsl, usl):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 5: return np.nan, np.nan, np.nan
    p_above = (np.sum(x > usl) + 0.5) / (n + 1.0)
    p_below = (np.sum(x < lsl) + 0.5) / (n + 1.0)
    z_upper = stats.norm.isf(p_above)
    z_lower = stats.norm.isf(p_below)
    return np.nanmin([z_upper, z_lower]) / 3.0, p_below, p_above

def capability_indices(x, lsl, usl, target=np.nan, use_mr=True):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
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

    return {
        "n": n, "mean": mu, "stdev_overall": s_overall,
        "min": np.min(x), "max": np.max(x),
        "Pp": Pp, "Ppk": Ppk,
        "ppm_below": ppm_lower, "ppm_above": ppm_upper,
        "ppm_total": (ppm_lower + ppm_upper) if np.isfinite(ppm_lower) and np.isfinite(ppm_upper) else np.nan,
        "Cp": Cp, "Cpk": Cpk, "sigma_within_mR": sigma_within,
        "Ppk_nonparametric": ppk_np, "p_below_emp": p_below, "p_above_emp": p_above,
        "normaltest_K2_stat": k2_stat, "normaltest_p": k2_p,
        "anderson_stat": ad_stat, "anderson_crit": ad_crit
    }

def iqr_filter(x, k=1.5):
    x = pd.Series(x, dtype=float)
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return x[(x >= lo) & (x <= hi)].values

def plot_hist_with_specs(x, lsl, usl, bins=30, title="Histogram with Spec Lines"):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    mu, s = np.mean(x), np.std(x, ddof=1)
    counts, bin_edges, _ = plt.hist(x, bins=bins, alpha=0.6, edgecolor='k')
    plt.axvline(lsl, linestyle='--', linewidth=2, label='LSL')
    plt.axvline(mu,  linestyle='-',  linewidth=2, label='Mean')
    plt.axvline(usl, linestyle='--', linewidth=2, label='USL')
    if s > 0:
        xs = np.linspace(bin_edges[0], bin_edges[-1], 400)
        pdf = stats.norm.pdf(xs, loc=mu, scale=s)
        area = np.sum(counts) * (bin_edges[1] - bin_edges[0])
        plt.plot(xs, pdf * area, linewidth=2, label='Normal Overlay')
    plt.title(title); plt.legend(); plt.xlabel('Value'); plt.ylabel('Frequency'); plt.show()

def plot_qq(x, title="Q-Q Plot (Normal)"):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    stats.probplot(x, dist="norm", plot=plt); plt.title(title); plt.show()

def plot_time_series(x, lsl, usl, title="Run Chart with Specs"):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    plt.plot(range(1, len(x)+1), x, marker='o', linewidth=1)
    plt.axhline(lsl, linestyle='--', linewidth=2, label='LSL')
    plt.axhline(usl, linestyle='--', linewidth=2, label='USL')
    plt.title(title); plt.xlabel('Sample Index'); plt.ylabel('Value'); plt.legend(); plt.show()

# ---------- Tk App ----------
class CapabilityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Capability Study App")
        self.root.geometry("1100x720")
        self.df = None

        style = ttk.Style()
        style.theme_use('clam')

        self.main = ttk.Frame(root, padding=20)
        self.main.pack(expand=True, fill=tk.BOTH)

        # Top: file info + button
        top = ttk.Frame(self.main)
        top.pack(fill=tk.X)
        self.file_label = ttk.Label(top, text="No file loaded")
        self.file_label.pack(side=tk.LEFT)
        ttk.Button(top, text="Open CSV/XLSX…", command=self.choose_file).pack(side=tk.RIGHT)

        # --- Form layout ---
        form = ttk.Frame(self.main)
        form.pack(fill=tk.X, pady=10)
        for i in range(5):
            form.grid_columnconfigure(i, weight=0)
        form.grid_columnconfigure(1, weight=1)  # make the column selector stretch

        ttk.Label(form, text="Column:").grid(row=0, column=0, sticky="w", padx=(0,6), pady=3)
        self.col_var = tk.StringVar()
        self.col_combo = ttk.Combobox(form, textvariable=self.col_var, width=50, state="disabled")
        self.col_combo.grid(row=0, column=1, columnspan=4, sticky="we", pady=3)

        ttk.Label(form, text="LSL:").grid(row=1, column=0, sticky="w", padx=(0,6), pady=3)
        self.lsl_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.lsl_var, width=12).grid(row=1, column=1, sticky="w", pady=3)

        ttk.Label(form, text="USL:").grid(row=1, column=2, sticky="w", padx=(12,6), pady=3)
        self.usl_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.usl_var, width=12).grid(row=1, column=3, sticky="w", pady=3)

        ttk.Label(form, text="Target (optional):").grid(row=2, column=0, sticky="w", padx=(0,6), pady=3)
        self.tgt_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.tgt_var, width=12).grid(row=2, column=1, sticky="w", pady=3)

        # --- Apply IQR + k entry on the same row ---
        self.use_outlier = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Apply IQR outlier filter", variable=self.use_outlier,
                        command=lambda: self.iqr_k_entry.configure(
                            state=("normal" if self.use_outlier.get() else "disabled"))
                        ).grid(row=2, column=2, sticky="w", padx=(12,8), pady=3)

        ttk.Label(form, text="k:").grid(row=2, column=3, sticky="e", pady=3)
        self.iqr_k_var = tk.StringVar(value="1.5")
        self.iqr_k_entry = ttk.Entry(form, textvariable=self.iqr_k_var, width=6)
        self.iqr_k_entry.grid(row=2, column=4, sticky="w", padx=(4,0), pady=3)
        self.iqr_k_entry.configure(state="disabled")  # disabled until checkbox ticked

        # --- mR option on a new line to avoid crowding ---
        self.use_mr = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, text="Use moving range for within sigma (Cp/Cpk)",
                        variable=self.use_mr).grid(row=3, column=0, columnspan=5,
                                                   sticky="w", pady=(6,0))

        # Actions
        actions = ttk.Frame(self.main)
        actions.pack(fill=tk.X, pady=10)
        ttk.Button(actions, text="Compute Capability", command=self.compute).pack(side=tk.LEFT)

        # Results
        self.output = tk.Text(self.main, height=18)
        self.output.pack(fill=tk.BOTH, expand=True)
        self.output.configure(state="disabled")

        # Open file chooser automatically once the window is visible
        self.root.after(200, self.choose_file)

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Select dataset (CSV or Excel)",
            filetypes=[("CSV files","*.csv"), ("Excel files","*.xlsx;*.xls"), ("All files","*.*")]
        )
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                self.df = pd.read_csv(path)
            else:
                self.df = pd.read_excel(path)
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load file:\n{e}")
            return

        self.file_label.config(text=f"Loaded: {os.path.basename(path)} "
                                    f"({self.df.shape[0]} rows, {self.df.shape[1]} cols)")
        # Populate column chooser with: "index — column_name"
        opts = [f"{i}: {name}" for i, name in enumerate(self.df.columns)]
        self.col_combo["values"] = opts
        self.col_combo.state(["!disabled"])
        if opts:
            self.col_combo.current(0)

    def _append(self, text):
        self.output.configure(state="normal")
        self.output.insert("end", text + "\n")
        self.output.configure(state="disabled")
        self.output.see("end")

    def compute(self):
        if self.df is None:
            messagebox.showwarning("No data", "Please open a dataset first.")
            return
        if not self.col_var.get():
            messagebox.showwarning("Column", "Please select a column.")
            return

        # Parse column index from "i: name"
        try:
            col_index = int(self.col_var.get().split(":")[0])
        except Exception:
            messagebox.showwarning("Column", "Invalid column selection.")
            return

        try:
            lsl = float(self.lsl_var.get())
            usl = float(self.usl_var.get())
        except Exception:
            messagebox.showwarning("Specs", "Enter numeric LSL and USL.")
            return
        if lsl >= usl:
            messagebox.showwarning("Specs", "LSL must be < USL.")
            return

        tgt = np.nan
        if self.tgt_var.get().strip():
            try:
                tgt = float(self.tgt_var.get())
            except Exception:
                tgt = np.nan

        series = pd.to_numeric(self.df.iloc[:, col_index], errors="coerce").values.astype(float)
        x = series[~np.isnan(series)]
        if self.use_outlier.get():
            try:
                 k_value = float(self.iqr_k_var.get())
            except Exception:
                 k_value = 1.5
            
                 x = iqr_filter(x, k=k_value)

        if len(x) < 5:
            messagebox.showwarning("Data", "Not enough data after filtering (need >= 5).")
            return

        res = capability_indices(x, lsl, usl, target=tgt, use_mr=self.use_mr.get())

        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")

        self._append(f"n={res['n']}, mean={res['mean']:.6g}, stdev_overall={res['stdev_overall']:.6g}, "
                     f"min={res['min']:.6g}, max={res['max']:.6g}")
        self._append(f"Overall (long-term):  Pp={res['Pp']:.4g},  Ppk={res['Ppk']:.4g}")
        self._append(f"Normal-theory PPM:    below LSL={res['ppm_below']:.3f}, above USL={res['ppm_above']:.3f}, "
                     f"total={res['ppm_total']:.3f}")
        self._append(f"Within (short-term via mR):  Cp={res['Cp']:.4g},  Cpk={res['Cpk']:.4g},  "
                     f"sigma_within(mR)={res['sigma_within_mR']:.6g}")
        self._append(f"Non-parametric Ppk (empirical tails): {res['Ppk_nonparametric']:.4g}  "
                     f"(p_below={res['p_below_emp']:.4f}, p_above={res['p_above_emp']:.4f})")
        self._append(f"Normality tests:")
        self._append(f"  D'Agostino K2: stat={res['normaltest_K2_stat']:.4g}, p={res['normaltest_p']:.4g}  "
                     f"(p<0.05 suggests non-normal)")
        if res['anderson_crit']:
            self._append(f"  Anderson-Darling: stat={res['anderson_stat']:.4g}; critical values @ sig levels:")
            for sl, cv in res['anderson_crit']:
                self._append(f"    {sl:.0f}% -> {cv:.4g}")

        # Plots
        plt.rcParams['figure.figsize'] = (8, 5)
        plt.rcParams['axes.grid'] = True
        plot_hist_with_specs(x, lsl, usl, bins=30, title=f"Histogram: {self.df.columns[col_index]}")
        plot_qq(x, title=f"Q-Q Plot: {self.df.columns[col_index]}")
        plot_time_series(x, lsl, usl, title=f"Run Chart: {self.df.columns[col_index]}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CapabilityApp(root)
    root.mainloop()
