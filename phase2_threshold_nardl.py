"""
=============================================================================
 Phase 2: Threshold NARDL Pipeline (Option D)
 Tests H6, H7 on the POST-2013 sample (T = 159) with regime_post2013.

 Why post-2013 only?
   The NBS Product Inventory series has a structural break at 2012-01 (level
   shifts from ~30 to ~76 RMB bn) due to a methodological revision. Using the
   post-2013 sample ensures the regime variable reflects economic state, not
   measurement scope.

 Strategy:
   1. Hansen (2000) sample-splitting on regime_post2013 to find threshold
   2. Estimate threshold-NARDL with regime-specific long-run coefficients
   3. Test cross-regime amplification (H6) and regime-dependent mediation (H7)

 Outputs:
   tables/threshold_estimation.txt
   tables/tnardl_results.txt
   tables/regime_decomposition.txt
   figures/threshold_ssr.pdf
   figures/regime_irf_oil.pdf
   figures/regime_irf_sr.pdf
=============================================================================
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.api import VAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase2")

DATA_FILE = Path("data/rubber_petrochemical_monthly_model_dataset_v3.xlsx")
TABLES    = Path("tables");  TABLES.mkdir(exist_ok=True)
FIGURES   = Path("figures"); FIGURES.mkdir(exist_ok=True)

RNG = np.random.default_rng(20260101)


def wald_scalar(w):
    s = np.asarray(w.statistic).ravel()
    p = np.asarray(w.pvalue).ravel()
    return float(s[0]), float(p[0])


# =========================================================================
# 1. Load + restrict to post-2013
# =========================================================================

df_all = pd.read_excel(DATA_FILE, sheet_name="Monthly_Model_Data")
if pd.api.types.is_numeric_dtype(df_all["date"]):
    df_all["date"] = pd.to_datetime(df_all["date"], unit="D", origin="1899-12-30")

df = df_all[(df_all["core_model_flag"] == 1)
            & (df_all["date"] >= "2013-01-01")].copy()
df = df.sort_values("date").reset_index(drop=True)
log.info(f"Post-2013 sample: T = {len(df)}, "
         f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")

# Map columns
df["lnNR"]   = df["ln_nr_rss_rmbton"]
df["lnOIL"]  = df["ln_oil_opec"]
df["lnBD"]   = df["ln_butadiene"]
df["lnSR"]   = df["ln_sbr_monthly"]
df["dlnNR"]  = df["dln_nr_rss"]
df["dlnOIL"] = df["dln_oil_opec"]
df["dlnBD"]  = df["dln_butadiene"]
df["dlnSR"]  = df["dln_sbr_monthly"]
df["lnOIL_pos"] = df["oil_opec_pos_csum"]
df["lnOIL_neg"] = df["oil_opec_neg_csum"]
df["lnSR_pos"]  = df["sbr_pos_csum"]
df["lnSR_neg"]  = df["sbr_neg_csum"]
df["dlnOIL_pos"] = df["oil_opec_pos_change"]
df["dlnOIL_neg"] = df["oil_opec_neg_change"]
df["dlnSR_pos"]  = df["sbr_pos_change"]
df["dlnSR_neg"]  = df["sbr_neg_change"]
df["fx_cny"] = np.log(df["fx_china_lcu_per_usd"])
df["fx_thb"] = np.log(df["fx_thailand_lcu_per_usd"])

# Regime variable (already pre-computed in v3)
# Use regime_post2013 (12m deviation from rolling mean)
log.info(f"regime_post2013 coverage: {df['regime_post2013'].notna().sum()}/{len(df)}")

CONTROLS = ["ln_china_auto", "fx_cny", "fx_thb"]   # nbs_break_dummy is 1 throughout post-2013, drop


# =========================================================================
# 2. Hansen (2000) threshold estimation
# =========================================================================

def build_tnardl_design(d: pd.DataFrame, tau: float, p: int = 2, q: int = 2):
    """Build threshold-NARDL design matrix given threshold tau on regime_post2013."""
    e = d.copy()
    e["IL"] = (e["regime_post2013"] <= tau).astype(int)
    e["IH"] = 1 - e["IL"]

    # Long-run regressors (lagged levels)
    base_lr = ["lnNR", "lnOIL_pos", "lnOIL_neg", "lnSR_pos", "lnSR_neg", "lnBD"]
    lr_cols_L = []
    lr_cols_H = []
    for v in base_lr:
        e[f"{v}_lag1"] = e[v].shift(1)
        e[f"{v}_lag1_L"] = e[f"{v}_lag1"] * e["IL"]
        e[f"{v}_lag1_H"] = e[f"{v}_lag1"] * e["IH"]
        lr_cols_L.append(f"{v}_lag1_L")
        lr_cols_H.append(f"{v}_lag1_H")

    # Short-run lags (same in both regimes for parsimony)
    sr_cols = []
    for i in range(1, p + 1):
        c = f"dlnNR_lag{i}"; e[c] = e["dlnNR"].shift(i); sr_cols.append(c)
    for i in range(0, q):
        for v in ["dlnOIL_pos", "dlnOIL_neg", "dlnSR_pos", "dlnSR_neg"]:
            c = f"{v}_lag{i}"; e[c] = e[v].shift(i); sr_cols.append(c)

    cols = ["IL", "IH"] + lr_cols_L + lr_cols_H + sr_cols + CONTROLS
    e = e.dropna(subset=["dlnNR"] + cols).reset_index(drop=True)
    return e, cols, lr_cols_L, lr_cols_H


def estimate_threshold(d: pd.DataFrame, n_grid: int = 80) -> dict:
    """Hansen (2000) sample-splitting: minimise SSR over a grid."""
    z = d["regime_post2013"].dropna()
    q_lo, q_hi = float(z.quantile(0.15)), float(z.quantile(0.85))
    cands = np.linspace(q_lo, q_hi, n_grid)
    ssrs = np.full(n_grid, np.inf)

    for k, tau in enumerate(cands):
        try:
            e, cols, *_ = build_tnardl_design(d, tau)
            if len(e) < 30:
                continue
            fit = OLS(e["dlnNR"], e[cols]).fit()  # no intercept (IL+IH)
            ssrs[k] = float(np.sum(fit.resid ** 2))
        except Exception:
            continue

    tau_hat = float(cands[np.argmin(ssrs)])
    log.info(f"Estimated threshold tau_hat = {tau_hat:.4f}  "
             f"(grid range [{q_lo:.4f}, {q_hi:.4f}])")
    return {"tau_hat": tau_hat, "cands": cands, "ssrs": ssrs}


def plot_ssr_profile(th: dict, save: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(th["cands"], th["ssrs"], color="black", linewidth=1.4)
    ax.axvline(th["tau_hat"], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Candidate threshold τ on regime_post2013")
    ax.set_ylabel("Sum of squared residuals")
    ax.set_title("Hansen (2000) threshold estimation")
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    log.info(f"Saved {save}")


# =========================================================================
# 3. Estimate threshold-NARDL
# =========================================================================

def fit_tnardl(d: pd.DataFrame, tau_hat: float) -> dict:
    e, cols, lrL, lrH = build_tnardl_design(d, tau_hat, p=2, q=2)
    fit = OLS(e["dlnNR"], e[cols]).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(np.ceil(len(e) ** 0.25))})

    # Recover regime-specific long-run coefficients
    rho_L = fit.params["lnNR_lag1_L"]
    rho_H = fit.params["lnNR_lag1_H"]
    LR_low = {
        "rho":       rho_L,
        "L_OIL_pos": -fit.params["lnOIL_pos_lag1_L"] / rho_L,
        "L_OIL_neg": -fit.params["lnOIL_neg_lag1_L"] / rho_L,
        "L_SR_pos":  -fit.params["lnSR_pos_lag1_L"]  / rho_L,
        "L_SR_neg":  -fit.params["lnSR_neg_lag1_L"]  / rho_L,
    }
    LR_high = {
        "rho":       rho_H,
        "L_OIL_pos": -fit.params["lnOIL_pos_lag1_H"] / rho_H,
        "L_OIL_neg": -fit.params["lnOIL_neg_lag1_H"] / rho_H,
        "L_SR_pos":  -fit.params["lnSR_pos_lag1_H"]  / rho_H,
        "L_SR_neg":  -fit.params["lnSR_neg_lag1_H"]  / rho_H,
    }

    # Asymmetry within regime
    F_low_oil, p_low_oil = wald_scalar(fit.wald_test(
        "lnOIL_pos_lag1_L = lnOIL_neg_lag1_L", use_f=True))
    F_low_sr, p_low_sr   = wald_scalar(fit.wald_test(
        "lnSR_pos_lag1_L = lnSR_neg_lag1_L", use_f=True))
    F_high_oil, p_high_oil = wald_scalar(fit.wald_test(
        "lnOIL_pos_lag1_H = lnOIL_neg_lag1_H", use_f=True))
    F_high_sr, p_high_sr   = wald_scalar(fit.wald_test(
        "lnSR_pos_lag1_H = lnSR_neg_lag1_H", use_f=True))

    # H6: cross-regime amplification (point estimate;
    # for inference, use bootstrap below)
    amp_oil_pt = abs(LR_low["L_OIL_pos"] - LR_low["L_OIL_neg"]) \
                 - abs(LR_high["L_OIL_pos"] - LR_high["L_OIL_neg"])
    amp_sr_pt  = abs(LR_low["L_SR_pos"] - LR_low["L_SR_neg"]) \
                 - abs(LR_high["L_SR_pos"] - LR_high["L_SR_neg"])

    return {"fit": fit, "LR_low": LR_low, "LR_high": LR_high,
            "test_low_oil": (F_low_oil, p_low_oil),
            "test_low_sr":  (F_low_sr, p_low_sr),
            "test_high_oil": (F_high_oil, p_high_oil),
            "test_high_sr":  (F_high_sr, p_high_sr),
            "amp_oil": amp_oil_pt, "amp_sr": amp_sr_pt,
            "data": e, "tau": tau_hat}


def write_tnardl_table(res: dict, save: Path):
    LR_low, LR_high = res["LR_low"], res["LR_high"]
    with open(save, "w") as f:
        f.write("Threshold NARDL Results (post-2013 sample)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Threshold tau_hat = {res['tau']:.4f}\n")
        f.write(f"Sample: T = {len(res['data'])}\n")
        f.write(f"R^2   = {res['fit'].rsquared:.3f}\n\n")

        f.write(f"{'':<14}{'Low-inv':>14}{'High-inv':>14}\n")
        f.write(f"{'-'*42}\n")
        f.write(f"{'rho':<14}{LR_low['rho']:>14.3f}{LR_high['rho']:>14.3f}\n")
        f.write(f"{'L_OIL+':<14}{LR_low['L_OIL_pos']:>14.3f}{LR_high['L_OIL_pos']:>14.3f}\n")
        f.write(f"{'L_OIL-':<14}{LR_low['L_OIL_neg']:>14.3f}{LR_high['L_OIL_neg']:>14.3f}\n")
        f.write(f"{'L_SR+':<14}{LR_low['L_SR_pos']:>14.3f}{LR_high['L_SR_pos']:>14.3f}\n")
        f.write(f"{'L_SR-':<14}{LR_low['L_SR_neg']:>14.3f}{LR_high['L_SR_neg']:>14.3f}\n")
        f.write("\n")
        f.write("Within-regime asymmetry tests:\n")
        f.write(f"  Low,  L_OIL+ = L_OIL-:  F = {res['test_low_oil'][0]:.2f}, "
                f"p = {res['test_low_oil'][1]:.4f}\n")
        f.write(f"  Low,  L_SR+  = L_SR-:   F = {res['test_low_sr'][0]:.2f}, "
                f"p = {res['test_low_sr'][1]:.4f}\n")
        f.write(f"  High, L_OIL+ = L_OIL-:  F = {res['test_high_oil'][0]:.2f}, "
                f"p = {res['test_high_oil'][1]:.4f}\n")
        f.write(f"  High, L_SR+  = L_SR-:   F = {res['test_high_sr'][0]:.2f}, "
                f"p = {res['test_high_sr'][1]:.4f}\n\n")

        f.write("Cross-regime amplification (H6):\n")
        f.write(f"  |L+ - L-|^Low - |L+ - L-|^High  (oil) = {res['amp_oil']:+.3f}\n")
        f.write(f"  |L+ - L-|^Low - |L+ - L-|^High  (SR)  = {res['amp_sr']:+.3f}\n")
        f.write("  H6 supported if both > 0.\n")
    log.info(f"Saved {save}")


# =========================================================================
# 4. Bootstrap confidence intervals for amplification
# =========================================================================

def bootstrap_amplification(d: pd.DataFrame, tau_hat: float,
                            B: int = 1000) -> dict:
    """Block bootstrap CIs for cross-regime amplification."""
    n = len(d)
    block = max(4, int(np.ceil(n ** (1/3))))
    n_blocks = int(np.ceil(n / block))

    amp_oil_b = np.full(B, np.nan)
    amp_sr_b  = np.full(B, np.nan)

    for b in range(B):
        # Stationary block bootstrap
        starts = RNG.integers(0, n - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        idx = np.clip(idx, 0, n - 1)
        d_b = d.iloc[idx].reset_index(drop=True)
        try:
            res = fit_tnardl(d_b, tau_hat)
            amp_oil_b[b] = res["amp_oil"]
            amp_sr_b[b]  = res["amp_sr"]
        except Exception:
            continue

    amp_oil_b = amp_oil_b[~np.isnan(amp_oil_b)]
    amp_sr_b  = amp_sr_b[~np.isnan(amp_sr_b)]
    return {
        "ci_oil": np.quantile(amp_oil_b, [0.025, 0.975]) if len(amp_oil_b) > 50 else (np.nan, np.nan),
        "ci_sr":  np.quantile(amp_sr_b, [0.025, 0.975]) if len(amp_sr_b) > 50 else (np.nan, np.nan),
        "B_eff_oil": len(amp_oil_b),
        "B_eff_sr":  len(amp_sr_b),
        "p_oil_one_sided": float(np.mean(amp_oil_b <= 0)) if len(amp_oil_b) > 50 else np.nan,
        "p_sr_one_sided":  float(np.mean(amp_sr_b <= 0)) if len(amp_sr_b) > 50 else np.nan,
    }


# =========================================================================
# 5. Regime-specific channel decomposition (H7)
# =========================================================================

def regime_decomposition(d: pd.DataFrame, tau_hat: float,
                         B: int = 2000) -> dict:
    """Estimate channel decomposition separately within each regime."""
    Z = ["fx_cny", "fx_thb", "ln_china_auto"]
    e = d.copy()
    e["IL"] = (e["regime_post2013"] <= tau_hat).astype(int)
    e["IH"] = 1 - e["IL"]
    e = e.dropna(subset=["lnNR", "lnSR", "lnOIL"] + Z)

    def decomp(sub):
        s1 = OLS(sub["lnSR"], add_constant(sub[["lnOIL"] + Z], has_constant="add")).fit()
        s2 = OLS(sub["lnNR"], add_constant(sub[["lnOIL", "lnSR"] + Z], has_constant="add")).fit()
        a1 = s1.params["lnOIL"]; k1 = s2.params["lnOIL"]; k2 = s2.params["lnSR"]
        ind = a1 * k2; tot = k1 + ind
        return {"direct": k1, "indirect": ind, "total": tot,
                "medshare": ind / tot if tot != 0 else np.nan,
                "alpha1": a1, "kappa1": k1, "kappa2": k2}

    sub_low  = e[e["IL"] == 1]
    sub_high = e[e["IH"] == 1]
    log.info(f"Low-inv N = {len(sub_low)}, High-inv N = {len(sub_high)}")

    if len(sub_low) < 20 or len(sub_high) < 20:
        log.warning("Insufficient observations in one regime; skipping decomposition")
        return {}

    dec_low  = decomp(sub_low)
    dec_high = decomp(sub_high)

    # Bootstrap difference in mediation share (H7)
    diffs = np.full(B, np.nan)
    for b in range(B):
        try:
            db_low  = sub_low.sample(n=len(sub_low),  replace=True, random_state=RNG.integers(1e9))
            db_high = sub_high.sample(n=len(sub_high), replace=True, random_state=RNG.integers(1e9))
            d_low  = decomp(db_low)
            d_high = decomp(db_high)
            diffs[b] = d_low["medshare"] - d_high["medshare"]
        except Exception:
            continue
    diffs = diffs[~np.isnan(diffs)]
    ci = np.quantile(diffs, [0.025, 0.975]) if len(diffs) > 50 else (np.nan, np.nan)
    p_one = float(np.mean(diffs <= 0)) if len(diffs) > 50 else np.nan

    return {"low": dec_low, "high": dec_high,
            "diff": dec_low["medshare"] - dec_high["medshare"],
            "ci": ci, "p_one_sided": p_one, "B_eff": len(diffs)}


def write_decomp_table(res: dict, save: Path):
    if not res:
        return
    with open(save, "w") as f:
        f.write("Regime-Specific Channel Decomposition (H7)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'':<18}{'Low-inv':>14}{'High-inv':>14}\n")
        f.write(f"{'-'*46}\n")
        for k in ["alpha1", "kappa1", "kappa2",
                  "direct", "indirect", "total", "medshare"]:
            f.write(f"{k:<18}{res['low'][k]:>14.4f}{res['high'][k]:>14.4f}\n")
        f.write("\n")
        f.write(f"H7 (mediation_low > mediation_high):\n")
        f.write(f"  Diff = {res['diff']:+.4f}\n")
        if not np.isnan(res['ci'][0]):
            f.write(f"  95% CI = [{res['ci'][0]:+.4f}, {res['ci'][1]:+.4f}]  "
                    f"(B = {res['B_eff']})\n")
            f.write(f"  One-sided p-value = {res['p_one_sided']:.4f}\n")
            f.write(f"  H7 supported: {res['p_one_sided'] < 0.10}\n")
    log.info(f"Saved {save}")


# =========================================================================
# 6. Regime-specific local projections (H6 dynamic)
# =========================================================================

def regime_lp(d: pd.DataFrame, tau_hat: float) -> pd.DataFrame:
    """Run asymmetric LP separately within each regime."""
    var_data = d[["dlnOIL", "dlnBD", "dlnSR", "dlnNR"]].dropna()
    var_fit = VAR(var_data).fit(maxlags=6, ic="aic")
    log.info(f"Regime LP: VAR(p={var_fit.k_ar}) on T={len(var_data)}")

    resid = np.asarray(var_fit.resid)
    n_resid = resid.shape[0]
    df_lp = d.iloc[-n_resid:].reset_index(drop=True).copy()
    df_lp["eps_OIL"] = resid[:, 0]
    df_lp["eps_SR"]  = resid[:, 2]
    df_lp["eps_OIL_pos"] = df_lp["eps_OIL"].clip(lower=0)
    df_lp["eps_OIL_neg"] = df_lp["eps_OIL"].clip(upper=0)
    df_lp["eps_SR_pos"]  = df_lp["eps_SR"].clip(lower=0)
    df_lp["eps_SR_neg"]  = df_lp["eps_SR"].clip(upper=0)
    df_lp["IL"] = (df_lp["regime_post2013"] <= tau_hat).astype(int)

    rows = []
    for h in range(1, 19):   # shorter horizon for small sample
        for regime, mask in [("Low", df_lp["IL"] == 1),
                             ("High", df_lp["IL"] == 0)]:
            sub = df_lp[mask].copy()
            sub["y_h"] = sub["lnNR"].shift(-h) - sub["lnNR"]
            cols = ["eps_OIL_pos", "eps_OIL_neg", "eps_SR_pos", "eps_SR_neg"]
            for lag in (1, 2):
                for v in ["dlnNR", "dlnOIL", "dlnSR", "dlnBD"]:
                    c = f"{v}_lag{lag}"
                    sub[c] = sub[v].shift(lag)
                    cols.append(c)
            sub = sub.dropna(subset=["y_h"] + cols)
            if len(sub) < 20:
                continue
            X = add_constant(sub[cols], has_constant="add")
            try:
                fit = OLS(sub["y_h"], X).fit(
                    cov_type="HAC", cov_kwds={"maxlags": h + 1})
                for v in ["eps_OIL_pos", "eps_OIL_neg", "eps_SR_pos", "eps_SR_neg"]:
                    rows.append({"horizon": h, "regime": regime, "var": v,
                                 "est": fit.params[v], "se": fit.bse[v]})
            except Exception:
                continue

    return pd.DataFrame(rows)


def plot_regime_irf(lp: pd.DataFrame, shock: str, title: str, save: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, regime in zip(axes, ["Low", "High"]):
        for sign, ls, lab in [(f"{shock}_pos", "-", "Positive"),
                              (f"{shock}_neg", "--", "Negative")]:
            sub = lp[(lp["regime"] == regime) & (lp["var"] == sign)]
            if len(sub) == 0: continue
            ax.plot(sub["horizon"], sub["est"], color="black", linestyle=ls,
                    linewidth=1.5, label=f"{lab} shock")
            ax.fill_between(sub["horizon"], sub["est"] - 1.96 * sub["se"],
                            sub["est"] + 1.96 * sub["se"],
                            color="grey", alpha=0.18)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"{regime}-inventory regime")
        ax.set_xlabel("Horizon (months)")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Cumulative response of ln NR")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    log.info(f"Saved {save}")


# =========================================================================
# Main
# =========================================================================

def main():
    log.info("\n=== Step 1: Hansen (2000) threshold estimation ===")
    th = estimate_threshold(df, n_grid=80)
    plot_ssr_profile(th, FIGURES / "threshold_ssr.pdf")

    n_low = (df["regime_post2013"] <= th["tau_hat"]).sum()
    n_high = ((df["regime_post2013"] > th["tau_hat"]) & df["regime_post2013"].notna()).sum()
    log.info(f"Regime split: Low = {n_low}, High = {n_high}")

    with open(TABLES / "threshold_estimation.txt", "w") as f:
        f.write("Hansen (2000) Threshold Estimation\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Sample: post-2013, T = {len(df)}\n")
        f.write(f"Regime variable: regime_post2013 (12-month deviation)\n")
        f.write(f"Grid range: [P15, P85] of regime variable\n\n")
        f.write(f"tau_hat = {th['tau_hat']:.4f}\n")
        f.write(f"Low-inventory observations:  {n_low}\n")
        f.write(f"High-inventory observations: {n_high}\n")

    log.info("\n=== Step 2: Threshold-NARDL estimation ===")
    res = fit_tnardl(df, th["tau_hat"])
    write_tnardl_table(res, TABLES / "tnardl_results.txt")

    log.info("\n=== Step 3: Bootstrap CIs for amplification (H6) ===")
    boot_res = bootstrap_amplification(df, th["tau_hat"], B=500)
    log.info(f"H6 amp(oil) = {res['amp_oil']:+.3f}, "
             f"95% CI [{boot_res['ci_oil'][0]:+.3f}, {boot_res['ci_oil'][1]:+.3f}], "
             f"p = {boot_res['p_oil_one_sided']:.3f}")
    log.info(f"H6 amp(SR)  = {res['amp_sr']:+.3f}, "
             f"95% CI [{boot_res['ci_sr'][0]:+.3f}, {boot_res['ci_sr'][1]:+.3f}], "
             f"p = {boot_res['p_sr_one_sided']:.3f}")

    # Append bootstrap results to tnardl table
    with open(TABLES / "tnardl_results.txt", "a") as f:
        f.write("\nBootstrap inference (block bootstrap, B = 500):\n")
        f.write(f"  amp(oil) = {res['amp_oil']:+.3f}, "
                f"95% CI [{boot_res['ci_oil'][0]:+.3f}, {boot_res['ci_oil'][1]:+.3f}], "
                f"one-sided p = {boot_res['p_oil_one_sided']:.4f}\n")
        f.write(f"  amp(SR)  = {res['amp_sr']:+.3f}, "
                f"95% CI [{boot_res['ci_sr'][0]:+.3f}, {boot_res['ci_sr'][1]:+.3f}], "
                f"one-sided p = {boot_res['p_sr_one_sided']:.4f}\n")

    log.info("\n=== Step 4: Regime-specific channel decomposition (H7) ===")
    dec = regime_decomposition(df, th["tau_hat"], B=2000)
    write_decomp_table(dec, TABLES / "regime_decomposition.txt")
    if dec:
        log.info(f"H7: medshare_low - medshare_high = {dec['diff']:+.3f}, "
                 f"p = {dec['p_one_sided']:.3f}")

    log.info("\n=== Step 5: Regime-specific Local Projections ===")
    lp = regime_lp(df, th["tau_hat"])
    if len(lp) > 0:
        lp.to_csv(TABLES / "regime_lp_results.csv", index=False)
        plot_regime_irf(lp, "eps_OIL", "Regime-specific LP: oil shock",
                        FIGURES / "regime_irf_oil.pdf")
        plot_regime_irf(lp, "eps_SR", "Regime-specific LP: synthetic-rubber shock",
                        FIGURES / "regime_irf_sr.pdf")

    log.info("\n[DONE] Phase 2 complete.")


if __name__ == "__main__":
    main()
