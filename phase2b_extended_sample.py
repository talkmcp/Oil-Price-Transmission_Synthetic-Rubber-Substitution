"""
=============================================================================
 Phase 2b: Threshold NARDL with Extended Sample (Path 1)
 Strategy:
   - Use YoY log change of inventory as regime variable (instead of dev12)
   - Add nbs_break_period dummy (= 1 for Jan 2012 - Dec 2012, the 12-month
     window during which YoY is contaminated by the methodology revision)
   - Drop SR partial sums from the threshold equation (parsimony)
   - Use p=q=1 lag order (parsimony)
   - Sample: 2004-04 to 2026-03 (T ≈ 264 after first lag, vs 155 in Phase 2)

 Why this works:
   - YoY recovers naturally one year after the level break
   - Break-period dummy absorbs the 12-month spike
   - Smaller threshold model -> more degrees of freedom in Low regime
   - Low regime expected to grow from 46 -> ~70 obs

 Compare with Phase 2:
   - Phase 2:  T=155, Low=46, High=109   (regime_post2013, p=q=2, 6 LR regs)
   - Phase 2b: T=~264 expected, Low=~80, High=~180   (parsimonious, more obs)
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
log = logging.getLogger("phase2b")

DATA_FILE = Path("data/rubber_petrochemical_monthly_model_dataset_v3.xlsx")
TABLES    = Path("tables");  TABLES.mkdir(exist_ok=True)
FIGURES   = Path("figures"); FIGURES.mkdir(exist_ok=True)

RNG = np.random.default_rng(20260101)


def wald_scalar(w):
    s = np.asarray(w.statistic).ravel()
    p = np.asarray(w.pvalue).ravel()
    return float(s[0]), float(p[0])


# =========================================================================
# 1. Load + construct extended-sample regime variable
# =========================================================================

df = pd.read_excel(DATA_FILE, sheet_name="Monthly_Model_Data")
if pd.api.types.is_numeric_dtype(df["date"]):
    df["date"] = pd.to_datetime(df["date"], unit="D", origin="1899-12-30")
df = df[df["core_model_flag"] == 1].sort_values("date").reset_index(drop=True)
log.info(f"Core sample: T = {len(df)}")

# Construct YoY regime variable on the FULL core sample
inv = df["product_inv_rubber_only_rmbbn"]
df["regime_yoy"] = np.log(inv) - np.log(inv.shift(12))

# Break-period dummy: the 12-month window after the NBS revision (Jan-Dec 2012)
# during which YoY is contaminated by the level shift
df["nbs_break_period"] = ((df["date"] >= "2012-01-01")
                          & (df["date"] <= "2012-12-01")).astype(int)

# Standard variable mapping
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

# Restrict to rows where regime_yoy is available
df_ext = df.dropna(subset=["regime_yoy"]).reset_index(drop=True)
log.info(f"Extended sample: T = {len(df_ext)}, "
         f"{df_ext['date'].min():%Y-%m} to {df_ext['date'].max():%Y-%m}")
log.info(f"Break-period months in sample: {df_ext['nbs_break_period'].sum()}")

CONTROLS = ["ln_china_auto", "fx_cny", "fx_thb", "nbs_break_period"]


# =========================================================================
# 2. Parsimonious Threshold NARDL builder
# =========================================================================

def build_tnardl_design(d: pd.DataFrame, tau: float, p: int = 1, q: int = 1):
    """
    Parsimonious threshold-NARDL design:
      - Long-run partial sums for OIL only (drop SR from LR; keep in mediation)
      - Pooled short-run dynamics (same in both regimes)
      - p=q=1 by default
    """
    e = d.copy()
    e["IL"] = (e["regime_yoy"] <= tau).astype(int)
    e["IH"] = 1 - e["IL"]

    # Regime-specific long-run regressors (lagged levels, OIL only + BD)
    base_lr = ["lnNR", "lnOIL_pos", "lnOIL_neg", "lnBD"]
    lr_cols_L, lr_cols_H = [], []
    for v in base_lr:
        e[f"{v}_lag1"] = e[v].shift(1)
        e[f"{v}_lag1_L"] = e[f"{v}_lag1"] * e["IL"]
        e[f"{v}_lag1_H"] = e[f"{v}_lag1"] * e["IH"]
        lr_cols_L.append(f"{v}_lag1_L")
        lr_cols_H.append(f"{v}_lag1_H")

    # Pooled short-run dynamics (NOT regime-split)
    sr_cols = []
    for i in range(1, p + 1):
        c = f"dlnNR_lag{i}"; e[c] = e["dlnNR"].shift(i); sr_cols.append(c)
    for i in range(0, q):
        for v in ["dlnOIL_pos", "dlnOIL_neg"]:
            c = f"{v}_lag{i}"; e[c] = e[v].shift(i); sr_cols.append(c)

    cols = ["IL", "IH"] + lr_cols_L + lr_cols_H + sr_cols + CONTROLS
    e = e.dropna(subset=["dlnNR"] + cols).reset_index(drop=True)
    return e, cols, lr_cols_L, lr_cols_H


# =========================================================================
# 3. Hansen (2000) threshold estimation
# =========================================================================

def estimate_threshold(d: pd.DataFrame, n_grid: int = 80) -> dict:
    z = d["regime_yoy"].dropna()
    q_lo, q_hi = float(z.quantile(0.15)), float(z.quantile(0.85))
    cands = np.linspace(q_lo, q_hi, n_grid)
    ssrs = np.full(n_grid, np.inf)

    for k, tau in enumerate(cands):
        try:
            e, cols, *_ = build_tnardl_design(d, tau)
            if len(e) < 30:
                continue
            fit = OLS(e["dlnNR"], e[cols]).fit()
            ssrs[k] = float(np.sum(fit.resid ** 2))
        except Exception:
            continue

    tau_hat = float(cands[np.argmin(ssrs)])
    log.info(f"tau_hat = {tau_hat:.4f}   (grid range [{q_lo:.4f}, {q_hi:.4f}])")
    return {"tau_hat": tau_hat, "cands": cands, "ssrs": ssrs}


# =========================================================================
# 4. Threshold-NARDL fit
# =========================================================================

def fit_tnardl(d: pd.DataFrame, tau_hat: float) -> dict:
    e, cols, lrL, lrH = build_tnardl_design(d, tau_hat, p=1, q=1)
    fit = OLS(e["dlnNR"], e[cols]).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(np.ceil(len(e) ** 0.25))})

    rho_L = fit.params["lnNR_lag1_L"]
    rho_H = fit.params["lnNR_lag1_H"]
    LR_low = {
        "rho":       rho_L,
        "L_OIL_pos": -fit.params["lnOIL_pos_lag1_L"] / rho_L,
        "L_OIL_neg": -fit.params["lnOIL_neg_lag1_L"] / rho_L,
    }
    LR_high = {
        "rho":       rho_H,
        "L_OIL_pos": -fit.params["lnOIL_pos_lag1_H"] / rho_H,
        "L_OIL_neg": -fit.params["lnOIL_neg_lag1_H"] / rho_H,
    }

    F_low_oil, p_low_oil   = wald_scalar(fit.wald_test(
        "lnOIL_pos_lag1_L = lnOIL_neg_lag1_L", use_f=True))
    F_high_oil, p_high_oil = wald_scalar(fit.wald_test(
        "lnOIL_pos_lag1_H = lnOIL_neg_lag1_H", use_f=True))

    amp_oil = abs(LR_low["L_OIL_pos"] - LR_low["L_OIL_neg"]) \
              - abs(LR_high["L_OIL_pos"] - LR_high["L_OIL_neg"])

    n_low  = int(e["IL"].sum())
    n_high = int(e["IH"].sum())

    return {"fit": fit, "LR_low": LR_low, "LR_high": LR_high,
            "test_low_oil":  (F_low_oil, p_low_oil),
            "test_high_oil": (F_high_oil, p_high_oil),
            "amp_oil": amp_oil, "data": e, "tau": tau_hat,
            "n_low": n_low, "n_high": n_high}


# =========================================================================
# 5. Bootstrap CI
# =========================================================================

def bootstrap_amplification(d: pd.DataFrame, tau_hat: float,
                            B: int = 1000) -> dict:
    n = len(d)
    block = max(4, int(np.ceil(n ** (1/3))))
    n_blocks = int(np.ceil(n / block))
    amp_b = np.full(B, np.nan)

    for b in range(B):
        starts = RNG.integers(0, max(1, n - block), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        idx = np.clip(idx, 0, n - 1)
        d_b = d.iloc[idx].reset_index(drop=True)
        try:
            res = fit_tnardl(d_b, tau_hat)
            amp_b[b] = res["amp_oil"]
        except Exception:
            continue

    amp_b = amp_b[~np.isnan(amp_b)]
    if len(amp_b) < 50:
        return {"ci": (np.nan, np.nan), "p_one_sided": np.nan,
                "B_eff": len(amp_b)}
    return {
        "ci": np.quantile(amp_b, [0.025, 0.975]),
        "p_one_sided": float(np.mean(amp_b <= 0)),
        "B_eff": len(amp_b),
    }


# =========================================================================
# 6. Regime-specific channel decomposition (H7)
# =========================================================================

def regime_decomposition(d: pd.DataFrame, tau_hat: float, B: int = 2000) -> dict:
    Z = ["fx_cny", "fx_thb", "ln_china_auto", "nbs_break_period"]
    e = d.copy()
    e["IL"] = (e["regime_yoy"] <= tau_hat).astype(int)
    e["IH"] = 1 - e["IL"]
    e = e.dropna(subset=["lnNR", "lnSR", "lnOIL"] + Z)

    def decomp(sub):
        s1 = OLS(sub["lnSR"],
                 add_constant(sub[["lnOIL"] + Z], has_constant="add")).fit()
        s2 = OLS(sub["lnNR"],
                 add_constant(sub[["lnOIL", "lnSR"] + Z], has_constant="add")).fit()
        a1, k1, k2 = s1.params["lnOIL"], s2.params["lnOIL"], s2.params["lnSR"]
        ind, tot = a1 * k2, k1 + a1 * k2
        return {"alpha1": a1, "kappa1": k1, "kappa2": k2,
                "direct": k1, "indirect": ind, "total": tot,
                "medshare": ind / tot if tot != 0 else np.nan}

    sub_L = e[e["IL"] == 1]
    sub_H = e[e["IH"] == 1]
    log.info(f"Decomposition: Low N={len(sub_L)}, High N={len(sub_H)}")
    if len(sub_L) < 20 or len(sub_H) < 20:
        return {}

    dec_L, dec_H = decomp(sub_L), decomp(sub_H)

    diffs = np.full(B, np.nan)
    for b in range(B):
        try:
            db_L = sub_L.sample(n=len(sub_L), replace=True,
                                random_state=RNG.integers(1e9))
            db_H = sub_H.sample(n=len(sub_H), replace=True,
                                random_state=RNG.integers(1e9))
            diffs[b] = decomp(db_L)["medshare"] - decomp(db_H)["medshare"]
        except Exception:
            continue
    diffs = diffs[~np.isnan(diffs)]
    ci = np.quantile(diffs, [0.025, 0.975]) if len(diffs) > 50 else (np.nan, np.nan)
    p_one = float(np.mean(diffs <= 0)) if len(diffs) > 50 else np.nan

    return {"low": dec_L, "high": dec_H,
            "diff": dec_L["medshare"] - dec_H["medshare"],
            "ci": ci, "p_one_sided": p_one, "B_eff": len(diffs)}


# =========================================================================
# 7. Output
# =========================================================================

def write_results(th, res, boot, dec):
    out = TABLES / "tnardl_extended_results.txt"
    with open(out, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PHASE 2b: Threshold NARDL with Extended Sample (Path 1)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Strategy: YoY regime variable + NBS break-period dummy\n")
        f.write("          Parsimonious specification (oil partial sums only)\n")
        f.write("          p = q = 1 (vs p = q = 2 in Phase 2)\n\n")

        f.write("--- Threshold ---\n")
        f.write(f"  tau_hat = {res['tau']:.4f}\n")
        f.write(f"  N (Low)  = {res['n_low']}\n")
        f.write(f"  N (High) = {res['n_high']}\n")
        f.write(f"  Total T  = {len(res['data'])}\n\n")

        f.write("--- Long-run coefficients ---\n")
        f.write(f"{'':<14}{'Low-inv':>14}{'High-inv':>14}\n")
        f.write(f"{'-'*42}\n")
        f.write(f"{'rho':<14}{res['LR_low']['rho']:>14.3f}"
                f"{res['LR_high']['rho']:>14.3f}\n")
        f.write(f"{'L_OIL+':<14}{res['LR_low']['L_OIL_pos']:>14.3f}"
                f"{res['LR_high']['L_OIL_pos']:>14.3f}\n")
        f.write(f"{'L_OIL-':<14}{res['LR_low']['L_OIL_neg']:>14.3f}"
                f"{res['LR_high']['L_OIL_neg']:>14.3f}\n\n")

        f.write("--- Within-regime asymmetry (oil) ---\n")
        f.write(f"  Low:  F = {res['test_low_oil'][0]:.2f}, "
                f"p = {res['test_low_oil'][1]:.4f}\n")
        f.write(f"  High: F = {res['test_high_oil'][0]:.2f}, "
                f"p = {res['test_high_oil'][1]:.4f}\n\n")

        f.write("--- H6: Cross-regime amplification ---\n")
        f.write(f"  amp = |L+ - L-|^Low - |L+ - L-|^High = {res['amp_oil']:+.3f}\n")
        if not np.isnan(boot["ci"][0]):
            f.write(f"  95% CI = [{boot['ci'][0]:+.3f}, {boot['ci'][1]:+.3f}]  "
                    f"(B = {boot['B_eff']})\n")
            f.write(f"  One-sided p-value = {boot['p_one_sided']:.4f}\n")
            sig = "YES" if boot["p_one_sided"] < 0.10 else "NO"
            f.write(f"  H6 significant at 10%: {sig}\n\n")

        if dec:
            f.write("--- H7: Regime-specific mediation ---\n")
            f.write(f"{'':<14}{'Low-inv':>14}{'High-inv':>14}\n")
            for k in ["alpha1", "kappa1", "kappa2",
                      "direct", "indirect", "medshare"]:
                f.write(f"{k:<14}{dec['low'][k]:>14.4f}{dec['high'][k]:>14.4f}\n")
            f.write(f"\nMediation share difference: {dec['diff']:+.4f}\n")
            if not np.isnan(dec["ci"][0]):
                f.write(f"95% CI = [{dec['ci'][0]:+.4f}, {dec['ci'][1]:+.4f}]  "
                        f"(B = {dec['B_eff']})\n")
                f.write(f"One-sided p-value = {dec['p_one_sided']:.4f}\n")
                sig = "YES" if dec["p_one_sided"] < 0.10 else "NO"
                f.write(f"H7 significant at 10%: {sig}\n")

    log.info(f"Saved {out}")


def plot_ssr(th, save):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(th["cands"], th["ssrs"], color="black", linewidth=1.4)
    ax.axvline(th["tau_hat"], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Candidate threshold τ on regime_yoy")
    ax.set_ylabel("Sum of squared residuals")
    ax.set_title("Hansen (2000) threshold profile (Phase 2b)")
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    log.info(f"Saved {save}")


# =========================================================================
# Main
# =========================================================================

def main():
    log.info("\n=== Phase 2b: Hansen threshold ===")
    th = estimate_threshold(df_ext)
    plot_ssr(th, FIGURES / "threshold_ssr_extended.pdf")

    log.info("\n=== Phase 2b: Fit threshold NARDL ===")
    res = fit_tnardl(df_ext, th["tau_hat"])
    log.info(f"Long-run (Low):  L_OIL+ = {res['LR_low']['L_OIL_pos']:+.3f}, "
             f"L_OIL- = {res['LR_low']['L_OIL_neg']:+.3f}")
    log.info(f"Long-run (High): L_OIL+ = {res['LR_high']['L_OIL_pos']:+.3f}, "
             f"L_OIL- = {res['LR_high']['L_OIL_neg']:+.3f}")
    log.info(f"Within-regime asymmetry: Low p = {res['test_low_oil'][1]:.4f}, "
             f"High p = {res['test_high_oil'][1]:.4f}")
    log.info(f"H6 amplification = {res['amp_oil']:+.3f}")

    log.info("\n=== Phase 2b: Bootstrap CI for H6 ===")
    boot = bootstrap_amplification(df_ext, th["tau_hat"], B=1000)
    if not np.isnan(boot["ci"][0]):
        log.info(f"H6: amp = {res['amp_oil']:+.3f}, "
                 f"95% CI [{boot['ci'][0]:+.3f}, {boot['ci'][1]:+.3f}], "
                 f"p = {boot['p_one_sided']:.4f}")

    log.info("\n=== Phase 2b: Regime-specific decomposition (H7) ===")
    dec = regime_decomposition(df_ext, th["tau_hat"], B=2000)
    if dec:
        log.info(f"H7: medshare(Low) = {dec['low']['medshare']:.3f}, "
                 f"medshare(High) = {dec['high']['medshare']:.3f}")
        log.info(f"    diff = {dec['diff']:+.3f}, p = {dec['p_one_sided']:.4f}")

    write_results(th, res, boot, dec)

    log.info("\n[DONE] Phase 2b complete. See tables/tnardl_extended_results.txt")


if __name__ == "__main__":
    main()
