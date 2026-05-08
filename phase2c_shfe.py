"""
Phase 2c: Threshold NARDL with SHFE regime variable
Companion to Phase 2 (NBS post-2013) and Phase 2b (NBS YoY extended).

Strategy:
  - Use SHFE warehouse stock as raw-rubber storage measure
  - Williams-Wright storage-buffer interpretation
  - Sample: Dec 2014+ (T=131 after lags)
  - Same parsimonious specification as Phase 2b
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase2c")

DATA_FILE = Path("data/rubber_petrochemical_monthly_model_dataset_v3.xlsx")
SHFE_FILE = Path("data/shfe_rubber_stock_monthly.csv")
TABLES    = Path("tables");  TABLES.mkdir(exist_ok=True)
FIGURES   = Path("figures"); FIGURES.mkdir(exist_ok=True)

RNG = np.random.default_rng(20260101)


def wald_scalar(w):
    s = np.asarray(w.statistic).ravel()
    p = np.asarray(w.pvalue).ravel()
    return float(s[0]), float(p[0])


# =========================================================================
# Load + merge SHFE
# =========================================================================

df = pd.read_excel(DATA_FILE, sheet_name="Monthly_Model_Data")
if pd.api.types.is_numeric_dtype(df["date"]):
    df["date"] = pd.to_datetime(df["date"], unit="D", origin="1899-12-30")
df = df[df["core_model_flag"] == 1].sort_values("date").reset_index(drop=True)
df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

shfe = pd.read_csv(SHFE_FILE, parse_dates=["date"])
shfe["date"] = shfe["date"].dt.to_period("M").dt.to_timestamp()
df = df.merge(shfe, on="date", how="left")

# Construct SHFE regime variable
df["log_shfe"] = np.log(df["shfe_stock_ton"])
df["regime_shfe"] = (df["log_shfe"]
                     - df["log_shfe"].rolling(12, min_periods=6).mean())

log.info(f"SHFE coverage in core: {df['regime_shfe'].notna().sum()}/{len(df)}")

# Standard mapping
df["lnNR"]   = df["ln_nr_rss_rmbton"]
df["lnOIL"]  = df["ln_oil_opec"]
df["lnBD"]   = df["ln_butadiene"]
df["lnSR"]   = df["ln_sbr_monthly"]
df["dlnNR"]  = df["dln_nr_rss"]
df["dlnOIL"] = df["dln_oil_opec"]
df["lnOIL_pos"] = df["oil_opec_pos_csum"]
df["lnOIL_neg"] = df["oil_opec_neg_csum"]
df["dlnOIL_pos"] = df["oil_opec_pos_change"]
df["dlnOIL_neg"] = df["oil_opec_neg_change"]
df["fx_cny"] = np.log(df["fx_china_lcu_per_usd"])
df["fx_thb"] = np.log(df["fx_thailand_lcu_per_usd"])

# Restrict to rows with regime_shfe
df_shfe = df.dropna(subset=["regime_shfe"]).reset_index(drop=True)
log.info(f"SHFE sample: T = {len(df_shfe)}, "
         f"{df_shfe['date'].min():%Y-%m} to {df_shfe['date'].max():%Y-%m}")

CONTROLS = ["ln_china_auto", "fx_cny", "fx_thb"]


# =========================================================================
# Threshold NARDL builder (parsimonious)
# =========================================================================

def build_design(d, tau, p=1, q=1):
    e = d.copy()
    e["IL"] = (e["regime_shfe"] <= tau).astype(int)
    e["IH"] = 1 - e["IL"]

    base_lr = ["lnNR", "lnOIL_pos", "lnOIL_neg", "lnBD"]
    lr_L, lr_H = [], []
    for v in base_lr:
        e[f"{v}_lag1"] = e[v].shift(1)
        e[f"{v}_lag1_L"] = e[f"{v}_lag1"] * e["IL"]
        e[f"{v}_lag1_H"] = e[f"{v}_lag1"] * e["IH"]
        lr_L.append(f"{v}_lag1_L"); lr_H.append(f"{v}_lag1_H")

    sr = []
    for i in range(1, p + 1):
        c = f"dlnNR_lag{i}"; e[c] = e["dlnNR"].shift(i); sr.append(c)
    for i in range(0, q):
        for v in ["dlnOIL_pos", "dlnOIL_neg"]:
            c = f"{v}_lag{i}"; e[c] = e[v].shift(i); sr.append(c)

    cols = ["IL", "IH"] + lr_L + lr_H + sr + CONTROLS
    e = e.dropna(subset=["dlnNR"] + cols).reset_index(drop=True)
    return e, cols


def estimate_threshold(d, n_grid=80):
    z = d["regime_shfe"].dropna()
    q_lo, q_hi = float(z.quantile(0.15)), float(z.quantile(0.85))
    cands = np.linspace(q_lo, q_hi, n_grid)
    ssrs = np.full(n_grid, np.inf)
    for k, tau in enumerate(cands):
        try:
            e, cols = build_design(d, tau)
            if len(e) < 25: continue
            fit = OLS(e["dlnNR"], e[cols]).fit()
            ssrs[k] = float(np.sum(fit.resid ** 2))
        except Exception:
            continue
    tau_hat = float(cands[np.argmin(ssrs)])
    return {"tau_hat": tau_hat, "cands": cands, "ssrs": ssrs,
            "q_lo": q_lo, "q_hi": q_hi}


def fit_tnardl(d, tau):
    e, cols = build_design(d, tau, p=1, q=1)
    fit = OLS(e["dlnNR"], e[cols]).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(np.ceil(len(e) ** 0.25))})
    rho_L = fit.params["lnNR_lag1_L"]
    rho_H = fit.params["lnNR_lag1_H"]
    LR_L = {"rho": rho_L,
            "L_OIL_pos": -fit.params["lnOIL_pos_lag1_L"] / rho_L,
            "L_OIL_neg": -fit.params["lnOIL_neg_lag1_L"] / rho_L}
    LR_H = {"rho": rho_H,
            "L_OIL_pos": -fit.params["lnOIL_pos_lag1_H"] / rho_H,
            "L_OIL_neg": -fit.params["lnOIL_neg_lag1_H"] / rho_H}
    F_L, p_L = wald_scalar(fit.wald_test(
        "lnOIL_pos_lag1_L = lnOIL_neg_lag1_L", use_f=True))
    F_H, p_H = wald_scalar(fit.wald_test(
        "lnOIL_pos_lag1_H = lnOIL_neg_lag1_H", use_f=True))
    amp = abs(LR_L["L_OIL_pos"] - LR_L["L_OIL_neg"]) \
          - abs(LR_H["L_OIL_pos"] - LR_H["L_OIL_neg"])
    return {"fit": fit, "LR_L": LR_L, "LR_H": LR_H,
            "p_L": p_L, "p_H": p_H, "amp": amp,
            "n_low": int(e["IL"].sum()), "n_high": int(e["IH"].sum()),
            "data": e, "tau": tau}


def boot_amp(d, tau, B=1000):
    n = len(d)
    block = max(4, int(np.ceil(n ** (1/3))))
    nb = int(np.ceil(n / block))
    a = np.full(B, np.nan)
    for b in range(B):
        starts = RNG.integers(0, max(1, n-block), size=nb)
        idx = np.concatenate([np.arange(s, s+block) for s in starts])[:n]
        idx = np.clip(idx, 0, n-1)
        try:
            r = fit_tnardl(d.iloc[idx].reset_index(drop=True), tau)
            a[b] = r["amp"]
        except Exception:
            continue
    a = a[~np.isnan(a)]
    if len(a) < 50:
        return {"ci": (np.nan, np.nan), "p_one": np.nan, "B_eff": len(a)}
    return {"ci": np.quantile(a, [0.025, 0.975]),
            "p_one": float(np.mean(a <= 0)), "B_eff": len(a)}


def regime_decomp(d, tau, B=2000):
    Z = ["fx_cny", "fx_thb", "ln_china_auto"]
    e = d.copy()
    e["IL"] = (e["regime_shfe"] <= tau).astype(int)
    e["IH"] = 1 - e["IL"]
    e = e.dropna(subset=["lnNR", "lnSR", "lnOIL"] + Z)

    def decomp(sub):
        s1 = OLS(sub["lnSR"], add_constant(sub[["lnOIL"] + Z], has_constant="add")).fit()
        s2 = OLS(sub["lnNR"], add_constant(sub[["lnOIL", "lnSR"] + Z], has_constant="add")).fit()
        a1, k1, k2 = s1.params["lnOIL"], s2.params["lnOIL"], s2.params["lnSR"]
        ind, tot = a1 * k2, k1 + a1 * k2
        return {"alpha1": a1, "kappa1": k1, "kappa2": k2,
                "direct": k1, "indirect": ind, "total": tot,
                "medshare": ind/tot if tot != 0 else np.nan}

    sub_L = e[e["IL"] == 1]; sub_H = e[e["IH"] == 1]
    log.info(f"Decomp: Low N={len(sub_L)}, High N={len(sub_H)}")
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
    return {"low": dec_L, "high": dec_H,
            "diff": dec_L["medshare"] - dec_H["medshare"],
            "ci": np.quantile(diffs, [0.025, 0.975]) if len(diffs) > 50 else (np.nan, np.nan),
            "p_one": float(np.mean(diffs <= 0)) if len(diffs) > 50 else np.nan,
            "B_eff": len(diffs)}


# =========================================================================
# Main
# =========================================================================

log.info("\n=== Phase 2c: Hansen threshold (SHFE) ===")
th = estimate_threshold(df_shfe)
log.info(f"tau_hat = {th['tau_hat']:.4f}, range [{th['q_lo']:.4f}, {th['q_hi']:.4f}]")

log.info("\n=== Phase 2c: Threshold NARDL ===")
res = fit_tnardl(df_shfe, th["tau_hat"])
log.info(f"Sample after lag: T = {len(res['data'])}, "
         f"Low = {res['n_low']}, High = {res['n_high']}")
log.info(f"Long-run (Low):  L_OIL+ = {res['LR_L']['L_OIL_pos']:+.3f}, "
         f"L_OIL- = {res['LR_L']['L_OIL_neg']:+.3f}, rho = {res['LR_L']['rho']:.3f}")
log.info(f"Long-run (High): L_OIL+ = {res['LR_H']['L_OIL_pos']:+.3f}, "
         f"L_OIL- = {res['LR_H']['L_OIL_neg']:+.3f}, rho = {res['LR_H']['rho']:.3f}")
log.info(f"Within-regime asymmetry (oil): Low p = {res['p_L']:.4f}, High p = {res['p_H']:.4f}")
log.info(f"H6 amp = {res['amp']:+.3f}")

log.info("\n=== Phase 2c: Bootstrap H6 ===")
boot = boot_amp(df_shfe, th["tau_hat"], B=1000)
if not np.isnan(boot["ci"][0]):
    log.info(f"H6: amp = {res['amp']:+.3f}, "
             f"95% CI [{boot['ci'][0]:+.3f}, {boot['ci'][1]:+.3f}], "
             f"p = {boot['p_one']:.4f}")

log.info("\n=== Phase 2c: H7 channel decomposition ===")
dec = regime_decomp(df_shfe, th["tau_hat"], B=2000)
if dec:
    log.info(f"H7: medshare(Low) = {dec['low']['medshare']:.3f}, "
             f"medshare(High) = {dec['high']['medshare']:.3f}")
    log.info(f"    diff = {dec['diff']:+.3f}, p = {dec['p_one']:.4f}")

# Write summary
out = TABLES / "tnardl_shfe_results.txt"
with open(out, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("PHASE 2c: Threshold NARDL with SHFE regime variable\n")
    f.write("=" * 60 + "\n\n")
    f.write("Source: SHFE Deliverable Stock: Natural Rubber (CEIC)\n")
    f.write("Theory: Williams-Wright (1991) storage-buffer\n")
    f.write("Sample: Dec 2014+ (raw rubber, no break)\n\n")

    f.write(f"--- Threshold ---\n")
    f.write(f"  tau_hat = {res['tau']:.4f}\n")
    f.write(f"  N (Low)  = {res['n_low']}\n")
    f.write(f"  N (High) = {res['n_high']}\n")
    f.write(f"  Total T  = {len(res['data'])}\n\n")

    f.write("--- Long-run coefficients ---\n")
    f.write(f"{'':<14}{'Low':>14}{'High':>14}\n")
    f.write(f"{'-'*42}\n")
    f.write(f"{'rho':<14}{res['LR_L']['rho']:>14.3f}{res['LR_H']['rho']:>14.3f}\n")
    f.write(f"{'L_OIL+':<14}{res['LR_L']['L_OIL_pos']:>14.3f}{res['LR_H']['L_OIL_pos']:>14.3f}\n")
    f.write(f"{'L_OIL-':<14}{res['LR_L']['L_OIL_neg']:>14.3f}{res['LR_H']['L_OIL_neg']:>14.3f}\n\n")

    f.write("--- Within-regime asymmetry (oil) ---\n")
    f.write(f"  Low:  p = {res['p_L']:.4f}\n")
    f.write(f"  High: p = {res['p_H']:.4f}\n\n")

    f.write("--- H6 (cross-regime amplification) ---\n")
    f.write(f"  amp = {res['amp']:+.3f}\n")
    if not np.isnan(boot["ci"][0]):
        f.write(f"  95% CI = [{boot['ci'][0]:+.3f}, {boot['ci'][1]:+.3f}]\n")
        f.write(f"  One-sided p = {boot['p_one']:.4f}\n\n")

    if dec:
        f.write("--- H7 (regime-specific mediation) ---\n")
        f.write(f"{'':<14}{'Low':>14}{'High':>14}\n")
        for k in ["alpha1", "kappa1", "kappa2", "direct", "indirect", "medshare"]:
            f.write(f"{k:<14}{dec['low'][k]:>14.4f}{dec['high'][k]:>14.4f}\n")
        f.write(f"\nDiff = {dec['diff']:+.4f}\n")
        if not np.isnan(dec["ci"][0]):
            f.write(f"95% CI = [{dec['ci'][0]:+.4f}, {dec['ci'][1]:+.4f}]\n")
            f.write(f"One-sided p = {dec['p_one']:.4f}\n")
            f.write(f"H7 sig at 10%: {dec['p_one'] < 0.10}\n")

log.info(f"\nSaved {out}")
log.info("[DONE] Phase 2c complete.")
