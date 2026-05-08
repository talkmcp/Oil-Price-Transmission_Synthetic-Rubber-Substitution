"""
=============================================================================
 Phase 1: Baseline NARDL Pipeline (Option D)
 Tests H1 through H5 on the FULL sample (T = 252) with NBS break dummy.

 Mapping (paper -> dataset):
   ln NR    = ln_nr_rss_rmbton
   ln OIL   = ln_oil_opec
   ln BD    = ln_butadiene
   ln SR    = ln_sbr_monthly
   ln OIL+  = oil_opec_pos_csum    (cumulative partial sums pre-computed)
   ln OIL-  = oil_opec_neg_csum
   ln SR+   = sbr_pos_csum
   ln SR-   = sbr_neg_csum

 Outputs:
   tables/eq33_petrochem_chain.txt      H1: oil + BD -> SR
   tables/eq34_substitution.txt         H2: SR -> NR
   tables/bounds_test.txt               PSS bounds test
   tables/nardl_baseline.txt            Eq.(3.5) NARDL with break dummy
   tables/long_run_coefs.csv            L+, L- from NARDL
   tables/asymmetry_tests.txt           H3 Wald tests
   tables/lp_results.csv                Local projections (H4)
   tables/channel_decomposition.txt     H5 mediation
   figures/lp_oil.pdf                   IRF plot
   figures/lp_sr.pdf
   figures/dynamic_multipliers.pdf
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

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase1")

DATA_FILE = Path("data/rubber_petrochemical_monthly_model_dataset_v3.xlsx")
TABLES    = Path("tables");  TABLES.mkdir(exist_ok=True)
FIGURES   = Path("figures"); FIGURES.mkdir(exist_ok=True)

RNG = np.random.default_rng(20260101)


# =========================================================================
# 1. Load and prepare
# =========================================================================

def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE, sheet_name="Monthly_Model_Data")
    if pd.api.types.is_numeric_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], unit="D", origin="1899-12-30")
    df = df[df["core_model_flag"] == 1].sort_values("date").reset_index(drop=True)
    log.info(f"Core sample: T = {len(df)}, "
             f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")
    return df


df = load_data()

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

# Controls
df["fx_cny"] = np.log(df["fx_china_lcu_per_usd"])
df["fx_thb"] = np.log(df["fx_thailand_lcu_per_usd"])
# nbs_break_dummy already in dataset

CONTROLS = ["ln_china_auto", "fx_cny", "fx_thb", "nbs_break_dummy"]


# =========================================================================
# 2. Helpers
# =========================================================================

def fit_hac(y, X, lags=None):
    Xc = add_constant(X, has_constant="add")
    n = len(Xc)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    return OLS(y, Xc, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})



def wald_scalar(w):
    """Return scalar (F, p-value) from a Wald test result, robust to statsmodels version."""
    import numpy as np
    s = np.asarray(w.statistic).ravel()
    p = np.asarray(w.pvalue).ravel()
    return float(s[0]), float(p[0])

def write_summary(fit, path: Path, title: str):
    with open(path, "w") as f:
        f.write(f"{title}\n{'=' * len(title)}\n\n")
        f.write(str(fit.summary()))
    log.info(f"Saved {path.name}")


# =========================================================================
# 3. Stationarity
# =========================================================================

def stationarity_table() -> pd.DataFrame:
    rows = []
    for v in ["lnNR", "lnOIL", "lnBD", "lnSR",
              "dlnNR", "dlnOIL", "dlnBD", "dlnSR"]:
        x = df[v].dropna()
        adf_p  = adfuller(x, autolag="AIC")[1]
        kpss_p = kpss(x, regression="c", nlags="auto")[1]
        rows.append({"variable": v,
                     "ADF_p": round(adf_p, 4),
                     "KPSS_p": round(kpss_p, 4),
                     "I(0) consensus": (adf_p < 0.05) and (kpss_p > 0.05)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stationarity.csv", index=False)
    log.info(f"Stationarity:\n{out.to_string(index=False)}")
    return out


# =========================================================================
# 4. H1: Petrochemical chain
# =========================================================================

def run_eq33():
    X = df[["lnOIL", "lnBD"] + CONTROLS]
    fit = fit_hac(df["lnSR"], X)
    write_summary(fit, TABLES / "eq33_petrochem_chain.txt",
                  "Equation (3.3): Petrochemical chain  ln SR = β0 + β1·lnOIL + β2·lnBD + controls")
    H1 = fit.wald_test("(lnOIL = 0), (lnBD = 0)", use_f=True)
    log.info(f"H1 joint Wald (β1=β2=0): F={wald_scalar(H1)[0]:.3f}, p={wald_scalar(H1)[1]:.4f}")
    return fit


# =========================================================================
# 5. H2: Substitution
# =========================================================================

def run_eq34():
    X = df[["lnOIL", "lnBD", "lnSR"] + CONTROLS]
    fit = fit_hac(df["lnNR"], X)
    write_summary(fit, TABLES / "eq34_substitution.txt",
                  "Equation (3.4): Substitution  ln NR = δ0 + δ1·lnOIL + δ2·lnBD + δ3·lnSR + controls")
    H2 = fit.wald_test("(lnSR = 0)", use_f=True)
    log.info(f"H2 (δ3 = 0): F={wald_scalar(H2)[0]:.3f}, p={wald_scalar(H2)[1]:.4f}")
    return fit


# =========================================================================
# 6. PSS Bounds Test
# =========================================================================

def bounds_f_test(p: int = 4, q: int = 4) -> dict:
    """Conditional ECM with cumulative partial sums for PSS bounds test."""
    d = df.copy()
    lr_cols = []
    for v, name in [("lnNR", "lnNR_lag1"),
                    ("lnOIL_pos", "lnOIL_pos_lag1"),
                    ("lnOIL_neg", "lnOIL_neg_lag1"),
                    ("lnSR_pos", "lnSR_pos_lag1"),
                    ("lnSR_neg", "lnSR_neg_lag1"),
                    ("lnBD", "lnBD_lag1")]:
        d[name] = d[v].shift(1)
        lr_cols.append(name)

    sr_cols = []
    for i in range(1, p):
        c = f"dlnNR_lag{i}"; d[c] = d["dlnNR"].shift(i); sr_cols.append(c)
    for i in range(0, q):
        for v in ["dlnOIL_pos", "dlnOIL_neg", "dlnSR_pos", "dlnSR_neg"]:
            c = f"{v}_lag{i}"; d[c] = d[v].shift(i); sr_cols.append(c)

    cols = lr_cols + sr_cols + CONTROLS
    d = d.dropna(subset=["dlnNR"] + cols).reset_index(drop=True)
    y = d["dlnNR"]
    X = add_constant(d[cols], has_constant="add")
    fit = OLS(y, X).fit(cov_type="HAC",
                        cov_kwds={"maxlags": int(np.ceil(len(d)**0.25))})

    restr = ", ".join([f"({c} = 0)" for c in lr_cols])
    F = fit.wald_test(restr, use_f=True)
    t_rho = fit.tvalues["lnNR_lag1"]

    # Pesaran-Shin-Smith (2001) Table CI(iii) Case III k=5 (5 regressors)
    crit_F = {
        "10pct": (2.26, 3.35),
        "5pct":  (2.62, 3.79),
        "1pct":  (3.41, 4.68),
    }
    crit_t = {
        "10pct": (-2.57, -3.86),
        "5pct":  (-2.86, -4.19),
        "1pct":  (-3.43, -4.79),
    }

    results = {
        "F_stat": wald_scalar(F)[0],
        "t_rho":  float(t_rho),
        "n_obs":  len(d),
        "F_crit": crit_F,
        "t_crit": crit_t,
        "fit":    fit,
    }

    # Save formatted output
    with open(TABLES / "bounds_test.txt", "w") as f:
        f.write("PSS Bounds Test for Cointegration (Asymmetric)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Sample: T = {len(d)}\n")
        f.write(f"H0: ρ = θ_OIL_pos = θ_OIL_neg = θ_SR_pos = θ_SR_neg = θ_BD = 0\n\n")
        f.write(f"F-statistic = {results['F_stat']:.3f}\n")
        f.write(f"t-statistic on ρ = {results['t_rho']:.3f}\n\n")
        f.write("Critical values (Pesaran-Shin-Smith 2001, k=5, Case III):\n")
        f.write(f"{'Level':<8} {'F: I(0)':<10} {'F: I(1)':<10} "
                f"{'t: I(0)':<10} {'t: I(1)':<10}\n")
        for lvl in ["10pct", "5pct", "1pct"]:
            f.write(f"{lvl:<8} "
                    f"{crit_F[lvl][0]:<10.2f} {crit_F[lvl][1]:<10.2f} "
                    f"{crit_t[lvl][0]:<10.2f} {crit_t[lvl][1]:<10.2f}\n")
        f.write("\nDecision rules:\n")
        f.write(" F > upper bound: reject no cointegration\n")
        f.write(" F < lower bound: cannot reject\n")
        f.write(" between: inconclusive\n")
        f.write(" Same logic for t-statistic (more negative is rejection)\n")

    log.info(f"Bounds test: F = {results['F_stat']:.3f}, t = {results['t_rho']:.3f}")
    return results


# =========================================================================
# 7. NARDL (eq. 3.5)  ->  H3 magnitude asymmetry
# =========================================================================

def run_nardl(p: int = 4, q: int = 4) -> dict:
    d = df.copy()
    # Build long-run regressors
    d["lnNR_lag1"]      = d["lnNR"].shift(1)
    d["lnOIL_pos_lag1"] = d["lnOIL_pos"].shift(1)
    d["lnOIL_neg_lag1"] = d["lnOIL_neg"].shift(1)
    d["lnSR_pos_lag1"]  = d["lnSR_pos"].shift(1)
    d["lnSR_neg_lag1"]  = d["lnSR_neg"].shift(1)
    d["lnBD_lag1"]      = d["lnBD"].shift(1)

    lr_cols = ["lnNR_lag1", "lnOIL_pos_lag1", "lnOIL_neg_lag1",
               "lnSR_pos_lag1", "lnSR_neg_lag1", "lnBD_lag1"]

    # Short-run lags
    sr_cols = []
    for i in range(1, p + 1):
        c = f"dlnNR_lag{i}"; d[c] = d["dlnNR"].shift(i); sr_cols.append(c)
    for i in range(0, q):
        for v in ["dlnOIL_pos", "dlnOIL_neg", "dlnSR_pos", "dlnSR_neg"]:
            c = f"{v}_lag{i}"; d[c] = d[v].shift(i); sr_cols.append(c)

    cols = lr_cols + sr_cols + CONTROLS
    d = d.dropna(subset=["dlnNR"] + cols).reset_index(drop=True)
    y = d["dlnNR"]
    X = add_constant(d[cols], has_constant="add")
    fit = OLS(y, X).fit(cov_type="HAC",
                        cov_kwds={"maxlags": int(np.ceil(len(d)**0.25))})

    write_summary(fit, TABLES / "nardl_baseline.txt",
                  f"Equation (3.5): Baseline NARDL ARDL({p}, {q})  ΔlnNR = ...")

    # Long-run coefficients
    rho = fit.params["lnNR_lag1"]
    LR = {
        "L_OIL_pos": -fit.params["lnOIL_pos_lag1"] / rho,
        "L_OIL_neg": -fit.params["lnOIL_neg_lag1"] / rho,
        "L_SR_pos":  -fit.params["lnSR_pos_lag1"]  / rho,
        "L_SR_neg":  -fit.params["lnSR_neg_lag1"]  / rho,
    }

    # Wald tests for asymmetry (H3)
    H3_oil = fit.wald_test("(lnOIL_pos_lag1 = lnOIL_neg_lag1)", use_f=True)
    H3_sr  = fit.wald_test("(lnSR_pos_lag1  = lnSR_neg_lag1)",  use_f=True)

    # Also test short-run asymmetry
    sr_oil_pos = [c for c in cols if c.startswith("dlnOIL_pos_lag")]
    sr_oil_neg = [c for c in cols if c.startswith("dlnOIL_neg_lag")]
    if len(sr_oil_pos) == len(sr_oil_neg):
        restr = " + ".join(sr_oil_pos) + " = " + " + ".join(sr_oil_neg)
        H3_sr_oil_short = fit.wald_test(restr, use_f=True)
    else:
        H3_sr_oil_short = None

    # Save asymmetry tests + LR coefficients
    with open(TABLES / "asymmetry_tests.txt", "w") as f:
        f.write("Asymmetry Tests (H3)\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Sample: T = {len(d)} (after lag-induced loss)\n\n")
        f.write("Long-run coefficients (recovered):\n")
        for k, v in LR.items():
            f.write(f"  {k:<12} = {v:+.4f}\n")
        f.write(f"\nLong-run asymmetry:\n")
        f.write(f"  H0: L_OIL+ = L_OIL-   F = {wald_scalar(H3_oil)[0]:.3f}, "
                f"p = {wald_scalar(H3_oil)[1]:.4f}\n")
        f.write(f"  H0: L_SR+  = L_SR-    F = {wald_scalar(H3_sr)[0]:.3f}, "
                f"p = {wald_scalar(H3_sr)[1]:.4f}\n")
        if H3_sr_oil_short:
            f.write(f"\nShort-run asymmetry (oil):\n")
            f.write(f"  F = {wald_scalar(H3_sr_oil_short)[0]:.3f}, "
                    f"p = {wald_scalar(H3_sr_oil_short)[1]:.4f}\n")

    pd.DataFrame([LR]).to_csv(TABLES / "long_run_coefs.csv", index=False)

    log.info(f"NARDL fitted: T={len(d)}, R²={fit.rsquared:.3f}")
    log.info(f"  Long-run: L_OIL+={LR['L_OIL_pos']:+.3f}, L_OIL-={LR['L_OIL_neg']:+.3f}")
    log.info(f"            L_SR+ ={LR['L_SR_pos']:+.3f}, L_SR- ={LR['L_SR_neg']:+.3f}")
    log.info(f"  H3 (oil): p = {wald_scalar(H3_oil)[1]:.4f}")
    log.info(f"  H3 (SR):  p = {wald_scalar(H3_sr)[1]:.4f}")

    return {"fit": fit, "LR": LR, "H3_oil": H3_oil, "H3_sr": H3_sr,
            "data": d, "p": p, "q": q}


# =========================================================================
# 8. Asymmetric Local Projections (eq. 3.7)  ->  H4 timing asymmetry
# =========================================================================

def run_local_projections() -> pd.DataFrame:
    """LP with shocks identified from recursive VAR (Cholesky OIL->BD->SR->NR)."""
    var_data = df[["dlnOIL", "dlnBD", "dlnSR", "dlnNR"]].dropna()
    var_fit = VAR(var_data).fit(maxlags=12, ic="aic")
    log.info(f"VAR(p={var_fit.k_ar}) fitted on T={len(var_data)}")

    # var_fit.resid in modern statsmodels is a DataFrame with original index;
    # we extract the underlying array and align by position with the last
    # len(resid) rows of df.
    resid_arr = np.asarray(var_fit.resid)
    n_resid = resid_arr.shape[0]

    df_lp = df.iloc[-n_resid:].reset_index(drop=True).copy()
    df_lp["eps_OIL"] = resid_arr[:, 0]
    df_lp["eps_BD"]  = resid_arr[:, 1]
    df_lp["eps_SR"]  = resid_arr[:, 2]
    df_lp["eps_NR"]  = resid_arr[:, 3]
    df_lp["eps_OIL_pos"] = df_lp["eps_OIL"].clip(lower=0)
    df_lp["eps_OIL_neg"] = df_lp["eps_OIL"].clip(upper=0)
    df_lp["eps_SR_pos"]  = df_lp["eps_SR"].clip(lower=0)
    df_lp["eps_SR_neg"]  = df_lp["eps_SR"].clip(upper=0)

    rows = []
    for h in range(1, 25):
        sub = df_lp.copy()
        sub["y_h"] = sub["lnNR"].shift(-h) - sub["lnNR"]
        cols = ["eps_OIL_pos", "eps_OIL_neg", "eps_SR_pos", "eps_SR_neg",
                "nbs_break_dummy"]
        for lag in (1, 2):
            for v in ["dlnNR", "dlnOIL", "dlnSR", "dlnBD"]:
                c = f"{v}_lag{lag}"
                sub[c] = sub[v].shift(lag)
                cols.append(c)
        sub = sub.dropna(subset=["y_h"] + cols)
        if len(sub) < 30:
            continue
        X = add_constant(sub[cols], has_constant="add")
        fit = OLS(sub["y_h"], X).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        for v in ["eps_OIL_pos", "eps_OIL_neg", "eps_SR_pos", "eps_SR_neg"]:
            rows.append({"horizon": h, "var": v,
                         "est": fit.params[v], "se": fit.bse[v],
                         "tstat": fit.tvalues[v]})

    lp = pd.DataFrame(rows)
    if len(lp) == 0:
        log.error("No LP horizons fitted; check sample size and lag structure.")
        return lp
    lp.to_csv(TABLES / "lp_results.csv", index=False)
    log.info(f"LP fitted across {lp['horizon'].nunique()} horizons")
    return lp


def plot_irf(lp: pd.DataFrame, shock: str, title: str, save: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for sign, ls, lab in [(f"{shock}_pos", "-", "Positive shock"),
                          (f"{shock}_neg", "--", "Negative shock")]:
        sub = lp[lp["var"] == sign]
        ax.plot(sub["horizon"], sub["est"], color="black", linestyle=ls,
                linewidth=1.5, label=lab)
        ax.fill_between(sub["horizon"], sub["est"] - 1.96 * sub["se"],
                        sub["est"] + 1.96 * sub["se"], color="grey", alpha=0.18)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Cumulative response of ln NR")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    log.info(f"Saved {save}")


# =========================================================================
# 9. Channel Decomposition (eq. 3.8)  ->  H5
# =========================================================================

def channel_decomposition(B: int = 5000) -> dict:
    Z = ["fx_cny", "fx_thb", "ln_china_auto", "nbs_break_dummy"]
    d = df.dropna(subset=["lnSR", "lnNR", "lnOIL"] + Z).reset_index(drop=True)

    # Stage 1: oil -> SR
    s1 = OLS(d["lnSR"], add_constant(d[["lnOIL"] + Z], has_constant="add")).fit()
    # Stage 2: oil + SR -> NR
    s2 = OLS(d["lnNR"], add_constant(d[["lnOIL", "lnSR"] + Z], has_constant="add")).fit()

    a1 = s1.params["lnOIL"]
    k1 = s2.params["lnOIL"]
    k2 = s2.params["lnSR"]
    direct, indirect = k1, a1 * k2
    total = direct + indirect
    medshare = indirect / total if total != 0 else np.nan

    # Bootstrap percentile CI for indirect effect and medshare
    n = len(d)
    boot_indirect = np.zeros(B)
    boot_medshare = np.zeros(B)
    for b in range(B):
        idx = RNG.integers(0, n, size=n)
        db = d.iloc[idx]
        try:
            s1b = OLS(db["lnSR"], add_constant(db[["lnOIL"] + Z], has_constant="add")).fit()
            s2b = OLS(db["lnNR"], add_constant(db[["lnOIL", "lnSR"] + Z], has_constant="add")).fit()
            ind_b = s1b.params["lnOIL"] * s2b.params["lnSR"]
            tot_b = s2b.params["lnOIL"] + ind_b
            boot_indirect[b] = ind_b
            boot_medshare[b] = ind_b / tot_b if tot_b != 0 else np.nan
        except Exception:
            boot_indirect[b] = np.nan
            boot_medshare[b] = np.nan

    boot_indirect = boot_indirect[~np.isnan(boot_indirect)]
    boot_medshare = boot_medshare[~np.isnan(boot_medshare)]
    ci_ind = np.quantile(boot_indirect, [0.025, 0.975])
    ci_med = np.quantile(boot_medshare, [0.025, 0.975])

    with open(TABLES / "channel_decomposition.txt", "w") as f:
        f.write("Channel Decomposition (H5)\n" + "=" * 30 + "\n\n")
        f.write(f"Sample: T = {n}\n")
        f.write(f"Bootstrap replications: B = {B}\n\n")
        f.write(f"α1 (oil -> SR):              {a1:+.4f}\n")
        f.write(f"κ1 (direct, oil -> NR):       {k1:+.4f}\n")
        f.write(f"κ2 (SR -> NR):                {k2:+.4f}\n\n")
        f.write(f"Direct effect:    {direct:+.4f}\n")
        f.write(f"Indirect effect:  {indirect:+.4f}  "
                f"95% CI [{ci_ind[0]:+.4f}, {ci_ind[1]:+.4f}]\n")
        f.write(f"Total effect:     {total:+.4f}\n")
        f.write(f"Mediation share:  {medshare:+.4f}  "
                f"95% CI [{ci_med[0]:+.4f}, {ci_med[1]:+.4f}]\n\n")
        f.write("H5 (mediation share > 0.5)?  " +
                ("YES" if ci_med[0] > 0.5 else "INCONCLUSIVE") + "\n")

    log.info(f"Channel: direct={direct:+.3f}, indirect={indirect:+.3f}, "
             f"medshare={medshare:.3f} [CI {ci_med[0]:.3f}, {ci_med[1]:.3f}]")
    return {"direct": direct, "indirect": indirect, "total": total,
            "medshare": medshare, "ci_med": ci_med}


# =========================================================================
# Main
# =========================================================================

def main():
    log.info("\n=== Stationarity tests ===")
    stationarity_table()

    log.info("\n=== H1: Petrochemical chain (Eq. 3.3) ===")
    run_eq33()

    log.info("\n=== H2: Substitution model (Eq. 3.4) ===")
    run_eq34()

    log.info("\n=== PSS Bounds Test ===")
    bounds_f_test()

    log.info("\n=== H3: NARDL with magnitude asymmetry (Eq. 3.5) ===")
    nardl_res = run_nardl(p=4, q=4)

    log.info("\n=== H4: Asymmetric Local Projections (Eq. 3.7) ===")
    lp = run_local_projections()
    plot_irf(lp, "eps_OIL", "Asymmetric LP: oil shock", FIGURES / "lp_oil.pdf")
    plot_irf(lp, "eps_SR", "Asymmetric LP: synthetic-rubber shock", FIGURES / "lp_sr.pdf")

    log.info("\n=== H5: Channel Decomposition (Eq. 3.8) ===")
    channel_decomposition(B=5000)

    log.info("\n[DONE] Phase 1 complete.  See tables/ and figures/ for outputs.")


if __name__ == "__main__":
    main()
