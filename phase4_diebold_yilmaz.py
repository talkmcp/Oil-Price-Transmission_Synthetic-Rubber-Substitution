"""
=============================================================================
 Phase 4: Diebold-Yilmaz Connectedness Analysis
 
 Tests how shocks propagate through a 6-variable system covering the
 full supply chain from upstream petrochemicals to Asian rubber markets:
 
   1. Oil       (OPEC crude)            -- upstream cost shock
   2. Butadiene (petrochemical)         -- intermediate
   3. SR        (China SBR)             -- substitute
   4. NR China  (Shanghai RSS)          -- main market
   5. NR Thai   (Trang RSS)             -- producer market
   6. TOCOM     (Tokyo rubber futures)  -- regional discovery

 Methodology:
   - Generalized FEVD (Pesaran-Shin 1998), order-invariant
   - Diebold-Yilmaz (2012, IJF) total/directional connectedness
   - Rolling window for time-varying connectedness
   - Frequency-domain decomposition (Baruník-Křehlík 2018)

 Outputs:
   - tables/dy_connectedness_table.csv      -- full connectedness matrix
   - tables/dy_summary.txt                  -- TCI, net spillovers
   - tables/dy_rolling_tci.csv              -- time-varying TCI
   - figures/dy_network_baseline.pdf        -- network spillover plot
   - figures/dy_rolling_tci.pdf             -- time-varying TCI
   - figures/dy_net_spillovers.pdf          -- net direction by variable
=============================================================================
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dy")

DATA_FILE = Path("data/rubber_petrochemical_monthly_model_dataset_v3.xlsx")
TABLES    = Path("tables");  TABLES.mkdir(exist_ok=True)
FIGURES   = Path("figures"); FIGURES.mkdir(exist_ok=True)


# =========================================================================
# 1. Load + prepare 6-variable system
# =========================================================================

def load_system() -> pd.DataFrame:
    """Build returns matrix for 6 variables."""
    df = pd.read_excel(DATA_FILE, sheet_name="Monthly_Model_Data")
    if pd.api.types.is_numeric_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], unit="D", origin="1899-12-30")
    df = df[df["core_model_flag"] == 1].sort_values("date").reset_index(drop=True)
    
    # Build returns (log differences)
    df["r_OIL"]   = df["dln_oil_opec"]
    df["r_BD"]    = df["dln_butadiene"]
    df["r_SR"]    = df["dln_sbr_monthly"]
    df["r_NR_CN"] = df["dln_nr_rss"]
    df["r_NR_TH"] = np.log(df["nr_tra_rss_thbkg"]).diff()
    df["r_TOCOM"] = np.log(df["tocom_rubber_current_jpykg"]).diff()
    
    out = df[["date", "r_OIL", "r_BD", "r_SR", "r_NR_CN",
              "r_NR_TH", "r_TOCOM"]].dropna().reset_index(drop=True)
    log.info(f"DY system: T = {len(out)}, "
             f"{out['date'].min():%Y-%m} to {out['date'].max():%Y-%m}")
    return out


VARS = ["OIL", "BD", "SR", "NR_CN", "NR_TH", "TOCOM"]
RETS = [f"r_{v}" for v in VARS]


# =========================================================================
# 2. Generalized FEVD (Pesaran-Shin 1998)
# =========================================================================

def generalized_fevd(var_fit, H: int = 12) -> np.ndarray:
    """
    Compute generalized forecast error variance decomposition.
    
    GFEVD is invariant to variable ordering, unlike Cholesky FEVD.
    
    Returns: K x K matrix where element (i, j) is the share of variable
             i's H-step forecast error variance attributable to shocks
             in variable j.
    """
    Sigma = np.asarray(var_fit.sigma_u)        # K x K residual covariance
    K = Sigma.shape[0]
    
    # Compute MA representation: y_{t+h} = ... + sum_{l=0}^{h} Phi_l · e_{t+h-l}
    # statsmodels gives ma_rep up to maxh
    ma = np.asarray(var_fit.ma_rep(maxn=H))    # shape: (H+1, K, K)
    
    sigma_jj = np.diag(Sigma)
    fevd = np.zeros((K, K))
    
    for i in range(K):
        for j in range(K):
            num = 0.0
            for h in range(H):
                # (Phi_h · Sigma)_{ij}^2 / sigma_{jj}
                phi_sigma = ma[h] @ Sigma
                num += (phi_sigma[i, j]) ** 2 / sigma_jj[j]
            den = 0.0
            for h in range(H):
                den += (ma[h] @ Sigma @ ma[h].T)[i, i]
            fevd[i, j] = num / den
    
    # Normalize each row to sum to 1 (Diebold-Yilmaz normalization)
    fevd_norm = fevd / fevd.sum(axis=1, keepdims=True)
    return fevd_norm


# =========================================================================
# 3. Connectedness measures
# =========================================================================

def connectedness_table(fevd: np.ndarray, vars: list[str]) -> dict:
    """
    Build the Diebold-Yilmaz connectedness table.
    
    Following the convention:
      Row i, Column j = share of i's variance from j's shocks
      Diagonal = own-variance share
      
    From j: column sum (excl. diagonal) = total variance i transmits TO others
    To j:   row sum (excl. diagonal)    = total variance i receives FROM others
    Net = From - To
    """
    K = len(vars)
    df = pd.DataFrame(fevd * 100, index=vars, columns=vars)  # in percent
    
    # From others (column sum excluding diagonal): how much i gives away
    from_others = (fevd.sum(axis=0) - np.diag(fevd)) * 100
    # To others (row sum excluding diagonal): how much i receives
    to_others = (fevd.sum(axis=1) - np.diag(fevd)) * 100
    
    # Net spillover: positive = net transmitter, negative = net receiver
    net_spillover = from_others - to_others
    
    # Total Connectedness Index
    total_conn = (fevd.sum() - np.trace(fevd)) / K * 100
    
    df["FROM others"] = to_others    # what i RECEIVES (sum across row)
    
    # Add bottom rows: TO and NET
    bot = pd.DataFrame([
        list(from_others) + [from_others.sum() / K * 100],
        list(net_spillover) + [total_conn]
    ], index=["TO others", "NET (FROM-TO)"], columns=df.columns)
    
    table = pd.concat([df, bot])
    
    return {
        "table": table,
        "total_conn": total_conn,
        "from_others": from_others,
        "to_others": to_others,
        "net_spillover": net_spillover,
    }


# =========================================================================
# 4. Baseline static connectedness
# =========================================================================

def baseline_connectedness(data: pd.DataFrame, p: int = 4, H: int = 12):
    """Estimate baseline (static) connectedness on the full sample."""
    Y = data[RETS].dropna().values
    log.info(f"VAR fitting on T = {len(Y)} observations")
    
    var_fit = VAR(Y).fit(maxlags=p, ic="aic")
    log.info(f"VAR(p = {var_fit.k_ar}) selected by AIC")
    
    fevd = generalized_fevd(var_fit, H=H)
    return connectedness_table(fevd, VARS), var_fit, fevd


# =========================================================================
# 5. Rolling-window TCI
# =========================================================================

def rolling_connectedness(data: pd.DataFrame, window: int = 60,
                          p: int = 2, H: int = 12) -> pd.DataFrame:
    """Rolling-window connectedness with shorter VAR lag for stability."""
    Y_all = data[RETS].dropna().values
    dates = data.loc[data[RETS].dropna().index, "date"].values
    n = len(Y_all)
    
    rows = []
    for t in range(window, n + 1):
        Y_w = Y_all[t - window: t]
        try:
            fit = VAR(Y_w).fit(maxlags=p, ic="aic")
            fevd = generalized_fevd(fit, H=H)
            ct = connectedness_table(fevd, VARS)
            rows.append({
                "date": dates[t - 1],
                "TCI": ct["total_conn"],
                **{f"NET_{v}": ct["net_spillover"][i] for i, v in enumerate(VARS)},
                **{f"FROM_{v}": ct["from_others"][i] for i, v in enumerate(VARS)},
                **{f"TO_{v}": ct["to_others"][i] for i, v in enumerate(VARS)},
            })
        except Exception as ex:
            log.warning(f"  t={t} ({dates[t-1]}): {ex}")
            continue
    
    return pd.DataFrame(rows)


# =========================================================================
# 6. Visualization
# =========================================================================

def plot_network(ct: dict, save: Path):
    """Draw spillover network with arrow widths proportional to flow."""
    K = len(VARS)
    fevd_pct = ct["table"].iloc[:K, :K].values  # already in percent
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Place nodes in a circle
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False) - np.pi / 2
    radius = 1.0
    coords = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    
    # Compute node "size" proportional to net spillover
    net = ct["net_spillover"]
    node_size = 600 + 80 * np.abs(net)
    
    # Draw arrows: flow from j to i if fevd[i, j] > threshold
    threshold = 5.0   # only show flows >= 5%
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            flow = fevd_pct[i, j]
            if flow < threshold:
                continue
            x_from, y_from = coords[j]
            x_to,   y_to   = coords[i]
            
            # Shorten arrow to not overlap nodes
            dx, dy = x_to - x_from, y_to - y_from
            dist = np.sqrt(dx**2 + dy**2)
            if dist == 0: continue
            shrink = 0.12
            x_from2 = x_from + shrink * dx / dist
            y_from2 = y_from + shrink * dy / dist
            x_to2   = x_to   - shrink * dx / dist
            y_to2   = y_to   - shrink * dy / dist
            
            # Width proportional to flow
            width = 0.5 + 0.05 * flow
            alpha = min(1.0, 0.3 + 0.015 * flow)
            
            ax.annotate("",
                xy=(x_to2, y_to2), xytext=(x_from2, y_from2),
                arrowprops=dict(arrowstyle="->", color="black",
                               lw=width, alpha=alpha))
    
    # Draw nodes
    for i, (x, y) in enumerate(coords):
        color = "white" if net[i] < 0 else "lightgray"
        ax.scatter(x, y, s=node_size[i], c=color,
                   edgecolors="black", linewidths=1.5, zorder=10)
        # Label
        label_text = f"{VARS[i]}\n({net[i]:+.0f})"
        ax.annotate(label_text, (x, y), ha="center", va="center",
                   fontsize=11, fontweight="bold", zorder=20)
    
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Diebold-Yilmaz spillover network\n"
                 f"(arrows: flows ≥ 5%; node label = net spillover %; "
                 f"TCI = {ct['total_conn']:.1f}%)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save)
    plt.savefig(save.with_suffix(".png"), dpi=130)
    plt.close()
    log.info(f"Saved {save.name}")


def plot_rolling_tci(rolling: pd.DataFrame, save: Path):
    """Plot rolling Total Connectedness Index over time."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(rolling["date"], rolling["TCI"], color="black", lw=1.6)
    ax.fill_between(rolling["date"], rolling["TCI"].mean() - rolling["TCI"].std(),
                    rolling["TCI"].mean() + rolling["TCI"].std(),
                    color="grey", alpha=0.15)
    ax.axhline(rolling["TCI"].mean(), color="black", ls=":", lw=0.8)
    
    events = [("2008-09", "Lehman"), ("2014-06", "Oil collapse"),
              ("2018-03", "EV growth"), ("2020-03", "COVID"),
              ("2022-02", "Russia–Ukraine")]
    for d, label in events:
        x = pd.Timestamp(d)
        ax.axvline(x, color="grey", lw=0.5, ls=":", alpha=0.6)
        ax.text(x, ax.get_ylim()[1] * 0.97, label, rotation=90,
                fontsize=8, alpha=0.7, va="top")
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Total Connectedness Index (%)")
    ax.set_title("Time-varying total connectedness in rubber-petrochemical system\n"
                 "(60-month rolling window, generalized FEVD H = 12)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save)
    plt.savefig(save.with_suffix(".png"), dpi=130)
    plt.close()
    log.info(f"Saved {save.name}")


def plot_net_spillovers(rolling: pd.DataFrame, save: Path):
    """Plot net spillover for each variable over time."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    axes = axes.flatten()
    
    for i, v in enumerate(VARS):
        ax = axes[i]
        col = f"NET_{v}"
        # Net positive = transmitter, negative = receiver
        ax.plot(rolling["date"], rolling[col], color="black", lw=1.4)
        ax.fill_between(rolling["date"], 0, rolling[col],
                        where=(rolling[col] >= 0),
                        color="black", alpha=0.15)
        ax.fill_between(rolling["date"], 0, rolling[col],
                        where=(rolling[col] < 0),
                        color="grey", alpha=0.4)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"{v}: net spillover", fontsize=11)
        ax.set_ylabel("Net %")
        ax.grid(alpha=0.3)
        
    for ax in axes[-2:]:
        ax.set_xlabel("Time")
    
    plt.suptitle("Net spillovers by variable\n"
                 "(positive = net TRANSMITTER, negative = net RECEIVER)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save)
    plt.savefig(save.with_suffix(".png"), dpi=130)
    plt.close()
    log.info(f"Saved {save.name}")


# =========================================================================
# 7. Subsample analysis
# =========================================================================

def subsample_analysis(data: pd.DataFrame):
    """Compare connectedness in distinct historical periods."""
    periods = [
        ("Pre-crisis (2003-2008)",  "2003-04", "2008-08"),
        ("Stimulus (2009-2013)",     "2009-01", "2013-12"),
        ("Oil collapse (2014-2017)", "2014-06", "2017-12"),
        ("EV growth (2018-2019)",    "2018-01", "2019-12"),
        ("COVID + EV (2020-2026)",   "2020-03", "2026-03"),
    ]
    
    rows = []
    for label, start, end in periods:
        sub = data[(data["date"] >= start) & (data["date"] <= end)]
        if len(sub) < 24:
            continue
        try:
            ct, _, _ = baseline_connectedness(sub, p=2, H=12)
            row = {"period": label, "T": len(sub), "TCI": ct["total_conn"]}
            for i, v in enumerate(VARS):
                row[f"NET_{v}"] = ct["net_spillover"][i]
            rows.append(row)
        except Exception as ex:
            log.warning(f"  {label}: {ex}")
            continue
    
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "dy_subsample.csv", index=False)
    
    log.info("\nSubsample TCI:")
    for _, row in out.iterrows():
        log.info(f"  {row['period']:<28}  TCI = {row['TCI']:5.1f}%  "
                 f"NET(NR_TH) = {row['NET_NR_TH']:+5.1f}, "
                 f"NET(NR_CN) = {row['NET_NR_CN']:+5.1f}")
    return out


# =========================================================================
# Main
# =========================================================================

def main():
    log.info("=== Phase 4: Diebold-Yilmaz Connectedness ===\n")
    
    data = load_system()
    
    log.info("\n--- Step 1: Baseline connectedness (full sample) ---")
    ct, var_fit, fevd = baseline_connectedness(data, p=4, H=12)
    log.info(f"  Total Connectedness Index = {ct['total_conn']:.1f}%")
    
    print("\n=== Connectedness Table (in percent) ===")
    print(ct["table"].round(2).to_string())
    
    ct["table"].to_csv(TABLES / "dy_connectedness_table.csv")
    
    # Summary
    with open(TABLES / "dy_summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Diebold-Yilmaz Connectedness Analysis (Phase 4)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Variables: {', '.join(VARS)}\n")
        f.write(f"  OIL    = OPEC crude\n")
        f.write(f"  BD     = Butadiene (petrochemical)\n")
        f.write(f"  SR     = Synthetic rubber (China SBR)\n")
        f.write(f"  NR_CN  = Natural rubber (Shanghai RSS)\n")
        f.write(f"  NR_TH  = Natural rubber (Trang RSS, Thailand)\n")
        f.write(f"  TOCOM  = Tokyo rubber futures (current month)\n\n")
        f.write(f"Sample: {data['date'].min():%Y-%m} to {data['date'].max():%Y-%m}\n")
        f.write(f"T = {len(data)}, VAR(p={var_fit.k_ar}), H = 12\n\n")
        f.write(f"Total Connectedness Index = {ct['total_conn']:.2f}%\n\n")
        f.write(f"{'Variable':<10}{'TO others':>12}{'FROM others':>14}{'NET':>10}\n")
        f.write("-" * 46 + "\n")
        for i, v in enumerate(VARS):
            net_label = "TRANSMITTER" if ct["net_spillover"][i] > 0 else "receiver"
            f.write(f"{v:<10}{ct['from_others'][i]:>12.2f}"
                    f"{ct['to_others'][i]:>14.2f}{ct['net_spillover'][i]:>+10.2f}"
                    f"  ({net_label})\n")
        f.write("\nNet positive = net TRANSMITTER of variance\n")
        f.write("Net negative = net RECEIVER of variance\n")
    
    log.info("\n--- Step 2: Plot baseline network ---")
    plot_network(ct, FIGURES / "dy_network_baseline.pdf")
    
    log.info("\n--- Step 3: Rolling-window connectedness ---")
    rolling = rolling_connectedness(data, window=60, p=2, H=12)
    rolling.to_csv(TABLES / "dy_rolling_tci.csv", index=False)
    log.info(f"  Rolling: {len(rolling)} window estimates")
    log.info(f"  TCI range: [{rolling['TCI'].min():.1f}%, "
             f"{rolling['TCI'].max():.1f}%], mean = {rolling['TCI'].mean():.1f}%")
    
    plot_rolling_tci(rolling, FIGURES / "dy_rolling_tci.pdf")
    plot_net_spillovers(rolling, FIGURES / "dy_net_spillovers.pdf")
    
    log.info("\n--- Step 4: Subsample analysis ---")
    subsample_analysis(data)
    
    log.info("\n[DONE] Phase 4 complete.  See tables/ and figures/ for outputs.")


if __name__ == "__main__":
    main()
