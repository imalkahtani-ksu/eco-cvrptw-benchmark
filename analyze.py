import os, json, itertools
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGOS = ["GLS", "IGLS", "RGLS", "ALNS", "HGLS"]
PROP = "HGLS"
OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "font.size": 9, "axes.linewidth": 0.8,
                     "savefig.dpi": 400, "savefig.bbox": "tight"})
COL = {"GLS": "#7f7f7f", "IGLS": "#1f77b4", "RGLS": "#d62728",
       "ALNS": "#2ca02c", "HGLS": "#000000"}
MK = {"GLS": "s", "IGLS": "^", "RGLS": "v", "ALNS": "D", "HGLS": "o"}


def cls(name):
    for p in ("RC1", "RC2", "C1", "C2", "R1", "R2"):
        if name.startswith(p):
            return p
    return "?"


def load(path):
    df = pd.read_csv(path)
    df = df[df["error"].isna() | (df["error"] == "")]
    df["class"] = df["instance"].map(cls)
    df["family"] = df["class"].str[:-1]
    return df


def aggregate(df):
    g = (df.groupby(["instance", "class", "family", "algo"])
           .agg(nv=("nv", "mean"), dist=("dist", "mean"), fuel=("fuel", "mean"),
                emis=("emis", "mean"), IF=("IF", "mean"), plow=("plow", "mean"),
                pmed=("pmed", "mean"), phigh=("phigh", "mean"),
                lam=("lam", "mean"), runtime=("runtime", "mean"),
                fuel_best=("fuel", "min"), fuel_sd=("fuel", "std"))
           .reset_index())
    best = g.groupby("instance")["fuel"].transform("min")
    g["RelGapZ"] = 100 * (g["fuel"] - best) / best
    bd = g.groupby("instance")["dist"].transform("min")
    g["RelGapD"] = 100 * (g["dist"] - bd) / bd
    return g


def paired_tests(g):
    piv = g.pivot(index="instance", columns="algo", values="fuel").dropna()
    rows, pvals = [], []
    for a in ALGOS:
        if a == PROP:
            continue
        x, y = piv[a].values, piv[PROP].values
        wins = int((y < x - 1e-9).sum())
        ties = int((np.abs(y - x) <= 1e-9).sum())
        loss = int((y > x + 1e-9).sum())
        try:
            st, p = stats.wilcoxon(x, y, alternative="greater")
        except ValueError:
            st, p = float("nan"), 1.0
        d = x - y
        nz = d[d != 0]
        pos = float((nz > 0).sum()); neg = float((nz < 0).sum())
        rb = (pos - neg) / max(1.0, (pos + neg))
        pen = 100 * (x - y) / y
        bs = np.array([np.median(np.random.choice(pen, len(pen), replace=True))
                       for _ in range(4000)])
        rows.append(dict(comparator=a, n=len(x), wins=wins, ties=ties,
                         losses=loss, med_gap=float(np.median(pen)),
                         ci_lo=float(np.percentile(bs, 2.5)),
                         ci_hi=float(np.percentile(bs, 97.5)),
                         p_raw=float(p), rb=float(rb)))
        pvals.append(p)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    prev = 0.0
    for r, i in enumerate(order):
        v = min(1.0, (m - r) * pvals[i])
        prev = max(prev, v)
        adj[i] = prev
    for i, row in enumerate(rows):
        row["p_holm"] = adj[i]
    return pd.DataFrame(rows), piv


def friedman_all(g, col="fuel"):
    piv = g.pivot(index="instance", columns="algo", values=col).dropna()
    arrs = [piv[a].values for a in ALGOS]
    chi, p = stats.friedmanchisquare(*arrs)
    ranks = piv[ALGOS].rank(axis=1).mean().to_dict()
    return dict(chi2=float(chi), df=len(ALGOS) - 1, p=float(p), n=len(piv),
                mean_rank={k: float(v) for k, v in ranks.items()})


def decomposition(g):
    rows = []
    allgaps = []
    for a in ALGOS:
        if a == PROP:
            continue
        sub = g[g.algo.isin([a, PROP])]
        pz = sub.pivot(index="instance", columns="algo", values="fuel").dropna()
        pd_ = sub.pivot(index="instance", columns="algo", values="dist").dropna()
        dz = 100 * (pz[a] - pz[PROP]) / pz[PROP]
        dd = 100 * (pd_[a] - pd_[PROP]) / pd_[PROP]
        gap = (dz - dd).values
        allgaps.append(gap)
        bs = np.array([np.median(np.random.choice(gap, len(gap), replace=True))
                       for _ in range(4000)])
        rows.append(dict(comparator=a, med_dD=float(np.median(dd)),
                         med_dZ=float(np.median(dz)),
                         med_gap=float(np.median(gap)),
                         ci_lo=float(np.percentile(bs, 2.5)),
                         ci_hi=float(np.percentile(bs, 97.5))))
    pooled = np.concatenate(allgaps)
    k = int((pooled > 0).sum()); n = int((pooled != 0).sum())
    sp = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
    return pd.DataFrame(rows), dict(k=k, n=n, p=float(sp))


def intensity_model(g):
    d = g.groupby(["family", "algo"])[["IF", "phigh", "plow", "lam"]].mean().reset_index()
    r1 = stats.spearmanr(d["phigh"], d["IF"])
    r2 = stats.spearmanr(d["plow"], d["IF"])
    r3 = stats.spearmanr(d["lam"], d["IF"])
    X = np.column_stack([np.ones(len(d)), d["phigh"], d["lam"]])
    y = d["IF"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(n=len(d),
                sp_phigh=[float(r1.statistic), float(r1.pvalue)],
                sp_plow=[float(r2.statistic), float(r2.pvalue)],
                sp_lam=[float(r3.statistic), float(r3.pvalue)],
                beta=[float(b) for b in beta], r2=float(ss)), d


def fig_iso(g, path):
    d = g.groupby(["class", "algo"])[["dist", "fuel", "IF"]].median().reset_index()
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    lo, hi = d["dist"].min() * 0.90, d["dist"].max() * 1.08
    xs = np.linspace(lo, hi, 50)
    levels = np.linspace(d["IF"].min(), d["IF"].max(), 5)
    for t, lev in enumerate(levels):
        ax.plot(xs, lev * xs, lw=0.6, ls=":", color="#b0b0b0", zorder=1)
        xf = lo + (0.30 + 0.16 * t) * (hi - lo)
        ax.annotate("%.3f" % lev, (xf, lev * xf), fontsize=6.2, color="#808080",
                    ha="center", va="bottom", rotation=0,
                    bbox=dict(fc="white", ec="none", pad=0.6))
    for a in ALGOS:
        s = d[d.algo == a]
        ax.scatter(s["dist"], s["fuel"], s=30, marker=MK[a], color=COL[a],
                   label=a, zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Distance D (km)")
    ax.set_ylabel("Fuel Z (L)")
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper left",
              handletextpad=0.3, columnspacing=1.0)
    ax.set_title("Class medians with iso-intensity lines (L/km)", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def fig_scatter(g, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    lim = 0
    for a in ALGOS:
        if a == PROP:
            continue
        sub = g[g.algo.isin([a, PROP])]
        pz = sub.pivot(index="instance", columns="algo", values="fuel").dropna()
        pdd = sub.pivot(index="instance", columns="algo", values="dist").dropna()
        dz = 100 * (pz[a] - pz[PROP]) / pz[PROP]
        dd = 100 * (pdd[a] - pdd[PROP]) / pdd[PROP]
        ax.scatter(dd, dz, s=13, alpha=0.65, marker=MK[a], color=COL[a], label=a,
                   edgecolor="none")
        lim = max(lim, float(np.nanmax(np.abs(dd))), float(np.nanmax(np.abs(dz))))
    lim *= 1.08
    ax.plot([-lim, lim], [-lim, lim], lw=0.8, color="#333333", ls="--",
            label="45$^\\circ$")
    ax.axhline(0, lw=0.5, color="#999999")
    ax.axvline(0, lw=0.5, color="#999999")
    ax.set_xlabel("Distance penalty vs HGLS (%)")
    ax.set_ylabel("Fuel penalty vs HGLS (%)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def fig_gaps(g, path):
    fams = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
    piv = g.groupby(["class", "algo"])["RelGapZ"].median().unstack()
    piv = piv.reindex(fams)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    w = 0.15
    xs = np.arange(len(fams))
    for i, a in enumerate(ALGOS):
        ax.bar(xs + (i - 2) * w, piv[a].values, w, label=a, color=COL[a],
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(fams)
    ax.set_ylabel("Median relative fuel gap (%)")
    ax.set_xlabel("Solomon class")
    ax.legend(frameon=False, fontsize=7, ncol=5)
    ax.axhline(0, lw=0.6, color="black")
    fig.savefig(path)
    plt.close(fig)


def fig_sensitivity(path, csvp):
    if not os.path.exists(csvp) or os.path.getsize(csvp) < 50:
        return False
    d = load(csvp)
    if d.empty:
        return False
    d["spread"] = d["phi"].map(lambda s: round(
        tuple(float(v) for v in s.strip("() ").split(","))[2]
        - tuple(float(v) for v in s.strip("() ").split(","))[0], 3))
    a = d.groupby(["spread", "algo"])[["fuel", "IF", "plow", "phigh"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for al in sorted(a["algo"].unique()):
        s = a[a.algo == al].sort_values("spread")
        axes[0].plot(s["spread"], s["IF"], marker=MK.get(al, "o"),
                     color=COL.get(al, "#555"), lw=1.0, ms=4, label=al)
        axes[1].plot(s["spread"], s["plow"], marker=MK.get(al, "o"),
                     color=COL.get(al, "#555"), lw=1.0, ms=4, label=al)
    axes[0].set_xlabel("Speed-factor spread $\\phi_{high}-\\phi_{low}$")
    axes[0].set_ylabel("Fuel intensity (L/km)")
    axes[1].set_xlabel("Speed-factor spread $\\phi_{high}-\\phi_{low}$")
    axes[1].set_ylabel("Distance at low speed (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)
    return True


def main(path="results_main.csv", sens="results_sens.csv"):
    df = load(path)
    g = aggregate(df)
    g.to_csv("agg_per_instance.csv", index=False)

    tests, piv = paired_tests(g)
    fr_fuel = friedman_all(g, "fuel")
    fr_if = friedman_all(g, "IF")
    dec, sign = decomposition(g)
    imod, cent = intensity_model(g)

    fam = (g.groupby(["family", "algo"])
             .agg(n=("instance", "count"), dist=("dist", "median"),
                  fuel=("fuel", "median"), IF=("IF", "median"),
                  gap=("RelGapZ", "median"), nv=("nv", "mean"),
                  plow=("plow", "mean"), pmed=("pmed", "mean"),
                  phigh=("phigh", "mean"), lam=("lam", "mean"),
                  rt=("runtime", "mean")).reset_index())
    fam.to_csv("agg_family.csv", index=False)
    cent.to_csv("agg_centroid.csv", index=False)
    tests.to_csv("stats_wilcoxon.csv", index=False)
    dec.to_csv("stats_decomposition.csv", index=False)

    summary = dict(n_instances=int(g["instance"].nunique()),
                   friedman_fuel=fr_fuel, friedman_IF=fr_if,
                   sign_test=sign, intensity_model=imod,
                   overall=g.groupby("algo")[["dist", "fuel", "IF", "nv",
                                              "RelGapZ", "runtime"]]
                            .median().round(4).to_dict())
    json.dump(summary, open("stats_summary.json", "w"), indent=1)

    fig_iso(g, "fig3_iso.png")
    fig_scatter(g, "fig4_scatter.png")
    fig_gaps(g, "fig5_gaps.png")
    has_s = fig_sensitivity("fig6_sensitivity.png", sens)

    print("instances:", g["instance"].nunique())
    print("\n-- median over all instances --")
    print(g.groupby("algo")[["nv", "dist", "fuel", "IF", "RelGapZ", "runtime"]]
            .median().round(3).reindex(ALGOS))
    print("\n-- paired Wilcoxon vs HGLS (fuel) --")
    print(tests.round(4).to_string(index=False))
    print("\n-- Friedman (fuel) --", fr_fuel)
    print("-- Friedman (IF)   --", fr_if)
    print("\n-- decomposition --")
    print(dec.round(3).to_string(index=False))
    print("sign test:", sign)
    print("\n-- intensity model --", imod)
    print("\nsensitivity figure:", has_s)


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:] or []))


def signature(g):
    d = g.copy()
    for c in ("dist", "IF", "plow", "phigh", "lam", "fuel", "nv"):
        d["d_" + c] = d.groupby("instance")[c].transform(
            lambda s: (s - s.mean()) / s.mean() * 100 if s.mean() else 0.0)
    rows = []
    for a in ALGOS:
        s = d[d.algo == a]
        row = {"algo": a}
        for c in ("dist", "IF", "fuel", "plow", "nv"):
            v = s["d_" + c].values
            try:
                _, p = stats.wilcoxon(v)
            except ValueError:
                p = 1.0
            row[c] = float(np.mean(v))
            row[c + "_p"] = float(p)
            row[c + "_sd"] = float(np.std(v, ddof=1))
        rows.append(row)
    sig_df = pd.DataFrame(rows)

    fried = {}
    for c in ("fuel", "dist", "IF", "plow", "nv"):
        piv = g.pivot(index="instance", columns="algo", values=c).dropna()
        try:
            chi, p = stats.friedmanchisquare(*[piv[a].values for a in ALGOS])
        except ValueError:
            chi, p = float("nan"), 1.0
        fried[c] = dict(chi2=float(chi), df=len(ALGOS) - 1, p=float(p),
                        n=int(len(piv)),
                        mean_rank={k: float(v) for k, v in
                                   piv[ALGOS].rank(axis=1).mean().items()})
    r, p = stats.pearsonr(d["d_dist"], d["d_IF"])
    coupling = dict(r=float(r), p=float(p), n=int(len(d)),
                    sd_dist=float(d["d_dist"].std()),
                    sd_IF=float(d["d_IF"].std()),
                    sd_fuel=float(d["d_fuel"].std()))
    return sig_df, fried, coupling


def fig_signature(g, path):
    sig_df, _, _ = signature(g)
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for _, r in sig_df.iterrows():
        a = r["algo"]
        ax.errorbar(r["dist"], r["IF"],
                    xerr=r["dist_sd"] / np.sqrt(56), yerr=r["IF_sd"] / np.sqrt(56),
                    fmt=MK[a], color=COL[a], ms=7, capsize=2.5, lw=0.9,
                    markeredgecolor="white", markeredgewidth=0.5, label=a)
    ax.axhline(0, lw=0.6, color="#999999", zorder=0)
    ax.axvline(0, lw=0.6, color="#999999", zorder=0)
    lim = max(abs(np.r_[ax.get_xlim(), ax.get_ylim()])) * 1.05
    xs = np.linspace(-lim, lim, 20)
    ax.plot(xs, -xs, ls="--", lw=0.7, color="#444444", zorder=0)
    ax.annotate("constant fuel", (lim * 0.55, -lim * 0.55), fontsize=6.5,
                color="#444444", rotation=-45, ha="center", va="bottom")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Distance deviation from instance mean (%)")
    ax.set_ylabel("Intensity deviation from instance mean (%)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
    fig.savefig(path)
    plt.close(fig)
